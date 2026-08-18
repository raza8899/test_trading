"""
Zerodha Kite Connect + OpenAI autonomous intraday bot (V3)

Design
------
1. Daily Kite access token is obtained using login.py.
2. Download Zerodha's NSE instrument dump once at startup.
3. Dynamically build a broad NSE cash-equity universe (no universe.txt).
4. Subscribe to the broad universe through KiteTicker WebSocket in QUOTE mode.
5. Rank active "stocks in play" from live WebSocket ticks:
      - traded value / liquidity
      - absolute intraday move
      - intraday range expansion
6. Enrich only the preliminary shortlist with ONE /quote REST request:
      - bid/ask spread
      - circuit limits
7. Fetch 5-minute historical candles only for the strongest candidates.
8. Apply deterministic setup rules:
      - 15-minute opening range
      - fresh breakout / breakdown
      - VWAP
      - EMA 9/20
      - RSI
      - ATR
      - recent relative volume
      - candle close quality
      - NIFTY regime
9. GPT-5.6 is a FINAL REVIEWER only.
10. Hard Python risk engine determines quantity, stop, target and limits.
11. Place MIS entry with automatic market protection.
12. Immediately place exchange-side SL-M protection with market protection.
13. Exit at target, stop, daily kill switch, or 15:10 IST.

Important
---------
- LIVE_TRADING defaults to false.
- No strategy can guarantee profit.
- Use the static IPv4 registered in your Kite Connect developer account
  before enabling LIVE_TRADING.
"""

from __future__ import annotations

import ipaddress
import hashlib
import json
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests
import signal
from dotenv import load_dotenv
from kiteconnect import KiteConnect, KiteTicker
from openai import OpenAI
from pydantic import BaseModel, Field
from trading_core import (
    OrderSnapshot,
    SingleInstanceLock,
    StateFileError,
    atomic_write_json,
    estimate_nse_equity_intraday_cost,
    gross_pnl,
    load_json_strict,
    strict_finite_float,
    strict_integral,
)


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

IST = "Asia/Kolkata"

# Kite
KITE_API_KEY = os.getenv("KITE_API_KEY", "").strip()
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "").strip()
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "").strip()
KITE_STATIC_IP = os.getenv("KITE_STATIC_IP", "").strip()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6").strip()
OPENAI_TIMEOUT_SECONDS = float(
    os.getenv("OPENAI_TIMEOUT_SECONDS", "15")
)
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "0"))
OPENAI_REASONING_EFFORT = os.getenv(
    "OPENAI_REASONING_EFFORT",
    "low",
).strip().lower()
AI_MODE = os.getenv("AI_MODE", "off").strip().lower()
AI_IDEA_MODE = os.getenv("AI_IDEA_MODE", "shadow").strip().lower()
AI_IDEA_MAX_CANDIDATES = int(os.getenv("AI_IDEA_MAX_CANDIDATES", "8"))
MAX_AI_REVIEWS_PER_SCAN = int(os.getenv("MAX_AI_REVIEWS_PER_SCAN", "3"))
AI_PROMPT_VERSION = "nse-orb-review-v3"
AI_IDEA_PROMPT_VERSION = "nse-candidate-ideas-v1"

# Safety switch
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"
LIVE_TRADING_CONFIRM = os.getenv(
    "LIVE_TRADING_CONFIRM",
    "",
).strip()
EXECUTION_MODE = "live" if LIVE_TRADING else "paper"

# Capital/risk
CAPITAL_LIMIT = float(os.getenv("CAPITAL_LIMIT", "100000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.0020"))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.25"))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.008"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "5"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "2"))
MAX_PORTFOLIO_STOP_RISK_PCT = float(
    os.getenv("MAX_PORTFOLIO_STOP_RISK_PCT", "0.004")
)
MAX_GROSS_EXPOSURE_PCT = float(
    os.getenv("MAX_GROSS_EXPOSURE_PCT", "0.50")
)

# Candidate selection
PRELIMINARY_POOL_SIZE = int(os.getenv("PRELIMINARY_POOL_SIZE", "100"))
CANDIDATE_POOL_SIZE = int(os.getenv("CANDIDATE_POOL_SIZE", "35"))
TECH_MIN_SCORE = float(os.getenv("TECH_MIN_SCORE", "74"))
AI_MIN_CONFIDENCE = int(os.getenv("AI_MIN_CONFIDENCE", "80"))

MIN_PRICE = float(os.getenv("MIN_PRICE", "50"))
MAX_PRICE = float(os.getenv("MAX_PRICE", "10000"))
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", "18"))
MIN_ABS_CHANGE_PCT = float(os.getenv("MIN_ABS_CHANGE_PCT", "0.35"))
MAX_ABS_CHANGE_PCT = float(os.getenv("MAX_ABS_CHANGE_PCT", "10.0"))
MIN_DAY_RANGE_PCT = float(os.getenv("MIN_DAY_RANGE_PCT", "0.60"))
MIN_CIRCUIT_BUFFER_PCT = float(os.getenv("MIN_CIRCUIT_BUFFER_PCT", "1.0"))
MAX_GAP_PCT = float(os.getenv("MAX_GAP_PCT", "8.0"))

# Setup filters
MIN_RVOL = float(os.getenv("MIN_RVOL", "1.35"))
MIN_ATR_PCT = float(os.getenv("MIN_ATR_PCT", "0.0025"))
MAX_ATR_PCT = float(os.getenv("MAX_ATR_PCT", "0.025"))
MAX_VWAP_DISTANCE_ATR = float(os.getenv("MAX_VWAP_DISTANCE_ATR", "1.8"))
MIN_BREAKOUT_DISTANCE_ATR = float(
    os.getenv("MIN_BREAKOUT_DISTANCE_ATR", "0.04")
)
MAX_BREAKOUT_DISTANCE_ATR = float(
    os.getenv("MAX_BREAKOUT_DISTANCE_ATR", "0.80")
)

# Trade construction
ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER", "1.20"))
TARGET_R_MULTIPLE = float(os.getenv("TARGET_R_MULTIPLE", "1.80"))
MIN_AFTER_COST_PAYOFF_RATIO = float(
    os.getenv("MIN_AFTER_COST_PAYOFF_RATIO", "1.20")
)
CIRCUIT_HEADROOM_BPS = float(os.getenv("CIRCUIT_HEADROOM_BPS", "10"))

# Execution safety. Signals are revalidated after the bounded AI review.
MAX_SIGNAL_AGE_SECONDS = int(
    os.getenv("MAX_SIGNAL_AGE_SECONDS", "240")
)
MAX_ENTRY_DRIFT_ATR = float(
    os.getenv("MAX_ENTRY_DRIFT_ATR", "0.30")
)
ENTRY_FILL_TIMEOUT_SECONDS = float(
    os.getenv("ENTRY_FILL_TIMEOUT_SECONDS", "15")
)
STOP_ARM_TIMEOUT_SECONDS = float(
    os.getenv("STOP_ARM_TIMEOUT_SECONDS", "10")
)
EXIT_FILL_TIMEOUT_SECONDS = float(
    os.getenv("EXIT_FILL_TIMEOUT_SECONDS", "15")
)
ORDER_POLL_SECONDS = float(os.getenv("ORDER_POLL_SECONDS", "0.50"))

# Paper execution uses adverse directional slippage and explicit fees.
PAPER_SLIPPAGE_BPS = float(os.getenv("PAPER_SLIPPAGE_BPS", "5"))
RISK_SLIPPAGE_BPS = float(
    os.getenv("RISK_SLIPPAGE_BPS", str(PAPER_SLIPPAGE_BPS))
)
MAX_CONSECUTIVE_LOSSES = int(
    os.getenv("MAX_CONSECUTIVE_LOSSES", "3")
)
MAX_EXIT_ATTEMPTS = int(os.getenv("MAX_EXIT_ATTEMPTS", "3"))

# Timing
MARKET_OPEN = dtime(9, 15)
SIGNAL_START = dtime(9, 35)
LAST_ENTRY = dtime(14, 30)
FORCE_EXIT = dtime(15, 10)
SESSION_END = dtime(15, 30)

FULL_SCAN_EVERY_SECONDS = int(os.getenv("FULL_SCAN_EVERY_SECONDS", "180"))
POSITION_MONITOR_EVERY_SECONDS = int(
    os.getenv("POSITION_MONITOR_EVERY_SECONDS", "5")
)
ENTRY_CUTOFF_GUARD_SECONDS = int(
    os.getenv("ENTRY_CUTOFF_GUARD_SECONDS", "10")
)

# REST rate safety
# Kite historical API = 3 req/sec. Stay slightly below it.
CANDLE_DELAY_SECONDS = float(os.getenv("CANDLE_DELAY_SECONDS", "0.36"))
INDICATOR_LOOKBACK_DAYS = int(
    os.getenv("INDICATOR_LOOKBACK_DAYS", "21")
)

# WebSocket
# Kite currently permits up to 3 concurrent WebSocket connections per API key
# and up to 3000 instrument subscriptions per connection.
# Use 2800 per connection as a small operational buffer.
MAX_WS_CONNECTIONS = 3
WS_MAX_INSTRUMENTS_PER_CONNECTION = int(
    os.getenv("WS_MAX_INSTRUMENTS_PER_CONNECTION", "2800")
)
WS_SUBSCRIBE_BATCH = 500
WS_ACTIVE_TICK_MAX_AGE_SECONDS = int(
    os.getenv("WS_ACTIVE_TICK_MAX_AGE_SECONDS", "30")
)
WS_WARMUP_SECONDS = int(os.getenv("WS_WARMUP_SECONDS", "15"))
MAX_WS_DISCONNECT_SECONDS = int(
    os.getenv("MAX_WS_DISCONNECT_SECONDS", "30")
)

# Avoid compulsory-delivery / special series where identifiable from symbols.
BLOCKED_SERIES_SUFFIXES = (
    "-BE", "-BZ", "-BL", "-BT", "-SM", "-ST", "-MT", "-SG", "-GB", "-GS"
)

DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = DATA_DIR / "bot_state.json"
LOCK_FILE = DATA_DIR / "bot.lock"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

STRATEGY_VERSION = "3.2-paper-research-20260811"
SOURCE_SHA256 = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (
        Path(__file__),
        BASE_DIR / "trading_core.py",
        BASE_DIR / "requirements.txt",
    )
}
CODE_SHA256 = SOURCE_SHA256[Path(__file__).name]

# This manifest deliberately excludes credentials while including every
# setting that can materially change selection, sizing, execution or AI
# attribution.  It is journalled at session start and hashed into active state.
RUNTIME_MANIFEST: dict[str, Any] = {
    "strategy_version": STRATEGY_VERSION,
    "source_sha256": SOURCE_SHA256,
    "execution_mode": EXECUTION_MODE,
    "ai": {
        "mode": AI_MODE,
        "idea_mode": AI_IDEA_MODE,
        "model": OPENAI_MODEL,
        "prompt_version": AI_PROMPT_VERSION,
        "idea_prompt_version": AI_IDEA_PROMPT_VERSION,
        "reasoning_effort": OPENAI_REASONING_EFFORT,
        "timeout_seconds": OPENAI_TIMEOUT_SECONDS,
        "max_retries": OPENAI_MAX_RETRIES,
        "min_confidence": AI_MIN_CONFIDENCE,
        "max_reviews_per_scan": MAX_AI_REVIEWS_PER_SCAN,
        "idea_max_candidates": AI_IDEA_MAX_CANDIDATES,
    },
    "risk": {
        "capital_limit": CAPITAL_LIMIT,
        "risk_per_trade_pct": RISK_PER_TRADE_PCT,
        "max_position_pct": MAX_POSITION_PCT,
        "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
        "max_portfolio_stop_risk_pct": MAX_PORTFOLIO_STOP_RISK_PCT,
        "max_gross_exposure_pct": MAX_GROSS_EXPOSURE_PCT,
        "max_trades": MAX_TRADES_PER_DAY,
        "max_positions": MAX_OPEN_POSITIONS,
        "max_consecutive_losses": MAX_CONSECUTIVE_LOSSES,
    },
    "selection": {
        "preliminary_pool_size": PRELIMINARY_POOL_SIZE,
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "technical_min_score": TECH_MIN_SCORE,
        "price": [MIN_PRICE, MAX_PRICE],
        "spread_bps": MAX_SPREAD_BPS,
        "abs_change_pct": [MIN_ABS_CHANGE_PCT, MAX_ABS_CHANGE_PCT],
        "min_day_range_pct": MIN_DAY_RANGE_PCT,
        "min_circuit_buffer_pct": MIN_CIRCUIT_BUFFER_PCT,
        "max_gap_pct": MAX_GAP_PCT,
    },
    "setup": {
        "min_rvol": MIN_RVOL,
        "atr_pct": [MIN_ATR_PCT, MAX_ATR_PCT],
        "max_vwap_distance_atr": MAX_VWAP_DISTANCE_ATR,
        "breakout_atr": [
            MIN_BREAKOUT_DISTANCE_ATR,
            MAX_BREAKOUT_DISTANCE_ATR,
        ],
        "stop_atr": ATR_STOP_MULTIPLIER,
        "target_r": TARGET_R_MULTIPLE,
        "min_after_cost_payoff_ratio": MIN_AFTER_COST_PAYOFF_RATIO,
        "circuit_headroom_bps": CIRCUIT_HEADROOM_BPS,
    },
    "execution": {
        "max_signal_age_seconds": MAX_SIGNAL_AGE_SECONDS,
        "max_entry_drift_atr": MAX_ENTRY_DRIFT_ATR,
        "entry_fill_timeout_seconds": ENTRY_FILL_TIMEOUT_SECONDS,
        "stop_arm_timeout_seconds": STOP_ARM_TIMEOUT_SECONDS,
        "exit_fill_timeout_seconds": EXIT_FILL_TIMEOUT_SECONDS,
        "order_poll_seconds": ORDER_POLL_SECONDS,
        "paper_slippage_bps": PAPER_SLIPPAGE_BPS,
        "risk_slippage_bps": RISK_SLIPPAGE_BPS,
        "max_exit_attempts": MAX_EXIT_ATTEMPTS,
    },
    "schedule": {
        "market_open": MARKET_OPEN.isoformat(),
        "signal_start": SIGNAL_START.isoformat(),
        "last_entry": LAST_ENTRY.isoformat(),
        "force_exit": FORCE_EXIT.isoformat(),
        "session_end": SESSION_END.isoformat(),
        "entry_cutoff_guard_seconds": ENTRY_CUTOFF_GUARD_SECONDS,
        "full_scan_every_seconds": FULL_SCAN_EVERY_SECONDS,
        "position_monitor_every_seconds": POSITION_MONITOR_EVERY_SECONDS,
    },
    "data": {
        "candle_delay_seconds": CANDLE_DELAY_SECONDS,
        "indicator_lookback_days": INDICATOR_LOOKBACK_DAYS,
        "ws_instruments_per_connection": WS_MAX_INSTRUMENTS_PER_CONNECTION,
        "ws_active_tick_max_age_seconds": WS_ACTIVE_TICK_MAX_AGE_SECONDS,
        "ws_warmup_seconds": WS_WARMUP_SECONDS,
        "max_ws_disconnect_seconds": MAX_WS_DISCONNECT_SECONDS,
    },
}
RUNTIME_CONFIG_FINGERPRINT = hashlib.sha256(
    json.dumps(
        RUNTIME_MANIFEST,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()[:20]


# =============================================================================
# Models
# =============================================================================

@dataclass
class Instrument:
    symbol: str
    name: str
    token: int
    tick_size: float


@dataclass
class Quote:
    symbol: str
    token: int
    ltp: float
    open: float
    high: float
    low: float
    prev_close: float
    pct_change: float
    trade_volume: float
    turnover_crore: float
    spread_bps: float
    day_range_pct: float
    gap_pct: float
    circuit_buffer_pct: float
    lower_circuit_limit: float = 0.0
    upper_circuit_limit: float = 0.0
    stock_in_play_score: float = 0.0


@dataclass
class Setup:
    symbol: str
    token: int
    side: Literal["LONG", "SHORT"]

    price: float
    prev_close: float
    day_change_pct: float
    gap_pct: float
    turnover_crore: float
    spread_bps: float
    stock_in_play_score: float

    opening_range_high: float
    opening_range_low: float

    vwap: float
    ema9: float
    ema20: float
    rsi: float
    atr: float
    atr_pct: float
    rvol: float

    breakout_distance_atr: float
    vwap_distance_atr: float
    candle_body_ratio: float
    candle_close_location: float

    nifty_regime: str
    nifty_return_pct: float

    technical_score: float
    signal_at: str
    lower_circuit_limit: float = 0.0
    upper_circuit_limit: float = 0.0


@dataclass
class Trade:
    symbol: str
    token: int
    side: Literal["LONG", "SHORT"]
    qty: int

    entry_price: float
    initial_risk_per_share: float
    stop_price: float
    target_price: float

    entry_order_id: str | None = None
    stop_order_id: str | None = None
    exit_order_id: str | None = None
    exit_order_ids: list[str] = field(default_factory=list)

    status: str = "PLANNED"
    opened_at: str = ""
    closed_at: str = ""
    exit_reason: str = ""
    client_tag: str = ""
    entry_tag: str = ""
    stop_tag: str = ""
    exit_tag: str = ""
    exit_tags: list[str] = field(default_factory=list)
    exit_attempts: int = 0
    exit_intent_qty: int = 0
    exit_intent_transaction: str = ""
    requested_qty: int = 0
    entry_status: str = ""
    stop_status: str = ""
    exit_status: str = ""
    exit_price: float = 0.0
    gross_pnl: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0
    r_multiple: float = 0.0
    planned_risk_amount: float = 0.0
    reserved_risk_amount: float = 0.0
    reserved_notional_amount: float = 0.0
    planned_target_profit_amount: float = 0.0
    planned_after_cost_payoff: float = 0.0
    execution_mode: str = ""
    idea_id: str = ""
    ai_review_idea_id: str = ""
    ai_decision: str = ""
    ai_mode: str = ""
    ai_valid: bool = False
    ai_error: str = ""
    ai_response_model: str = ""
    ai_response_id: str = ""
    ai_prompt_version: str = ""
    ai_decision_id: str = ""
    ai_input_sha256: str = ""
    ai_input_tokens: int = 0
    ai_output_tokens: int = 0
    ai_total_tokens: int = 0
    accounting_uncertain: bool = False
    accounting_note: str = ""


class AIDecision(BaseModel):
    decision: Literal["APPROVE", "REJECT", "ERROR"]
    confidence: int = Field(ge=0, le=100)
    quality_score: int = Field(ge=0, le=100)
    reason: str
    risk_flags: list[str]


@dataclass(frozen=True)
class ExecutionSnapshot:
    ltp: float
    best_bid: float
    best_ask: float
    spread_bps: float
    lower_circuit: float
    upper_circuit: float
    observed_at: str


@dataclass(frozen=True)
class AfterCostOutcome:
    entry_fill: float
    stop_reference: float
    target_reference: float
    stop_fill: float
    target_fill: float
    stop_loss: float
    target_profit: float
    payoff_ratio: float


@dataclass(frozen=True)
class EntryCapacity:
    open_reserved_risk: float
    open_gross_notional: float
    daily_risk_remaining: float
    portfolio_risk_remaining: float
    gross_notional_remaining: float
    candidate_risk_budget: float
    candidate_notional_budget: float
    allowed: bool
    reason: str


@dataclass(frozen=True)
class TradeBuildResult:
    trade: Trade | None
    reason: str
    outcome: AfterCostOutcome | None = None


# =============================================================================
# Generic helpers
# =============================================================================

def now_ist() -> datetime:
    return pd.Timestamp.now(tz=IST).to_pydatetime()


def current_execution_mode() -> str:
    return "live" if LIVE_TRADING else "paper"


def log(message: str) -> None:
    print(
        f"[{now_ist().strftime('%Y-%m-%d %H:%M:%S %Z')}] {message}",
        flush=True,
    )


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_ai_candidate_payload(
    setup: Setup,
    trade: Trade,
    capacity: EntryCapacity,
) -> dict[str, Any]:
    observed = now_ist()
    signal_at = datetime.fromisoformat(setup.signal_at)
    if signal_at.tzinfo is None:
        signal_at = pd.Timestamp(signal_at, tz=IST).to_pydatetime()
    market_open = datetime.combine(
        observed.date(),
        MARKET_OPEN,
        tzinfo=observed.tzinfo,
    )
    return {
        "setup": asdict(setup),
        "context": {
            "minutes_since_open": max(
                0.0,
                (observed - market_open).total_seconds() / 60,
            ),
            "signal_age_seconds": max(
                0.0,
                (observed - signal_at).total_seconds(),
            ),
        },
        "economics": {
            "entry": trade.entry_price,
            "stop": trade.stop_price,
            "target": trade.target_price,
            "qty": trade.qty,
            "planned_risk": trade.planned_risk_amount,
            "reserved_notional": trade.reserved_notional_amount,
            "planned_target_profit": trade.planned_target_profit_amount,
            "after_cost_payoff": trade.planned_after_cost_payoff,
        },
        "capacity": asdict(capacity),
        "config_fingerprint": RUNTIME_CONFIG_FINGERPRINT,
    }


def identifier_stripped_ai_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    """Remove identity/time/composite scores from an online model request."""
    setup = dict(candidate["setup"])
    for key in (
        "symbol",
        "token",
        "signal_at",
        "technical_score",
        "stock_in_play_score",
    ):
        setup.pop(key, None)
    return {
        "setup": setup,
        "context": candidate["context"],
        "economics": candidate["economics"],
    }


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def parse_mis_position_quantities(rows: Any) -> dict[str, int]:
    """Normalize authoritative NSE/MIS positions without synthetic zeros."""
    if not isinstance(rows, list):
        raise RuntimeError("broker positions payload must be a list")
    quantities: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"broker position row {index} is not an object")
        exchange = str(row.get("exchange") or "").strip().upper()
        product = str(row.get("product") or "").strip().upper()
        if exchange != "NSE" or product != "MIS":
            continue
        symbol = str(row.get("tradingsymbol") or "").strip().upper()
        if not symbol:
            raise RuntimeError(f"broker NSE/MIS position row {index} has no symbol")
        if symbol in quantities:
            raise RuntimeError(f"duplicate NSE/MIS position row for {symbol}")
        try:
            quantities[symbol] = strict_integral(
                row.get("quantity"),
                field=f"position[{symbol}].quantity",
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    return quantities


def broker_mis_position_quantities(broker: Any) -> dict[str, int]:
    method = getattr(broker, "mis_position_quantities", None)
    if method is not None:
        result = method()
        if not isinstance(result, dict):
            raise RuntimeError("broker MIS position map is invalid")
        return {
            str(symbol).upper(): strict_integral(
                quantity,
                field=f"position[{symbol}].quantity",
            )
            for symbol, quantity in result.items()
        }
    return parse_mis_position_quantities(broker.positions())


def verify_live_account_matches_state(broker: Any, state: dict) -> None:
    """Fail closed when live NSE/MIS exposure is not exclusively state-owned."""
    actual = {
        symbol: quantity
        for symbol, quantity in broker_mis_position_quantities(broker).items()
        if quantity != 0
    }
    expected: dict[str, int] = {}
    for symbol, record in state.get("trades", {}).items():
        if not isinstance(record, dict):
            raise RuntimeError(f"invalid active trade record for {symbol}")
        status = str(record.get("status", "")).upper()
        if status not in ACTIVE_TRADE_STATUSES and not status.startswith("OPEN"):
            continue
        trade = trade_from_dict(record)
        if status != "OPEN_PROTECTED":
            raise RuntimeError(
                f"unreconciled {trade.symbol} lifecycle state {status}"
            )
        if trade.qty <= 0:
            raise RuntimeError(f"invalid active quantity for {trade.symbol}")
        signed_quantity = trade.qty if trade.side == "LONG" else -trade.qty
        normalized_symbol = trade.symbol.upper()
        if normalized_symbol in expected:
            raise RuntimeError(f"duplicate active trade for {normalized_symbol}")
        expected[normalized_symbol] = signed_quantity
    if actual != expected:
        raise RuntimeError(
            "broker NSE/MIS positions do not match durable active state: "
            f"expected={expected}, actual={actual}"
        )


def round_to_tick(price: float, tick_size: float) -> float:
    tick_size = tick_size if tick_size > 0 else 0.05
    return round(round(price / tick_size) * tick_size, 2)


def entry_deadline(value: datetime) -> datetime:
    cutoff = datetime.combine(
        value.date(),
        LAST_ENTRY,
        tzinfo=value.tzinfo,
    )
    return cutoff - timedelta(seconds=ENTRY_CUTOFF_GUARD_SECONDS)


def entry_window_open(value: datetime | None = None) -> bool:
    observed = value or now_ist()
    start = datetime.combine(
        observed.date(),
        SIGNAL_START,
        tzinfo=observed.tzinfo,
    )
    return start <= observed < entry_deadline(observed)


def new_order_tag(role: str) -> str:
    """Return a unique alphanumeric Kite tag (maximum 20 characters)."""
    clean_role = "".join(ch for ch in role.upper() if ch.isalnum())[:3]
    return f"AI{clean_role}{uuid.uuid4().hex[:12]}"[:20]


def is_generated_order_tag(tag: str) -> bool:
    value = str(tag or "")
    return bool(
        len(value) == 17
        and value[:5] in {"AIENT", "AISTP", "AIEXT", "AITRD"}
        and all(character in "0123456789abcdef" for character in value[5:])
    )


def allowed_broker_order_types(requested_order_type: str) -> frozenset[str]:
    """Return documented effective types for protected Kite order intents."""
    requested = str(requested_order_type or "").strip().upper()
    if requested == "MARKET":
        # Kite market protection represents the exchange mutation as bounded
        # limit behaviour; recovery must accept either broker representation.
        return frozenset({"MARKET", "LIMIT"})
    if requested in {"SL-M", "SLM"}:
        # A protected stop may be represented as SL-M while armed and as a
        # MARKET/LIMIT order after trigger or explicit conversion.
        return frozenset({"SL-M", "SLM", "MARKET", "LIMIT"})
    return frozenset({requested}) if requested else frozenset()


def trade_from_dict(data: dict) -> Trade:
    """Load current or legacy persisted trade data without accepting junk."""
    allowed = {field.name for field in fields(Trade)}
    return Trade(**{key: value for key, value in data.items() if key in allowed})


def infer_trade_execution_mode(data: dict) -> str | None:
    explicit = str(data.get("execution_mode") or "").strip().lower()
    order_ids = [
        data.get("entry_order_id"),
        data.get("stop_order_id"),
        data.get("exit_order_id"),
        *(data.get("exit_order_ids") or []),
    ]
    concrete = [str(value) for value in order_ids if value]
    derived: str | None = None
    if concrete:
        dry = [value.startswith("DRY-") for value in concrete]
        if any(dry) and not all(dry):
            raise RuntimeError("trade mixes DRY and real broker order identities")
        derived = "paper" if all(dry) else "live"
    if explicit and explicit not in {"paper", "live"}:
        raise RuntimeError("trade execution mode is invalid")
    if explicit and derived and explicit != derived:
        raise RuntimeError(
            f"trade is labelled {explicit} but its order IDs are {derived}"
        )
    return explicit or derived


ACTIVE_TRADE_STATUSES = {
    "ENTRY_INTENT",
    "ENTRY_SUBMITTED",
    "ENTRY_PARTIAL",
    "ENTRY_FILLED",
    "STOP_SUBMITTED",
    "OPEN_PROTECTED",
    "EXIT_PENDING",
    "HALTED_UNCERTAIN",
}


def validate_configuration() -> None:
    errors: list[str] = []

    if AI_MODE not in {"off", "shadow", "gate"}:
        errors.append("AI_MODE must be off, shadow, or gate")
    if AI_IDEA_MODE not in {"off", "shadow"}:
        errors.append("AI_IDEA_MODE must be off or shadow")
    if AI_MODE == "gate" and not OPENAI_API_KEY:
        errors.append("AI_MODE=gate requires OPENAI_API_KEY")
    if not 1 <= AI_IDEA_MAX_CANDIDATES <= 25:
        errors.append("AI_IDEA_MAX_CANDIDATES must be between 1 and 25")
    if not 1 <= MAX_AI_REVIEWS_PER_SCAN <= 10:
        errors.append("MAX_AI_REVIEWS_PER_SCAN must be between 1 and 10")

    if OPENAI_REASONING_EFFORT not in {
        "none", "low", "medium", "high", "xhigh", "max"
    }:
        errors.append("OPENAI_REASONING_EFFORT is invalid")

    if LIVE_TRADING and LIVE_TRADING_CONFIRM != "I_UNDERSTAND_REAL_MONEY":
        errors.append(
            "LIVE_TRADING=true also requires "
            "LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY"
        )

    if LIVE_TRADING and AI_MODE == "shadow":
        errors.append(
            "Live mode requires an explicit AI_MODE=off or AI_MODE=gate; "
            "shadow is a research default"
        )
    if LIVE_TRADING and AI_IDEA_MODE != "off":
        errors.append("AI idea generation is research-only in live mode")

    if not 0 < RISK_PER_TRADE_PCT <= 0.02:
        errors.append("RISK_PER_TRADE_PCT must be in (0, 0.02]")
    if not 0 < MAX_DAILY_LOSS_PCT <= 0.05:
        errors.append("MAX_DAILY_LOSS_PCT must be in (0, 0.05]")
    if not 0 < MAX_POSITION_PCT <= 1:
        errors.append("MAX_POSITION_PCT must be in (0, 1]")
    if not 0 < MAX_PORTFOLIO_STOP_RISK_PCT <= MAX_DAILY_LOSS_PCT:
        errors.append(
            "MAX_PORTFOLIO_STOP_RISK_PCT must be positive and no greater "
            "than MAX_DAILY_LOSS_PCT"
        )
    if not 0 < MAX_GROSS_EXPOSURE_PCT <= 1:
        errors.append("MAX_GROSS_EXPOSURE_PCT must be in (0, 1]")
    if MAX_OPEN_POSITIONS < 1 or MAX_TRADES_PER_DAY < 1:
        errors.append("position/trade limits must be positive")
    if PRELIMINARY_POOL_SIZE > 250:
        errors.append("PRELIMINARY_POOL_SIZE exceeds Kite /quote capacity")
    if not 1 <= CANDIDATE_POOL_SIZE <= PRELIMINARY_POOL_SIZE:
        errors.append("CANDIDATE_POOL_SIZE must fit the preliminary pool")
    if not 1 <= WS_MAX_INSTRUMENTS_PER_CONNECTION <= 3000:
        errors.append("WebSocket capacity/socket must be between 1 and 3000")
    if MIN_PRICE <= 0 or MAX_PRICE <= MIN_PRICE:
        errors.append("price bounds are invalid")
    if MIN_ATR_PCT <= 0 or MAX_ATR_PCT <= MIN_ATR_PCT:
        errors.append("ATR bounds are invalid")
    if not 0 < TARGET_R_MULTIPLE <= 10:
        errors.append("TARGET_R_MULTIPLE must be in (0, 10]")
    if not 0 < MIN_AFTER_COST_PAYOFF_RATIO <= 10:
        errors.append("MIN_AFTER_COST_PAYOFF_RATIO must be in (0, 10]")
    if not 0 <= CIRCUIT_HEADROOM_BPS <= 1000:
        errors.append("CIRCUIT_HEADROOM_BPS must be between 0 and 1000")
    if MAX_SIGNAL_AGE_SECONDS <= 0 or MAX_ENTRY_DRIFT_ATR <= 0:
        errors.append("signal freshness limits must be positive")
    if min(
        ENTRY_FILL_TIMEOUT_SECONDS,
        STOP_ARM_TIMEOUT_SECONDS,
        EXIT_FILL_TIMEOUT_SECONDS,
        ORDER_POLL_SECONDS,
        OPENAI_TIMEOUT_SECONDS,
    ) <= 0:
        errors.append("timeouts and polling intervals must be positive")
    if not 0 <= PAPER_SLIPPAGE_BPS <= 100:
        errors.append("PAPER_SLIPPAGE_BPS must be between 0 and 100")
    if not 0 <= RISK_SLIPPAGE_BPS <= 100:
        errors.append("RISK_SLIPPAGE_BPS must be between 0 and 100")
    if RISK_SLIPPAGE_BPS < PAPER_SLIPPAGE_BPS:
        errors.append(
            "RISK_SLIPPAGE_BPS must be at least PAPER_SLIPPAGE_BPS"
        )
    if MAX_CONSECUTIVE_LOSSES < 1 or MAX_EXIT_ATTEMPTS < 1:
        errors.append("loss and exit-attempt limits must be positive")
    if not 1 <= WS_ACTIVE_TICK_MAX_AGE_SECONDS <= 60:
        errors.append("WS_ACTIVE_TICK_MAX_AGE_SECONDS must be between 1 and 60")
    if MAX_WS_DISCONNECT_SECONDS < 1:
        errors.append("MAX_WS_DISCONNECT_SECONDS must be positive")
    if INDICATOR_LOOKBACK_DAYS < 7:
        errors.append("INDICATOR_LOOKBACK_DAYS must be at least 7")
    if not 0 <= ENTRY_CUTOFF_GUARD_SECONDS < 300:
        errors.append("ENTRY_CUTOFF_GUARD_SECONDS must be in [0, 300)")

    finite_values = {
        "CAPITAL_LIMIT": CAPITAL_LIMIT,
        "RISK_PER_TRADE_PCT": RISK_PER_TRADE_PCT,
        "MAX_POSITION_PCT": MAX_POSITION_PCT,
        "MAX_DAILY_LOSS_PCT": MAX_DAILY_LOSS_PCT,
        "MAX_PORTFOLIO_STOP_RISK_PCT": MAX_PORTFOLIO_STOP_RISK_PCT,
        "MAX_GROSS_EXPOSURE_PCT": MAX_GROSS_EXPOSURE_PCT,
        "MIN_AFTER_COST_PAYOFF_RATIO": MIN_AFTER_COST_PAYOFF_RATIO,
        "CIRCUIT_HEADROOM_BPS": CIRCUIT_HEADROOM_BPS,
        "PAPER_SLIPPAGE_BPS": PAPER_SLIPPAGE_BPS,
        "RISK_SLIPPAGE_BPS": RISK_SLIPPAGE_BPS,
    }
    if any(not math.isfinite(value) for value in finite_values.values()):
        errors.append("numeric risk configuration must be finite")
    if not math.isfinite(CAPITAL_LIMIT) or CAPITAL_LIMIT <= 0:
        errors.append("CAPITAL_LIMIT must be finite and positive")

    if errors:
        raise RuntimeError(
            "Invalid configuration:\n- " + "\n- ".join(errors)
        )


def percent_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    if len(series) <= 1:
        return pd.Series([1.0] * len(series), index=series.index)
    return series.rank(pct=True, ascending=ascending, method="average")


def dynamic_min_turnover_crore() -> float:
    now = now_ist()
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    elapsed_min = max(0.0, (now - start).total_seconds() / 60)
    return max(2.0, min(15.0, elapsed_min * 0.04))


_JOURNAL_LOCK = threading.Lock()


def journal(event: str, **fields) -> None:
    """
    JSONL avoids the changing-column problem that CSV journals get when OPEN,
    CLOSE and AI_REVIEW events have different fields.
    """
    path = LOG_DIR / f"trades_{now_ist().strftime('%Y%m%d')}.jsonl"
    payload = {
        "timestamp": now_ist().isoformat(),
        "event": event,
        "strategy_version": STRATEGY_VERSION,
        "code_sha256": CODE_SHA256,
        "config_fingerprint": RUNTIME_CONFIG_FINGERPRINT,
        "execution_mode": current_execution_mode(),
        **fields,
    }
    encoded = json.dumps(
        payload,
        default=str,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"
    with _JOURNAL_LOCK:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                fd = -1
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fd >= 0:
                os.close(fd)


def journal_best_effort(event: str, **fields) -> bool:
    """Record telemetry without allowing it to block a safety action."""
    try:
        journal(event, **fields)
        return True
    except Exception as exc:
        log(f"JOURNAL FAILURE for {event}: {type(exc).__name__}: {exc}")
        return False


# =============================================================================
# State
# =============================================================================

def fresh_state() -> dict:
    return {
        "schema_version": 3,
        "date": str(now_ist().date()),
        "execution_mode": current_execution_mode(),
        "trades_today": 0,
        "blocked_symbols": [],
        "trades": {},
        "kill_switch": False,
        "realized_pnl": 0.0,
        "fees_paid": 0.0,
        "consecutive_losses": 0,
        "halt_reason": "",
        "config_fingerprint": RUNTIME_CONFIG_FINGERPRINT,
    }


def load_state() -> dict:
    if not STATE_FILE.exists():
        return fresh_state()

    try:
        state = load_json_strict(STATE_FILE)
    except StateFileError as exc:
        raise RuntimeError(
            f"State file is corrupt or unreadable: {STATE_FILE}. "
            "Refusing to start; restore or reconcile it manually."
        ) from exc

    raw_trades = state.get("trades")
    trade_records = raw_trades if isinstance(raw_trades, dict) else {}
    has_active_state = any(
        str(record.get("status", "")).upper() in ACTIVE_TRADE_STATUSES
        or str(record.get("status", "")).upper().startswith("OPEN")
        for record in trade_records.values()
        if isinstance(record, dict)
    )
    if state.get("date") != str(now_ist().date()):
        if has_active_state:
            raise RuntimeError(
                "Previous-date active state exists; refusing to discard possible "
                "broker exposure. Manual reconciliation is required."
            )
        return fresh_state()

    stored_fingerprint = state.get("config_fingerprint")
    stored_mode = str(state.get("execution_mode") or "").strip().lower()
    inferred_modes = {
        mode
        for record in trade_records.values()
        if isinstance(record, dict)
        and str(record.get("status", "")).upper() in ACTIVE_TRADE_STATUSES
        for mode in [infer_trade_execution_mode(record)]
        if mode
    }
    if len(inferred_modes) > 1:
        raise RuntimeError("Active state mixes live and paper order identities.")
    inferred_mode = next(iter(inferred_modes), None)
    if stored_mode not in {"paper", "live"}:
        stored_mode = inferred_mode or ""
    if has_active_state:
        if not stored_mode:
            raise RuntimeError(
                "Active legacy state has ambiguous execution mode; manual "
                "reconciliation is required."
            )
        if inferred_mode and inferred_mode != stored_mode:
            raise RuntimeError("Active state execution-mode identity is inconsistent.")
        if stored_mode != current_execution_mode():
            raise RuntimeError(
                f"Active {stored_mode} state cannot start in "
                f"{current_execution_mode()} mode; "
                "manual broker reconciliation is required."
            )
        if stored_fingerprint != RUNTIME_CONFIG_FINGERPRINT:
            raise RuntimeError(
                "Code/config fingerprint changed while active state exists; "
                "manual broker reconciliation is required."
            )

    defaults = fresh_state()
    for key, value in defaults.items():
        state.setdefault(key, value)
    state["schema_version"] = 3
    state["execution_mode"] = current_execution_mode()
    state["config_fingerprint"] = RUNTIME_CONFIG_FINGERPRINT

    if not isinstance(state.get("trades"), dict):
        raise RuntimeError("State field 'trades' must be an object.")
    if not isinstance(state.get("blocked_symbols"), list):
        raise RuntimeError("State field 'blocked_symbols' must be a list.")

    if (
        isinstance(state.get("trades_today"), bool)
        or not isinstance(state.get("trades_today"), int)
        or state["trades_today"] < 0
    ):
        raise RuntimeError("State field 'trades_today' must be non-negative.")
    if not isinstance(state.get("kill_switch"), bool):
        raise RuntimeError("State field 'kill_switch' must be boolean.")
    if any(
        not isinstance(symbol, str) or not symbol
        for symbol in state["blocked_symbols"]
    ):
        raise RuntimeError("State blocked symbols must be non-empty strings.")
    for key in ("realized_pnl", "fees_paid"):
        value = state.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"State field '{key}' must be numeric.")
        if not math.isfinite(float(value)):
            raise RuntimeError(f"State field '{key}' must be finite.")
    for symbol, record in state["trades"].items():
        if not isinstance(symbol, str) or not isinstance(record, dict):
            raise RuntimeError("State trades must map symbols to objects.")
        try:
            trade = trade_from_dict(record)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid persisted trade for {symbol}.") from exc
        if trade.symbol != symbol or trade.side not in {"LONG", "SHORT"}:
            raise RuntimeError(f"Persisted trade identity is invalid for {symbol}.")
        trade_mode = trade.execution_mode or infer_trade_execution_mode(record)
        if not trade_mode:
            trade_mode = state["execution_mode"]
        if trade_mode not in {"paper", "live"}:
            raise RuntimeError(f"Persisted {symbol} execution mode is invalid.")
        if str(trade.status).upper() in ACTIVE_TRADE_STATUSES and (
            trade_mode != state["execution_mode"]
        ):
            raise RuntimeError(f"Persisted {symbol} execution mode changed.")
        if trade.execution_mode != trade_mode:
            trade.execution_mode = trade_mode
            state["trades"][symbol] = asdict(trade)
        for key, value in {
            "qty": trade.qty,
            "requested_qty": trade.requested_qty,
            "exit_attempts": trade.exit_attempts,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"Persisted {symbol} {key} is invalid.")
        if isinstance(trade.exit_intent_qty, bool) or not isinstance(
            trade.exit_intent_qty,
            int,
        ):
            raise RuntimeError(f"Persisted {symbol} exit intent quantity is invalid.")
        for key, value in {
            "entry_price": trade.entry_price,
            "initial_risk_per_share": trade.initial_risk_per_share,
            "stop_price": trade.stop_price,
            "target_price": trade.target_price,
            "planned_risk_amount": trade.planned_risk_amount,
            "reserved_risk_amount": trade.reserved_risk_amount,
            "reserved_notional_amount": trade.reserved_notional_amount,
            "planned_target_profit_amount": trade.planned_target_profit_amount,
            "planned_after_cost_payoff": trade.planned_after_cost_payoff,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise RuntimeError(f"Persisted {symbol} {key} is invalid.")
        if not isinstance(trade.exit_order_ids, list) or not isinstance(
            trade.exit_tags,
            list,
        ):
            raise RuntimeError(f"Persisted {symbol} order history is invalid.")
        if trade.exit_intent_transaction not in {"", "BUY", "SELL"}:
            raise RuntimeError(f"Persisted {symbol} exit transaction is invalid.")
        if not isinstance(trade.accounting_uncertain, bool):
            raise RuntimeError(f"Persisted {symbol} accounting flag is invalid.")

    return state


def save_state(state: dict) -> None:
    atomic_write_json(STATE_FILE, state)


def open_trade_count(state: dict) -> int:
    return sum(
        1
        for value in state["trades"].values()
        if (
            str(value.get("status", "")).upper()
            in ACTIVE_TRADE_STATUSES
            or str(value.get("status", "")).upper().startswith("OPEN")
        )
    )


# =============================================================================
# Indicators
# =============================================================================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    result = pd.Series(50.0, index=series.index, dtype=float)
    gain_only = (avg_gain > 0) & (avg_loss == 0)
    loss_only = (avg_gain == 0) & (avg_loss > 0)
    regular = (avg_gain > 0) & (avg_loss > 0)

    result.loc[gain_only] = 100.0
    result.loc[loss_only] = 0.0

    rs = avg_gain.loc[regular] / avg_loss.loc[regular]
    result.loc[regular] = 100 - (100 / (1 + rs))
    return result


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = atr(df, 14)

    session = df["date"].dt.date
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_pv = (typical * df["volume"]).groupby(session).cumsum()
    cumulative_volume = (
        df["volume"].groupby(session).cumsum().replace(0, np.nan)
    )
    df["vwap"] = cumulative_pv / cumulative_volume

    # Compare each bar with the same clock-time bar in earlier sessions.
    # This removes the strong U-shaped intraday volume seasonality.
    bar_slot = df["date"].dt.strftime("%H:%M")
    baseline_volume = df.groupby(bar_slot)["volume"].transform(
        lambda values: values.shift(1).rolling(
            window=10,
            min_periods=3,
        ).median()
    )
    df["rvol"] = df["volume"] / baseline_volume.replace(0, np.nan)

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (
        (df["close"] - df["open"]).abs() / candle_range
    )
    df["close_location"] = (
        (df["close"] - df["low"]) / candle_range
    )

    return df


def keep_only_closed_5m_candles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    now = pd.Timestamp.now(tz=IST)
    bucket_start = now.floor("5min")

    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize(IST)
    else:
        df["date"] = df["date"].dt.tz_convert(IST)

    return df[df["date"] < bucket_start].reset_index(drop=True)


# =============================================================================
# WebSocket tick store
# =============================================================================

class LiveTickStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._ticks: dict[int, dict] = {}

    def update_many(self, ticks: list[dict]) -> None:
        received_at = time.monotonic()

        with self._lock:
            for tick in ticks:
                token = int(tick["instrument_token"])
                self._ticks[token] = {
                    **tick,
                    "_received_at": received_at,
                }

    def active_snapshot(
        self,
        max_age_seconds: int,
    ) -> dict[int, dict]:
        cutoff = time.monotonic() - max_age_seconds

        with self._lock:
            return {
                token: dict(tick)
                for token, tick in self._ticks.items()
                if tick.get("_received_at", 0) >= cutoff
            }

    def count(self) -> int:
        with self._lock:
            return len(self._ticks)


# =============================================================================
# Zerodha broker
# =============================================================================

class KiteBroker:
    def __init__(self):
        self._validate_env()
        self._validate_static_ip_if_live()

        self.kite = KiteConnect(
            api_key=KITE_API_KEY,
            timeout=10,
        )
        self.kite.set_access_token(KITE_ACCESS_TOKEN)

        profile = self.kite.profile()
        log(
            "Connected to Zerodha as "
            f"{profile.get('user_name', profile.get('user_id'))}"
        )

        self.tick_store = LiveTickStore()
        self._quote_lock = threading.Lock()
        self._last_quote_request_at = 0.0
        self._order_condition = threading.Condition()
        self._order_updates: dict[str, OrderSnapshot] = {}

        self.instruments, self.nifty_token = (
            self._build_dynamic_universe()
        )

        log(
            f"Dynamic NSE cash-equity universe: "
            f"{len(self.instruments):,} instruments"
        )

        self.token_to_instrument = {
            inst.token: inst
            for inst in self.instruments.values()
        }

        self.kws_connections: list[KiteTicker] = []
        self.ws_connected = threading.Event()
        self._ws_lock = threading.Lock()
        self._ws_expected_connections = 0
        self._ws_live_connections: set[int] = set()

        self._start_websockets()

    def _validate_env(self):
        required = {
            "KITE_API_KEY": KITE_API_KEY,
            "KITE_ACCESS_TOKEN": KITE_ACCESS_TOKEN,
        }

        missing = [
            key for key, value in required.items()
            if not value
        ]

        if missing:
            raise RuntimeError(
                "Missing: "
                + ", ".join(missing)
                + ". Run login.py if KITE_ACCESS_TOKEN is empty."
            )

    def _validate_static_ip_if_live(self):
        if not LIVE_TRADING:
            return

        if not KITE_STATIC_IP:
            raise RuntimeError(
                "LIVE_TRADING=true requires KITE_STATIC_IP in .env. "
                "It must match the IPv4 registered in developers.kite.trade."
            )

        try:
            configured_ip = ipaddress.ip_address(KITE_STATIC_IP)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid KITE_STATIC_IP: {KITE_STATIC_IP}"
            ) from exc

        if configured_ip.version != 4:
            raise RuntimeError(
                "KITE_STATIC_IP must be a registered static IPv4 address."
            )

        try:
            public_ip = requests.get(
                "https://api.ipify.org",
                timeout=5,
            ).text.strip()
        except Exception as exc:
            raise RuntimeError(
                "Could not verify public IP; refusing live trading."
            ) from exc

        if public_ip != KITE_STATIC_IP:
            raise RuntimeError(
                "Current public IP does not match the registered "
                "KITE_STATIC_IP.\n"
                f"Current : {public_ip}\n"
                f"Expected: {KITE_STATIC_IP}\n"
                "Refusing live orders."
            )

        log(f"Static IP check passed: {public_ip}")

    def _build_dynamic_universe(
        self,
    ) -> tuple[dict[str, Instrument], int]:
        rows = self.kite.instruments("NSE")

        instruments: dict[str, Instrument] = {}
        nifty_token: int | None = None

        for row in rows:
            symbol = str(row.get("tradingsymbol", "")).strip()
            segment = str(row.get("segment", "")).upper()
            instrument_type = str(
                row.get("instrument_type", "")
            ).upper()

            if symbol == "NIFTY 50":
                nifty_token = int(row["instrument_token"])

            if segment != "NSE":
                continue

            if instrument_type != "EQ":
                continue

            if not symbol:
                continue

            if any(
                symbol.endswith(suffix)
                for suffix in BLOCKED_SERIES_SUFFIXES
            ):
                continue

            instruments[symbol] = Instrument(
                symbol=symbol,
                name=str(row.get("name", "")).strip(),
                token=int(row["instrument_token"]),
                tick_size=safe_float(row.get("tick_size"), 0.05),
            )

        # NIFTY 50 index token is historically stable, but prefer dynamic lookup.
        if nifty_token is None:
            nifty_token = 256265

        if len(instruments) < 500:
            raise RuntimeError(
                f"Unexpectedly small NSE EQ universe: "
                f"{len(instruments)}"
            )

        total_capacity = (
            MAX_WS_CONNECTIONS
            * WS_MAX_INSTRUMENTS_PER_CONNECTION
        )

        if len(instruments) > total_capacity:
            raise RuntimeError(
                f"NSE EQ universe ({len(instruments)}) exceeds configured "
                f"WebSocket capacity ({total_capacity} = "
                f"{MAX_WS_CONNECTIONS} connections × "
                f"{WS_MAX_INSTRUMENTS_PER_CONNECTION} instruments)."
            )

        return instruments, nifty_token

    def instrument(
        self,
        symbol: str,
    ) -> Instrument | None:
        return self.instruments.get(symbol)

    def all_instruments(self) -> list[Instrument]:
        return list(self.instruments.values())

    def _start_websockets(self) -> None:
        """Shard the complete NSE EQ universe across Kite WebSockets."""
        tokens = [
            inst.token
            for inst in self.instruments.values()
        ]

        token_groups = list(
            chunks(tokens, WS_MAX_INSTRUMENTS_PER_CONNECTION)
        )

        if len(token_groups) > MAX_WS_CONNECTIONS:
            raise RuntimeError(
                f"Need {len(token_groups)} WebSocket connections for "
                f"{len(tokens):,} instruments, but configured maximum is "
                f"{MAX_WS_CONNECTIONS}."
            )

        self._ws_expected_connections = len(token_groups)

        log(
            f"WebSocket plan: {len(tokens):,} instruments across "
            f"{self._ws_expected_connections} connection(s): "
            + ", ".join(
                f"WS{i + 1}={len(group):,}"
                for i, group in enumerate(token_groups)
            )
        )

        def mark_connected(connection_index: int) -> None:
            with self._ws_lock:
                self._ws_live_connections.add(connection_index)
                if len(self._ws_live_connections) == self._ws_expected_connections:
                    self.ws_connected.set()

        def mark_disconnected(connection_index: int) -> None:
            with self._ws_lock:
                self._ws_live_connections.discard(connection_index)
                self.ws_connected.clear()

        for connection_index, token_group in enumerate(token_groups):
            kws = KiteTicker(
                KITE_API_KEY,
                KITE_ACCESS_TOKEN,
                reconnect=True,
                reconnect_max_tries=50,
                reconnect_max_delay=60,
            )

            def on_ticks(ws, ticks, idx=connection_index):
                self.tick_store.update_many(ticks)

            def on_connect(ws, response, idx=connection_index, group=token_group):
                log(
                    f"Kite WebSocket #{idx + 1} connected; "
                    f"subscribing to {len(group):,} instruments."
                )
                for batch in chunks(group, WS_SUBSCRIBE_BATCH):
                    ws.subscribe(batch)
                    ws.set_mode(ws.MODE_QUOTE, batch)
                mark_connected(idx)

            def on_close(ws, code, reason, idx=connection_index):
                mark_disconnected(idx)
                log(
                    f"Kite WebSocket #{idx + 1} closed: "
                    f"code={code}, reason={reason}"
                )

            def on_error(ws, code, reason, idx=connection_index):
                log(
                    f"Kite WebSocket #{idx + 1} error: "
                    f"code={code}, reason={reason}"
                )

            def on_reconnect(ws, attempts_count, idx=connection_index):
                log(
                    f"Kite WebSocket #{idx + 1} reconnect attempt "
                    f"{attempts_count}"
                )

            def on_noreconnect(ws, idx=connection_index):
                mark_disconnected(idx)
                log(
                    f"CRITICAL: Kite WebSocket #{idx + 1} exhausted "
                    "reconnect attempts."
                )

            def on_order_update(ws, data, idx=connection_index):
                self._record_order_update(data)
                log(
                    f"ORDER UPDATE [WS{idx + 1}]: "
                    f"{data.get('tradingsymbol')} "
                    f"{data.get('status')} "
                    f"id={data.get('order_id')}"
                )

            kws.on_ticks = on_ticks
            kws.on_connect = on_connect
            kws.on_close = on_close
            kws.on_error = on_error
            kws.on_reconnect = on_reconnect
            kws.on_noreconnect = on_noreconnect
            kws.on_order_update = on_order_update

            self.kws_connections.append(kws)
            kws.connect(threaded=True)
            time.sleep(0.5)

        if not self.ws_connected.wait(timeout=45):
            with self._ws_lock:
                connected = len(self._ws_live_connections)
            raise RuntimeError(
                f"Only {connected}/{self._ws_expected_connections} Kite "
                "WebSocket connections became ready within 45 seconds."
            )

        log(
            f"All {self._ws_expected_connections} Kite WebSocket "
            "connections are ready."
        )
        log(f"WebSocket warm-up for {WS_WARMUP_SECONDS}s...")
        time.sleep(WS_WARMUP_SECONDS)
        log(
            f"Initial tick coverage: {self.tick_store.count():,}/"
            f"{len(self.instruments):,}"
        )

    def close(self):
        for idx, kws in enumerate(self.kws_connections):
            try:
                kws.close()
            except Exception as exc:
                log(f"WebSocket #{idx + 1} close warning: {exc}")

    def _record_order_update(self, payload: dict) -> None:
        """Idempotently cache duplicate postbacks from sharded sockets."""
        try:
            snapshot = OrderSnapshot.from_payload(payload)
        except (TypeError, ValueError) as exc:
            log(f"Invalid order postback ignored: {exc}")
            return

        if not snapshot.order_id:
            return

        with self._order_condition:
            current = self._order_updates.get(snapshot.order_id)
            should_update = current is None
            if current is not None:
                if snapshot.filled < current.filled:
                    # Filled quantity is cumulative and cannot legitimately
                    # move backwards, even in an out-of-order postback.
                    should_update = False
                elif current.terminal:
                    # A delayed OPEN/UPDATE postback must never regress a
                    # terminal broker state. A later terminal with more
                    # fills is still material and must win.
                    should_update = (
                        snapshot.terminal
                        and (
                            snapshot.status == current.status
                            or snapshot.filled > current.filled
                        )
                    )
                elif snapshot.terminal:
                    should_update = True
                elif (
                    snapshot.filled == current.filled
                    and current.order_type in {"LIMIT", "MARKET"}
                    and snapshot.order_type in {"SL-M", "SLM"}
                ):
                    # A delayed pre-conversion stop postback must not regress a
                    # protected SL-M already represented as MARKET/LIMIT.
                    should_update = False
                else:
                    should_update = snapshot.filled >= current.filled

            if should_update:
                self._order_updates[snapshot.order_id] = snapshot
            self._order_condition.notify_all()

    # -------------------------------------------------------------------------
    # Market data
    # -------------------------------------------------------------------------

    def active_websocket_quotes(self) -> list[Quote]:
        snapshot = self.tick_store.active_snapshot(
            WS_ACTIVE_TICK_MAX_AGE_SECONDS
        )

        quotes: list[Quote] = []

        for token, tick in snapshot.items():
            inst = self.token_to_instrument.get(int(token))
            if not inst:
                continue

            ltp = safe_float(tick.get("last_price"))
            ohlc = tick.get("ohlc") or {}

            opn = safe_float(ohlc.get("open"))
            high = safe_float(ohlc.get("high"))
            low = safe_float(ohlc.get("low"))
            prev_close = safe_float(ohlc.get("close"))
            volume = safe_float(tick.get("volume_traded"))

            last_trade_time = tick.get("last_trade_time")
            if last_trade_time is not None:
                try:
                    last_trade_at = pd.Timestamp(last_trade_time)
                    if last_trade_at.tzinfo is None:
                        last_trade_at = last_trade_at.tz_localize(IST)
                    else:
                        last_trade_at = last_trade_at.tz_convert(IST)
                    trade_age = (
                        pd.Timestamp(now_ist()) - last_trade_at
                    ).total_seconds()
                    if (
                        trade_age < -5
                        or trade_age > WS_ACTIVE_TICK_MAX_AGE_SECONDS
                    ):
                        continue
                except (TypeError, ValueError):
                    continue

            if ltp <= 0 or prev_close <= 0:
                continue

            pct_change = (
                (ltp - prev_close) / prev_close * 100
            )

            turnover_crore = (
                ltp * volume / 10_000_000
            )

            day_range_pct = (
                (high - low) / prev_close * 100
                if high > 0 and low > 0
                else 0.0
            )

            gap_pct = (
                (opn - prev_close) / prev_close * 100
                if opn > 0
                else 0.0
            )

            # Spread/circuit are enriched later through ONE /quote call.
            quotes.append(
                Quote(
                    symbol=inst.symbol,
                    token=inst.token,
                    ltp=ltp,
                    open=opn,
                    high=high,
                    low=low,
                    prev_close=prev_close,
                    pct_change=pct_change,
                    trade_volume=volume,
                    turnover_crore=turnover_crore,
                    spread_bps=9999.0,
                    day_range_pct=day_range_pct,
                    gap_pct=gap_pct,
                    circuit_buffer_pct=999.0,
                )
            )

        return quotes

    def enrich_full_quotes(
        self,
        quotes: list[Quote],
    ) -> list[Quote]:
        if not quotes:
            return []

        keys = [
            f"NSE:{q.symbol}"
            for q in quotes
        ]

        # PRELIMINARY_POOL_SIZE is intentionally below the current 250-key
        # full-quote request limit,
        # so this is one REST /quote request per market scan.
        response = self._rate_limited_quote(keys)

        enriched: list[Quote] = []

        by_symbol = {
            q.symbol: q
            for q in quotes
        }

        for key, data in response.items():
            if ":" not in key:
                continue

            symbol = key.split(":", 1)[1]
            q = by_symbol.get(symbol)

            if not q:
                continue

            depth = data.get("depth") or {}
            buys = depth.get("buy") or []
            sells = depth.get("sell") or []

            best_bid = (
                safe_float(buys[0].get("price"))
                if buys else 0.0
            )
            best_ask = (
                safe_float(sells[0].get("price"))
                if sells else 0.0
            )

            if best_bid > 0 and best_ask >= best_bid:
                mid = (best_bid + best_ask) / 2
                spread_bps = (
                    (best_ask - best_bid)
                    / mid
                    * 10000
                )
            else:
                spread_bps = 9999.0

            upper = safe_float(
                data.get("upper_circuit_limit")
            )
            lower = safe_float(
                data.get("lower_circuit_limit")
            )
            ltp = safe_float(
                data.get("last_price"),
                q.ltp,
            )
            ohlc = data.get("ohlc") or {}
            refreshed_open = safe_float(ohlc.get("open"), q.open)
            refreshed_high = safe_float(ohlc.get("high"), q.high)
            refreshed_low = safe_float(ohlc.get("low"), q.low)
            refreshed_prev_close = safe_float(ohlc.get("close"), q.prev_close)
            refreshed_volume = safe_float(data.get("volume"), q.trade_volume)
            if ltp <= 0 or refreshed_prev_close <= 0:
                continue

            q.ltp = ltp
            q.open = refreshed_open
            q.high = refreshed_high
            q.low = refreshed_low
            q.prev_close = refreshed_prev_close
            q.trade_volume = refreshed_volume
            q.pct_change = (
                (ltp - refreshed_prev_close) / refreshed_prev_close * 100
            )
            q.turnover_crore = ltp * refreshed_volume / 10_000_000
            q.day_range_pct = (
                (refreshed_high - refreshed_low) / refreshed_prev_close * 100
                if refreshed_high > 0 and refreshed_low > 0
                else 0.0
            )
            q.gap_pct = (
                (refreshed_open - refreshed_prev_close)
                / refreshed_prev_close
                * 100
                if refreshed_open > 0
                else 0.0
            )

            circuit_buffer = -1.0
            if upper > ltp > lower > 0:
                circuit_buffer = min(
                    (upper - ltp) / ltp * 100,
                    (ltp - lower) / ltp * 100,
                )

            q.spread_bps = spread_bps
            q.circuit_buffer_pct = circuit_buffer
            q.lower_circuit_limit = lower
            q.upper_circuit_limit = upper

            enriched.append(q)

        return enriched

    def _rate_limited_quote(self, keys: list[str]) -> dict:
        """Serialize full-quote calls at Kite's documented 1 request/second."""
        with self._quote_lock:
            elapsed = time.monotonic() - self._last_quote_request_at
            if elapsed < 1.05:
                time.sleep(1.05 - elapsed)
            try:
                return self.kite.quote(keys)
            finally:
                self._last_quote_request_at = time.monotonic()

    def execution_snapshot(self, symbol: str) -> ExecutionSnapshot:
        """Refresh price, depth and both directional circuit limits."""
        key = f"NSE:{symbol}"
        data = self._rate_limited_quote([key]).get(key) or {}
        ltp = safe_float(data.get("last_price"))
        depth = data.get("depth") or {}
        buys = depth.get("buy") or []
        sells = depth.get("sell") or []
        bid = safe_float(buys[0].get("price")) if buys else 0.0
        ask = safe_float(sells[0].get("price")) if sells else 0.0
        if ltp <= 0 or bid <= 0 or ask < bid:
            raise RuntimeError("invalid live price/depth")
        spread_bps = (ask - bid) / ((ask + bid) / 2) * 10_000
        upper = safe_float(data.get("upper_circuit_limit"))
        lower = safe_float(data.get("lower_circuit_limit"))
        if not upper > ltp > lower > 0:
            raise RuntimeError("missing or invalid circuit limits")
        return ExecutionSnapshot(
            ltp=ltp,
            best_bid=bid,
            best_ask=ask,
            spread_bps=spread_bps,
            lower_circuit=lower,
            upper_circuit=upper,
            observed_at=now_ist().isoformat(),
        )

    def historical_candles(
        self,
        token: int,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        rows = self.kite.historical_data(
            instrument_token=token,
            from_date=start,
            to_date=end,
            interval="5minute",
            continuous=False,
            oi=False,
        )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        return df.dropna().reset_index(drop=True)

    def strategy_candles(
        self,
        token: int,
    ) -> pd.DataFrame:
        now = now_ist()
        start = now - timedelta(days=INDICATOR_LOOKBACK_DAYS)

        df = self.historical_candles(
            token,
            start,
            now,
        )

        return keep_only_closed_5m_candles(df)

    def ltp(
        self,
        symbol: str,
    ) -> float:
        response = self.kite.ltp(
            f"NSE:{symbol}"
        )
        return safe_float(
            response[f"NSE:{symbol}"]["last_price"]
        )

    # -------------------------------------------------------------------------
    # Portfolio / order APIs
    # -------------------------------------------------------------------------

    @staticmethod
    def _validated_order_quantity(qty: int) -> int:
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            raise ValueError("order quantity must be a positive integer")
        return qty

    @staticmethod
    def _validated_order_tag(tag: str) -> str:
        if not isinstance(tag, str):
            raise ValueError(
                "Kite order tag must be 1-20 ASCII alphanumeric characters"
            )
        value = tag.strip()
        if (
            not value
            or len(value) > 20
            or not value.isascii()
            or not value.isalnum()
        ):
            raise ValueError(
                "Kite order tag must be 1-20 ASCII alphanumeric characters"
            )
        return value

    @staticmethod
    def _validated_trade_side(side: str) -> str:
        value = str(side).strip().upper()
        if value not in {"LONG", "SHORT"}:
            raise ValueError("trade side must be LONG or SHORT")
        return value

    def orders(self) -> list[dict]:
        return self.kite.orders()

    def trades(self) -> list[dict]:
        return self.kite.trades()

    def positions(self) -> list[dict]:
        payload = self.kite.positions()
        if not isinstance(payload, dict) or not isinstance(payload.get("net"), list):
            raise RuntimeError("Kite positions response is missing the net list")
        return payload["net"]

    def mis_position_quantities(self) -> dict[str, int]:
        return parse_mis_position_quantities(self.positions())

    def current_intraday_pnl(self) -> float:
        total = 0.0
        seen: set[str] = set()

        for index, position in enumerate(self.positions()):
            if not isinstance(position, dict):
                raise RuntimeError(f"broker position row {index} is not an object")
            if (
                position.get("exchange") == "NSE"
                and position.get("product") == "MIS"
            ):
                symbol = str(position.get("tradingsymbol") or "").strip().upper()
                if not symbol or symbol in seen:
                    raise RuntimeError("invalid or duplicate NSE/MIS P&L position row")
                seen.add(symbol)
                try:
                    total += strict_finite_float(
                        position.get("pnl"),
                        field=f"position[{symbol}].pnl",
                    )
                except ValueError as exc:
                    raise RuntimeError(str(exc)) from exc

        return total

    def position_qty(
        self,
        symbol: str,
    ) -> int:
        return self.mis_position_quantities().get(symbol.upper(), 0)

    def latest_order(
        self,
        order_id: str,
    ) -> OrderSnapshot:
        order_id = str(order_id).strip()
        if not order_id:
            raise ValueError("order_id is required")

        try:
            history = self.kite.order_history(order_id)
        except Exception:
            # WebSocket postbacks are an independent state source. During a
            # transient REST failure, retain a known terminal/partial state.
            with self._order_condition:
                cached = self._order_updates.get(order_id)
            if cached is not None:
                return cached
            raise

        if history:
            snapshot = OrderSnapshot.from_payload(history[-1])
            self._record_order_update(history[-1])
            with self._order_condition:
                return self._order_updates.get(order_id, snapshot)

        with self._order_condition:
            snapshot = self._order_updates.get(str(order_id))
        if snapshot is None:
            raise RuntimeError(f"No broker state for order {order_id}.")
        return snapshot

    def wait_for_order(
        self,
        order_id: str,
        *,
        timeout_seconds: float,
        require_stop_armed: bool = False,
        return_on_partial: bool = False,
    ) -> OrderSnapshot:
        """Wait for a terminal, armed stop, or explicitly requested partial."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        deadline = time.monotonic() + timeout_seconds
        last: OrderSnapshot | None = None
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            try:
                last = self.latest_order(order_id)
                last_error = None
            except Exception as exc:
                last_error = exc

            if last is not None:
                if last.terminal:
                    return last
                if require_stop_armed and last.stop_armed:
                    return last
                if return_on_partial and last.filled > 0:
                    # Entry callers immediately cancel the unfilled remainder,
                    # then arm protection for the terminal confirmed quantity.
                    # This avoids leaving a partial fill exposed for the full
                    # entry timeout.
                    return last

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            with self._order_condition:
                self._order_condition.wait(
                    timeout=min(ORDER_POLL_SECONDS, remaining)
                )

        if last is not None:
            return last
        raise RuntimeError(
            f"Unable to resolve order {order_id}: {last_error}"
        )

    def find_exact_order_by_tag(
        self,
        *,
        tag: str,
        symbol: str,
        transaction_type: str,
        order_type: str,
        quantity: int,
    ) -> OrderSnapshot | None:
        matches: list[OrderSnapshot] = []
        allowed_types = allowed_broker_order_types(order_type)
        for payload in self.orders():
            try:
                snapshot = OrderSnapshot.from_payload(payload)
            except (TypeError, ValueError) as exc:
                if isinstance(payload, dict) and str(payload.get("tag") or "") == tag:
                    raise RuntimeError(
                        f"broker order matching tag {tag} is malformed"
                    ) from exc
                continue
            if (
                snapshot.tag == tag
                and snapshot.symbol == symbol.upper()
                and snapshot.exchange == "NSE"
                and snapshot.product == "MIS"
                and snapshot.transaction_type == transaction_type.upper()
                and snapshot.order_type in allowed_types
                and snapshot.qty == abs(quantity)
            ):
                matches.append(snapshot)

        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous broker mutation: {len(matches)} orders "
                f"matched tag {tag}."
            )
        return matches[0] if matches else None

    def place_market_entry(
        self,
        inst: Instrument,
        side: str,
        qty: int,
        tag: str,
    ) -> str:
        side = self._validated_trade_side(side)
        qty = self._validated_order_quantity(qty)
        tag = self._validated_order_tag(tag)
        transaction = (
            self.kite.TRANSACTION_TYPE_BUY
            if side == "LONG"
            else self.kite.TRANSACTION_TYPE_SELL
        )

        if not LIVE_TRADING:
            fake = (
                f"DRY-ENTRY-{inst.symbol}-"
                f"{tag}"
            )
            log(
                f"DRY RUN: {transaction} "
                f"{qty} {inst.symbol} MARKET"
            )
            return fake

        return self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=self.kite.EXCHANGE_NSE,
            tradingsymbol=inst.symbol,
            transaction_type=transaction,
            quantity=qty,
            product=self.kite.PRODUCT_MIS,
            order_type=self.kite.ORDER_TYPE_MARKET,
            validity=self.kite.VALIDITY_DAY,
            market_protection=-1,
            tag=tag,
        )

    def place_protective_stop(
        self,
        inst: Instrument,
        side: str,
        qty: int,
        trigger_price: float,
        tag: str,
    ) -> str:
        side = self._validated_trade_side(side)
        qty = self._validated_order_quantity(qty)
        tag = self._validated_order_tag(tag)
        if not math.isfinite(trigger_price) or trigger_price <= 0:
            raise ValueError("stop trigger price must be finite and positive")
        transaction = (
            self.kite.TRANSACTION_TYPE_SELL
            if side == "LONG"
            else self.kite.TRANSACTION_TYPE_BUY
        )

        if not LIVE_TRADING:
            fake = (
                f"DRY-STOP-{inst.symbol}-"
                f"{tag}"
            )
            log(
                f"DRY RUN: protective "
                f"{transaction} SL-M "
                f"{qty} {inst.symbol} "
                f"trigger={trigger_price:.2f}"
            )
            return fake

        return self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=self.kite.EXCHANGE_NSE,
            tradingsymbol=inst.symbol,
            transaction_type=transaction,
            quantity=qty,
            product=self.kite.PRODUCT_MIS,
            order_type=self.kite.ORDER_TYPE_SLM,
            trigger_price=trigger_price,
            validity=self.kite.VALIDITY_DAY,
            market_protection=-1,
            tag=tag,
        )

    def cancel_order_confirmed(
        self,
        order_id: str | None,
        *,
        timeout_seconds: float = 8,
    ) -> OrderSnapshot | None:
        if not order_id:
            return None

        if (
            not LIVE_TRADING
            or str(order_id).startswith("DRY-")
        ):
            return OrderSnapshot.from_payload({
                "order_id": str(order_id),
                "status": "CANCELLED",
            })

        before = self.latest_order(order_id)
        if before.terminal:
            return before

        try:
            self.kite.cancel_order(
                variety=self.kite.VARIETY_REGULAR,
                order_id=order_id,
            )
        except Exception as exc:
            log(
                f"Order cancellation request uncertain "
                f"for {order_id}: {exc}"
            )

        after = self.wait_for_order(
            order_id,
            timeout_seconds=timeout_seconds,
        )
        if not after.terminal:
            raise TimeoutError(
                f"Cancellation of {order_id} was not confirmed; "
                f"last status={after.status}, filled={after.filled}, "
                f"pending={after.pending}."
            )
        return after

    def convert_stop_to_market(
        self,
        order_id: str,
        qty: int,
    ) -> str:
        order_id = str(order_id).strip()
        if not order_id:
            raise ValueError("order_id is required")
        qty = self._validated_order_quantity(qty)

        if not LIVE_TRADING:
            log(
                f"DRY RUN: convert stop {order_id} to MARKET "
                f"qty={qty}"
            )
            return order_id
        return self.kite.modify_order(
            variety=self.kite.VARIETY_REGULAR,
            order_id=order_id,
            quantity=qty,
            order_type=self.kite.ORDER_TYPE_MARKET,
            validity=self.kite.VALIDITY_DAY,
            market_protection=-1,
        )

    def exit_market(
        self,
        inst: Instrument,
        signed_qty: int,
        tag: str,
    ) -> str:
        if (
            isinstance(signed_qty, bool)
            or not isinstance(signed_qty, int)
            or signed_qty == 0
        ):
            raise ValueError("signed exit quantity must be a non-zero integer")
        tag = self._validated_order_tag(tag)
        qty = abs(signed_qty)

        transaction = (
            self.kite.TRANSACTION_TYPE_SELL
            if signed_qty > 0
            else self.kite.TRANSACTION_TYPE_BUY
        )

        if not LIVE_TRADING:
            fake = (
                f"DRY-EXIT-{inst.symbol}-"
                f"{tag}"
            )
            log(
                f"DRY RUN: EXIT {transaction} "
                f"{qty} {inst.symbol} MARKET"
            )
            return fake

        return self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=self.kite.EXCHANGE_NSE,
            tradingsymbol=inst.symbol,
            transaction_type=transaction,
            quantity=qty,
            product=self.kite.PRODUCT_MIS,
            order_type=self.kite.ORDER_TYPE_MARKET,
            validity=self.kite.VALIDITY_DAY,
            market_protection=-1,
            tag=tag,
        )

    def wait_for_position_qty(
        self,
        symbol: str,
        expected_qty: int,
        *,
        timeout_seconds: float = 8,
    ) -> bool:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not LIVE_TRADING:
            return expected_qty == 0

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.position_qty(symbol) == expected_qty:
                return True
            time.sleep(ORDER_POLL_SECONDS)
        return self.position_qty(symbol) == expected_qty


# =============================================================================
# Broad-market scanner
# =============================================================================

def select_stocks_in_play(
    broker: KiteBroker,
    blocked_symbols: set[str],
) -> list[Quote]:
    """
    Stage A:
        Use live WebSocket QUOTE ticks for the broad universe.
        No REST requests here.

    Stage B:
        Rank a preliminary pool.

    Stage C:
        Use ONE full /quote REST request for preliminary names so we can add
        bid/ask spread and circuit-limit protection.
    """
    broad = broker.active_websocket_quotes()

    min_turnover = dynamic_min_turnover_crore()

    broad_filtered = [
        q
        for q in broad
        if (
            q.symbol not in blocked_symbols
            and MIN_PRICE <= q.ltp <= MAX_PRICE
            and q.turnover_crore >= min_turnover
            and MIN_ABS_CHANGE_PCT
                <= abs(q.pct_change)
                <= MAX_ABS_CHANGE_PCT
            and q.day_range_pct >= MIN_DAY_RANGE_PCT
            and abs(q.gap_pct) <= MAX_GAP_PCT
        )
    ]

    if not broad_filtered:
        return []

    df = pd.DataFrame(
        [asdict(q) for q in broad_filtered]
    )

    # Preliminary ranking without depth.
    df["turnover_rank"] = percent_rank(
        df["turnover_crore"]
    )
    df["move_rank"] = percent_rank(
        df["pct_change"].abs()
    )
    df["range_rank"] = percent_rank(
        df["day_range_pct"]
    )

    df["pre_score"] = 100 * (
        0.45 * df["turnover_rank"]
        + 0.30 * df["move_rank"]
        + 0.25 * df["range_rank"]
    )

    df = df.sort_values(
        "pre_score",
        ascending=False,
    ).head(PRELIMINARY_POOL_SIZE)

    preliminary = []

    for record in df.to_dict("records"):
        preliminary.append(
            Quote(
                symbol=record["symbol"],
                token=int(record["token"]),
                ltp=record["ltp"],
                open=record["open"],
                high=record["high"],
                low=record["low"],
                prev_close=record["prev_close"],
                pct_change=record["pct_change"],
                trade_volume=record["trade_volume"],
                turnover_crore=record["turnover_crore"],
                spread_bps=9999.0,
                day_range_pct=record["day_range_pct"],
                gap_pct=record["gap_pct"],
                circuit_buffer_pct=999.0,
            )
        )

    enriched = broker.enrich_full_quotes(
        preliminary
    )

    final_filtered = [
        q
        for q in enriched
        if (
            q.spread_bps <= MAX_SPREAD_BPS
            and q.circuit_buffer_pct
                >= MIN_CIRCUIT_BUFFER_PCT
        )
    ]

    if not final_filtered:
        return []

    final_df = pd.DataFrame(
        [asdict(q) for q in final_filtered]
    )

    final_df["turnover_rank"] = percent_rank(
        final_df["turnover_crore"]
    )
    final_df["move_rank"] = percent_rank(
        final_df["pct_change"].abs()
    )
    final_df["range_rank"] = percent_rank(
        final_df["day_range_pct"]
    )
    final_df["spread_rank"] = percent_rank(
        final_df["spread_bps"],
        ascending=False,
    )

    final_df["stock_in_play_score"] = 100 * (
        0.38 * final_df["turnover_rank"]
        + 0.27 * final_df["move_rank"]
        + 0.20 * final_df["range_rank"]
        + 0.15 * final_df["spread_rank"]
    )

    final_df = final_df.sort_values(
        "stock_in_play_score",
        ascending=False,
    ).head(CANDIDATE_POOL_SIZE)

    candidates = []

    for record in final_df.to_dict("records"):
        candidates.append(
            Quote(
                symbol=record["symbol"],
                token=int(record["token"]),
                ltp=record["ltp"],
                open=record["open"],
                high=record["high"],
                low=record["low"],
                prev_close=record["prev_close"],
                pct_change=record["pct_change"],
                trade_volume=record["trade_volume"],
                turnover_crore=record["turnover_crore"],
                spread_bps=record["spread_bps"],
                day_range_pct=record["day_range_pct"],
                gap_pct=record["gap_pct"],
                circuit_buffer_pct=record["circuit_buffer_pct"],
                lower_circuit_limit=record["lower_circuit_limit"],
                upper_circuit_limit=record["upper_circuit_limit"],
                stock_in_play_score=record["stock_in_play_score"],
            )
        )

    log(
        f"Market scan: "
        f"{len(broad):,} active WebSocket names -> "
        f"{len(broad_filtered):,} liquid movers -> "
        f"{len(preliminary)} depth-enriched -> "
        f"top {len(candidates)} candidates "
        f"(min turnover ₹{min_turnover:.1f} cr)"
    )

    return candidates


# =============================================================================
# NIFTY market regime
# =============================================================================

def get_nifty_regime(
    broker: KiteBroker,
) -> tuple[str, float]:
    df = broker.strategy_candles(
        broker.nifty_token
    )
    time.sleep(CANDLE_DELAY_SECONDS)

    if df.empty:
        return "NEUTRAL", 0.0

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    latest_session = df["date"].dt.date.max()
    if latest_session != now_ist().date():
        return "NEUTRAL", 0.0
    today = df[df["date"].dt.date == latest_session]

    if len(today) < 6 or len(df) < 20:
        return "NEUTRAL", 0.0

    close = df["close"]

    ema9 = close.ewm(
        span=9,
        adjust=False,
    ).mean().iloc[-1]

    ema20 = close.ewm(
        span=20,
        adjust=False,
    ).mean().iloc[-1]

    last = float(today["close"].iloc[-1])
    first_open = float(today["open"].iloc[0])

    ret_pct = (
        (last - first_open)
        / first_open
        * 100
    )

    if (
        last > ema20
        and ema9 > ema20
        and ret_pct >= 0.10
    ):
        return "BULL", ret_pct

    if (
        last < ema20
        and ema9 < ema20
        and ret_pct <= -0.10
    ):
        return "BEAR", ret_pct

    return "NEUTRAL", ret_pct


# =============================================================================
# Deterministic setup engine
# =============================================================================

def detect_setup(
    quote: Quote,
    df: pd.DataFrame,
    nifty_regime: str,
    nifty_return_pct: float,
) -> Setup | None:
    if len(df) < 20:
        return None

    df = add_indicators(df)
    latest_session = df["date"].dt.date.max()
    if latest_session != now_ist().date():
        return None
    today = df[df["date"].dt.date == latest_session].reset_index(drop=True)

    if len(today) < 4:
        return None

    expected_opening_times = [dtime(9, 15), dtime(9, 20), dtime(9, 25)]
    actual_opening_times = list(today["date"].iloc[:3].dt.time)
    if actual_opening_times != expected_opening_times:
        return None
    if pd.Timestamp(today["date"].iloc[-1]) - pd.Timestamp(
        today["date"].iloc[-2]
    ) != pd.Timedelta(minutes=5):
        return None

    opening = today.iloc[:3]

    if len(opening) < 3:
        return None

    opening_high = float(
        opening["high"].max()
    )
    opening_low = float(
        opening["low"].min()
    )

    last = today.iloc[-1]
    prev = today.iloc[-2]

    price = float(last["close"])
    ema9 = float(last["ema9"])
    ema20 = float(last["ema20"])
    vwap = float(last["vwap"])
    rsi_value = float(last["rsi"])
    atr_value = float(last["atr"])

    if (
        price <= 0
        or atr_value <= 0
        or math.isnan(vwap)
    ):
        return None

    rvol_value = (
        float(last["rvol"])
        if pd.notna(last["rvol"])
        else 0.0
    )

    body_ratio = (
        float(last["body_ratio"])
        if pd.notna(last["body_ratio"])
        else 0.0
    )

    close_location = (
        float(last["close_location"])
        if pd.notna(last["close_location"])
        else 0.5
    )

    atr_pct = atr_value / price

    if not (
        MIN_ATR_PCT
        <= atr_pct
        <= MAX_ATR_PCT
    ):
        return None

    if rvol_value < MIN_RVOL:
        return None

    if body_ratio < 0.35:
        return None

    long_fresh = (
        float(prev["close"])
        <= opening_high + 0.15 * atr_value
    )

    short_fresh = (
        float(prev["close"])
        >= opening_low - 0.15 * atr_value
    )

    long_breakout_atr = (
        price - opening_high
    ) / atr_value

    short_breakout_atr = (
        opening_low - price
    ) / atr_value

    vwap_distance_atr = (
        abs(price - vwap) / atr_value
    )

    if (
        vwap_distance_atr
        > MAX_VWAP_DISTANCE_ATR
    ):
        return None

    long_setup = (
        long_fresh
        and MIN_BREAKOUT_DISTANCE_ATR
            <= long_breakout_atr
            <= MAX_BREAKOUT_DISTANCE_ATR
        and price > vwap
        and ema9 > ema20
        and 52 <= rsi_value <= 75
        and close_location >= 0.60
        and quote.pct_change >= 0.35
        and nifty_regime != "BEAR"
    )

    short_setup = (
        short_fresh
        and MIN_BREAKOUT_DISTANCE_ATR
            <= short_breakout_atr
            <= MAX_BREAKOUT_DISTANCE_ATR
        and price < vwap
        and ema9 < ema20
        and 28 <= rsi_value <= 48
        and close_location <= 0.40
        and quote.pct_change <= -0.25
        and nifty_regime != "BULL"
    )

    if not long_setup and not short_setup:
        return None

    side: Literal["LONG", "SHORT"] = (
        "LONG"
        if long_setup
        else "SHORT"
    )

    breakout_distance_atr = (
        long_breakout_atr
        if side == "LONG"
        else short_breakout_atr
    )

    score = 0.0

    score += min(
        28.0,
        quote.stock_in_play_score * 0.28,
    )

    score += min(
        20.0,
        max(
            0.0,
            (rvol_value - 1.0)
            / 2.0
            * 20,
        ),
    )

    score += min(
        12.0,
        body_ratio * 12,
    )

    if (
        0.10
        <= breakout_distance_atr
        <= 0.55
    ):
        score += 12
    elif breakout_distance_atr <= 0.70:
        score += 8
    else:
        score += 4

    score += max(
        0.0,
        10.0 * (
            1
            - min(
                vwap_distance_atr,
                MAX_VWAP_DISTANCE_ATR,
            )
            / MAX_VWAP_DISTANCE_ATR
        ),
    )

    aligned = (
        side == "LONG"
        and nifty_regime == "BULL"
    ) or (
        side == "SHORT"
        and nifty_regime == "BEAR"
    )

    score += 10 if aligned else 6

    score += max(
        0.0,
        8 * (
            1
            - quote.spread_bps
            / MAX_SPREAD_BPS
        ),
    )

    score = min(100.0, score)

    return Setup(
        symbol=quote.symbol,
        token=quote.token,
        side=side,

        price=price,
        prev_close=quote.prev_close,
        day_change_pct=quote.pct_change,
        gap_pct=quote.gap_pct,
        turnover_crore=quote.turnover_crore,
        spread_bps=quote.spread_bps,
        stock_in_play_score=quote.stock_in_play_score,

        opening_range_high=opening_high,
        opening_range_low=opening_low,

        vwap=vwap,
        ema9=ema9,
        ema20=ema20,
        rsi=rsi_value,
        atr=atr_value,
        atr_pct=atr_pct,
        rvol=rvol_value,

        breakout_distance_atr=breakout_distance_atr,
        vwap_distance_atr=vwap_distance_atr,
        candle_body_ratio=body_ratio,
        candle_close_location=close_location,

        nifty_regime=nifty_regime,
        nifty_return_pct=nifty_return_pct,

        technical_score=score,
        signal_at=(
            pd.Timestamp(last["date"]) + pd.Timedelta(minutes=5)
        ).isoformat(),
        lower_circuit_limit=quote.lower_circuit_limit,
        upper_circuit_limit=quote.upper_circuit_limit,
    )


# =============================================================================
# AI final reviewer
# =============================================================================

AI_SYSTEM_PROMPT = """
You are the FINAL QUALITY AND RISK REVIEWER in an automated NSE cash-equity
intraday trading system.

A deterministic strategy has already:
- scanned a broad NSE cash-equity universe,
- filtered for liquidity and tight spreads,
- identified a fresh 15-minute opening-range breakout/breakdown,
- confirmed VWAP, EMA trend, RSI, ATR, recent volume and NIFTY regime.

You DO NOT choose stocks.
You DO NOT determine position size.
You DO NOT override hard risk rules.
You only APPROVE or REJECT the supplied numerical setup.

Be conservative. Reject when:
- the breakout is late or overextended,
- volume confirmation is weak,
- RSI is stretched,
- the stock is too far from VWAP,
- spread/execution quality is poor,
- the candle lacks directional conviction,
- NIFTY regime conflicts,
- the numerical setup is internally inconsistent.

Do not invent news, fundamentals, support/resistance or facts not supplied.
If uncertain, REJECT.
Return only APPROVE or REJECT; ERROR is reserved for local transport failures.

"confidence" means confidence in your APPROVE/REJECT judgment, not expected
percentage return.
"""


class AIFilter:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is missing."
            )

        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=OPENAI_MAX_RETRIES,
        )
        self.last_response_model = ""
        self.last_response_id = ""
        self.last_latency_ms = 0
        self.last_error = ""
        self.last_status = "NOT_RUN"
        self.last_decision_id = ""
        self.last_input_sha256 = ""
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_total_tokens = 0

    def review(
        self,
        candidate_payload: dict[str, Any],
    ) -> AIDecision:
        started = time.monotonic()
        self.last_response_model = ""
        self.last_response_id = ""
        self.last_latency_ms = 0
        self.last_error = ""
        self.last_status = "RUNNING"
        self.last_decision_id = uuid.uuid4().hex
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_total_tokens = 0
        self.last_input_sha256 = stable_json_sha256(candidate_payload)

        try:
            response = self.client.responses.parse(
                model=OPENAI_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": AI_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": (
                                "Review this candidate:\n"
                                + json.dumps(
                                candidate_payload,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                        ),
                    },
                ],
                text_format=AIDecision,
                reasoning={"effort": OPENAI_REASONING_EFFORT},
                max_output_tokens=600,
                store=False,
                metadata={
                    "workflow": "nse_intraday_review",
                    "prompt_version": AI_PROMPT_VERSION,
                },
                timeout=OPENAI_TIMEOUT_SECONDS,
            )

            decision = response.output_parsed

            if decision is None:
                raise RuntimeError(
                    "No structured AI output."
                )
            if decision.decision == "ERROR":
                raise RuntimeError("AI returned reserved ERROR decision")

            self.last_response_model = str(response.model)
            self.last_response_id = str(response.id)
            usage = getattr(response, "usage", None)
            self.last_input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            self.last_output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            self.last_total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            self.last_latency_ms = int(
                (time.monotonic() - started) * 1000
            )
            self.last_status = "OK"
            return decision

        except Exception as exc:
            self.last_latency_ms = int(
                (time.monotonic() - started) * 1000
            )
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_status = "ERROR"
            log(
                f"AI review unavailable: {self.last_error}. "
                "Gate mode blocks; shadow mode remains execution-neutral."
            )

            return AIDecision(
                decision="ERROR",
                confidence=0,
                quality_score=0,
                reason=(
                    "AI service/review failure; "
                    "fail-closed."
                ),
                risk_flags=["AI_FAILURE"],
            )


# =============================================================================
# Risk engine
# =============================================================================

def planned_after_cost_stop_loss(
    side: str,
    entry_price: float,
    stop_price: float,
    qty: int,
) -> float:
    """Conservative planned stop loss including slippage and current charges."""
    if qty <= 0:
        return 0.0
    slippage = RISK_SLIPPAGE_BPS / 10_000
    if side == "LONG":
        entry_fill = entry_price * (1 + slippage)
        exit_fill = stop_price * (1 - slippage)
        costs = estimate_nse_equity_intraday_cost(
            entry_fill * qty,
            exit_fill * qty,
        )
    else:
        entry_fill = entry_price * (1 - slippage)
        exit_fill = stop_price * (1 + slippage)
        costs = estimate_nse_equity_intraday_cost(
            exit_fill * qty,
            entry_fill * qty,
        )
    expected_gross = float(gross_pnl(side, entry_fill, exit_fill, qty))
    return max(0.0, -expected_gross + float(costs.total))


def estimate_after_cost_outcome(
    side: str,
    entry_price: float,
    stop_distance: float,
    target_distance: float,
    qty: int,
    tick_size: float,
    *,
    include_entry_slippage: bool = True,
) -> AfterCostOutcome:
    """Model the actual adverse-entry repricing, exits, and two-leg fees."""
    if qty <= 0:
        return AfterCostOutcome(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if (
        not math.isfinite(tick_size)
        or tick_size <= 0
        or not all(
            math.isfinite(value) and value > 0
            for value in (entry_price, stop_distance, target_distance)
        )
    ):
        raise ValueError("entry, distances and tick size must be positive")

    slippage = RISK_SLIPPAGE_BPS / 10_000
    entry_slippage = slippage if include_entry_slippage else 0.0
    if side == "LONG":
        entry_fill = entry_price * (1 + entry_slippage)
        stop_reference = round_to_tick(entry_fill - stop_distance, tick_size)
        target_reference = round_to_tick(entry_fill + target_distance, tick_size)
        stop_fill = stop_reference * (1 - slippage)
        target_fill = target_reference * (1 - slippage)
        stop_costs = estimate_nse_equity_intraday_cost(
            entry_fill * qty,
            stop_fill * qty,
        )
        target_costs = estimate_nse_equity_intraday_cost(
            entry_fill * qty,
            target_fill * qty,
        )
    elif side == "SHORT":
        entry_fill = entry_price * (1 - entry_slippage)
        stop_reference = round_to_tick(entry_fill + stop_distance, tick_size)
        target_reference = round_to_tick(entry_fill - target_distance, tick_size)
        stop_fill = stop_reference * (1 + slippage)
        target_fill = target_reference * (1 + slippage)
        stop_costs = estimate_nse_equity_intraday_cost(
            stop_fill * qty,
            entry_fill * qty,
        )
        target_costs = estimate_nse_equity_intraday_cost(
            target_fill * qty,
            entry_fill * qty,
        )
    else:
        raise ValueError("side must be LONG or SHORT")

    stop_gross = float(gross_pnl(side, entry_fill, stop_fill, qty))
    target_gross = float(gross_pnl(side, entry_fill, target_fill, qty))
    stop_loss = max(0.0, -stop_gross + float(stop_costs.total))
    target_profit = target_gross - float(target_costs.total)
    payoff = target_profit / stop_loss if stop_loss > 0 else 0.0
    return AfterCostOutcome(
        entry_fill=entry_fill,
        stop_reference=stop_reference,
        target_reference=target_reference,
        stop_fill=stop_fill,
        target_fill=target_fill,
        stop_loss=stop_loss,
        target_profit=target_profit,
        payoff_ratio=payoff,
    )


def validate_price_band_geometry(
    side: str,
    stop_price: float,
    target_price: float,
    lower_circuit: float,
    upper_circuit: float,
    ltp: float,
    tick_size: float,
) -> tuple[bool, str]:
    values = (
        stop_price,
        target_price,
        lower_circuit,
        upper_circuit,
        ltp,
        tick_size,
    )
    if any(not math.isfinite(value) or value <= 0 for value in values):
        return False, "INVALID_PRICE_BAND_DATA"
    if not lower_circuit < ltp < upper_circuit:
        return False, "INVALID_PRICE_BAND_ORDER"
    headroom = max(2 * tick_size, ltp * CIRCUIT_HEADROOM_BPS / 10_000)
    if side == "LONG":
        if stop_price < lower_circuit + headroom:
            return False, "LONG_STOP_OUTSIDE_PRICE_BAND"
        if target_price > upper_circuit - headroom:
            return False, "LONG_TARGET_OUTSIDE_PRICE_BAND"
    elif side == "SHORT":
        if stop_price > upper_circuit - headroom:
            return False, "SHORT_STOP_OUTSIDE_PRICE_BAND"
        if target_price < lower_circuit + headroom:
            return False, "SHORT_TARGET_OUTSIDE_PRICE_BAND"
    else:
        return False, "INVALID_TRADE_SIDE"
    return True, "OK"


def max_qty_within_stop_budget(
    side: str,
    entry_price: float,
    stop_price: float,
    max_qty: int,
    budget: float,
) -> int:
    """Binary-search the largest integer size whose planned net loss fits."""
    low, high, result = 1, max_qty, 0
    while low <= high:
        candidate = (low + high) // 2
        loss = planned_after_cost_stop_loss(
            side,
            entry_price,
            stop_price,
            candidate,
        )
        if loss <= budget:
            result = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    return result


def _candidate_reserved_risk(trade: Trade) -> float:
    for value in (trade.reserved_risk_amount, trade.planned_risk_amount):
        if math.isfinite(value) and value > 0:
            return float(value)
    estimated = planned_after_cost_stop_loss(
        trade.side,
        trade.entry_price,
        trade.stop_price,
        trade.qty,
    )
    return float(estimated)


def entry_capacity(
    state: dict,
    *,
    exclude_symbol: str | None = None,
) -> EntryCapacity:
    """Compute remaining loss and notional capacity from durable active state."""
    if state.get("kill_switch"):
        return EntryCapacity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, "KILL_SWITCH")

    open_risk = 0.0
    open_notional = 0.0
    try:
        for symbol, record in state.get("trades", {}).items():
            if exclude_symbol and str(symbol).upper() == exclude_symbol.upper():
                continue
            if not isinstance(record, dict):
                raise RuntimeError("INVALID_ACTIVE_TRADE_RECORD")
            status = str(record.get("status", "")).upper()
            if status not in ACTIVE_TRADE_STATUSES and not status.startswith("OPEN"):
                continue
            trade = trade_from_dict(record)
            risk = trade.reserved_risk_amount or trade.planned_risk_amount
            if not math.isfinite(risk) or risk <= 0:
                raise RuntimeError(f"{trade.symbol}_ACTIVE_RISK_UNRESOLVED")
            qty = trade.qty or trade.requested_qty
            notional = max(
                abs(trade.entry_price * qty),
                float(trade.reserved_notional_amount or 0.0),
            )
            if not math.isfinite(notional) or notional <= 0:
                raise RuntimeError(f"{trade.symbol}_ACTIVE_NOTIONAL_UNRESOLVED")
            open_risk += risk
            open_notional += notional
    except Exception as exc:
        return EntryCapacity(
            open_risk,
            open_notional,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            False,
            f"ACTIVE_RISK_INVALID_{type(exc).__name__}:{exc}",
        )

    try:
        realized = strict_finite_float(
            state.get("realized_pnl", 0.0),
            field="state.realized_pnl",
        )
    except ValueError as exc:
        return EntryCapacity(
            open_risk,
            open_notional,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            False,
            f"REALIZED_PNL_INVALID:{exc}",
        )

    # Profits never expand the day's risk allowance; losses consume it.
    realized_debit = min(0.0, realized)
    daily_remaining = max(
        0.0,
        CAPITAL_LIMIT * MAX_DAILY_LOSS_PCT + realized_debit - open_risk,
    )
    portfolio_remaining = max(
        0.0,
        CAPITAL_LIMIT * MAX_PORTFOLIO_STOP_RISK_PCT - open_risk,
    )
    gross_remaining = max(
        0.0,
        CAPITAL_LIMIT * MAX_GROSS_EXPOSURE_PCT - open_notional,
    )
    risk_budget = max(
        0.0,
        min(
            CAPITAL_LIMIT * RISK_PER_TRADE_PCT,
            daily_remaining,
            portfolio_remaining,
        ),
    )
    notional_budget = max(
        0.0,
        min(CAPITAL_LIMIT * MAX_POSITION_PCT, gross_remaining),
    )
    allowed = risk_budget > 0 and notional_budget > 0
    reason = "OK" if allowed else "PORTFOLIO_CAPACITY_EXHAUSTED"
    return EntryCapacity(
        open_reserved_risk=open_risk,
        open_gross_notional=open_notional,
        daily_risk_remaining=daily_remaining,
        portfolio_risk_remaining=portfolio_remaining,
        gross_notional_remaining=gross_remaining,
        candidate_risk_budget=risk_budget,
        candidate_notional_budget=notional_budget,
        allowed=allowed,
        reason=reason,
    )


def assess_trade_admission(
    state: dict,
    trade: Trade,
    *,
    exclude_symbol: str | None = None,
) -> tuple[bool, str]:
    capacity = entry_capacity(state, exclude_symbol=exclude_symbol)
    if not capacity.allowed:
        return False, capacity.reason
    candidate_risk = _candidate_reserved_risk(trade)
    candidate_notional = max(
        abs(trade.entry_price * (trade.qty or trade.requested_qty)),
        float(trade.reserved_notional_amount or 0.0),
    )
    if not math.isfinite(candidate_risk) or candidate_risk <= 0:
        return False, "CANDIDATE_RISK_INVALID"
    if not math.isfinite(candidate_notional) or candidate_notional <= 0:
        return False, "CANDIDATE_NOTIONAL_INVALID"
    if candidate_risk > capacity.candidate_risk_budget + 1e-9:
        return False, "REMAINING_RISK_BUDGET_EXCEEDED"
    if candidate_notional > capacity.candidate_notional_budget + 1e-9:
        return False, "REMAINING_GROSS_EXPOSURE_EXCEEDED"
    return True, "OK"

def revalidate_live_setup(
    broker: KiteBroker,
    setup: Setup,
) -> tuple[Setup | None, str]:
    """Recheck signal age and price after the potentially slow AI call."""
    try:
        signal_at = datetime.fromisoformat(setup.signal_at)
    except (TypeError, ValueError):
        return None, "INVALID_SIGNAL_TIMESTAMP"

    if signal_at.tzinfo is None:
        signal_at = pd.Timestamp(signal_at, tz=IST).to_pydatetime()

    age_seconds = (now_ist() - signal_at).total_seconds()
    if age_seconds < -5 or age_seconds > MAX_SIGNAL_AGE_SECONDS:
        return None, f"STALE_SIGNAL_{age_seconds:.0f}s"

    live_spread = setup.spread_bps
    lower_circuit = setup.lower_circuit_limit
    upper_circuit = setup.upper_circuit_limit
    try:
        execution_snapshot = getattr(broker, "execution_snapshot", None)
        if execution_snapshot is None:
            live_price = broker.ltp(setup.symbol)
        else:
            observed = execution_snapshot(setup.symbol)
            if isinstance(observed, ExecutionSnapshot):
                live_price = observed.ltp
                live_spread = observed.spread_bps
                lower_circuit = observed.lower_circuit
                upper_circuit = observed.upper_circuit
                circuit_buffer = min(
                    (upper_circuit - live_price) / live_price * 100,
                    (live_price - lower_circuit) / live_price * 100,
                )
            else:
                # Backward-compatible test/broker adapter shape. New real
                # adapters must return ExecutionSnapshot so directional bands
                # are preserved.
                live_price, live_spread, circuit_buffer = observed
            if live_spread > MAX_SPREAD_BPS:
                return None, f"LIVE_SPREAD_{live_spread:.1f}BPS"
            if circuit_buffer < MIN_CIRCUIT_BUFFER_PCT:
                return None, f"LIVE_CIRCUIT_BUFFER_{circuit_buffer:.2f}PCT"
    except Exception as exc:
        return None, f"EXECUTION_QUOTE_FAILED_{type(exc).__name__}"

    if live_price <= 0 or setup.atr <= 0:
        return None, "INVALID_LIVE_PRICE"

    drift_atr = abs(live_price - setup.price) / setup.atr
    if drift_atr > MAX_ENTRY_DRIFT_ATR:
        return None, f"PRICE_DRIFT_{drift_atr:.2f}ATR"

    if setup.side == "LONG":
        breakout_atr = (
            live_price - setup.opening_range_high
        ) / setup.atr
        if live_price <= max(setup.opening_range_high, setup.vwap):
            return None, "LONG_BREAKOUT_FAILED"
    else:
        breakout_atr = (
            setup.opening_range_low - live_price
        ) / setup.atr
        if live_price >= min(setup.opening_range_low, setup.vwap):
            return None, "SHORT_BREAKOUT_FAILED"

    if not (
        MIN_BREAKOUT_DISTANCE_ATR
        <= breakout_atr
        <= MAX_BREAKOUT_DISTANCE_ATR
    ):
        return None, f"LIVE_BREAKOUT_DISTANCE_{breakout_atr:.2f}ATR"

    return replace(
        setup,
        price=live_price,
        spread_bps=live_spread,
        breakout_distance_atr=breakout_atr,
        vwap_distance_atr=abs(live_price - setup.vwap) / setup.atr,
        lower_circuit_limit=lower_circuit,
        upper_circuit_limit=upper_circuit,
    ), "OK"


def build_trade_result(
    broker: KiteBroker,
    setup: Setup,
    *,
    risk_budget: float | None = None,
    notional_budget: float | None = None,
) -> TradeBuildResult:
    inst = broker.instrument(
        setup.symbol
    )

    if not inst:
        return TradeBuildResult(None, "INSTRUMENT_UNAVAILABLE")

    stop_distance = (
        setup.atr
        * ATR_STOP_MULTIPLIER
    )

    stop_pct = (
        stop_distance
        / setup.price
    )

    if (
        stop_pct < 0.003
        or stop_pct > 0.03
    ):
        return TradeBuildResult(None, f"STOP_DISTANCE_{stop_pct:.4f}")

    rupee_risk_budget = float(
        risk_budget
        if risk_budget is not None
        else CAPITAL_LIMIT * RISK_PER_TRADE_PCT
    )
    max_notional = float(
        notional_budget
        if notional_budget is not None
        else CAPITAL_LIMIT * MAX_POSITION_PCT
    )
    if rupee_risk_budget <= 0 or max_notional <= 0:
        return TradeBuildResult(None, "NO_REMAINING_ENTRY_CAPACITY")

    qty_by_risk = math.floor(
        rupee_risk_budget
        / stop_distance
    )

    conservative_notional_price = setup.price * (
        1 + RISK_SLIPPAGE_BPS / 10_000
    )
    qty_by_notional = math.floor(max_notional / conservative_notional_price)

    qty = max(
        0,
        min(
            qty_by_risk,
            qty_by_notional,
        ),
    )

    if setup.side == "LONG":
        raw_stop = (
            setup.price
            - stop_distance
        )
        raw_target = (
            setup.price
            + stop_distance
            * TARGET_R_MULTIPLE
        )
    else:
        raw_stop = (
            setup.price
            + stop_distance
        )
        raw_target = (
            setup.price
            - stop_distance
            * TARGET_R_MULTIPLE
        )

    stop_price = round_to_tick(raw_stop, inst.tick_size)
    target_price = round_to_tick(raw_target, inst.tick_size)
    band_ok, band_reason = validate_price_band_geometry(
        setup.side,
        stop_price,
        target_price,
        setup.lower_circuit_limit,
        setup.upper_circuit_limit,
        setup.price,
        inst.tick_size,
    )
    if not band_ok:
        return TradeBuildResult(None, band_reason)

    qty = max_qty_within_stop_budget(
        setup.side,
        setup.price,
        stop_price,
        qty,
        rupee_risk_budget,
    )

    if qty < 1:
        return TradeBuildResult(None, "NO_QUANTITY_WITHIN_AFTER_COST_RISK")

    planned_risk = planned_after_cost_stop_loss(
        setup.side,
        setup.price,
        stop_price,
        qty,
    )
    outcome = estimate_after_cost_outcome(
        setup.side,
        setup.price,
        stop_distance,
        stop_distance * TARGET_R_MULTIPLE,
        qty,
        inst.tick_size,
    )
    modeled_band_ok, modeled_band_reason = validate_price_band_geometry(
        setup.side,
        outcome.stop_reference,
        outcome.target_reference,
        setup.lower_circuit_limit,
        setup.upper_circuit_limit,
        outcome.entry_fill,
        inst.tick_size,
    )
    if not modeled_band_ok:
        return TradeBuildResult(None, f"MODELED_{modeled_band_reason}", outcome)
    if outcome.target_profit <= 0:
        return TradeBuildResult(None, "NONPOSITIVE_AFTER_COST_TARGET", outcome)
    if outcome.payoff_ratio < MIN_AFTER_COST_PAYOFF_RATIO:
        return TradeBuildResult(
            None,
            (
                f"AFTER_COST_PAYOFF_{outcome.payoff_ratio:.3f}_LT_"
                f"{MIN_AFTER_COST_PAYOFF_RATIO:.3f}"
            ),
            outcome,
        )

    trade = Trade(
        symbol=setup.symbol,
        token=setup.token,
        side=setup.side,
        qty=qty,
        entry_price=setup.price,
        initial_risk_per_share=stop_distance,
        stop_price=stop_price,
        target_price=target_price,
        opened_at=now_ist().isoformat(),
        client_tag=new_order_tag("TRD"),
        requested_qty=qty,
        planned_risk_amount=planned_risk,
        reserved_risk_amount=planned_risk,
        reserved_notional_amount=conservative_notional_price * qty,
        planned_target_profit_amount=outcome.target_profit,
        planned_after_cost_payoff=outcome.payoff_ratio,
        execution_mode=current_execution_mode(),
    )
    return TradeBuildResult(trade, "OK", outcome)


def build_trade(
    broker: KiteBroker,
    setup: Setup,
    *,
    risk_budget: float | None = None,
    notional_budget: float | None = None,
) -> Trade | None:
    """Backward-compatible wrapper for callers that only need the trade."""
    return build_trade_result(
        broker,
        setup,
        risk_budget=risk_budget,
        notional_budget=notional_budget,
    ).trade


# =============================================================================
# Execution / monitoring
# =============================================================================

def persist_trade(state: dict, trade: Trade) -> None:
    """Atomically mirror a trade to memory only if the durable write succeeds."""
    runtime_mode = current_execution_mode()
    trade.execution_mode = trade.execution_mode or runtime_mode
    if trade.execution_mode != runtime_mode:
        raise RuntimeError(
            f"refusing to persist {trade.execution_mode} trade in "
            f"{runtime_mode} runtime"
        )
    missing = object()
    previous_mode = state.get("execution_mode", missing)
    previous_trade = state["trades"].get(trade.symbol, missing)
    state["execution_mode"] = runtime_mode
    state["trades"][trade.symbol] = asdict(trade)
    try:
        save_state(state)
    except Exception:
        if previous_mode is missing:
            state.pop("execution_mode", None)
        else:
            state["execution_mode"] = previous_mode
        if previous_trade is missing:
            state["trades"].pop(trade.symbol, None)
        else:
            state["trades"][trade.symbol] = previous_trade
        raise


def halt_trading(state: dict, reason: str) -> None:
    state["kill_switch"] = True
    state["halt_reason"] = reason
    try:
        save_state(state)
    except Exception as exc:
        log(f"STATE WRITE FAILURE while halting: {type(exc).__name__}: {exc}")
    journal_best_effort("HALT", reason=reason)
    log(f"TRADING HALTED: {reason}")


def paper_fill_price(
    reference_price: float,
    transaction_type: str,
) -> float:
    adjustment = PAPER_SLIPPAGE_BPS / 10_000
    if transaction_type == "BUY":
        return reference_price * (1 + adjustment)
    return reference_price * (1 - adjustment)


def submit_or_recover_order(
    broker: KiteBroker,
    submit,
    *,
    tag: str,
    symbol: str,
    transaction_type: str,
    order_type: str,
    quantity: int,
) -> str:
    """Submit once; reconcile an ambiguous transport failure by unique tag."""
    try:
        return str(submit())
    except Exception as submit_error:
        if not LIVE_TRADING:
            raise

        deadline = time.monotonic() + 5
        last_lookup_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                match = broker.find_exact_order_by_tag(
                    tag=tag,
                    symbol=symbol,
                    transaction_type=transaction_type,
                    order_type=order_type,
                    quantity=quantity,
                )
                if match is not None:
                    log(
                        f"Recovered ambiguous {order_type} submission "
                        f"from tag {tag}: {match.order_id}"
                    )
                    return match.order_id
                last_lookup_error = None
            except Exception as exc:
                last_lookup_error = exc
            time.sleep(ORDER_POLL_SECONDS)

        raise RuntimeError(
            f"UNKNOWN broker submission for tag {tag}; do not retry. "
            f"submit={type(submit_error).__name__}: {submit_error}; "
            f"lookup={last_lookup_error}"
        ) from submit_error


def emergency_flatten_without_state(
    broker: KiteBroker,
    trade: Trade,
    reason: str,
    *,
    unresolved_order_intent: bool = False,
) -> bool:
    """Last-resort live flatten that does not depend on disk or journals."""
    if not LIVE_TRADING:
        return True
    inst = broker.instrument(trade.symbol)
    if inst is None:
        log(f"CRITICAL {trade.symbol}: emergency flatten has no instrument")
        return False
    if unresolved_order_intent:
        log(
            f"CRITICAL {trade.symbol}: emergency flatten cannot prove an "
            "ambiguous order intent absent; manual reconciliation required"
        )
        return False

    def stable_position(expected: int) -> bool:
        if broker.position_qty(trade.symbol) != expected:
            return False
        time.sleep(min(max(ORDER_POLL_SECONDS, 0.01), 0.50))
        return broker.position_qty(trade.symbol) == expected

    try:
        # Never race a separate exit against an owned entry, stop, or prior
        # exit. Unresolved cancellation is a manual-reconciliation condition.
        if not cancel_active_trade_orders(broker, trade):
            log(
                f"CRITICAL {trade.symbol}: emergency flatten blocked because "
                "an owned order is not confirmed terminal"
            )
            return False

        reducing_ids = list(
            dict.fromkeys(
                [trade.stop_order_id, trade.exit_order_id, *trade.exit_order_ids]
            )
        )
        if any(reducing_ids):
            reduced = broker_filled_quantity(broker, reducing_ids)
            if reduced > trade.qty:
                log(
                    f"CRITICAL {trade.symbol}: confirmed reducing fills exceed "
                    "the tracked entry; refusing another exit"
                )
                return False
            if reduced == trade.qty:
                # The broker already confirms a full reducing fill. A nonzero
                # position can be settlement lag; another order could reverse.
                return stable_position(0)

        signed_qty = broker.position_qty(trade.symbol)
        if signed_qty == 0:
            return stable_position(0)
        tag = new_order_tag("EXT")
        order_id = submit_or_recover_order(
            broker,
            lambda: broker.exit_market(inst, signed_qty, tag),
            tag=tag,
            symbol=trade.symbol,
            transaction_type="SELL" if signed_qty > 0 else "BUY",
            order_type="MARKET",
            quantity=abs(signed_qty),
        )
        trade.exit_tag = tag
        trade.exit_tags.append(tag)
        trade.exit_order_id = order_id
        if order_id not in trade.exit_order_ids:
            trade.exit_order_ids.append(order_id)
        result = broker.wait_for_order(
            order_id,
            timeout_seconds=EXIT_FILL_TIMEOUT_SECONDS,
        )
        if not result.terminal:
            result = broker.cancel_order_confirmed(
                order_id,
                timeout_seconds=EXIT_FILL_TIMEOUT_SECONDS,
            )
        flat = bool(
            result
            and result.terminal
            and broker.wait_for_position_qty(
                trade.symbol,
                0,
                timeout_seconds=EXIT_FILL_TIMEOUT_SECONDS,
            )
            and stable_position(0)
        )
        if not flat:
            log(
                f"CRITICAL {trade.symbol}: emergency flatten not verified ({reason})"
            )
        return flat
    except Exception as exc:
        log(
            f"CRITICAL {trade.symbol}: state-independent emergency flatten "
            f"failed ({reason}): {type(exc).__name__}: {exc}"
        )
        return False


def update_prices_from_entry(
    trade: Trade,
    inst: Instrument,
    actual_entry: float,
) -> None:
    trade.entry_price = actual_entry
    if trade.side == "LONG":
        trade.stop_price = round_to_tick(
            actual_entry - trade.initial_risk_per_share,
            inst.tick_size,
        )
        trade.target_price = round_to_tick(
            actual_entry
            + trade.initial_risk_per_share * TARGET_R_MULTIPLE,
            inst.tick_size,
        )
    else:
        trade.stop_price = round_to_tick(
            actual_entry + trade.initial_risk_per_share,
            inst.tick_size,
        )
        trade.target_price = round_to_tick(
            actual_entry
            - trade.initial_risk_per_share * TARGET_R_MULTIPLE,
            inst.tick_size,
        )


def refresh_trade_economics_after_fill(trade: Trade, inst: Instrument) -> None:
    """Recompute nonlinear fee/risk metrics for the confirmed filled quantity."""
    outcome = estimate_after_cost_outcome(
        trade.side,
        trade.entry_price,
        trade.initial_risk_per_share,
        trade.initial_risk_per_share * TARGET_R_MULTIPLE,
        trade.qty,
        inst.tick_size,
        include_entry_slippage=False,
    )
    trade.planned_risk_amount = outcome.stop_loss
    trade.reserved_risk_amount = outcome.stop_loss
    trade.reserved_notional_amount = abs(trade.entry_price * trade.qty)
    trade.planned_target_profit_amount = outcome.target_profit
    trade.planned_after_cost_payoff = outcome.payoff_ratio


# =============================================================================
# Verified execution / monitoring lifecycle
# =============================================================================

def broker_fill_average(
    broker: KiteBroker,
    order_ids: list[str | None],
) -> float:
    """Return the quantity-weighted broker execution price for known orders."""
    ids = {str(value) for value in order_ids if value}
    if not ids:
        return 0.0

    total_qty = 0
    total_value = 0.0
    try:
        rows = broker.trades()
        if not isinstance(rows, list):
            raise RuntimeError("broker trades payload must be a list")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(f"broker trade row {index} is not an object")
            if str(row.get("order_id")) not in ids:
                continue
            qty = strict_integral(
                row.get("quantity"),
                field=f"trade[{index}].quantity",
            )
            price_value = row.get("average_price") or row.get("price")
            price = strict_finite_float(
                price_value,
                field=f"trade[{index}].price",
            )
            if qty <= 0 or price <= 0:
                raise RuntimeError(f"broker trade row {index} has invalid fill")
            total_qty += qty
            total_value += qty * price
    except (ValueError, RuntimeError):
        raise
    except Exception:
        # A transport failure can use fully resolved order snapshots below.
        total_qty = 0
        total_value = 0.0
    if total_qty > 0:
        return total_value / total_qty

    snapshots: list[OrderSnapshot] = []
    for order_id in ids:
        try:
            snapshot = broker.latest_order(order_id)
            if snapshot.filled > 0 and snapshot.avg > 0:
                snapshots.append(snapshot)
        except Exception:
            continue
    denominator = sum(snapshot.filled for snapshot in snapshots)
    if denominator <= 0:
        return 0.0
    return sum(snapshot.filled * snapshot.avg for snapshot in snapshots) / denominator


def broker_filled_quantity(
    broker: KiteBroker,
    order_ids: list[str | None],
) -> int:
    """Return cumulative executed quantity once per known order ID."""
    ids = {str(value) for value in order_ids if value}
    if not ids:
        return 0
    quantities = {order_id: 0 for order_id in ids}
    resolved_ids: set[str] = set()
    try:
        rows = broker.trades()
        if not isinstance(rows, list):
            raise RuntimeError("broker trades payload must be a list")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(f"broker trade row {index} is not an object")
            order_id = str(row.get("order_id"))
            if order_id not in ids:
                continue
            quantity = strict_integral(
                row.get("quantity"),
                field=f"trade[{index}].quantity",
            )
            if quantity <= 0:
                raise RuntimeError(f"broker trade row {index} has invalid quantity")
            quantities[order_id] += quantity
            resolved_ids.add(order_id)
    except (ValueError, RuntimeError):
        raise
    except Exception:
        # REST unavailability may still be resolved from order snapshots.
        pass
    for order_id in ids:
        try:
            snapshot_filled = broker.latest_order(order_id).filled
            quantities[order_id] = max(quantities[order_id], snapshot_filled)
            resolved_ids.add(order_id)
        except Exception:
            continue
    if resolved_ids != ids:
        missing = ", ".join(sorted(ids - resolved_ids))
        raise RuntimeError(f"could not resolve filled quantity for {missing}")
    return sum(quantities.values())


def wait_for_known_fill_settlement(
    broker: KiteBroker,
    trade: Trade,
) -> bool:
    """Wait until positions reflect every confirmed bot reducing fill."""
    reducing_ids = list(
        dict.fromkeys(
            [trade.stop_order_id, trade.exit_order_id, *trade.exit_order_ids]
        )
    )
    try:
        reduced = broker_filled_quantity(broker, reducing_ids)
    except Exception as exc:
        trade.accounting_uncertain = True
        trade.accounting_note = f"reducing fill quantity unavailable: {exc}"
        return False
    if reduced > trade.qty:
        trade.accounting_uncertain = True
        trade.accounting_note = "confirmed reducing fills exceed entry quantity"
        return False
    remaining = trade.qty - reduced
    expected = remaining if trade.side == "LONG" else -remaining
    return broker.wait_for_position_qty(
        trade.symbol,
        expected,
        timeout_seconds=EXIT_FILL_TIMEOUT_SECONDS,
    )


def _all_trade_order_ids(trade: Trade) -> list[str]:
    values = [
        trade.entry_order_id,
        trade.stop_order_id,
        trade.exit_order_id,
        *trade.exit_order_ids,
    ]
    return list(dict.fromkeys(str(value) for value in values if value))


def cancel_active_trade_orders(
    broker: KiteBroker,
    trade: Trade,
) -> bool:
    """Confirm every known order terminal before declaring a flat trade closed."""
    if not LIVE_TRADING:
        trade.stop_status = "CANCELLED"
        return True

    inst = broker.instrument(trade.symbol)
    if not inst:
        return False

    for order_id in _all_trade_order_ids(trade):
        try:
            snapshot = broker.latest_order(order_id)
            if not snapshot.terminal:
                if not order_owned_for_cancellation(snapshot, trade, inst):
                    log(
                        f"{trade.symbol}: refusing to cancel unowned order "
                        f"{order_id}"
                    )
                    return False
                snapshot = broker.cancel_order_confirmed(
                    order_id,
                    timeout_seconds=EXIT_FILL_TIMEOUT_SECONDS,
                )
            if snapshot is None or not snapshot.terminal:
                return False
            if order_id == trade.stop_order_id:
                trade.stop_status = snapshot.status
        except Exception as exc:
            log(f"{trade.symbol}: cannot terminalize {order_id}: {exc}")
            return False
    return True


def resolve_order_intent(
    broker: KiteBroker,
    *,
    tag: str,
    symbol: str,
    transaction_type: str,
    order_type: str,
    quantities: list[int],
) -> OrderSnapshot | None:
    """Resolve a previously persisted mutation intent without resubmitting it."""
    if not tag:
        return None
    for quantity in dict.fromkeys(abs(value) for value in quantities if value):
        match = broker.find_exact_order_by_tag(
            tag=tag,
            symbol=symbol,
            transaction_type=transaction_type,
            order_type=order_type,
            quantity=quantity,
        )
        if match is not None:
            return match
    return None


def stop_identity_matches(
    stop: OrderSnapshot | None,
    trade: Trade,
    order_qty: int,
    inst: Instrument,
) -> bool:
    """Verify that an order is this trade's intended protective stop."""
    if stop is None or order_qty <= 0:
        return False
    expected_transaction = "SELL" if trade.side == "LONG" else "BUY"
    trigger_tolerance = max(inst.tick_size / 2, 0.01) + 1e-9
    return bool(
        stop.symbol == trade.symbol.upper()
        and stop.exchange == "NSE"
        and stop.product == "MIS"
        and stop.order_type in allowed_broker_order_types("SL-M")
        and stop.transaction_type == expected_transaction
        and stop.qty == order_qty
        and bool(trade.stop_tag)
        and stop.tag == trade.stop_tag
        and abs(stop.trigger_price - trade.stop_price) <= trigger_tolerance
    )


def entry_identity_matches(entry: OrderSnapshot, trade: Trade) -> bool:
    return bool(
        (not trade.entry_order_id or entry.order_id == str(trade.entry_order_id))
        and entry.symbol == trade.symbol.upper()
        and entry.exchange == "NSE"
        and entry.product == "MIS"
        and entry.order_type in allowed_broker_order_types("MARKET")
        and entry.transaction_type == ("BUY" if trade.side == "LONG" else "SELL")
        and entry.qty == (trade.requested_qty or trade.qty)
        and bool(trade.entry_tag)
        and entry.tag == trade.entry_tag
    )


def exit_identity_matches(exit_order: OrderSnapshot, trade: Trade) -> bool:
    expected_transaction = (
        trade.exit_intent_transaction
        or ("SELL" if trade.side == "LONG" else "BUY")
    )
    expected_quantity = abs(trade.exit_intent_qty) if trade.exit_intent_qty else 0
    return bool(
        (
            exit_order.order_id in set(trade.exit_order_ids + [trade.exit_order_id])
            or (
                not trade.exit_order_id
                and exit_order.tag in set(trade.exit_tags)
            )
        )
        and exit_order.symbol == trade.symbol.upper()
        and exit_order.exchange == "NSE"
        and exit_order.product == "MIS"
        and exit_order.order_type in allowed_broker_order_types("MARKET")
        and exit_order.transaction_type == expected_transaction
        and exit_order.qty > 0
        and (not expected_quantity or exit_order.qty == expected_quantity)
        and bool(exit_order.tag)
        and exit_order.tag in set(trade.exit_tags)
    )


def order_owned_for_cancellation(
    snapshot: OrderSnapshot,
    trade: Trade,
    inst: Instrument,
) -> bool:
    """Prove role ownership before any cancel mutation is sent."""
    if snapshot.order_id == trade.entry_order_id:
        return entry_identity_matches(snapshot, trade)
    if snapshot.order_id == trade.stop_order_id:
        expected_transaction = "SELL" if trade.side == "LONG" else "BUY"
        return bool(
            snapshot.symbol == trade.symbol.upper()
            and snapshot.exchange == "NSE"
            and snapshot.product == "MIS"
            and snapshot.order_type in allowed_broker_order_types("SL-M")
            and snapshot.transaction_type == expected_transaction
            and snapshot.qty == trade.qty
            and bool(trade.stop_tag)
            and snapshot.tag == trade.stop_tag
        )
    return exit_identity_matches(snapshot, trade)


def stop_exactly_protects(
    stop: OrderSnapshot | None,
    trade: Trade,
    signed_position_qty: int,
    inst: Instrument,
) -> bool:
    """Verify identity plus exact pending coverage of the signed position."""
    return bool(
        signed_position_qty
        and abs(signed_position_qty) == trade.qty
        and stop_identity_matches(stop, trade, abs(signed_position_qty), inst)
        and stop
        and stop.stop_armed
        and stop.pending == abs(signed_position_qty)
        and ((signed_position_qty > 0) == (trade.side == "LONG"))
    )


def apply_confirmed_entry_snapshot(
    broker: KiteBroker,
    trade: Trade,
    entry: OrderSnapshot,
    inst: Instrument,
) -> None:
    """Restore filled quantity and basis from the broker after a crash/retry."""
    trade.entry_status = entry.status
    if entry.filled <= 0:
        return
    average = entry.avg or broker_fill_average(broker, [entry.order_id])
    if average <= 0:
        trade.accounting_uncertain = True
        trade.accounting_note = "confirmed entry fill price unavailable"
        trade.qty = entry.filled
        return
    trade.qty = entry.filled
    update_prices_from_entry(trade, inst, average)
    refresh_trade_economics_after_fill(trade, inst)


def mark_trade_closed(
    state: dict,
    symbol: str,
    reason: str,
    *,
    exit_price: float = 0.0,
    exit_order_id: str | None = None,
) -> bool:
    """Persist operational closure and after-cost P&L from actual fills."""
    data = state["trades"].get(symbol)
    if not data:
        return False

    trade = trade_from_dict(data)
    accounting_before = {
        key: state.get(key)
        for key in (
            "realized_pnl",
            "fees_paid",
            "consecutive_losses",
            "kill_switch",
            "halt_reason",
        )
    }
    trade.closed_at = now_ist().isoformat()
    trade.exit_reason = reason
    trade.exit_order_id = exit_order_id or trade.exit_order_id
    if trade.exit_order_id and trade.exit_order_id not in trade.exit_order_ids:
        trade.exit_order_ids.append(trade.exit_order_id)

    if (
        trade.accounting_uncertain
        or not math.isfinite(exit_price)
        or exit_price <= 0
    ):
        trade.status = "CLOSED_UNPRICED"
        trade.exit_status = "FLAT_PRICE_UNRESOLVED"
        state["kill_switch"] = True
        state["halt_reason"] = (
            f"{symbol} is flat but its actual exit price is unresolved"
        )
        persist_trade(state, trade)
        journal_best_effort(
            "CLOSE",
            symbol=symbol,
            idea_id=trade.idea_id,
            ai_review_idea_id=trade.ai_review_idea_id,
            reason=reason,
            ai_mode=trade.ai_mode or AI_MODE,
            ai_decision=trade.ai_decision,
            execution_mode=trade.execution_mode or current_execution_mode(),
            pricing_status="UNRESOLVED",
            accounting_note=trade.accounting_note,
        )
        return True

    trade.exit_price = float(exit_price)
    trade.exit_status = "COMPLETE"
    trade.status = f"CLOSED_{reason}"
    trade.gross_pnl = float(
        gross_pnl(
            trade.side,
            trade.entry_price,
            trade.exit_price,
            trade.qty,
        )
    )
    entry_turnover = trade.entry_price * trade.qty
    exit_turnover = trade.exit_price * trade.qty
    costs = (
        estimate_nse_equity_intraday_cost(entry_turnover, exit_turnover)
        if trade.side == "LONG"
        else estimate_nse_equity_intraday_cost(exit_turnover, entry_turnover)
    )
    trade.fees = float(costs.total)
    trade.net_pnl = trade.gross_pnl - trade.fees
    risk_amount = (
        trade.planned_risk_amount
        if trade.planned_risk_amount > 0
        else trade.initial_risk_per_share * trade.qty
    )
    trade.r_multiple = trade.net_pnl / risk_amount if risk_amount > 0 else 0.0

    state["realized_pnl"] = safe_float(state.get("realized_pnl")) + trade.net_pnl
    state["fees_paid"] = safe_float(state.get("fees_paid")) + trade.fees
    if trade.net_pnl < 0:
        state["consecutive_losses"] = int(state.get("consecutive_losses", 0)) + 1
    else:
        state["consecutive_losses"] = 0

    if state["realized_pnl"] <= -(CAPITAL_LIMIT * MAX_DAILY_LOSS_PCT):
        state["kill_switch"] = True
        state["halt_reason"] = "realized daily loss limit reached"
    if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
        state["kill_switch"] = True
        state["halt_reason"] = "consecutive loss limit reached"

    try:
        persist_trade(state, trade)
    except Exception:
        for key, value in accounting_before.items():
            state[key] = value
        state["kill_switch"] = True
        state["halt_reason"] = (
            f"{symbol} closure accounting could not be persisted"
        )
        try:
            save_state(state)
        except Exception:
            pass
        raise
    journal_best_effort(
        "CLOSE",
        symbol=symbol,
        idea_id=trade.idea_id,
        ai_review_idea_id=trade.ai_review_idea_id,
        side=trade.side,
        qty=trade.qty,
        reason=reason,
        entry_price=trade.entry_price,
        exit_price=trade.exit_price,
        gross_pnl=trade.gross_pnl,
        fees=trade.fees,
        net_pnl=trade.net_pnl,
        r_multiple=trade.r_multiple,
        ai_mode=trade.ai_mode or AI_MODE,
        ai_decision=trade.ai_decision,
        ai_valid=trade.ai_valid,
        ai_error=trade.ai_error,
        ai_response_model=trade.ai_response_model,
        ai_response_id=trade.ai_response_id,
        ai_prompt_version=trade.ai_prompt_version,
        ai_decision_id=trade.ai_decision_id,
        ai_input_sha256=trade.ai_input_sha256,
        ai_input_tokens=trade.ai_input_tokens,
        ai_output_tokens=trade.ai_output_tokens,
        ai_total_tokens=trade.ai_total_tokens,
        execution_mode=trade.execution_mode or current_execution_mode(),
        entry_order_id=trade.entry_order_id,
        exit_order_ids=trade.exit_order_ids,
    )
    return True


def close_trade_market(
    broker: KiteBroker,
    state: dict,
    symbol: str,
    reason: str,
    *,
    reference_price: float | None = None,
) -> bool:
    """Flatten by signed broker quantity; CLOSED requires flat + terminal orders."""
    data = state["trades"].get(symbol)
    if not data:
        return False
    trade = trade_from_dict(data)
    if trade.status.startswith("CLOSED"):
        return True
    inst = broker.instrument(symbol)
    if not inst:
        halt_trading(state, f"{symbol}: missing instrument during exit")
        return False

    if not LIVE_TRADING:
        if trade.execution_mode not in {"", "paper"}:
            halt_trading(
                state,
                f"{symbol}: refusing paper close for {trade.execution_mode} trade",
            )
            return False
        if any(not order_id.startswith("DRY-") for order_id in _all_trade_order_ids(trade)):
            halt_trading(
                state,
                f"{symbol}: non-DRY order identity requires live reconciliation",
            )
            return False
        if reference_price is None:
            try:
                reference_price = broker.ltp(symbol)
            except Exception as exc:
                trade.accounting_uncertain = True
                trade.accounting_note = (
                    f"paper exit quote unavailable: {type(exc).__name__}: {exc}"
                )
                trade.exit_order_id = (
                    f"DRY-UNPRICED-{symbol}-{new_order_tag('EXT')}"
                )
                trade.exit_order_ids.append(trade.exit_order_id)
                state["trades"][symbol] = asdict(trade)
                result = mark_trade_closed(
                    state,
                    symbol,
                    reason,
                    exit_price=0.0,
                    exit_order_id=trade.exit_order_id,
                )
                if result:
                    log(f"PAPER CLOSED_UNPRICED {symbol}: {reason}")
                return result
        if reference_price is None or not math.isfinite(reference_price) or reference_price <= 0:
            trade.accounting_uncertain = True
            trade.accounting_note = "paper exit reference price is invalid"
            state["trades"][symbol] = asdict(trade)
            return mark_trade_closed(
                state,
                symbol,
                reason,
                exit_price=0.0,
                exit_order_id=trade.exit_order_id,
            )
        transaction = "SELL" if trade.side == "LONG" else "BUY"
        exit_price = paper_fill_price(float(reference_price), transaction)
        trade.exit_order_id = f"DRY-EXIT-{symbol}-{new_order_tag('EXT')}"
        trade.exit_order_ids.append(trade.exit_order_id)
        trade.stop_status = "COMPLETE" if "STOP" in reason else "CANCELLED"
        state["trades"][symbol] = asdict(trade)
        result = mark_trade_closed(
            state,
            symbol,
            reason,
            exit_price=exit_price,
            exit_order_id=trade.exit_order_id,
        )
        if result:
            log(f"PAPER CLOSED {symbol}: {reason} @ {exit_price:.2f}")
        return result

    trade.status = "EXIT_PENDING"
    try:
        persist_trade(state, trade)
    except Exception as exc:
        halt_trading(
            state,
            f"{symbol}: exit intent persistence failed: {type(exc).__name__}: {exc}",
        )
        emergency_flatten_without_state(
            broker,
            trade,
            "EXIT_INTENT_PERSISTENCE_FAILURE",
        )
        return False

    entry_snapshot: OrderSnapshot | None = None
    if not trade.entry_order_id and trade.entry_tag:
        entry_snapshot = resolve_order_intent(
            broker,
            tag=trade.entry_tag,
            symbol=symbol,
            transaction_type="BUY" if trade.side == "LONG" else "SELL",
            order_type="MARKET",
            quantities=[trade.requested_qty or trade.qty],
        )
        if entry_snapshot is None:
            halt_trading(
                state,
                f"{symbol}: entry submission remains ambiguous; refusing another mutation",
            )
            return False
        trade.entry_order_id = entry_snapshot.order_id
        persist_trade(state, trade)
    elif trade.entry_order_id:
        try:
            entry_snapshot = broker.latest_order(trade.entry_order_id)
        except Exception as exc:
            halt_trading(state, f"{symbol}: entry state unavailable during exit: {exc}")
            return False

    if entry_snapshot is not None and not entry_identity_matches(
        entry_snapshot,
        trade,
    ):
        halt_trading(state, f"{symbol}: entry order identity mismatch")
        return False

    entry_cancelled_now = False
    if entry_snapshot is not None and not entry_snapshot.terminal:
        try:
            entry_snapshot = broker.cancel_order_confirmed(
                trade.entry_order_id,
                timeout_seconds=ENTRY_FILL_TIMEOUT_SECONDS,
            )
            trade.entry_status = entry_snapshot.status if entry_snapshot else "UNKNOWN"
            entry_cancelled_now = True
            persist_trade(state, trade)
        except Exception as exc:
            halt_trading(state, f"{symbol}: entry cancellation unresolved: {exc}")
            return False

    if entry_snapshot is not None:
        apply_confirmed_entry_snapshot(broker, trade, entry_snapshot, inst)
        persist_trade(state, trade)
        if (
            entry_cancelled_now
            and entry_snapshot.filled > 0
            and not wait_for_known_fill_settlement(broker, trade)
        ):
            halt_trading(
                state,
                f"{symbol}: entry cancellation fills have not settled",
            )
            return False

    qty_now = broker.position_qty(symbol)

    if not trade.stop_order_id and trade.stop_tag:
        stop_intent = resolve_order_intent(
            broker,
            tag=trade.stop_tag,
            symbol=symbol,
            transaction_type="SELL" if trade.side == "LONG" else "BUY",
            order_type="SL-M",
            quantities=[trade.qty, abs(qty_now)],
        )
        if stop_intent is None:
            halt_trading(
                state,
                f"{symbol}: stop submission remains ambiguous; manual reconciliation required",
            )
            return False
        trade.stop_order_id = stop_intent.order_id
        trade.stop_status = stop_intent.status
        persist_trade(state, trade)

    if (
        not trade.exit_order_id
        and trade.exit_tag
        and trade.exit_tags
        and trade.status == "EXIT_PENDING"
    ):
        exit_intent = resolve_order_intent(
            broker,
            tag=trade.exit_tag,
            symbol=symbol,
            transaction_type=(
                trade.exit_intent_transaction
                or (
                    "SELL"
                    if qty_now > 0 or (qty_now == 0 and trade.side == "LONG")
                    else "BUY"
                )
            ),
            order_type="MARKET",
            quantities=(
                [abs(trade.exit_intent_qty)]
                if trade.exit_intent_qty
                else [abs(qty_now), trade.qty]
            ),
        )
        if exit_intent is None:
            halt_trading(
                state,
                f"{symbol}: exit submission remains ambiguous; refusing duplicate exit",
            )
            return False
        trade.exit_order_id = exit_intent.order_id
        if exit_intent.order_id not in trade.exit_order_ids:
            trade.exit_order_ids.append(exit_intent.order_id)
        persist_trade(state, trade)

    if trade.exit_order_id and trade.exit_order_id != trade.stop_order_id:
        try:
            prior_exit = broker.latest_order(trade.exit_order_id)
            if not exit_identity_matches(prior_exit, trade):
                raise RuntimeError("previous exit identity mismatch")
            if not prior_exit.terminal:
                prior_exit = broker.cancel_order_confirmed(
                    trade.exit_order_id,
                    timeout_seconds=EXIT_FILL_TIMEOUT_SECONDS,
                )
            if prior_exit is None or not prior_exit.terminal:
                raise RuntimeError("previous exit is still active")
            trade.exit_status = prior_exit.status
            persist_trade(state, trade)
            if prior_exit.status == "COMPLETE" and not broker.wait_for_position_qty(
                symbol, 0, timeout_seconds=EXIT_FILL_TIMEOUT_SECONDS
            ):
                if not wait_for_known_fill_settlement(broker, trade):
                    raise RuntimeError("previous exit fills have not settled")
            elif prior_exit.filled > 0 and not wait_for_known_fill_settlement(
                broker,
                trade,
            ):
                raise RuntimeError("previous exit fills have not settled")
            if prior_exit.status == "COMPLETE" and broker.position_qty(symbol) != 0:
                trade.accounting_uncertain = True
                trade.accounting_note = (
                    "completed exit did not flatten the tracked entry quantity"
                )
                persist_trade(state, trade)
                raise RuntimeError("previous COMPLETE exit left a position")
            if prior_exit.status == "REJECTED":
                log(
                    f"{symbol}: prior exit rejected; a bounded signed-residual "
                    f"retry may be attempted: {prior_exit.message}"
                )
            if prior_exit.status in {"CANCELLED", "REJECTED"}:
                if broker.position_qty(symbol) != 0:
                    # Preserve the terminal order in exit_order_ids for fills
                    # and fees, but clear the current slot before a new intent.
                    trade.exit_order_id = None
                    trade.exit_tag = ""
                    persist_trade(state, trade)
        except Exception as exc:
            halt_trading(state, f"{symbol}: prior exit reconciliation failed: {exc}")
            return False

    qty_now = broker.position_qty(symbol)

    if qty_now != 0 and trade.stop_order_id:
        try:
            stop = broker.latest_order(trade.stop_order_id)
        except Exception as exc:
            halt_trading(state, f"{symbol}: stop state unavailable during exit: {exc}")
            return False

        original_stop_identity = stop_identity_matches(
            stop,
            trade,
            trade.qty,
            inst,
        )
        converted_stop_identity = bool(
            trade.exit_order_id == trade.stop_order_id
            and order_owned_for_cancellation(stop, trade, inst)
        )
        if not (original_stop_identity or converted_stop_identity):
            halt_trading(
                state,
                f"{symbol}: protective stop identity mismatch; refusing mutation",
            )
            return False

        if stop.status == "COMPLETE" and stop.filled == 0:
            halt_trading(
                state,
                f"{symbol}: stop COMPLETE without a confirmed fill quantity",
            )
            return False
        if stop.terminal and stop.filled > 0:
            if not (original_stop_identity or converted_stop_identity):
                halt_trading(
                    state,
                    f"{symbol}: terminal stop identity does not match the trade",
                )
                return False
            if not wait_for_known_fill_settlement(broker, trade):
                halt_trading(
                    state,
                    f"{symbol}: stop fill/position settlement is inconsistent; "
                    "refusing a second exit",
                )
                return False
            qty_now = broker.position_qty(symbol)

        can_convert = (
            original_stop_identity
            and stop_exactly_protects(stop, trade, qty_now, inst)
        )
        if can_convert:
            trade.exit_order_id = trade.stop_order_id
            if trade.stop_order_id not in trade.exit_order_ids:
                trade.exit_order_ids.append(trade.stop_order_id)
            persist_trade(state, trade)
            try:
                broker.convert_stop_to_market(trade.stop_order_id, abs(qty_now))
            except Exception as exc:
                log(f"{symbol}: stop-to-market request uncertain: {exc}")
            final_stop = broker.wait_for_order(
                trade.stop_order_id,
                timeout_seconds=EXIT_FILL_TIMEOUT_SECONDS,
            )
            trade.stop_status = final_stop.status
            trade.exit_status = final_stop.status
            persist_trade(state, trade)
            if not final_stop.terminal:
                halt_trading(
                    state,
                    f"{symbol}: stop-to-market unresolved ({final_stop.status})",
                )
                return False
            if final_stop.status == "REJECTED":
                halt_trading(state, f"{symbol}: stop-to-market rejected")
                return False
            if final_stop.filled > 0 and not wait_for_known_fill_settlement(
                broker,
                trade,
            ):
                halt_trading(
                    state,
                    f"{symbol}: stop-to-market fills have not settled",
                )
                return False
        elif not stop.terminal:
            try:
                stop = broker.cancel_order_confirmed(
                    trade.stop_order_id,
                    timeout_seconds=EXIT_FILL_TIMEOUT_SECONDS,
                )
                trade.stop_status = stop.status if stop else "UNKNOWN"
                persist_trade(state, trade)
                if stop and stop.filled > 0 and not wait_for_known_fill_settlement(
                    broker,
                    trade,
                ):
                    raise RuntimeError("stop cancellation fills have not settled")
            except Exception as exc:
                halt_trading(state, f"{symbol}: stop cancellation unresolved: {exc}")
                emergency_flatten_without_state(
                    broker,
                    trade,
                    "STOP_CANCELLATION_UNRESOLVED",
                )
                return False

    qty_now = broker.position_qty(symbol)
    if qty_now != 0:
        if trade.exit_attempts >= MAX_EXIT_ATTEMPTS:
            halt_trading(
                state,
                f"{symbol}: exhausted {MAX_EXIT_ATTEMPTS} verified exit attempts",
            )
            return False
        trade.exit_order_id = None
        trade.exit_status = ""
        trade.exit_tag = new_order_tag("EXT")
        trade.exit_tags.append(trade.exit_tag)
        trade.exit_attempts += 1
        trade.exit_intent_qty = qty_now
        trade.exit_intent_transaction = "SELL" if qty_now > 0 else "BUY"
        try:
            persist_trade(state, trade)  # durable intent before mutation
        except Exception as exc:
            halt_trading(
                state,
                f"{symbol}: exit order intent persistence failed: {exc}",
            )
            emergency_flatten_without_state(
                broker,
                trade,
                "EXIT_ORDER_INTENT_PERSISTENCE_FAILURE",
            )
            return False
        transaction = trade.exit_intent_transaction
        try:
            exit_order_id = submit_or_recover_order(
                broker,
                lambda: broker.exit_market(inst, qty_now, trade.exit_tag),
                tag=trade.exit_tag,
                symbol=symbol,
                transaction_type=transaction,
                order_type="MARKET",
                quantity=abs(qty_now),
            )
            trade.exit_order_id = exit_order_id
            if exit_order_id not in trade.exit_order_ids:
                trade.exit_order_ids.append(exit_order_id)
            persist_trade(state, trade)
            exit_order = broker.wait_for_order(
                exit_order_id,
                timeout_seconds=EXIT_FILL_TIMEOUT_SECONDS,
            )
            if not exit_order.terminal:
                exit_order = broker.cancel_order_confirmed(
                    exit_order_id,
                    timeout_seconds=EXIT_FILL_TIMEOUT_SECONDS,
                )
            if exit_order is None or not exit_order.terminal:
                raise RuntimeError("exit order did not become terminal")
            trade.exit_status = exit_order.status
            persist_trade(state, trade)
            if exit_order.status == "REJECTED":
                raise RuntimeError(f"exit rejected: {exit_order.message}")
            if exit_order.filled > 0 and not wait_for_known_fill_settlement(
                broker,
                trade,
            ):
                raise RuntimeError(
                    f"exit {exit_order.status} fills have not settled"
                )
            if broker.position_qty(symbol) != 0:
                raise RuntimeError(
                    f"exit {exit_order.status} left a verified residual position"
                )
        except Exception as exc:
            halt_trading(
                state,
                f"{symbol}: exit state uncertain: {type(exc).__name__}: {exc}",
            )
            emergency_flatten_without_state(
                broker,
                trade,
                "EXIT_STATE_UNCERTAIN",
                unresolved_order_intent=bool(
                    trade.exit_tag and not trade.exit_order_id
                ),
            )
            return False

    if broker.position_qty(symbol) != 0:
        halt_trading(state, f"{symbol}: refusing CLOSED while position is nonzero")
        return False

    if (
        entry_snapshot is not None
        and entry_snapshot.filled == 0
        and not trade.stop_order_id
    ):
        trade.status = "ABORTED"
        trade.entry_status = entry_snapshot.status
        persist_trade(state, trade)
        journal(
            "ENTRY_ABORTED",
            symbol=symbol,
            order_id=trade.entry_order_id,
            status=entry_snapshot.status,
        )
        return True
    if not cancel_active_trade_orders(broker, trade):
        halt_trading(state, f"{symbol}: orphan order could not be terminalized")
        return False
    if broker.position_qty(symbol) != 0:
        halt_trading(state, f"{symbol}: position changed during final reconciliation")
        return False

    exit_price = broker_fill_average(
        broker,
        [*trade.exit_order_ids, trade.exit_order_id, trade.stop_order_id],
    )
    result = mark_trade_closed(
        state,
        symbol,
        reason,
        exit_price=exit_price,
        exit_order_id=trade.exit_order_id,
    )
    if result:
        log(f"CLOSED {symbol}: {reason}")
    return result


def fail_safe_trade_lifecycle(
    broker: KiteBroker,
    state: dict,
    trade: Trade,
    reason: str,
    *,
    unresolved_order_intent: bool = False,
) -> None:
    """Halt and guarantee best-effort protection/flatten after any mutation."""
    trade.status = "HALTED_UNCERTAIN"
    trade.execution_mode = trade.execution_mode or current_execution_mode()
    state.setdefault("trades", {})[trade.symbol] = asdict(trade)
    try:
        save_state(state)
    except Exception as exc:
        log(
            f"{trade.symbol}: state unavailable during failure handling: "
            f"{type(exc).__name__}: {exc}"
        )
    halt_trading(state, reason)
    try:
        close_trade_market(
            broker,
            state,
            trade.symbol,
            "EXECUTION_FAILURE",
        )
    except Exception as exc:
        log(
            f"{trade.symbol}: normal failure flatten unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
    if LIVE_TRADING:
        try:
            still_open = broker.position_qty(trade.symbol) != 0
        except Exception:
            still_open = True
        if still_open:
            emergency_flatten_without_state(
                broker,
                trade,
                reason,
                unresolved_order_intent=unresolved_order_intent,
            )


def execute_trade(
    broker: KiteBroker,
    trade: Trade,
    setup: Setup,
    ai_decision: AIDecision,
    state: dict,
) -> None:
    """Persist each intent before mutation and verify entry + protection."""
    inst = broker.instrument(trade.symbol)
    if not inst:
        log(f"{trade.symbol}: instrument disappeared before entry.")
        return
    if state.get("kill_switch"):
        log(f"{trade.symbol}: entry blocked by kill switch.")
        return
    if not entry_window_open():
        journal_best_effort(
            "SIGNAL_REJECTED",
            symbol=trade.symbol,
            side=trade.side,
            reason="ENTRY_CUTOFF",
            evaluated_at=now_ist().isoformat(),
            deadline=entry_deadline(now_ist()).isoformat(),
        )
        return

    trade.execution_mode = trade.execution_mode or current_execution_mode()
    trade.requested_qty = trade.requested_qty or trade.qty
    if trade.reserved_risk_amount <= 0:
        trade.reserved_risk_amount = _candidate_reserved_risk(trade)
    if trade.planned_risk_amount <= 0:
        trade.planned_risk_amount = trade.reserved_risk_amount
    admitted, admission_reason = assess_trade_admission(state, trade)
    if not admitted:
        journal_best_effort(
            "SIGNAL_REJECTED",
            symbol=trade.symbol,
            side=trade.side,
            reason=admission_reason,
        )
        return
    if LIVE_TRADING:
        try:
            verify_live_account_matches_state(broker, state)
            for payload in broker.orders():
                snapshot = OrderSnapshot.from_payload(payload)
                if (
                    not snapshot.terminal
                    and snapshot.exchange == "NSE"
                    and snapshot.product == "MIS"
                    and snapshot.symbol == trade.symbol.upper()
                ):
                    raise RuntimeError(
                        f"active same-symbol order exists: {snapshot.order_id}"
                    )
        except Exception as exc:
            halt_trading(
                state,
                f"{trade.symbol}: just-in-time account preflight failed: {exc}",
            )
            return

    # Account calls above can consume the remaining entry window or portfolio
    # budget. This second check is the authoritative pre-mutation gate.
    if not entry_window_open():
        journal_best_effort(
            "SIGNAL_REJECTED",
            symbol=trade.symbol,
            side=trade.side,
            reason="ENTRY_CUTOFF_AFTER_PREFLIGHT",
        )
        return
    admitted, admission_reason = assess_trade_admission(state, trade)
    if not admitted:
        journal_best_effort(
            "SIGNAL_REJECTED",
            symbol=trade.symbol,
            side=trade.side,
            reason=admission_reason,
        )
        return

    trade.ai_decision = ai_decision.decision if AI_MODE != "off" else "OFF"
    trade.entry_tag = new_order_tag("ENT")
    trade.stop_tag = new_order_tag("STP")
    trade.status = "ENTRY_INTENT"
    try:
        persist_trade(state, trade)
        journal(
            "ORDER_INTENT",
            role="ENTRY",
            symbol=trade.symbol,
            side=trade.side,
            qty=trade.requested_qty,
            tag=trade.entry_tag,
            reserved_risk=trade.reserved_risk_amount,
            planned_stop_tag=trade.stop_tag,
        )
    except Exception as exc:
        halt_trading(
            state,
            f"{trade.symbol}: durable entry intent failed before submission: {exc}",
        )
        return

    # Durable telemetry above can cross the guarded cutoff. No broker mutation
    # is allowed after that boundary, even though an entry intent exists.
    if not entry_window_open():
        trade.status = "ABORTED"
        trade.reserved_risk_amount = 0.0
        try:
            persist_trade(state, trade)
        except Exception as exc:
            halt_trading(
                state,
                f"{trade.symbol}: cutoff-abort persistence failed: {exc}",
            )
            return
        journal_best_effort(
            "SIGNAL_REJECTED",
            symbol=trade.symbol,
            side=trade.side,
            reason="ENTRY_CUTOFF_AFTER_INTENT",
        )
        return

    if LIVE_TRADING:
        transaction = "BUY" if trade.side == "LONG" else "SELL"
        try:
            trade.entry_order_id = submit_or_recover_order(
                broker,
                lambda: broker.place_market_entry(
                    inst,
                    trade.side,
                    trade.requested_qty,
                    trade.entry_tag,
                ),
                tag=trade.entry_tag,
                symbol=trade.symbol,
                transaction_type=transaction,
                order_type="MARKET",
                quantity=trade.requested_qty,
            )
            trade.status = "ENTRY_SUBMITTED"
            persist_trade(state, trade)
            entry = broker.wait_for_order(
                trade.entry_order_id,
                timeout_seconds=ENTRY_FILL_TIMEOUT_SECONDS,
                return_on_partial=True,
            )
            if not entry.terminal:
                entry = broker.cancel_order_confirmed(
                    trade.entry_order_id,
                    timeout_seconds=ENTRY_FILL_TIMEOUT_SECONDS,
                )
            if entry is None or not entry.terminal:
                raise RuntimeError("entry order did not reach a terminal state")
            if not entry_identity_matches(entry, trade):
                raise RuntimeError("entry order identity mismatch")
            trade.entry_status = entry.status
            if entry.filled <= 0:
                trade.qty = 0
                trade.reserved_risk_amount = 0.0
                trade.status = "ABORTED"
                persist_trade(state, trade)
                journal_best_effort(
                    "ENTRY_ABORTED",
                    symbol=trade.symbol,
                    order_id=trade.entry_order_id,
                    status=entry.status,
                    message=entry.message,
                )
                return
            trade.qty = entry.filled
            actual_entry = entry.avg or broker_fill_average(
                broker,
                [trade.entry_order_id],
            )
            if actual_entry <= 0:
                raise RuntimeError("confirmed entry fill has no execution price")
        except Exception as exc:
            fail_safe_trade_lifecycle(
                broker,
                state,
                trade,
                f"{trade.symbol} entry state uncertain: {type(exc).__name__}: {exc}",
                unresolved_order_intent=bool(
                    trade.entry_tag and not trade.entry_order_id
                ),
            )
            return
    else:
        trade.entry_order_id = f"DRY-ENTRY-{trade.symbol}-{trade.entry_tag}"
        trade.entry_status = "COMPLETE"
        trade.qty = trade.requested_qty
        actual_entry = paper_fill_price(
            trade.entry_price,
            "BUY" if trade.side == "LONG" else "SELL",
        )

    update_prices_from_entry(trade, inst, actual_entry)
    refresh_trade_economics_after_fill(trade, inst)
    actual_admitted, actual_admission_reason = assess_trade_admission(
        state,
        trade,
        exclude_symbol=trade.symbol,
    )
    if not actual_admitted:
        fail_safe_trade_lifecycle(
            broker,
            state,
            trade,
            (
                f"{trade.symbol}: actual fill exceeds admission limits: "
                f"{actual_admission_reason}"
            ),
        )
        return
    band_ok, band_reason = validate_price_band_geometry(
        trade.side,
        trade.stop_price,
        trade.target_price,
        setup.lower_circuit_limit,
        setup.upper_circuit_limit,
        trade.entry_price,
        inst.tick_size,
    )
    if not band_ok:
        fail_safe_trade_lifecycle(
            broker,
            state,
            trade,
            f"{trade.symbol}: actual fill invalidated price band: {band_reason}",
        )
        return

    # Persist the complete protection intent before any optional telemetry.
    trade.status = "STOP_SUBMITTED"
    state["trades_today"] += 1
    if trade.symbol not in state["blocked_symbols"]:
        state["blocked_symbols"].append(trade.symbol)
    try:
        persist_trade(state, trade)
    except Exception as exc:
        fail_safe_trade_lifecycle(
            broker,
            state,
            trade,
            f"{trade.symbol}: protection intent persistence failed: {exc}",
        )
        return
    journal_best_effort(
        "ENTRY_FILLED",
        symbol=trade.symbol,
        side=trade.side,
        requested_qty=trade.requested_qty,
        filled_qty=trade.qty,
        entry=trade.entry_price,
        order_id=trade.entry_order_id,
        status=trade.entry_status,
        reserved_risk=trade.reserved_risk_amount,
    )
    journal_best_effort(
        "ORDER_INTENT",
        role="STOP",
        symbol=trade.symbol,
        qty=trade.qty,
        trigger=trade.stop_price,
        tag=trade.stop_tag,
    )

    if LIVE_TRADING:
        stop_transaction = "SELL" if trade.side == "LONG" else "BUY"
        try:
            trade.stop_order_id = submit_or_recover_order(
                broker,
                lambda: broker.place_protective_stop(
                    inst,
                    trade.side,
                    trade.qty,
                    trade.stop_price,
                    trade.stop_tag,
                ),
                tag=trade.stop_tag,
                symbol=trade.symbol,
                transaction_type=stop_transaction,
                order_type="SL-M",
                quantity=trade.qty,
            )
            persist_trade(state, trade)
            stop = broker.wait_for_order(
                trade.stop_order_id,
                timeout_seconds=STOP_ARM_TIMEOUT_SECONDS,
                require_stop_armed=True,
            )
            trade.stop_status = stop.status
            persist_trade(state, trade)
            expected_qty = trade.qty if trade.side == "LONG" else -trade.qty
            position_matches = broker.wait_for_position_qty(
                trade.symbol,
                expected_qty,
                timeout_seconds=STOP_ARM_TIMEOUT_SECONDS,
            )
            if not (
                stop_exactly_protects(stop, trade, expected_qty, inst)
                and position_matches
            ):
                if stop.status == "COMPLETE" and broker.wait_for_position_qty(
                    trade.symbol,
                    0,
                    timeout_seconds=EXIT_FILL_TIMEOUT_SECONDS,
                ):
                    exit_price = stop.avg or broker_fill_average(
                        broker,
                        [trade.stop_order_id],
                    )
                    mark_trade_closed(
                        state,
                        trade.symbol,
                        "STOP_IMMEDIATE",
                        exit_price=exit_price,
                        exit_order_id=trade.stop_order_id,
                    )
                    return
                raise RuntimeError(
                    "protective stop not verified: "
                    f"status={stop.status}, pending={stop.pending}, "
                    f"position_matches={position_matches}"
                )
        except Exception as exc:
            fail_safe_trade_lifecycle(
                broker,
                state,
                trade,
                (
                    f"{trade.symbol} protection failure: "
                    f"{type(exc).__name__}: {exc}"
                ),
                unresolved_order_intent=bool(
                    trade.stop_tag and not trade.stop_order_id
                ),
            )
            return
    else:
        trade.stop_order_id = f"DRY-STOP-{trade.symbol}-{trade.stop_tag}"
        trade.stop_status = "TRIGGER PENDING"

    trade.status = "OPEN_PROTECTED"
    try:
        persist_trade(state, trade)
    except Exception as exc:
        fail_safe_trade_lifecycle(
            broker,
            state,
            trade,
            f"{trade.symbol}: protected-state persistence failed: {exc}",
        )
        return
    journal_best_effort(
        "OPEN",
        symbol=trade.symbol,
        idea_id=trade.idea_id,
        ai_review_idea_id=trade.ai_review_idea_id,
        side=trade.side,
        qty=trade.qty,
        entry=trade.entry_price,
        stop=trade.stop_price,
        target=trade.target_price,
        entry_order_id=trade.entry_order_id,
        stop_order_id=trade.stop_order_id,
        technical_score=setup.technical_score,
        ai_mode=trade.ai_mode or AI_MODE,
        ai_decision=trade.ai_decision,
        ai_valid=trade.ai_valid,
        ai_error=trade.ai_error,
        ai_response_model=trade.ai_response_model,
        ai_response_id=trade.ai_response_id,
        ai_prompt_version=trade.ai_prompt_version,
        ai_decision_id=trade.ai_decision_id,
        ai_input_sha256=trade.ai_input_sha256,
        ai_input_tokens=trade.ai_input_tokens,
        ai_output_tokens=trade.ai_output_tokens,
        ai_total_tokens=trade.ai_total_tokens,
        planned_risk=trade.planned_risk_amount,
        reserved_risk=trade.reserved_risk_amount,
        planned_target_profit=trade.planned_target_profit_amount,
        planned_after_cost_payoff=trade.planned_after_cost_payoff,
        ai_confidence=ai_decision.confidence,
        ai_quality=ai_decision.quality_score,
        ai_reason=ai_decision.reason,
    )
    log(
        f"OPENED/PROTECTED {trade.symbol}: entry={trade.entry_price:.2f} "
        f"stop={trade.stop_price:.2f} target={trade.target_price:.2f}"
    )


def monitor_open_trades(
    broker: KiteBroker,
    state: dict,
) -> None:
    for symbol, data in list(state["trades"].items()):
        trade = trade_from_dict(data)
        if trade.status != "OPEN_PROTECTED":
            if str(trade.status).upper() in ACTIVE_TRADE_STATUSES:
                halt_trading(
                    state,
                    f"{symbol}: runtime recovery required for {trade.status}",
                )
                if LIVE_TRADING:
                    close_trade_market(
                        broker,
                        state,
                        symbol,
                        "RUNTIME_LIFECYCLE_RECOVERY",
                    )
            continue
        if LIVE_TRADING:
            qty_now = broker.position_qty(symbol)
            if qty_now == 0:
                close_trade_market(
                    broker,
                    state,
                    symbol,
                    "STOP_OR_EXTERNAL_EXIT",
                )
                continue
            try:
                stop = (
                    broker.latest_order(trade.stop_order_id)
                    if trade.stop_order_id
                    else None
                )
            except Exception as exc:
                stop = None
                log(f"{symbol}: protective-stop check failed: {exc}")
            inst = broker.instrument(symbol)
            exactly_protected = bool(
                inst and stop_exactly_protects(stop, trade, qty_now, inst)
            )
            if not exactly_protected:
                if abs(qty_now) != trade.qty:
                    trade.accounting_uncertain = True
                    trade.accounting_note = (
                        "broker position quantity changed outside the tracked lifecycle"
                    )
                    try:
                        persist_trade(state, trade)
                    except Exception as exc:
                        halt_trading(
                            state,
                            f"{symbol}: protection-loss state persistence failed: {exc}",
                        )
                        emergency_flatten_without_state(
                            broker,
                            trade,
                            "PROTECTION_LOSS_PERSISTENCE_FAILURE",
                        )
                        continue
                halt_trading(
                    state,
                    f"{symbol}: live position is no longer exactly protected",
                )
                close_trade_market(
                    broker,
                    state,
                    symbol,
                    "PROTECTION_LOST",
                )
                continue
        try:
            price = broker.ltp(symbol)
        except Exception as exc:
            log(f"{symbol}: monitor LTP error: {exc}")
            continue

        stop_hit = (
            trade.side == "LONG" and price <= trade.stop_price
        ) or (
            trade.side == "SHORT" and price >= trade.stop_price
        )
        target_hit = (
            trade.side == "LONG" and price >= trade.target_price
        ) or (
            trade.side == "SHORT" and price <= trade.target_price
        )
        if stop_hit:
            stop_reference = (
                min(price, trade.stop_price)
                if trade.side == "LONG"
                else max(price, trade.stop_price)
            )
            close_trade_market(
                broker,
                state,
                symbol,
                "STOP",
                reference_price=stop_reference,
            )
        elif target_hit:
            close_trade_market(
                broker,
                state,
                symbol,
                "TARGET",
                reference_price=trade.target_price,
            )


def flatten_all(
    broker: KiteBroker,
    state: dict,
    reason: str,
) -> bool:
    log(f"FLATTEN ALL: {reason}")
    success = True
    for symbol, data in list(state["trades"].items()):
        if str(data.get("status", "")).upper() in ACTIVE_TRADE_STATUSES:
            try:
                success = close_trade_market(
                    broker,
                    state,
                    symbol,
                    reason,
                ) and success
            except Exception as exc:
                success = False
                log(f"CRITICAL flatten failure {symbol}: {exc}")
    if LIVE_TRADING:
        remaining = sorted(
            symbol
            for symbol, quantity in broker_mis_position_quantities(broker).items()
            if quantity != 0
        )
        if remaining:
            success = False
            halt_trading(
                state,
                "positions remain after flatten: " + ", ".join(remaining),
            )
    return success


def enforce_daily_pnl_limit(broker: KiteBroker, state: dict) -> None:
    """Fail closed when authoritative daily P&L is unavailable or breached."""
    try:
        pnl = (
            broker.current_intraday_pnl()
            if LIVE_TRADING
            else strict_finite_float(
                state.get("realized_pnl"),
                field="state.realized_pnl",
            )
        )
    except Exception as exc:
        halt_trading(
            state,
            f"authoritative daily P&L unavailable: {type(exc).__name__}: {exc}",
        )
        if open_trade_count(state) > 0:
            flatten_all(broker, state, "PNL_DATA_UNAVAILABLE")
        return

    if pnl <= -(CAPITAL_LIMIT * MAX_DAILY_LOSS_PCT):
        log(f"DAILY LOSS LIMIT reached: ₹{pnl:,.2f}")
        halt_trading(state, "daily loss limit reached")
        flatten_all(broker, state, "DAILY_LOSS_LIMIT")


def reconcile_startup(
    broker: KiteBroker,
    state: dict,
) -> None:
    """Reconcile positions and protection before the event loop accepts work."""
    if not LIVE_TRADING:
        real_positions = sorted(
            symbol
            for symbol, quantity in broker_mis_position_quantities(broker).items()
            if quantity != 0
        )
        active_bot_orders: list[str] = []
        for payload in broker.orders():
            snapshot = OrderSnapshot.from_payload(payload)
            if not snapshot.terminal and is_generated_order_tag(snapshot.tag):
                active_bot_orders.append(snapshot.order_id)
        if real_positions or active_bot_orders:
            details = []
            if real_positions:
                details.append("positions=" + ",".join(real_positions))
            if active_bot_orders:
                details.append("orders=" + ",".join(active_bot_orders))
            raise RuntimeError(
                "Paper mode found real bot/account exposure; authenticated live "
                "reconciliation is required (" + "; ".join(details) + ")"
            )
        return
    tracked = {
        symbol
        for symbol, trade in state["trades"].items()
        if str(trade.get("status", "")).upper() in ACTIVE_TRADE_STATUSES
    }
    positions = {
        symbol: quantity
        for symbol, quantity in broker_mis_position_quantities(broker).items()
        if quantity != 0
    }
    untracked = sorted(set(positions) - tracked)
    if untracked:
        raise RuntimeError(
            "Live MIS positions exist that this bot does not track: "
            + ", ".join(untracked)
            + ". Refusing to start."
        )

    known_ids: dict[str, str] = {}
    known_tags: dict[str, str] = {}
    for owner_symbol, data in state["trades"].items():
        owner = trade_from_dict(data)
        for order_id in _all_trade_order_ids(owner):
            known_ids[order_id] = owner_symbol
        for tag in [
            owner.entry_tag,
            owner.stop_tag,
            owner.exit_tag,
            *owner.exit_tags,
        ]:
            if tag:
                known_tags[tag] = owner_symbol

    try:
        startup_orders = [
            OrderSnapshot.from_payload(payload)
            for payload in broker.orders()
        ]
    except Exception as exc:
        raise RuntimeError("Cannot reconcile the broker order book at startup") from exc

    for snapshot in startup_orders:
        if snapshot.terminal:
            continue
        owner_symbol = known_ids.get(snapshot.order_id) or known_tags.get(snapshot.tag)
        bot_owned = bool(owner_symbol or is_generated_order_tag(snapshot.tag))
        if not bot_owned:
            if snapshot.exchange == "NSE" and snapshot.product == "MIS":
                raise RuntimeError(
                    f"Unowned active NSE/MIS order {snapshot.order_id} exists; "
                    "use an account dedicated to this bot."
                )
            continue
        if owner_symbol in tracked:
            owner = trade_from_dict(state["trades"][owner_symbol])
            owner_inst = broker.instrument(owner_symbol)
            if not owner_inst:
                raise RuntimeError(f"Missing instrument for startup owner {owner_symbol}")
            expected_stop = (
                snapshot.order_id == owner.stop_order_id
                or (
                    not owner.stop_order_id
                    and bool(owner.stop_tag)
                    and snapshot.tag == owner.stop_tag
                )
            )
            if expected_stop:
                # Preserve only the one candidate protection order; all
                # entries, exits, duplicates, and extras are terminalized.
                if not stop_identity_matches(
                    snapshot,
                    owner,
                    owner.qty,
                    owner_inst,
                ):
                    raise RuntimeError(
                        f"Persisted stop identity mismatch for {owner_symbol}"
                    )
                continue
            if not order_owned_for_cancellation(snapshot, owner, owner_inst):
                raise RuntimeError(
                    f"Persisted order identity mismatch for {owner_symbol}: "
                    f"{snapshot.order_id}"
                )
        try:
            terminal = broker.cancel_order_confirmed(
                snapshot.order_id,
                timeout_seconds=EXIT_FILL_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not cancel orphan bot order {snapshot.order_id}"
            ) from exc
        if terminal is None or not terminal.terminal:
            raise RuntimeError(
                f"Orphan bot order {snapshot.order_id} remains active"
            )

    # A cancellation can race a fill. Re-fetch after every orphan mutation
    # before deciding startup is safe.
    positions = {
        symbol: quantity
        for symbol, quantity in broker_mis_position_quantities(broker).items()
        if quantity != 0
    }
    untracked_after_cancels = sorted(set(positions) - tracked)
    if untracked_after_cancels:
        raise RuntimeError(
            "An orphan order changed broker exposure during startup: "
            + ", ".join(untracked_after_cancels)
            + ". Refusing to start."
        )

    for symbol in sorted(tracked):
        trade = trade_from_dict(state["trades"][symbol])
        if trade.status in {
            "ENTRY_INTENT",
            "ENTRY_SUBMITTED",
            "ENTRY_PARTIAL",
            "ENTRY_FILLED",
            "EXIT_PENDING",
            "HALTED_UNCERTAIN",
        }:
            halt_trading(
                state,
                f"{symbol}: recovering incomplete lifecycle state {trade.status}",
            )
            if not close_trade_market(
                broker,
                state,
                symbol,
                "STARTUP_INCOMPLETE_LIFECYCLE",
            ):
                raise RuntimeError(
                    f"{symbol}: incomplete lifecycle requires manual reconciliation"
                )
            continue

        if not trade.entry_order_id:
            raise RuntimeError(f"{symbol}: active trade has no entry order ID")
        try:
            entry = broker.latest_order(trade.entry_order_id)
            if not entry.terminal:
                entry = broker.cancel_order_confirmed(
                    trade.entry_order_id,
                    timeout_seconds=ENTRY_FILL_TIMEOUT_SECONDS,
                )
            if entry is None or not entry.terminal:
                raise RuntimeError("entry remains active")
            if not entry_identity_matches(entry, trade):
                raise RuntimeError("entry identity mismatch")
            inst = broker.instrument(symbol)
            if not inst:
                raise RuntimeError("instrument unavailable")
            apply_confirmed_entry_snapshot(broker, trade, entry, inst)
            persist_trade(state, trade)
        except Exception as exc:
            raise RuntimeError(
                f"{symbol}: entry order could not be terminalized at startup"
            ) from exc

        qty_now = broker.position_qty(symbol)
        if qty_now == 0:
            if not cancel_active_trade_orders(broker, trade):
                raise RuntimeError(
                    f"{symbol}: flat startup reconciliation left an active order"
                )
            if trade.status in {"ENTRY_INTENT", "ENTRY_SUBMITTED"}:
                trade.status = "ABORTED"
                trade.entry_status = "CANCELLED"
                persist_trade(state, trade)
                continue
            exit_price = broker_fill_average(
                broker,
                [*trade.exit_order_ids, trade.exit_order_id, trade.stop_order_id],
            )
            mark_trade_closed(
                state,
                symbol,
                "STARTUP_FLAT_RECONCILED",
                exit_price=exit_price,
                exit_order_id=trade.exit_order_id or trade.stop_order_id,
            )
            continue

        expected_sign = 1 if trade.side == "LONG" else -1
        if qty_now * expected_sign <= 0:
            halt_trading(state, f"{symbol}: broker position direction mismatch")
            if not close_trade_market(
                broker,
                state,
                symbol,
                "STARTUP_DIRECTION_MISMATCH",
            ):
                raise RuntimeError(f"{symbol}: failed emergency startup flatten")
            continue
        try:
            stop = broker.latest_order(trade.stop_order_id) if trade.stop_order_id else None
        except Exception:
            stop = None
        inst = broker.instrument(symbol)
        protected = bool(
            inst and stop_exactly_protects(stop, trade, qty_now, inst)
        )
        if not protected:
            if abs(qty_now) != trade.qty:
                trade.accounting_uncertain = True
                trade.accounting_note = (
                    "startup broker quantity differs from confirmed entry fill"
                )
                persist_trade(state, trade)
            halt_trading(state, f"{symbol}: startup position is not exactly protected")
            if not close_trade_market(
                broker,
                state,
                symbol,
                "STARTUP_UNPROTECTED",
            ):
                raise RuntimeError(f"{symbol}: failed emergency startup flatten")
        else:
            trade.stop_status = stop.status
            trade.status = "OPEN_PROTECTED"
            persist_trade(state, trade)


# =============================================================================
# Full scan
# =============================================================================

def scan_for_new_trades(
    broker: KiteBroker,
    ai: AIFilter | None,
    state: dict,
) -> None:
    if not entry_window_open():
        return
    if (
        state["trades_today"]
        >= MAX_TRADES_PER_DAY
    ):
        return

    if (
        open_trade_count(state)
        >= MAX_OPEN_POSITIONS
    ):
        return

    if not broker.ws_connected.is_set():
        log(
            "WebSocket is not connected. "
            "Skipping new entries."
        )
        return

    blocked = set(
        state.get(
            "blocked_symbols",
            [],
        )
    )

    candidates = select_stocks_in_play(
        broker,
        blocked,
    )

    if not candidates:
        log(
            "No liquid stocks-in-play "
            "passed the broad scan."
        )
        return

    nifty_regime, nifty_return_pct = (
        get_nifty_regime(broker)
    )

    log(
        f"NIFTY regime: {nifty_regime} "
        f"({nifty_return_pct:+.2f}% from open)"
    )

    qualified: list[Setup] = []

    for quote in candidates:
        if not entry_window_open():
            break
        if (
            state["trades_today"]
            >= MAX_TRADES_PER_DAY
        ):
            break

        if (
            open_trade_count(state)
            >= MAX_OPEN_POSITIONS
        ):
            break

        try:
            if open_trade_count(state) > 0:
                monitor_open_trades(broker, state)
            df = broker.strategy_candles(
                quote.token
            )

            time.sleep(
                CANDLE_DELAY_SECONDS
            )

            setup = detect_setup(
                quote,
                df,
                nifty_regime,
                nifty_return_pct,
            )

            if setup is None:
                continue

            if (
                setup.technical_score
                < TECH_MIN_SCORE
            ):
                continue

            qualified.append(setup)

        except Exception as exc:
            log(
                f"{quote.symbol}: "
                f"setup error: {exc}"
            )

    if not qualified:
        log(
            "No fresh high-quality "
            "breakout setups this scan."
        )
        return

    qualified.sort(
        key=lambda x: (-x.technical_score, x.signal_at, x.symbol),
    )

    ai_reviews_used = 0
    for setup in qualified:
        if (
            state["trades_today"]
            >= MAX_TRADES_PER_DAY
        ):
            break

        if (
            open_trade_count(state)
            >= MAX_OPEN_POSITIONS
        ):
            break
        if not entry_window_open():
            journal_best_effort(
                "SIGNAL_REJECTED",
                symbol=setup.symbol,
                side=setup.side,
                reason="ENTRY_CUTOFF_DURING_SCAN",
            )
            break

        log(
            f"CANDIDATE "
            f"{setup.symbol} "
            f"{setup.side}: "
            f"Tech={setup.technical_score:.1f}, "
            f"SIP={setup.stock_in_play_score:.1f}, "
            f"RVOL={setup.rvol:.2f}, "
            f"Turnover=₹{setup.turnover_crore:.1f}cr, "
            f"Spread={setup.spread_bps:.1f}bps"
        )

        refreshed_setup, refresh_reason = revalidate_live_setup(
            broker,
            setup,
        )
        if refreshed_setup is None:
            journal_best_effort(
                "SIGNAL_REJECTED",
                symbol=setup.symbol,
                side=setup.side,
                reason=refresh_reason,
            )
            log(f"{setup.symbol}: live revalidation rejected: {refresh_reason}")
            continue
        setup = refreshed_setup

        capacity = entry_capacity(state)
        if not capacity.allowed:
            journal_best_effort(
                "SIGNAL_REJECTED",
                symbol=setup.symbol,
                side=setup.side,
                reason=capacity.reason,
                capacity=asdict(capacity),
            )
            break
        built = build_trade_result(
            broker,
            setup,
            risk_budget=capacity.candidate_risk_budget,
            notional_budget=capacity.candidate_notional_budget,
        )
        if built.trade is None:
            journal_best_effort(
                "SIGNAL_REJECTED",
                symbol=setup.symbol,
                side=setup.side,
                reason=built.reason,
                setup=asdict(setup),
                outcome=asdict(built.outcome) if built.outcome else None,
                capacity=asdict(capacity),
            )
            continue
        trade = built.trade
        review_candidate = build_ai_candidate_payload(setup, trade, capacity)
        review_idea_id = stable_json_sha256(review_candidate)[:24]
        if AI_MODE != "off":
            review_candidate_logged = journal_best_effort(
                "AI_REVIEW_CANDIDATE",
                idea_id=review_idea_id,
                symbol=setup.symbol,
                side=setup.side,
                candidate=review_candidate,
                candidate_sha256=stable_json_sha256(review_candidate),
                ai_mode=AI_MODE,
            )
            if not review_candidate_logged and AI_MODE == "gate":
                continue

        review_trace: AIFilter | None = None
        if AI_MODE == "off":
            decision = AIDecision(
                decision="APPROVE",
                confidence=100,
                quality_score=min(100, int(round(setup.technical_score))),
                reason="AI disabled; deterministic strategy controls execution.",
                risk_flags=[],
            )
        elif ai is None:
            decision = AIDecision(
                decision="ERROR",
                confidence=0,
                quality_score=0,
                reason="AI unavailable.",
                risk_flags=["AI_UNAVAILABLE"],
            )
        elif ai_reviews_used >= MAX_AI_REVIEWS_PER_SCAN:
            decision = AIDecision(
                decision="ERROR",
                confidence=0,
                quality_score=0,
                reason="Per-scan AI review budget exhausted.",
                risk_flags=["AI_BUDGET_SKIPPED"],
            )
        else:
            decision = ai.review(
                identifier_stripped_ai_payload(review_candidate)
            )
            ai_reviews_used += 1
            review_trace = ai

        log(
            f"AI {setup.symbol}: "
            f"{decision.decision} "
            f"confidence={decision.confidence} "
            f"quality={decision.quality_score} | "
            f"{decision.reason}"
        )

        review_logged = journal_best_effort(
            "AI_REVIEW",
            idea_id=review_idea_id,
            decision_id=getattr(review_trace, "last_decision_id", ""),
            symbol=setup.symbol,
            side=setup.side,
            technical_score=setup.technical_score,
            decision=decision.decision,
            confidence=decision.confidence,
            quality=decision.quality_score,
            reason=decision.reason,
            risk_flags=decision.risk_flags,
            ai_mode=AI_MODE,
            prompt_version=AI_PROMPT_VERSION,
            response_model=getattr(review_trace, "last_response_model", ""),
            response_id=getattr(review_trace, "last_response_id", ""),
            review_status=getattr(review_trace, "last_status", "NOT_RUN"),
            latency_ms=getattr(review_trace, "last_latency_ms", 0),
            error=getattr(review_trace, "last_error", ""),
            input_sha256=getattr(review_trace, "last_input_sha256", ""),
            input_tokens=getattr(review_trace, "last_input_tokens", 0),
            output_tokens=getattr(review_trace, "last_output_tokens", 0),
            total_tokens=getattr(review_trace, "last_total_tokens", 0),
            setup=asdict(setup),
        )

        if (
            AI_MODE == "gate"
            and (
                decision.decision != "APPROVE"
                or decision.confidence < AI_MIN_CONFIDENCE
                or decision.quality_score < 75
                or not review_logged
            )
        ):
            continue

        if not entry_window_open():
            journal_best_effort(
                "SIGNAL_REJECTED",
                symbol=setup.symbol,
                side=setup.side,
                reason="ENTRY_CUTOFF_AFTER_AI",
            )
            break

        final_setup, refresh_reason = revalidate_live_setup(
            broker,
            setup,
        )
        if final_setup is None:
            journal_best_effort(
                "SIGNAL_REJECTED",
                symbol=setup.symbol,
                side=setup.side,
                reason=refresh_reason,
            )
            log(f"{setup.symbol}: live revalidation rejected: {refresh_reason}")
            continue
        setup = final_setup

        final_capacity = entry_capacity(state)
        if not final_capacity.allowed:
            journal_best_effort(
                "SIGNAL_REJECTED",
                symbol=setup.symbol,
                side=setup.side,
                reason=final_capacity.reason,
                capacity=asdict(final_capacity),
            )
            break
        rebuilt = build_trade_result(
            broker,
            setup,
            risk_budget=final_capacity.candidate_risk_budget,
            notional_budget=final_capacity.candidate_notional_budget,
        )
        if rebuilt.trade is None:
            journal_best_effort(
                "SIGNAL_REJECTED",
                symbol=setup.symbol,
                side=setup.side,
                reason=rebuilt.reason,
                outcome=asdict(rebuilt.outcome) if rebuilt.outcome else None,
            )
            continue
        final_trade = rebuilt.trade
        trade = final_trade

        final_candidate = build_ai_candidate_payload(
            setup,
            trade,
            final_capacity,
        )
        trade.idea_id = stable_json_sha256(final_candidate)[:24]
        trade.ai_review_idea_id = review_idea_id
        should_log_ai_candidate = (
            AI_MODE != "off" or AI_IDEA_MODE == "shadow"
        )
        if should_log_ai_candidate:
            candidate_logged = journal_best_effort(
                "AI_CANDIDATE",
                idea_id=trade.idea_id,
                ai_review_idea_id=review_idea_id,
                symbol=setup.symbol,
                side=setup.side,
                candidate=final_candidate,
                candidate_sha256=stable_json_sha256(final_candidate),
                idea_mode=AI_IDEA_MODE,
            )
            if not candidate_logged and AI_MODE == "gate":
                continue

        trade.ai_mode = AI_MODE
        trade.ai_valid = bool(
            AI_MODE in {"shadow", "gate"}
            and review_trace is not None
            and not getattr(review_trace, "last_error", "")
            and decision.decision in {"APPROVE", "REJECT"}
        )
        trade.ai_error = getattr(review_trace, "last_error", "")
        if decision.decision == "ERROR" and not trade.ai_error:
            trade.ai_error = decision.reason
        trade.ai_response_model = getattr(review_trace, "last_response_model", "")
        trade.ai_response_id = getattr(review_trace, "last_response_id", "")
        trade.ai_prompt_version = AI_PROMPT_VERSION
        trade.ai_decision_id = getattr(review_trace, "last_decision_id", "")
        trade.ai_input_sha256 = getattr(review_trace, "last_input_sha256", "")
        trade.ai_input_tokens = getattr(review_trace, "last_input_tokens", 0)
        trade.ai_output_tokens = getattr(review_trace, "last_output_tokens", 0)
        trade.ai_total_tokens = getattr(review_trace, "last_total_tokens", 0)

        execute_trade(
            broker,
            trade,
            setup,
            decision,
            state,
        )


# =============================================================================
# Main
# =============================================================================

def _run_main() -> None:
    log("=" * 78)
    log(
        "ZERODHA KITE + GPT "
        "AI INTRADAY BOT V3"
    )
    log(
        f"LIVE_TRADING={LIVE_TRADING}"
    )
    log(
        f"OpenAI model={OPENAI_MODEL}"
    )
    log(
        f"Capital ceiling="
        f"₹{CAPITAL_LIMIT:,.0f}"
    )
    log(
        f"Risk/trade="
        f"{RISK_PER_TRADE_PCT:.2%} | "
        f"Daily kill="
        f"{MAX_DAILY_LOSS_PCT:.2%} | "
        f"Max trades="
        f"{MAX_TRADES_PER_DAY}"
    )
    log(
        "Universe=dynamic NSE cash equities; "
        "broad scan via Kite WebSocket"
    )
    log("=" * 78)

    if now_ist().weekday() >= 5:
        log("Weekend. Exiting.")
        return

    broker = None
    state = None

    try:
        state = load_state()
        broker = KiteBroker()
        reconcile_startup(broker, state)
        journal(
            "SESSION_START",
            manifest=RUNTIME_MANIFEST,
            manifest_sha256=stable_json_sha256(RUNTIME_MANIFEST),
        )
        ai = (
            AIFilter()
            if AI_MODE != "off" and OPENAI_API_KEY
            else None
        )
        if AI_MODE == "shadow" and ai is None:
            log("AI shadow review unavailable; deterministic paper logic continues.")

        next_full_scan = 0.0
        next_monitor = 0.0
        next_pnl_check = 0.0
        ws_disconnect_since: float | None = None

        while True:
            now = now_ist()
            current = now.time()
            monotonic_now = (
                time.monotonic()
            )

            if current < MARKET_OPEN:
                log(
                    "Waiting for market open..."
                )
                time.sleep(15)
                continue

            if current >= SESSION_END:
                if open_trade_count(state) > 0:
                    flattened = flatten_all(
                        broker,
                        state,
                        "SESSION_END_1530",
                    )
                    if not flattened:
                        halt_trading(
                            state,
                            "session ended with unresolved exposure; manual action required",
                        )
                log(
                    "Market session finished."
                )
                break

            if broker.ws_connected.is_set():
                ws_disconnect_since = None
            else:
                if ws_disconnect_since is None:
                    ws_disconnect_since = monotonic_now
                disconnected_for = monotonic_now - ws_disconnect_since
                if (
                    disconnected_for >= MAX_WS_DISCONNECT_SECONDS
                    and open_trade_count(state) > 0
                    and not state.get("kill_switch")
                ):
                    halt_trading(
                        state,
                        f"market-data WebSocket disconnected for "
                        f"{disconnected_for:.0f}s",
                    )
                    flatten_all(
                        broker,
                        state,
                        "MARKET_DATA_DISCONNECT",
                    )

            if (
                monotonic_now
                >= next_monitor
                and open_trade_count(state)
                > 0
            ):
                try:
                    monitor_open_trades(
                        broker,
                        state,
                    )
                except Exception as exc:
                    log(
                        "Position monitor error: "
                        f"{exc}"
                    )

                next_monitor = (
                    monotonic_now
                    + POSITION_MONITOR_EVERY_SECONDS
                )

            # Don't hammer positions() every second.
            if (
                not state.get("kill_switch")
                and monotonic_now
                >= next_pnl_check
            ):
                enforce_daily_pnl_limit(broker, state)

                next_pnl_check = (
                    monotonic_now + 10
                )

            if current >= FORCE_EXIT:
                flattened = flatten_all(
                    broker,
                    state,
                    "FORCE_EXIT_1510",
                )
                if flattened and open_trade_count(state) == 0:
                    log("15:10 IST safety exit verified complete.")
                    break
                halt_trading(
                    state,
                    "15:10 exit not verified; manual action may be required",
                )
                time.sleep(2)
                continue

            if state.get("kill_switch"):
                if open_trade_count(state) > 0:
                    flatten_all(
                        broker,
                        state,
                        "KILL_SWITCH",
                    )
                time.sleep(2)
                continue

            entry_window = entry_window_open(now)

            if (
                entry_window
                and monotonic_now
                >= next_full_scan
                and state["trades_today"]
                < MAX_TRADES_PER_DAY
                and open_trade_count(state)
                < MAX_OPEN_POSITIONS
            ):
                try:
                    scan_for_new_trades(
                        broker,
                        ai,
                        state,
                    )

                except Exception as exc:
                    log(
                        "Full market scan error: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                next_full_scan = (
                    time.monotonic()
                    + FULL_SCAN_EVERY_SECONDS
                )

            time.sleep(1)

    except BaseException:
        if broker is not None and state is not None and open_trade_count(state) > 0:
            try:
                flatten_all(broker, state, "FATAL_PROCESS_ERROR")
            except Exception as flatten_exc:
                log(f"CRITICAL fatal-error flatten failed: {flatten_exc}")
        raise
    finally:
        if broker:
            broker.close()


def main() -> None:
    validate_configuration()
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_graceful_shutdown(signum, frame) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, request_graceful_shutdown)
    try:
        with SingleInstanceLock(LOCK_FILE):
            _run_main()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
