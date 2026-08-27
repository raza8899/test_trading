#!/usr/bin/env python3
"""Strict, cost-aware replay for complete point-in-time opportunity tapes.

The engine is intentionally offline: it imports neither the broker adapter nor
OpenAI.  It re-evaluates frozen scalar setup rules, applies portfolio capacity,
and resolves a recorded five-minute path conservatively.  Its output is always
labelled low fidelity because OHLC bars cannot prove intrabar ordering or fill
availability.

This is not an importer for the bot's trade journals.  Those journals omit the
complete opportunity set and market path.  Missing contract data causes an
``INSUFFICIENT_POINT_IN_TIME_DATA`` error instead of a fabricated backtest.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, time, timedelta
import hashlib
import heapq
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

from strategy_rules import (
    SetupRuleConfig,
    SetupRuleInput,
    SetupRuleResult,
    evaluate_setup_rules,
)
from trading_core import (
    NSE_EQUITY_INTRADAY_FEE_MODEL_VERSION,
    estimate_nse_equity_intraday_cost,
    gross_pnl,
)


IST = ZoneInfo("Asia/Kolkata")
DATASET_SCHEMA_VERSION = "orb-replay-dataset-v1"
OPPORTUNITY_SCHEMA_VERSION = "orb-replay-opportunity-v1"
BAR_SCHEMA_VERSION = "orb-replay-bar-v1"
TRIAL_REGISTRY_SCHEMA_VERSION = "orb-replay-trials-v1"
DECISION_SCOPE = "all_structural_setups_before_rules_and_capacity"
FIDELITY = "DERIVED_LEDGER_BAR_ONLY_LOW_FIDELITY"
MARKET_OPEN = time(9, 15)
SIGNAL_START = time(9, 35)
LAST_ENTRY = time(14, 30)
FORCE_EXIT = time(15, 10)
BAR_INTERVAL = timedelta(minutes=5)
FEE_MODEL_EFFECTIVE_DATE = date(2026, 3, 1)
REPLAY_ENGINE_VERSION = "orb-derived-ledger-replay-v1"


class ReplayDataError(ValueError):
    """Raised when data cannot support an honest point-in-time replay."""

    def __init__(self, detail: str):
        super().__init__(f"INSUFFICIENT_POINT_IN_TIME_DATA: {detail}")


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    created_at: datetime
    sessions: tuple[date, ...]
    source_strategy_fingerprint: str
    raw_tape_sha256: str
    session_rules_version: str
    fee_model_version: str
    opportunities_path: Path
    bars_path: Path


@dataclass(frozen=True, slots=True)
class ReplayOpportunity:
    opportunity_id: str
    scan_id: str
    session_date: date
    symbol: str
    token: int
    decision_at: datetime
    signal_bar_closed_at: datetime
    features_available_at: datetime
    quote_available_at: datetime
    best_bid: float
    best_ask: float
    best_bid_qty: int
    best_ask_qty: int
    tick_size: float
    lower_circuit: float
    upper_circuit: float
    source_data_sha256: str
    rule_input: SetupRuleInput


@dataclass(frozen=True, slots=True)
class MarketBar:
    session_date: date
    symbol: str
    bar_start: datetime
    available_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source_data_sha256: str

    @property
    def bar_end(self) -> datetime:
        return self.bar_start + BAR_INTERVAL


@dataclass(frozen=True)
class ReplayDataset:
    manifest: DatasetManifest
    opportunities: tuple[ReplayOpportunity, ...]
    bars: dict[tuple[date, str], tuple[MarketBar, ...]]

    def subset(self, sessions: Iterable[date]) -> "ReplayDataset":
        selected = frozenset(sessions)
        return ReplayDataset(
            manifest=self.manifest,
            opportunities=tuple(
                item for item in self.opportunities
                if item.session_date in selected
            ),
            bars={
                key: value for key, value in self.bars.items()
                if key[0] in selected
            },
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayDataError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReplayDataError(f"non-finite JSON number: {value}")


def _parse_json(text: str, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ReplayDataError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ReplayDataError(f"malformed JSON in {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReplayDataError(f"{source} must contain a JSON object")
    return payload


def _exact_fields(payload: dict[str, Any], expected: set[str], source: str) -> None:
    missing = sorted(expected - payload.keys())
    extra = sorted(payload.keys() - expected)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if extra:
            detail.append(f"unknown={extra}")
        raise ReplayDataError(f"{source} schema mismatch ({', '.join(detail)})")


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplayDataError(f"{field_name} must be non-empty text")
    return value.strip()


def _sha256_text(value: Any, *, field_name: str) -> str:
    text_value = _text(value, field_name=field_name).lower()
    if len(text_value) != 64 or any(ch not in "0123456789abcdef" for ch in text_value):
        raise ReplayDataError(f"{field_name} must be a lowercase SHA-256 digest")
    return text_value


def _number(value: Any, *, field_name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayDataError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ReplayDataError(f"{field_name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ReplayDataError(f"{field_name} must be >= {minimum}")
    return result


def _integer(value: Any, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReplayDataError(f"{field_name} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ReplayDataError(f"{field_name} must be boolean")
    return value


def _session_date(value: Any, *, field_name: str) -> date:
    text_value = _text(value, field_name=field_name)
    try:
        parsed = date.fromisoformat(text_value)
    except ValueError as exc:
        raise ReplayDataError(f"{field_name} must be YYYY-MM-DD") from exc
    if parsed.isoformat() != text_value:
        raise ReplayDataError(f"{field_name} must be canonical YYYY-MM-DD")
    return parsed


def _timestamp(value: Any, *, field_name: str) -> datetime:
    text_value = _text(value, field_name=field_name)
    try:
        parsed = datetime.fromisoformat(text_value)
    except ValueError as exc:
        raise ReplayDataError(f"{field_name} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReplayDataError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(IST)


def _safe_dataset_file(root: Path, relative_value: Any, *, field_name: str) -> Path:
    relative = Path(_text(relative_value, field_name=field_name))
    if relative.is_absolute() or ".." in relative.parts:
        raise ReplayDataError(f"{field_name} must stay inside the dataset directory")
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    if candidate.parent != root_resolved and root_resolved not in candidate.parents:
        raise ReplayDataError(f"{field_name} escapes the dataset directory")
    if not candidate.is_file():
        raise ReplayDataError(f"missing dataset file: {relative}")
    return candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    raise ReplayDataError(f"blank JSONL line in {path.name}:{line_number}")
                records.append(
                    _parse_json(raw_line, source=f"{path.name}:{line_number}")
                )
    except OSError as exc:
        raise ReplayDataError(f"cannot read {path}: {exc}") from exc
    if not records:
        raise ReplayDataError(f"{path.name} contains no records")
    return records


def _parse_rule_input(payload: Any, *, source: str) -> SetupRuleInput:
    if not isinstance(payload, dict):
        raise ReplayDataError(f"{source}.rule_input must be an object")
    names = {item.name for item in fields(SetupRuleInput)}
    _exact_fields(payload, names, f"{source}.rule_input")
    values: dict[str, Any] = {}
    optional = {
        "prior_post_opening_max_close",
        "prior_post_opening_min_close",
    }
    for name in names:
        value = payload[name]
        if name in optional and value is None:
            values[name] = None
        elif name == "nifty_regime":
            values[name] = _text(value, field_name=f"{source}.rule_input.{name}")
        else:
            values[name] = _number(value, field_name=f"{source}.rule_input.{name}")
    return SetupRuleInput(**values)


def _parse_opportunity(payload: dict[str, Any], *, source: str) -> ReplayOpportunity:
    expected = {
        "schema_version", "opportunity_id", "scan_id", "session_date",
        "symbol", "token", "decision_at", "signal_bar_closed_at",
        "features_available_at", "quote_available_at", "best_bid",
        "best_ask", "best_bid_qty", "best_ask_qty", "tick_size",
        "lower_circuit", "upper_circuit", "source_data_sha256",
        "rule_input",
    }
    _exact_fields(payload, expected, source)
    if payload["schema_version"] != OPPORTUNITY_SCHEMA_VERSION:
        raise ReplayDataError(f"{source} has unsupported opportunity schema")
    session = _session_date(payload["session_date"], field_name=f"{source}.session_date")
    opportunity = ReplayOpportunity(
        opportunity_id=_text(payload["opportunity_id"], field_name=f"{source}.opportunity_id"),
        scan_id=_text(payload["scan_id"], field_name=f"{source}.scan_id"),
        session_date=session,
        symbol=_text(payload["symbol"], field_name=f"{source}.symbol").upper(),
        token=_integer(payload["token"], field_name=f"{source}.token", minimum=1),
        decision_at=_timestamp(payload["decision_at"], field_name=f"{source}.decision_at"),
        signal_bar_closed_at=_timestamp(payload["signal_bar_closed_at"], field_name=f"{source}.signal_bar_closed_at"),
        features_available_at=_timestamp(payload["features_available_at"], field_name=f"{source}.features_available_at"),
        quote_available_at=_timestamp(payload["quote_available_at"], field_name=f"{source}.quote_available_at"),
        best_bid=_number(payload["best_bid"], field_name=f"{source}.best_bid", minimum=0.0),
        best_ask=_number(payload["best_ask"], field_name=f"{source}.best_ask", minimum=0.0),
        best_bid_qty=_integer(payload["best_bid_qty"], field_name=f"{source}.best_bid_qty"),
        best_ask_qty=_integer(payload["best_ask_qty"], field_name=f"{source}.best_ask_qty"),
        tick_size=_number(payload["tick_size"], field_name=f"{source}.tick_size", minimum=0.0),
        lower_circuit=_number(payload["lower_circuit"], field_name=f"{source}.lower_circuit", minimum=0.0),
        upper_circuit=_number(payload["upper_circuit"], field_name=f"{source}.upper_circuit", minimum=0.0),
        source_data_sha256=_sha256_text(payload["source_data_sha256"], field_name=f"{source}.source_data_sha256"),
        rule_input=_parse_rule_input(payload["rule_input"], source=source),
    )
    if opportunity.decision_at.date() != session:
        raise ReplayDataError(f"{source} decision date differs from session_date")
    if opportunity.signal_bar_closed_at.date() != session:
        raise ReplayDataError(f"{source} signal date differs from session_date")
    if (
        opportunity.signal_bar_closed_at.second
        or opportunity.signal_bar_closed_at.microsecond
        or opportunity.signal_bar_closed_at.minute % 5
    ):
        raise ReplayDataError(f"{source} signal close is not five-minute aligned")
    if not SIGNAL_START <= opportunity.decision_at.time().replace(tzinfo=None) < LAST_ENTRY:
        raise ReplayDataError(f"{source} decision is outside the entry window")
    if not (
        opportunity.signal_bar_closed_at
        <= opportunity.features_available_at
        <= opportunity.decision_at
    ):
        raise ReplayDataError(f"{source} feature availability is non-causal")
    if not (
        opportunity.signal_bar_closed_at
        <= opportunity.quote_available_at
        <= opportunity.decision_at
    ):
        raise ReplayDataError(f"{source} quote availability is non-causal")
    if opportunity.decision_at <= opportunity.signal_bar_closed_at:
        raise ReplayDataError(f"{source} attempts a signal-close fill")
    if opportunity.best_bid <= 0 or opportunity.best_ask < opportunity.best_bid:
        raise ReplayDataError(f"{source} has invalid bid/ask geometry")
    if opportunity.best_bid_qty < 1 or opportunity.best_ask_qty < 1:
        raise ReplayDataError(f"{source} has no executable top-of-book quantity")
    if opportunity.tick_size <= 0:
        raise ReplayDataError(f"{source} tick_size must be positive")
    if not (
        0 < opportunity.lower_circuit
        < opportunity.best_bid
        <= opportunity.best_ask
        < opportunity.upper_circuit
    ):
        raise ReplayDataError(f"{source} has invalid circuit geometry")
    return opportunity


def _parse_bar(payload: dict[str, Any], *, source: str) -> MarketBar:
    expected = {
        "schema_version", "session_date", "symbol", "bar_start",
        "available_at", "open", "high", "low", "close", "volume",
        "source_data_sha256",
    }
    _exact_fields(payload, expected, source)
    if payload["schema_version"] != BAR_SCHEMA_VERSION:
        raise ReplayDataError(f"{source} has unsupported bar schema")
    session = _session_date(payload["session_date"], field_name=f"{source}.session_date")
    bar = MarketBar(
        session_date=session,
        symbol=_text(payload["symbol"], field_name=f"{source}.symbol").upper(),
        bar_start=_timestamp(payload["bar_start"], field_name=f"{source}.bar_start"),
        available_at=_timestamp(payload["available_at"], field_name=f"{source}.available_at"),
        open=_number(payload["open"], field_name=f"{source}.open", minimum=0.0),
        high=_number(payload["high"], field_name=f"{source}.high", minimum=0.0),
        low=_number(payload["low"], field_name=f"{source}.low", minimum=0.0),
        close=_number(payload["close"], field_name=f"{source}.close", minimum=0.0),
        volume=_number(payload["volume"], field_name=f"{source}.volume", minimum=0.0),
        source_data_sha256=_sha256_text(payload["source_data_sha256"], field_name=f"{source}.source_data_sha256"),
    )
    if bar.bar_start.date() != session:
        raise ReplayDataError(f"{source} bar date differs from session_date")
    if (
        bar.bar_start.second
        or bar.bar_start.microsecond
        or bar.bar_start.minute % 5
    ):
        raise ReplayDataError(f"{source} bar_start is not five-minute aligned")
    if bar.available_at < bar.bar_end:
        raise ReplayDataError(f"{source} was available before its bar closed")
    if min(bar.open, bar.high, bar.low, bar.close) <= 0:
        raise ReplayDataError(f"{source} prices must be positive")
    if bar.high < max(bar.open, bar.low, bar.close):
        raise ReplayDataError(f"{source} high is inconsistent")
    if bar.low > min(bar.open, bar.high, bar.close):
        raise ReplayDataError(f"{source} low is inconsistent")
    return bar


def _parse_file_descriptor(
    payload: Any,
    *,
    root: Path,
    field_name: str,
) -> Path:
    if not isinstance(payload, dict):
        raise ReplayDataError(f"{field_name} must be an object")
    _exact_fields(payload, {"path", "sha256"}, field_name)
    path = _safe_dataset_file(root, payload["path"], field_name=f"{field_name}.path")
    expected_hash = _sha256_text(payload["sha256"], field_name=f"{field_name}.sha256")
    actual_hash = _file_sha256(path)
    if actual_hash != expected_hash:
        raise ReplayDataError(
            f"{field_name} checksum mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return path


def load_replay_dataset(directory: str | Path) -> ReplayDataset:
    """Load and fully validate a checksummed replay dataset directory."""

    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ReplayDataError(
            f"{root} has no manifest.json; trade journals are not replay data"
        )
    try:
        manifest_payload = _parse_json(
            manifest_path.read_text(encoding="utf-8"),
            source=str(manifest_path),
        )
    except OSError as exc:
        raise ReplayDataError(f"cannot read {manifest_path}: {exc}") from exc

    expected_manifest_fields = {
        "schema_version", "dataset_id", "created_at", "timezone",
        "bar_interval_minutes", "bar_timestamp_semantics", "fidelity",
        "decision_scope", "point_in_time_universe", "survivorship_bias_free",
        "complete_decision_trace", "raw_as_traded_prices",
        "source_strategy_fingerprint", "raw_tape_sha256",
        "session_rules_version", "fee_model_version", "sessions", "files",
    }
    _exact_fields(manifest_payload, expected_manifest_fields, "manifest")
    if manifest_payload["schema_version"] != DATASET_SCHEMA_VERSION:
        raise ReplayDataError("unsupported dataset schema_version")
    if manifest_payload["timezone"] != "Asia/Kolkata":
        raise ReplayDataError("manifest timezone must be Asia/Kolkata")
    if manifest_payload["bar_interval_minutes"] != 5:
        raise ReplayDataError("only five-minute replay bars are supported")
    if manifest_payload["bar_timestamp_semantics"] != "start":
        raise ReplayDataError("bar timestamps must denote interval start")
    if manifest_payload["fidelity"] != FIDELITY:
        raise ReplayDataError(f"fidelity must be {FIDELITY}")
    if manifest_payload["decision_scope"] != DECISION_SCOPE:
        raise ReplayDataError(f"decision_scope must be {DECISION_SCOPE}")
    for name in (
        "point_in_time_universe",
        "survivorship_bias_free",
        "complete_decision_trace",
        "raw_as_traded_prices",
    ):
        if not _boolean(manifest_payload[name], field_name=f"manifest.{name}"):
            raise ReplayDataError(f"manifest.{name} must be true")
    if manifest_payload["fee_model_version"] != NSE_EQUITY_INTRADAY_FEE_MODEL_VERSION:
        raise ReplayDataError(
            "fee_model_version does not match the implemented effective schedule"
        )

    raw_sessions = manifest_payload["sessions"]
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ReplayDataError("manifest.sessions must be a non-empty array")
    sessions = tuple(
        _session_date(value, field_name=f"manifest.sessions[{index}]")
        for index, value in enumerate(raw_sessions)
    )
    if tuple(sorted(set(sessions))) != sessions:
        raise ReplayDataError("manifest.sessions must be unique and chronological")
    if min(sessions) < FEE_MODEL_EFFECTIVE_DATE:
        raise ReplayDataError(
            "the implemented fee schedule is not effective for every session"
        )
    files_payload = manifest_payload["files"]
    if not isinstance(files_payload, dict):
        raise ReplayDataError("manifest.files must be an object")
    _exact_fields(files_payload, {"opportunities", "bars"}, "manifest.files")
    opportunities_path = _parse_file_descriptor(
        files_payload["opportunities"],
        root=root,
        field_name="manifest.files.opportunities",
    )
    bars_path = _parse_file_descriptor(
        files_payload["bars"],
        root=root,
        field_name="manifest.files.bars",
    )
    manifest = DatasetManifest(
        dataset_id=_text(manifest_payload["dataset_id"], field_name="manifest.dataset_id"),
        created_at=_timestamp(manifest_payload["created_at"], field_name="manifest.created_at"),
        sessions=sessions,
        source_strategy_fingerprint=_text(
            manifest_payload["source_strategy_fingerprint"],
            field_name="manifest.source_strategy_fingerprint",
        ),
        raw_tape_sha256=_sha256_text(
            manifest_payload["raw_tape_sha256"],
            field_name="manifest.raw_tape_sha256",
        ),
        session_rules_version=_text(
            manifest_payload["session_rules_version"],
            field_name="manifest.session_rules_version",
        ),
        fee_model_version=manifest_payload["fee_model_version"],
        opportunities_path=opportunities_path,
        bars_path=bars_path,
    )

    opportunities: list[ReplayOpportunity] = []
    seen_opportunity_ids: set[str] = set()
    seen_scan_symbols: set[tuple[date, str, str]] = set()
    scan_decision_times: dict[tuple[date, str], datetime] = {}
    symbols_by_session_token: dict[tuple[date, int], str] = {}
    tokens_by_session_symbol: dict[tuple[date, str], int] = {}
    for index, payload in enumerate(_load_jsonl(opportunities_path), start=1):
        opportunity = _parse_opportunity(
            payload,
            source=f"{opportunities_path.name}:{index}",
        )
        if opportunity.opportunity_id in seen_opportunity_ids:
            raise ReplayDataError(
                f"duplicate opportunity_id: {opportunity.opportunity_id}"
            )
        seen_opportunity_ids.add(opportunity.opportunity_id)
        if opportunity.session_date not in sessions:
            raise ReplayDataError(
                f"opportunity session {opportunity.session_date} is absent from manifest"
            )
        scan_key = (opportunity.session_date, opportunity.scan_id)
        prior_decision_at = scan_decision_times.setdefault(
            scan_key,
            opportunity.decision_at,
        )
        if prior_decision_at != opportunity.decision_at:
            raise ReplayDataError(
                f"scan {opportunity.scan_id} does not have one shared decision timestamp"
            )
        scan_symbol_key = (
            opportunity.session_date,
            opportunity.scan_id,
            opportunity.symbol,
        )
        if scan_symbol_key in seen_scan_symbols:
            raise ReplayDataError(
                f"duplicate scan symbol: {opportunity.scan_id}/{opportunity.symbol}"
            )
        seen_scan_symbols.add(scan_symbol_key)
        token_key = (opportunity.session_date, opportunity.token)
        prior_symbol = symbols_by_session_token.setdefault(
            token_key,
            opportunity.symbol,
        )
        if prior_symbol != opportunity.symbol:
            raise ReplayDataError(
                f"token {opportunity.token} maps to multiple symbols in {opportunity.session_date}"
            )
        symbol_key = (opportunity.session_date, opportunity.symbol)
        prior_token = tokens_by_session_symbol.setdefault(
            symbol_key,
            opportunity.token,
        )
        if prior_token != opportunity.token:
            raise ReplayDataError(
                f"symbol {opportunity.symbol} maps to multiple tokens in {opportunity.session_date}"
            )
        opportunities.append(opportunity)

    grouped_bars: dict[tuple[date, str], list[MarketBar]] = {}
    seen_bar_keys: set[tuple[date, str, datetime]] = set()
    for index, payload in enumerate(_load_jsonl(bars_path), start=1):
        bar = _parse_bar(payload, source=f"{bars_path.name}:{index}")
        if bar.session_date not in sessions:
            raise ReplayDataError(
                f"bar session {bar.session_date} is absent from manifest"
            )
        key = (bar.session_date, bar.symbol, bar.bar_start)
        if key in seen_bar_keys:
            raise ReplayDataError(
                f"duplicate bar: {bar.session_date}/{bar.symbol}/{bar.bar_start.isoformat()}"
            )
        seen_bar_keys.add(key)
        grouped_bars.setdefault((bar.session_date, bar.symbol), []).append(bar)

    frozen_bars = {
        key: tuple(sorted(values, key=lambda item: item.bar_start))
        for key, values in grouped_bars.items()
    }
    for opportunity in opportunities:
        path_key = (opportunity.session_date, opportunity.symbol)
        path = frozen_bars.get(path_key)
        if not path:
            raise ReplayDataError(
                f"no market path for {opportunity.session_date}/{opportunity.symbol}"
            )
        first_required = opportunity.decision_at.replace(
            minute=opportunity.decision_at.minute
            - opportunity.decision_at.minute % 5,
            second=0,
            microsecond=0,
        )
        last_required = datetime.combine(
            opportunity.session_date,
            FORCE_EXIT,
            IST,
        ) - BAR_INTERVAL
        expected_starts = []
        cursor = first_required
        while cursor <= last_required:
            expected_starts.append(cursor)
            cursor += BAR_INTERVAL
        actual_starts = [
            bar.bar_start for bar in path
            if first_required <= bar.bar_start <= last_required
        ]
        if actual_starts != expected_starts:
            raise ReplayDataError(
                f"incomplete five-minute path for "
                f"{opportunity.session_date}/{opportunity.symbol}"
            )

    opportunities.sort(
        key=lambda item: (
            item.session_date,
            item.decision_at,
            item.scan_id,
            item.symbol,
            item.opportunity_id,
        )
    )
    return ReplayDataset(manifest, tuple(opportunities), frozen_bars)


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Frozen setup, execution, sizing, and portfolio assumptions."""

    trial_id: str = "baseline-v1"
    setup_rules: SetupRuleConfig = field(default_factory=SetupRuleConfig)
    min_technical_score: float = 74.0
    capital_limit: float = 100_000.0
    risk_per_trade_pct: float = 0.0020
    max_position_pct: float = 0.25
    max_daily_loss_pct: float = 0.008
    max_trades_per_day: int = 5
    max_open_positions: int = 2
    max_consecutive_losses: int = 3
    max_portfolio_stop_risk_pct: float = 0.004
    max_gross_exposure_pct: float = 0.50
    atr_stop_multiplier: float = 1.20
    target_r_multiple: float = 1.80
    min_after_cost_payoff_ratio: float = 1.20
    entry_slippage_bps: float = 5.0
    exit_slippage_bps: float = 5.0
    exit_spread_multiplier: float = 1.0
    top_of_book_participation: float = 0.25
    max_signal_age_seconds: float = 240.0
    max_quote_age_seconds: float = 10.0
    max_entry_drift_atr: float = 0.30
    entry_cutoff_guard_seconds: int = 10
    min_circuit_buffer_pct: float = 1.0
    circuit_headroom_bps: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.trial_id, str) or not self.trial_id.strip():
            raise ValueError("trial_id must be non-empty")
        numeric_names = (
            "min_technical_score", "capital_limit", "risk_per_trade_pct",
            "max_position_pct", "max_daily_loss_pct",
            "max_portfolio_stop_risk_pct", "max_gross_exposure_pct",
            "atr_stop_multiplier", "target_r_multiple",
            "min_after_cost_payoff_ratio", "entry_slippage_bps",
            "exit_slippage_bps", "exit_spread_multiplier",
            "top_of_book_participation",
            "max_signal_age_seconds", "max_quote_age_seconds",
            "max_entry_drift_atr", "min_circuit_buffer_pct",
            "circuit_headroom_bps",
        )
        for name in numeric_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.min_technical_score <= 100:
            raise ValueError("min_technical_score must be in [0, 100]")
        if self.capital_limit <= 0:
            raise ValueError("capital_limit must be positive")
        for name in (
            "risk_per_trade_pct", "max_position_pct", "max_daily_loss_pct",
            "max_portfolio_stop_risk_pct", "max_gross_exposure_pct",
        ):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.risk_per_trade_pct > 0.02:
            raise ValueError("risk_per_trade_pct cannot exceed 0.02")
        if self.max_daily_loss_pct > 0.05:
            raise ValueError("max_daily_loss_pct cannot exceed 0.05")
        if self.max_portfolio_stop_risk_pct > self.max_daily_loss_pct:
            raise ValueError("portfolio stop risk cannot exceed daily loss cap")
        if self.risk_per_trade_pct > self.max_portfolio_stop_risk_pct:
            raise ValueError("per-trade risk cannot exceed portfolio stop risk")
        for name in (
            "max_trades_per_day", "max_open_positions", "max_consecutive_losses",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.atr_stop_multiplier <= 0 or not 0 < self.target_r_multiple <= 10:
            raise ValueError("stop and target multipliers must be positive")
        if not 0 < self.min_after_cost_payoff_ratio <= 10:
            raise ValueError("minimum payoff ratio must be in (0, 10]")
        if not 0 <= self.entry_slippage_bps <= 100:
            raise ValueError("entry_slippage_bps must be in [0, 100]")
        if not 0 <= self.exit_slippage_bps <= 100:
            raise ValueError("exit_slippage_bps must be in [0, 100]")
        if not 1 <= self.exit_spread_multiplier <= 10:
            raise ValueError("exit_spread_multiplier must be in [1, 10]")
        if not 0 < self.top_of_book_participation <= 1:
            raise ValueError("top_of_book_participation must be in (0, 1]")
        if not 0 < self.max_signal_age_seconds < 300:
            raise ValueError("max_signal_age_seconds must be in (0, 300)")
        if not 1 <= self.max_quote_age_seconds <= 60:
            raise ValueError("max_quote_age_seconds must be in [1, 60]")
        if self.max_entry_drift_atr <= 0:
            raise ValueError("max_entry_drift_atr must be positive")
        if (
            isinstance(self.entry_cutoff_guard_seconds, bool)
            or not isinstance(self.entry_cutoff_guard_seconds, int)
            or not 0 <= self.entry_cutoff_guard_seconds < 300
        ):
            raise ValueError("entry_cutoff_guard_seconds must be an integer in [0, 300)")
        if not 0 <= self.min_circuit_buffer_pct <= 100:
            raise ValueError("min_circuit_buffer_pct must be in [0, 100]")
        if not 0 <= self.circuit_headroom_bps <= 1000:
            raise ValueError("circuit_headroom_bps must be in [0, 1000]")

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        payload.pop("trial_id", None)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _config_from_payload(payload: dict[str, Any], *, source: str) -> ReplayConfig:
    expected = {item.name for item in fields(ReplayConfig)}
    _exact_fields(payload, expected, source)
    rule_payload = payload["setup_rules"]
    if not isinstance(rule_payload, dict):
        raise ReplayDataError(f"{source}.setup_rules must be an object")
    rule_names = {item.name for item in fields(SetupRuleConfig)}
    _exact_fields(rule_payload, rule_names, f"{source}.setup_rules")
    try:
        rules = SetupRuleConfig(**rule_payload)
        values = dict(payload)
        values["setup_rules"] = rules
        return ReplayConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ReplayDataError(f"invalid replay config in {source}: {exc}") from exc


def load_trial_registry(path: str | Path) -> tuple[ReplayConfig, ...]:
    """Load a frozen, explicit list of every configuration in one trial."""

    registry_path = Path(path)
    try:
        payload = _parse_json(
            registry_path.read_text(encoding="utf-8"),
            source=str(registry_path),
        )
    except OSError as exc:
        raise ReplayDataError(f"cannot read trial registry: {exc}") from exc
    _exact_fields(
        payload,
        {"schema_version", "registry_id", "registered_at", "trials"},
        "trial registry",
    )
    if payload["schema_version"] != TRIAL_REGISTRY_SCHEMA_VERSION:
        raise ReplayDataError("unsupported trial registry schema")
    _text(payload["registry_id"], field_name="trial_registry.registry_id")
    _timestamp(payload["registered_at"], field_name="trial_registry.registered_at")
    raw_trials = payload["trials"]
    if not isinstance(raw_trials, list) or not raw_trials:
        raise ReplayDataError("trial_registry.trials must be a non-empty array")
    configs = tuple(
        _config_from_payload(item, source=f"trial_registry.trials[{index}]")
        if isinstance(item, dict)
        else (_ for _ in ()).throw(
            ReplayDataError(f"trial_registry.trials[{index}] must be an object")
        )
        for index, item in enumerate(raw_trials)
    )
    trial_ids = [config.trial_id for config in configs]
    if len(set(trial_ids)) != len(trial_ids):
        raise ReplayDataError("trial IDs must be unique")
    fingerprints = [config.fingerprint for config in configs]
    if len(set(fingerprints)) != len(fingerprints):
        raise ReplayDataError("duplicate parameter sets are not separate trials")
    return configs


@dataclass(frozen=True, slots=True)
class PlannedTrade:
    opportunity: ReplayOpportunity
    rules: SetupRuleResult
    side: str
    qty: int
    entry_reference: float
    entry_fill: float
    stop_price: float
    target_price: float
    exit_adverse_bps: float
    planned_stop_loss: float
    planned_target_profit: float
    planned_payoff_ratio: float
    reserved_notional: float


@dataclass(frozen=True, slots=True)
class ReplayTrade:
    opportunity_id: str
    scan_id: str
    session_date: date
    symbol: str
    side: str
    qty: int
    decision_at: datetime
    exit_at: datetime
    entry_reference: float
    entry_fill: float
    stop_price: float
    target_price: float
    exit_reference: float
    exit_fill: float
    exit_reason: str
    gross_pnl: float
    fees: float
    net_pnl: float
    planned_stop_loss: float
    r_multiple: float


@dataclass(frozen=True)
class ReplayResult:
    dataset_id: str
    source_strategy_fingerprint: str
    raw_tape_sha256: str
    fee_model_version: str
    trial_id: str
    config_fingerprint: str
    sessions: tuple[date, ...]
    trades: tuple[ReplayTrade, ...]
    rejection_reasons: dict[str, int]
    summary: dict[str, Any]
    fidelity: str = FIDELITY

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_strategy_fingerprint": self.source_strategy_fingerprint,
            "raw_tape_sha256": self.raw_tape_sha256,
            "fee_model_version": self.fee_model_version,
            "replay_engine_version": REPLAY_ENGINE_VERSION,
            "trial_id": self.trial_id,
            "config_fingerprint": self.config_fingerprint,
            "sessions": [item.isoformat() for item in self.sessions],
            "fidelity": self.fidelity,
            "summary": self.summary,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "trades": [
                {
                    **asdict(trade),
                    "session_date": trade.session_date.isoformat(),
                    "decision_at": trade.decision_at.isoformat(),
                    "exit_at": trade.exit_at.isoformat(),
                }
                for trade in self.trades
            ],
            "warnings": [
                "OHLC replay is low fidelity; stop/target order and fills are not observable.",
                (
                    f"{self.summary['entry_bar_conservative_stop_assumptions']} "
                    "trade(s) use an explicitly labelled worst-case entry-bar stop "
                    "assumption because pre/post-entry ordering is unknowable."
                ),
                "Results do not establish profitability or authorize live trading.",
            ],
        }


def _round_to_tick(price: float, tick_size: float) -> float:
    # Match the live order-price normalization exactly; the final decimal
    # rounding also prevents binary-float residue from missing an exact touch.
    return round(round(price / tick_size) * tick_size, 2)


def _adverse_fill(reference: float, bps: float, transaction: str) -> float:
    fraction = bps / 10_000.0
    if transaction == "BUY":
        return reference * (1.0 + fraction)
    if transaction == "SELL":
        return reference * (1.0 - fraction)
    raise ValueError("transaction must be BUY or SELL")


def _trade_costs(side: str, entry_fill: float, exit_fill: float, qty: int) -> float:
    if side == "LONG":
        estimate = estimate_nse_equity_intraday_cost(
            entry_fill * qty,
            exit_fill * qty,
        )
    else:
        estimate = estimate_nse_equity_intraday_cost(
            exit_fill * qty,
            entry_fill * qty,
        )
    return float(estimate.total)


def _outcome(
    side: str,
    entry_fill: float,
    exit_reference: float,
    qty: int,
    exit_adverse_bps: float,
) -> tuple[float, float, float, float]:
    exit_transaction = "SELL" if side == "LONG" else "BUY"
    exit_fill = _adverse_fill(exit_reference, exit_adverse_bps, exit_transaction)
    gross = float(gross_pnl(side, entry_fill, exit_fill, qty))
    fees = _trade_costs(side, entry_fill, exit_fill, qty)
    return exit_fill, gross, fees, gross - fees


def _price_band_reason(
    opportunity: ReplayOpportunity,
    side: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    config: ReplayConfig,
) -> str | None:
    if not opportunity.lower_circuit < entry_price < opportunity.upper_circuit:
        return "INVALID_PRICE_BAND_ORDER"
    headroom = max(
        2.0 * opportunity.tick_size,
        entry_price * config.circuit_headroom_bps / 10_000.0,
    )
    if side == "LONG":
        if stop_price < opportunity.lower_circuit + headroom:
            return "LONG_STOP_OUTSIDE_PRICE_BAND"
        if target_price > opportunity.upper_circuit - headroom:
            return "LONG_TARGET_OUTSIDE_PRICE_BAND"
    else:
        if stop_price > opportunity.upper_circuit - headroom:
            return "SHORT_STOP_OUTSIDE_PRICE_BAND"
        if target_price < opportunity.lower_circuit + headroom:
            return "SHORT_TARGET_OUTSIDE_PRICE_BAND"
    return None


def _execution_revalidation_reason(
    opportunity: ReplayOpportunity,
    rules: SetupRuleResult,
    config: ReplayConfig,
) -> str | None:
    if rules.side is None:
        return "INVALID_TRADE_SIDE"
    inputs = opportunity.rule_input
    price = opportunity.best_ask if rules.side == "LONG" else opportunity.best_bid
    spread_bps = (
        (opportunity.best_ask - opportunity.best_bid)
        / ((opportunity.best_ask + opportunity.best_bid) / 2.0)
        * 10_000.0
    )
    if spread_bps > config.setup_rules.max_spread_bps:
        return "EXECUTION_SPREAD_ABOVE_MAXIMUM"
    circuit_buffer_pct = min(
        (opportunity.upper_circuit - price) / price * 100.0,
        (price - opportunity.lower_circuit) / price * 100.0,
    )
    if circuit_buffer_pct < config.min_circuit_buffer_pct:
        return "EXECUTION_CIRCUIT_BUFFER_BELOW_MINIMUM"
    if abs(price - inputs.price) / inputs.atr > config.max_entry_drift_atr:
        return "ENTRY_DRIFT_ABOVE_MAXIMUM"
    if abs(price - inputs.vwap) / inputs.atr > config.setup_rules.max_vwap_distance_atr:
        return "EXECUTION_VWAP_DISTANCE_ABOVE_MAXIMUM"
    day_change = (price - inputs.prev_close) / inputs.prev_close * 100.0
    if rules.side == "LONG":
        breakout = (price - inputs.opening_high) / inputs.atr
        if price <= max(inputs.opening_high, inputs.vwap):
            return "LONG_BREAKOUT_FAILED"
        if day_change < config.setup_rules.long_min_day_change_pct:
            return "LONG_DAY_CHANGE_FAILED"
    else:
        breakout = (inputs.opening_low - price) / inputs.atr
        if price >= min(inputs.opening_low, inputs.vwap):
            return "SHORT_BREAKOUT_FAILED"
        if day_change > config.setup_rules.short_max_day_change_pct:
            return "SHORT_DAY_CHANGE_FAILED"
    if not (
        config.setup_rules.min_breakout_distance_atr
        <= breakout
        <= config.setup_rules.max_breakout_distance_atr
    ):
        return "EXECUTION_BREAKOUT_DISTANCE_OUT_OF_RANGE"
    return None


def _planned_stop_loss(
    side: str,
    entry_fill: float,
    stop_price: float,
    qty: int,
    exit_adverse_bps: float,
) -> float:
    _, _, _, net = _outcome(
        side,
        entry_fill,
        stop_price,
        qty,
        exit_adverse_bps,
    )
    return max(0.0, -net)


def _largest_qty_within_risk(
    side: str,
    entry_fill: float,
    stop_price: float,
    max_qty: int,
    risk_budget: float,
    exit_adverse_bps: float,
) -> int:
    low, high, result = 1, max_qty, 0
    while low <= high:
        candidate = (low + high) // 2
        loss = _planned_stop_loss(
            side,
            entry_fill,
            stop_price,
            candidate,
            exit_adverse_bps,
        )
        if loss <= risk_budget + 1e-9:
            result = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    return result


def _build_planned_trade(
    opportunity: ReplayOpportunity,
    rules: SetupRuleResult,
    config: ReplayConfig,
    *,
    risk_budget: float,
    notional_budget: float,
) -> tuple[PlannedTrade | None, str]:
    if not rules.accepted or rules.side is None:
        return None, f"SETUP_{rules.reason}"
    side = rules.side
    entry_reference = (
        opportunity.best_ask if side == "LONG" else opportunity.best_bid
    )
    entry_transaction = "BUY" if side == "LONG" else "SELL"
    entry_fill = _adverse_fill(
        entry_reference,
        config.entry_slippage_bps,
        entry_transaction,
    )
    observed_spread_bps = (
        (opportunity.best_ask - opportunity.best_bid)
        / ((opportunity.best_ask + opportunity.best_bid) / 2.0)
        * 10_000.0
    )
    # Bar OHLC is last-traded data, not an executable exit quote.  Charge at
    # least half the observed decision-time spread again on exit, in addition
    # to configured impact/latency slippage.  This remains a proxy and is why
    # every result is labelled low fidelity.
    exit_adverse_bps = (
        config.exit_slippage_bps
        + 0.5 * observed_spread_bps * config.exit_spread_multiplier
    )
    stop_distance = opportunity.rule_input.atr * config.atr_stop_multiplier
    stop_pct = stop_distance / entry_reference
    if not 0.003 <= stop_pct <= 0.03:
        return None, "STOP_DISTANCE_OUT_OF_RANGE"
    if side == "LONG":
        pre_fill_stop = _round_to_tick(
            entry_reference - stop_distance,
            opportunity.tick_size,
        )
        pre_fill_target = _round_to_tick(
            entry_reference + stop_distance * config.target_r_multiple,
            opportunity.tick_size,
        )
        stop_price = _round_to_tick(
            entry_fill - stop_distance,
            opportunity.tick_size,
        )
        target_price = _round_to_tick(
            entry_fill + stop_distance * config.target_r_multiple,
            opportunity.tick_size,
        )
        depth_qty = opportunity.best_ask_qty
    else:
        pre_fill_stop = _round_to_tick(
            entry_reference + stop_distance,
            opportunity.tick_size,
        )
        pre_fill_target = _round_to_tick(
            entry_reference - stop_distance * config.target_r_multiple,
            opportunity.tick_size,
        )
        stop_price = _round_to_tick(
            entry_fill + stop_distance,
            opportunity.tick_size,
        )
        target_price = _round_to_tick(
            entry_fill - stop_distance * config.target_r_multiple,
            opportunity.tick_size,
        )
        depth_qty = opportunity.best_bid_qty
    if min(pre_fill_stop, pre_fill_target, stop_price, target_price) <= 0:
        return None, "NONPOSITIVE_STOP_OR_TARGET"
    pre_fill_band_reason = _price_band_reason(
        opportunity,
        side,
        entry_reference,
        pre_fill_stop,
        pre_fill_target,
        config,
    )
    if pre_fill_band_reason:
        return None, pre_fill_band_reason
    modeled_band_reason = _price_band_reason(
        opportunity,
        side,
        entry_fill,
        stop_price,
        target_price,
        config,
    )
    if modeled_band_reason:
        return None, f"MODELED_{modeled_band_reason}"

    qty_by_raw_risk = math.floor(risk_budget / stop_distance)
    conservative_notional_price = entry_reference * (
        1.0 + config.entry_slippage_bps / 10_000.0
    )
    qty_by_notional = math.floor(notional_budget / conservative_notional_price)
    qty_by_depth = math.floor(depth_qty * config.top_of_book_participation)
    max_qty = max(0, min(qty_by_raw_risk, qty_by_notional, qty_by_depth))
    if max_qty < 1:
        return None, "NO_EXECUTABLE_QUANTITY"
    qty = _largest_qty_within_risk(
        side,
        entry_fill,
        pre_fill_stop,
        max_qty,
        risk_budget,
        exit_adverse_bps,
    )
    if qty < 1:
        return None, "NO_QUANTITY_WITHIN_AFTER_COST_RISK"
    # The live engine performs a second risk admission after the actual fill
    # and its stop/target repricing. Keep that stage explicit even though the
    # adverse pre-fill sizing should normally be the tighter constraint.
    qty = _largest_qty_within_risk(
        side,
        entry_fill,
        stop_price,
        qty,
        risk_budget,
        exit_adverse_bps,
    )
    if qty < 1:
        return None, "NO_QUANTITY_WITHIN_MODELED_FILL_RISK"
    planned_stop = _planned_stop_loss(
        side,
        entry_fill,
        stop_price,
        qty,
        exit_adverse_bps,
    )
    _, _, _, planned_target = _outcome(
        side,
        entry_fill,
        target_price,
        qty,
        exit_adverse_bps,
    )
    if planned_target <= 0:
        return None, "NONPOSITIVE_AFTER_COST_TARGET"
    payoff = planned_target / planned_stop if planned_stop > 0 else 0.0
    if payoff < config.min_after_cost_payoff_ratio:
        return None, "AFTER_COST_PAYOFF_BELOW_MINIMUM"
    return PlannedTrade(
        opportunity=opportunity,
        rules=rules,
        side=side,
        qty=qty,
        entry_reference=entry_reference,
        entry_fill=entry_fill,
        stop_price=stop_price,
        target_price=target_price,
        exit_adverse_bps=exit_adverse_bps,
        planned_stop_loss=planned_stop,
        planned_target_profit=planned_target,
        planned_payoff_ratio=payoff,
        reserved_notional=entry_fill * qty,
    ), "OK"


def _simulate_exit(
    plan: PlannedTrade,
    bars: Sequence[MarketBar],
    config: ReplayConfig,
) -> ReplayTrade:
    opportunity = plan.opportunity
    force_at = datetime.combine(opportunity.session_date, FORCE_EXIT, IST)
    relevant = [
        bar for bar in bars
        if bar.bar_start <= force_at and bar.bar_end > opportunity.decision_at
    ]
    if not relevant:
        raise ReplayDataError(
            f"no post-entry path for {opportunity.session_date}/{opportunity.symbol}"
        )
    for index, bar in enumerate(relevant):
        if plan.side == "LONG":
            stop_hit = bar.low <= plan.stop_price
            target_hit = bar.high >= plan.target_price
        else:
            stop_hit = bar.high >= plan.stop_price
            target_hit = bar.low <= plan.target_price

        # A decision can occur after this bar opened.  Its earlier high/low is
        # unknowable relative to entry.  Never grant a favorable target; if the
        # adverse extreme crosses the stop, book the worst case under an
        # explicit assumption label rather than claiming an observed fill.
        entry_bar_ambiguous = (
            index == 0 and bar.bar_start < opportunity.decision_at
        )
        if entry_bar_ambiguous:
            target_hit = False
        if stop_hit:
            if entry_bar_ambiguous:
                # The bar open predates the entry and cannot be a gap fill.
                exit_reference = plan.stop_price
                reason = "ENTRY_BAR_CONSERVATIVE_STOP_ASSUMPTION"
            elif plan.side == "LONG":
                exit_reference = min(plan.stop_price, bar.open)
                reason = "AMBIGUOUS_BAR_STOP_FIRST" if target_hit else "STOP"
            else:
                exit_reference = max(plan.stop_price, bar.open)
                reason = "AMBIGUOUS_BAR_STOP_FIRST" if target_hit else "STOP"
        elif target_hit:
            exit_reference = plan.target_price
            reason = "TARGET"
        elif bar.bar_end >= force_at:
            exit_reference = bar.close
            reason = "FORCE_EXIT_1510"
        else:
            continue

        exit_fill, gross, fees, net = _outcome(
            plan.side,
            plan.entry_fill,
            exit_reference,
            plan.qty,
            plan.exit_adverse_bps,
        )
        return ReplayTrade(
            opportunity_id=opportunity.opportunity_id,
            scan_id=opportunity.scan_id,
            session_date=opportunity.session_date,
            symbol=opportunity.symbol,
            side=plan.side,
            qty=plan.qty,
            decision_at=opportunity.decision_at,
            exit_at=bar.available_at,
            entry_reference=plan.entry_reference,
            entry_fill=plan.entry_fill,
            stop_price=plan.stop_price,
            target_price=plan.target_price,
            exit_reference=exit_reference,
            exit_fill=exit_fill,
            exit_reason=reason,
            gross_pnl=gross,
            fees=fees,
            net_pnl=net,
            planned_stop_loss=plan.planned_stop_loss,
            r_multiple=(
                net / plan.planned_stop_loss
                if plan.planned_stop_loss > 0
                else 0.0
            ),
        )
    raise ReplayDataError(
        f"no executable force-exit bar for "
        f"{opportunity.session_date}/{opportunity.symbol}"
    )


def _summarize(trades: Sequence[ReplayTrade]) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda item: (item.exit_at, item.opportunity_id))
    net_values = [trade.net_pnl for trade in ordered]
    r_values = [trade.r_multiple for trade in ordered]
    wins = sum(value > 0 for value in net_values)
    losses = sum(value < 0 for value in net_values)
    gross_profit = math.fsum(value for value in net_values if value > 0)
    gross_loss = math.fsum(value for value in net_values if value < 0)
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    index = 0
    while index < len(ordered):
        exit_at = ordered[index].exit_at
        group: list[float] = []
        while index < len(ordered) and ordered[index].exit_at == exit_at:
            group.append(ordered[index].net_pnl)
            index += 1
        # Bar exits sharing one availability timestamp have no observable
        # sequence. Apply all losses before gains for an ID-invariant,
        # conservative drawdown envelope.
        equity += math.fsum(value for value in group if value < 0)
        max_drawdown = max(max_drawdown, peak - equity)
        equity += math.fsum(value for value in group if value >= 0)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    profit_factor: float | None
    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = None
    else:
        profit_factor = 0.0
    return {
        "trades": len(ordered),
        "wins": wins,
        "losses": losses,
        "breakeven": len(ordered) - wins - losses,
        "win_rate": wins / len(ordered) if ordered else 0.0,
        "gross_pnl": math.fsum(trade.gross_pnl for trade in ordered),
        "fees": math.fsum(trade.fees for trade in ordered),
        "net_pnl": math.fsum(net_values),
        "expectancy": statistics.fmean(net_values) if net_values else 0.0,
        "average_r": statistics.fmean(r_values) if r_values else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "entry_bar_conservative_stop_assumptions": sum(
            trade.exit_reason == "ENTRY_BAR_CONSERVATIVE_STOP_ASSUMPTION"
            for trade in ordered
        ),
    }


@dataclass(slots=True)
class _SessionState:
    realized_pnl: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    blocked_symbols: set[str] = field(default_factory=set)
    active: dict[str, tuple[PlannedTrade, ReplayTrade]] = field(default_factory=dict)
    pending_exits: list[tuple[datetime, str]] = field(default_factory=list)


def _realize_due_exits(
    state: _SessionState,
    through: datetime,
    completed: list[ReplayTrade],
    config: ReplayConfig,
) -> None:
    while state.pending_exits and state.pending_exits[0][0] <= through:
        exit_at = state.pending_exits[0][0]
        group: list[ReplayTrade] = []
        while state.pending_exits and state.pending_exits[0][0] == exit_at:
            _, opportunity_id = heapq.heappop(state.pending_exits)
            record = state.active.pop(opportunity_id, None)
            if record is not None:
                group.append(record[1])
        completed.extend(group)
        state.realized_pnl += math.fsum(trade.net_pnl for trade in group)
        # Exact ordering inside one bar timestamp is unknowable. Wins first and
        # losses last is the conservative policy for the trailing-loss gate and
        # makes admissions invariant to arbitrary opportunity IDs.
        if any(trade.net_pnl > 0 for trade in group):
            state.consecutive_losses = 0
        state.consecutive_losses += sum(trade.net_pnl < 0 for trade in group)
        daily_limit_hit = (
            state.realized_pnl
            <= -config.capital_limit * config.max_daily_loss_pct
        )
        consecutive_limit_hit = (
            state.consecutive_losses >= config.max_consecutive_losses
        )
        if state.active and (daily_limit_hit or consecutive_limit_hit):
            # Live trading activates the kill switch and liquidates every other
            # position. Five-minute OHLC cannot reconstruct the emergency
            # executable quote at this instant, so continuing those positions
            # would be optimistic and fabricating a price would be misleading.
            raise ReplayDataError(
                "kill-switch limit breached with overlapping exposure; "
                "bar-only replay cannot price the mandatory live liquidation"
            )


def _candidate_sort_key(
    candidate: tuple[ReplayOpportunity, SetupRuleResult],
) -> tuple[Any, ...]:
    opportunity, rules = candidate
    return (
        opportunity.decision_at,
        opportunity.scan_id,
        -rules.technical_score,
        opportunity.signal_bar_closed_at,
        opportunity.symbol,
        opportunity.opportunity_id,
    )


def run_replay(
    dataset: ReplayDataset,
    config: ReplayConfig,
    *,
    sessions: Iterable[date] | None = None,
) -> ReplayResult:
    """Run a deterministic, chronological portfolio replay."""

    selected_sessions = tuple(
        sorted(set(sessions if sessions is not None else dataset.manifest.sessions))
    )
    unknown = set(selected_sessions) - set(dataset.manifest.sessions)
    if unknown:
        raise ReplayDataError(f"requested sessions are absent: {sorted(unknown)}")
    selected_set = set(selected_sessions)
    rejection_reasons: Counter[str] = Counter()
    completed: list[ReplayTrade] = []

    by_session: dict[date, list[tuple[ReplayOpportunity, SetupRuleResult]]] = {
        session: [] for session in selected_sessions
    }
    for opportunity in dataset.opportunities:
        if opportunity.session_date not in selected_set:
            continue
        entry_cutoff = datetime.combine(
            opportunity.session_date,
            LAST_ENTRY,
            IST,
        ) - timedelta(seconds=config.entry_cutoff_guard_seconds)
        if opportunity.decision_at >= entry_cutoff:
            rejection_reasons["ENTRY_CUTOFF_GUARD"] += 1
            continue
        rules = evaluate_setup_rules(opportunity.rule_input, config.setup_rules)
        if not rules.accepted:
            rejection_reasons[f"SETUP_{rules.reason}"] += 1
            continue
        if rules.technical_score < config.min_technical_score:
            rejection_reasons["TECHNICAL_SCORE_BELOW_MINIMUM"] += 1
            continue
        signal_age = (
            opportunity.decision_at - opportunity.signal_bar_closed_at
        ).total_seconds()
        if signal_age < 0 or signal_age > config.max_signal_age_seconds:
            rejection_reasons["STALE_SIGNAL"] += 1
            continue
        quote_age = (
            opportunity.decision_at - opportunity.quote_available_at
        ).total_seconds()
        if quote_age < 0 or quote_age > config.max_quote_age_seconds:
            rejection_reasons["STALE_EXECUTION_QUOTE"] += 1
            continue
        execution_reason = _execution_revalidation_reason(
            opportunity,
            rules,
            config,
        )
        if execution_reason:
            rejection_reasons[execution_reason] += 1
            continue
        by_session[opportunity.session_date].append((opportunity, rules))

    for session in selected_sessions:
        state = _SessionState()
        candidates = sorted(by_session.get(session, ()), key=_candidate_sort_key)
        for opportunity, rules in candidates:
            _realize_due_exits(
                state,
                opportunity.decision_at,
                completed,
                config,
            )
            if opportunity.symbol in state.blocked_symbols:
                rejection_reasons["SYMBOL_ALREADY_TRADED"] += 1
                continue
            if state.trades_today >= config.max_trades_per_day:
                rejection_reasons["MAX_TRADES_PER_DAY"] += 1
                continue
            if len(state.active) >= config.max_open_positions:
                rejection_reasons["MAX_OPEN_POSITIONS"] += 1
                continue
            if state.consecutive_losses >= config.max_consecutive_losses:
                rejection_reasons["MAX_CONSECUTIVE_LOSSES"] += 1
                continue
            if state.realized_pnl <= -config.capital_limit * config.max_daily_loss_pct:
                rejection_reasons["DAILY_LOSS_LIMIT"] += 1
                continue

            open_risk = math.fsum(
                plan.planned_stop_loss for plan, _ in state.active.values()
            )
            open_notional = math.fsum(
                plan.reserved_notional for plan, _ in state.active.values()
            )
            daily_remaining = max(
                0.0,
                config.capital_limit * config.max_daily_loss_pct
                + min(0.0, state.realized_pnl)
                - open_risk,
            )
            portfolio_remaining = max(
                0.0,
                config.capital_limit * config.max_portfolio_stop_risk_pct
                - open_risk,
            )
            gross_remaining = max(
                0.0,
                config.capital_limit * config.max_gross_exposure_pct
                - open_notional,
            )
            risk_budget = min(
                config.capital_limit * config.risk_per_trade_pct,
                daily_remaining,
                portfolio_remaining,
            )
            notional_budget = min(
                config.capital_limit * config.max_position_pct,
                gross_remaining,
            )
            if risk_budget <= 0 or notional_budget <= 0:
                rejection_reasons["PORTFOLIO_CAPACITY_EXHAUSTED"] += 1
                continue
            plan, reason = _build_planned_trade(
                opportunity,
                rules,
                config,
                risk_budget=risk_budget,
                notional_budget=notional_budget,
            )
            if plan is None:
                rejection_reasons[reason] += 1
                continue
            path = dataset.bars[(session, opportunity.symbol)]
            simulated = _simulate_exit(plan, path, config)
            state.active[opportunity.opportunity_id] = (plan, simulated)
            heapq.heappush(
                state.pending_exits,
                (simulated.exit_at, opportunity.opportunity_id),
            )
            state.blocked_symbols.add(opportunity.symbol)
            state.trades_today += 1

        far_future = datetime.combine(session + timedelta(days=1), time.min, IST)
        _realize_due_exits(state, far_future, completed, config)
        if state.active:
            raise ReplayDataError(f"session {session} ended with unresolved positions")

    completed.sort(key=lambda item: (item.exit_at, item.opportunity_id))
    return ReplayResult(
        dataset_id=dataset.manifest.dataset_id,
        source_strategy_fingerprint=(
            dataset.manifest.source_strategy_fingerprint
        ),
        raw_tape_sha256=dataset.manifest.raw_tape_sha256,
        fee_model_version=dataset.manifest.fee_model_version,
        trial_id=config.trial_id,
        config_fingerprint=config.fingerprint,
        sessions=selected_sessions,
        trades=tuple(completed),
        rejection_reasons=dict(rejection_reasons),
        summary=_summarize(completed),
    )


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold: int
    train_sessions: tuple[date, ...]
    purged_sessions: tuple[date, ...]
    test_sessions: tuple[date, ...]


@dataclass(frozen=True, slots=True)
class WalkForwardPlan:
    folds: tuple[WalkForwardFold, ...]
    holdout_sessions: tuple[date, ...]


def build_walk_forward_plan(
    sessions: Sequence[date],
    *,
    min_train_sessions: int,
    test_sessions: int,
    purge_sessions: int = 1,
    final_holdout_sessions: int = 0,
) -> WalkForwardPlan:
    """Create expanding chronological folds without shuffle or overlap."""

    ordered = tuple(sessions)
    if tuple(sorted(set(ordered))) != ordered:
        raise ValueError("sessions must be unique and chronological")
    for name, value, minimum in (
        ("min_train_sessions", min_train_sessions, 1),
        ("test_sessions", test_sessions, 1),
        ("purge_sessions", purge_sessions, 0),
        ("final_holdout_sessions", final_holdout_sessions, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if final_holdout_sessions >= len(ordered):
        raise ValueError("final holdout must leave development sessions")
    development_end = len(ordered) - final_holdout_sessions
    development = ordered[:development_end]
    holdout = ordered[development_end:]
    folds: list[WalkForwardFold] = []
    test_start = min_train_sessions + purge_sessions
    while test_start + test_sessions <= len(development):
        train_end = test_start - purge_sessions
        fold = WalkForwardFold(
            fold=len(folds) + 1,
            train_sessions=development[:train_end],
            purged_sessions=development[train_end:test_start],
            test_sessions=development[test_start:test_start + test_sessions],
        )
        if not fold.train_sessions or max(fold.train_sessions) >= min(fold.test_sessions):
            raise ValueError("walk-forward chronology invariant failed")
        if fold.purged_sessions and not (
            max(fold.train_sessions)
            < min(fold.purged_sessions)
            <= max(fold.purged_sessions)
            < min(fold.test_sessions)
        ):
            raise ValueError("walk-forward purge invariant failed")
        folds.append(fold)
        test_start += test_sessions
    if not folds:
        raise ValueError("not enough sessions for one complete walk-forward fold")
    return WalkForwardPlan(tuple(folds), holdout)


def _selection_score(result: ReplayResult, minimum_trades: int) -> float:
    values = [trade.r_multiple for trade in result.trades]
    if len(values) < minimum_trades:
        return -math.inf
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return mean - 1.645 * standard_error


def _compact_result(result: ReplayResult) -> dict[str, Any]:
    return {
        "trial_id": result.trial_id,
        "config_fingerprint": result.config_fingerprint,
        "sessions": [item.isoformat() for item in result.sessions],
        "summary": result.summary,
        "rejection_reasons": dict(sorted(result.rejection_reasons.items())),
    }


def run_walk_forward(
    dataset: ReplayDataset,
    configs: Sequence[ReplayConfig],
    plan: WalkForwardPlan,
    *,
    minimum_training_trades: int = 20,
    evaluate_holdout: bool = False,
) -> dict[str, Any]:
    """Select on training only, then evaluate frozen configurations OOS."""

    if not configs:
        raise ValueError("at least one registered config is required")
    if minimum_training_trades < 1:
        raise ValueError("minimum_training_trades must be positive")
    if len({item.trial_id for item in configs}) != len(configs):
        raise ValueError("registered trial IDs must be unique")
    if len({item.fingerprint for item in configs}) != len(configs):
        raise ValueError("registered parameter sets must be unique")
    if not plan.folds:
        raise ValueError("walk-forward plan requires at least one fold")
    ordered_sessions = dataset.manifest.sessions
    known_sessions = set(ordered_sessions)
    if tuple(fold.fold for fold in plan.folds) != tuple(
        range(1, len(plan.folds) + 1)
    ):
        raise ValueError("walk-forward fold IDs must be sequential")
    if tuple(sorted(set(plan.holdout_sessions))) != plan.holdout_sessions:
        raise ValueError("holdout sessions must be unique and chronological")
    reserved_holdout = set(plan.holdout_sessions)
    if not reserved_holdout <= known_sessions:
        raise ValueError("holdout contains sessions absent from the dataset")
    if plan.holdout_sessions and plan.holdout_sessions != ordered_sessions[
        -len(plan.holdout_sessions):
    ]:
        raise ValueError("holdout must be an exact final dataset suffix")
    seen_test_sessions: set[date] = set()
    prior_test_end: date | None = None
    for fold in plan.folds:
        train = set(fold.train_sessions)
        purged = set(fold.purged_sessions)
        test = set(fold.test_sessions)
        if not fold.train_sessions or not fold.test_sessions:
            raise ValueError("walk-forward folds require train and test sessions")
        for name, values in (
            ("train", fold.train_sessions),
            ("purged", fold.purged_sessions),
            ("test", fold.test_sessions),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(
                    f"walk-forward {name} sessions must be unique and chronological"
                )
        if not (train | purged | test) <= known_sessions:
            raise ValueError("walk-forward fold contains an unknown session")
        if train & purged or train & test or purged & test:
            raise ValueError("walk-forward fold partitions overlap")
        if reserved_holdout & (train | purged | test):
            raise ValueError("reserved holdout leaked into a development fold")
        if max(fold.train_sessions) >= min(fold.test_sessions):
            raise ValueError("training sessions must strictly precede test sessions")
        if fold.purged_sessions and not (
            max(fold.train_sessions)
            < min(fold.purged_sessions)
            <= max(fold.purged_sessions)
            < min(fold.test_sessions)
        ):
            raise ValueError("purged sessions must separate training and test")
        partition = (
            fold.train_sessions
            + fold.purged_sessions
            + fold.test_sessions
        )
        if partition != ordered_sessions[:len(partition)]:
            raise ValueError(
                "each walk-forward fold must be a chronological dataset prefix"
            )
        if prior_test_end is not None and min(fold.test_sessions) <= prior_test_end:
            raise ValueError("walk-forward test folds must be chronological")
        prior_test_end = max(fold.test_sessions)
        if seen_test_sessions & test:
            raise ValueError("out-of-sample test windows overlap")
        seen_test_sessions.update(test)
    fold_reports: list[dict[str, Any]] = []
    combined_oos: list[ReplayTrade] = []

    def choose(train_sessions: Sequence[date]) -> tuple[ReplayConfig, list[dict[str, Any]]]:
        candidates: list[tuple[float, str, ReplayConfig, ReplayResult]] = []
        audit: list[dict[str, Any]] = []
        for config in configs:
            result = run_replay(dataset, config, sessions=train_sessions)
            score = _selection_score(result, minimum_training_trades)
            audit.append({
                **_compact_result(result),
                "selection_score_lcb_r": None if not math.isfinite(score) else score,
            })
            candidates.append((score, config.fingerprint, config, result))
        eligible = [item for item in candidates if math.isfinite(item[0])]
        if not eligible:
            raise ReplayDataError(
                "no registered trial has enough training trades for selection"
            )
        # Fingerprint ordering makes exact-score ties deterministic without
        # relying on registry order.
        eligible.sort(key=lambda item: (-item[0], item[1]))
        return eligible[0][2], audit

    for fold in plan.folds:
        selected, training_audit = choose(fold.train_sessions)
        test_result = run_replay(dataset, selected, sessions=fold.test_sessions)
        combined_oos.extend(test_result.trades)
        fold_reports.append({
            "fold": fold.fold,
            "train_sessions": [item.isoformat() for item in fold.train_sessions],
            "purged_sessions": [item.isoformat() for item in fold.purged_sessions],
            "test_sessions": [item.isoformat() for item in fold.test_sessions],
            "training_trials": training_audit,
            "selected_trial_id": selected.trial_id,
            "selected_config_fingerprint": selected.fingerprint,
            "test": _compact_result(test_result),
        })

    holdout_report: dict[str, Any] = {
        "status": "RESERVED_NOT_EVALUATED",
        "sessions": [item.isoformat() for item in plan.holdout_sessions],
    }
    if evaluate_holdout:
        if not plan.holdout_sessions:
            raise ValueError("no final holdout was reserved")
        first_holdout = min(plan.holdout_sessions)
        development_sessions = tuple(
            session for session in dataset.manifest.sessions
            if session < first_holdout
        )
        selected, training_audit = choose(development_sessions)
        result = run_replay(dataset, selected, sessions=plan.holdout_sessions)
        holdout_report = {
            "status": "EVALUATED_EXPLICITLY",
            "sessions": [item.isoformat() for item in plan.holdout_sessions],
            "training_trials": training_audit,
            "selected_trial_id": selected.trial_id,
            "selected_config_fingerprint": selected.fingerprint,
            "result": _compact_result(result),
        }

    combined_oos.sort(key=lambda item: (item.exit_at, item.opportunity_id))
    combined_oos_summary = _summarize(combined_oos)
    report_entry_bar_assumptions = combined_oos_summary[
        "entry_bar_conservative_stop_assumptions"
    ]
    report: dict[str, Any] = {
        "dataset_id": dataset.manifest.dataset_id,
        "source_strategy_fingerprint": (
            dataset.manifest.source_strategy_fingerprint
        ),
        "raw_tape_sha256": dataset.manifest.raw_tape_sha256,
        "fee_model_version": dataset.manifest.fee_model_version,
        "replay_engine_version": REPLAY_ENGINE_VERSION,
        "fidelity": FIDELITY,
        "selection_metric": "mean_R_minus_1.645_standard_errors",
        "minimum_training_trades": minimum_training_trades,
        "folds": fold_reports,
        "combined_oos_summary": combined_oos_summary,
        "holdout": holdout_report,
        "warnings": [
            "All registered trials count toward strategy selection.",
            "The final holdout is excluded unless explicitly requested.",
            (
                f"{report_entry_bar_assumptions} OOS trade(s) use an explicitly "
                "labelled worst-case entry-bar stop assumption."
            ),
            "Bar-only replay cannot establish production fill fidelity or future profit.",
        ],
    }
    canonical = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict point-in-time, cost-aware intraday replay",
    )
    parser.add_argument("dataset", help="directory containing manifest.json")
    parser.add_argument(
        "--trials",
        help="frozen trial-registry JSON; default is one documented baseline",
    )
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--min-train-sessions", type=int, default=40)
    parser.add_argument("--test-sessions", type=int, default=10)
    parser.add_argument("--purge-sessions", type=int, default=1)
    parser.add_argument("--holdout-sessions", type=int, default=20)
    parser.add_argument("--minimum-training-trades", type=int, default=30)
    parser.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="explicitly unlock and report the reserved final holdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        dataset = load_replay_dataset(args.dataset)
        configs = (
            load_trial_registry(args.trials)
            if args.trials
            else (ReplayConfig(),)
        )
        if args.walk_forward:
            plan = build_walk_forward_plan(
                dataset.manifest.sessions,
                min_train_sessions=args.min_train_sessions,
                test_sessions=args.test_sessions,
                purge_sessions=args.purge_sessions,
                final_holdout_sessions=args.holdout_sessions,
            )
            report = run_walk_forward(
                dataset,
                configs,
                plan,
                minimum_training_trades=args.minimum_training_trades,
                evaluate_holdout=args.evaluate_holdout,
            )
        else:
            if len(configs) != 1:
                raise ValueError(
                    "multiple trials require --walk-forward to avoid in-sample ranking"
                )
            report = configs and run_replay(dataset, configs[0]).as_dict()
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ReplayDataError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
