from __future__ import annotations

import importlib
import os
import sys
import threading
import tempfile
import unittest
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import dotenv
import pandas as pd


def _import_bot_without_environment_file():
    """Import bot with deterministic settings and without reading project .env."""
    safe_environment = {
        "AI_MODE": "shadow",
        "KITE_ACCESS_TOKEN": "",
        "KITE_API_KEY": "",
        "KITE_API_SECRET": "",
        "KITE_STATIC_IP": "",
        "LIVE_TRADING": "false",
        "LIVE_TRADING_CONFIRM": "",
        "OPENAI_API_KEY": "",
    }

    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            with mock.patch.dict(os.environ, safe_environment, clear=True):
                with mock.patch.object(dotenv, "load_dotenv", return_value=False):
                    sys.modules.pop("bot", None)
                    return importlib.import_module("bot")
        finally:
            os.chdir(previous_cwd)


bot = _import_bot_without_environment_file()


FIXED_NOW = datetime(2026, 8, 11, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))


def make_setup(side: str = "LONG"):
    return bot.Setup(
        symbol="INFY",
        token=123,
        side=side,
        price=100.0,
        prev_close=98.0,
        day_change_pct=2.0 if side == "LONG" else -2.0,
        gap_pct=0.1,
        turnover_crore=100.0,
        spread_bps=2.0,
        stock_in_play_score=90.0,
        opening_range_high=99.0,
        opening_range_low=98.0,
        vwap=99.0,
        ema9=99.5,
        ema20=99.0,
        rsi=60.0 if side == "LONG" else 40.0,
        atr=2.0,
        atr_pct=0.02,
        rvol=2.0,
        breakout_distance_atr=0.5,
        vwap_distance_atr=0.5,
        candle_body_ratio=0.8,
        candle_close_location=0.9 if side == "LONG" else 0.1,
        nifty_regime="BULL" if side == "LONG" else "BEAR",
        nifty_return_pct=0.3 if side == "LONG" else -0.3,
        technical_score=90.0,
        signal_at=FIXED_NOW.isoformat(),
        lower_circuit_limit=90.0,
        upper_circuit_limit=110.0,
    )


def make_trade(*, side: str = "LONG", status: str = "PLANNED"):
    return bot.Trade(
        symbol="INFY",
        token=123,
        side=side,
        qty=10,
        entry_price=100.0,
        initial_risk_per_share=2.0,
        stop_price=98.0 if side == "LONG" else 102.0,
        target_price=103.6 if side == "LONG" else 96.4,
        status=status,
        opened_at=FIXED_NOW.isoformat(),
        client_tag="AITRD000000000001",
        requested_qty=10,
    )


def approved_decision():
    return bot.AIDecision(
        decision="APPROVE",
        confidence=90,
        quality_score=90,
        reason="deterministic test approval",
        risk_flags=[],
    )


class FakeBroker:
    """Scriptable broker surface used by paper and live lifecycle tests."""

    def __init__(self, *, price: float = 100.0, signed_qty: int = 0):
        self.current_price = price
        self.signed_qty = signed_qty
        self.flat_confirmation = True
        self.raise_after_accept_entry = False
        self.raise_after_accept_exit = False
        self.entry_calls: list[tuple] = []
        self.stop_calls: list[tuple] = []
        self.exit_calls: list[tuple] = []
        self.cancel_calls: list[str | None] = []
        self.wait_position_calls: list[tuple[str, int]] = []
        self.order_snapshots: dict[str, object] = {}
        self.cancel_snapshots: dict[str, object] = {}
        self.recovered_by_tag: dict[str, object] = {}
        self.startup_order_payloads: list[dict] = []
        self.trade_payloads: list[dict] = []
        self.order_metadata: dict[str, dict] = {}
        self.wait_options: list[dict] = []
        self.ws_connected = mock.Mock()
        self.ws_connected.is_set.return_value = True
        self._instrument = bot.Instrument(
            symbol="INFY",
            name="Infosys",
            token=123,
            tick_size=0.05,
        )

    def instrument(self, symbol: str):
        return self._instrument if symbol == "INFY" else None

    def ltp(self, symbol: str) -> float:
        if symbol != "INFY":
            raise KeyError(symbol)
        return self.current_price

    def position_qty(self, symbol: str) -> int:
        if symbol != "INFY":
            return 0
        return self.signed_qty

    def place_market_entry(self, inst, side, qty, tag):
        self.entry_calls.append((inst.symbol, side, qty, tag))
        self.order_metadata["ENTRY-1"] = {
            "symbol": inst.symbol,
            "exchange": "NSE",
            "product": "MIS",
            "transaction_type": "BUY" if side == "LONG" else "SELL",
            "order_type": "MARKET",
            "tag": tag,
        }
        self.signed_qty = qty if side == "LONG" else -qty
        if self.raise_after_accept_entry:
            snapshot = bot.OrderSnapshot.from_payload(
                {
                    "order_id": "ENTRY-RECOVERED",
                    "status": "COMPLETE",
                    "quantity": qty,
                    "filled_quantity": qty,
                    "pending_quantity": 0,
                    "average_price": self.current_price,
                    "transaction_type": "BUY" if side == "LONG" else "SELL",
                    "order_type": "MARKET",
                    "tradingsymbol": inst.symbol,
                    "exchange": "NSE",
                    "product": "MIS",
                    "tag": tag,
                }
            )
            self.recovered_by_tag[tag] = snapshot
            self.order_snapshots[snapshot.order_id] = snapshot
            raise TimeoutError("entry response lost after broker acceptance")
        return "ENTRY-1"

    def place_protective_stop(self, inst, side, qty, trigger_price, tag):
        self.stop_calls.append((inst.symbol, side, qty, trigger_price, tag))
        self.order_metadata["STOP-1"] = {
            "symbol": inst.symbol,
            "exchange": "NSE",
            "product": "MIS",
            "transaction_type": "SELL" if side == "LONG" else "BUY",
            "order_type": "SL-M",
            "tag": tag,
            "trigger_price": trigger_price,
        }
        return "STOP-1"

    def _enrich(self, order_id, snapshot):
        metadata = self.order_metadata.get(order_id, {})
        values = {
            key: (
                value
                if key == "trigger_price"
                else getattr(snapshot, key) or value
            )
            for key, value in metadata.items()
        }
        return replace(snapshot, **values) if values else snapshot

    def cancel_order_confirmed(self, order_id, *, timeout_seconds=8):
        self.cancel_calls.append(order_id)
        if order_id in self.cancel_snapshots:
            snapshot = self._enrich(order_id, self.cancel_snapshots[order_id])
            if snapshot.transaction_type == "BUY":
                self.signed_qty = snapshot.filled
            elif snapshot.transaction_type == "SELL":
                self.signed_qty = -snapshot.filled
            return snapshot
        return bot.OrderSnapshot.from_payload(
            {"order_id": order_id or "", "status": "CANCELLED"}
        )

    def exit_market(self, inst, signed_qty, tag):
        self.exit_calls.append((inst.symbol, signed_qty, tag))
        self.order_metadata["EXIT-1"] = {
            "symbol": inst.symbol,
            "exchange": "NSE",
            "product": "MIS",
            "transaction_type": "SELL" if signed_qty > 0 else "BUY",
            "order_type": "MARKET",
            "tag": tag,
        }
        if self.raise_after_accept_exit:
            snapshot = bot.OrderSnapshot.from_payload(
                {
                    "order_id": "EXIT-RECOVERED",
                    "status": "COMPLETE",
                    "quantity": abs(signed_qty),
                    "filled_quantity": abs(signed_qty),
                    "pending_quantity": 0,
                    "average_price": self.current_price,
                    "transaction_type": "SELL" if signed_qty > 0 else "BUY",
                    "order_type": "MARKET",
                    "tradingsymbol": inst.symbol,
                    "exchange": "NSE",
                    "product": "MIS",
                    "tag": tag,
                }
            )
            self.recovered_by_tag[tag] = snapshot
            self.order_snapshots[snapshot.order_id] = snapshot
            raise TimeoutError("exit response lost after broker acceptance")
        return "EXIT-1"

    def wait_for_order(
        self,
        order_id,
        *,
        timeout_seconds,
        require_stop_armed=False,
        return_on_partial=False,
    ):
        self.wait_options.append(
            {
                "order_id": order_id,
                "require_stop_armed": require_stop_armed,
                "return_on_partial": return_on_partial,
            }
        )
        if order_id in self.order_snapshots:
            return self._enrich(order_id, self.order_snapshots[order_id])
        if order_id == "STOP-1" or require_stop_armed:
            return self._enrich(order_id, bot.OrderSnapshot.from_payload(
                {
                    "order_id": order_id,
                    "status": "TRIGGER PENDING",
                    "quantity": 10,
                    "filled_quantity": 0,
                    "pending_quantity": 10,
                    "trigger_price": 98.0,
                }
            ))
        return self._enrich(order_id, bot.OrderSnapshot.from_payload(
            {
                "order_id": order_id,
                "status": "COMPLETE",
                "quantity": 10,
                "filled_quantity": 10,
                "pending_quantity": 0,
                "average_price": self.current_price,
            }
        ))

    def wait_for_position_qty(
        self,
        symbol,
        expected_qty,
        *,
        timeout_seconds=8,
    ) -> bool:
        self.wait_position_calls.append((symbol, expected_qty))
        if self.flat_confirmation:
            self.signed_qty = expected_qty
            return True
        return False

    def find_exact_order_by_tag(self, **kwargs):
        return self.recovered_by_tag.get(kwargs["tag"])

    def latest_order(self, order_id):
        return self.wait_for_order(order_id, timeout_seconds=1)

    def convert_stop_to_market(self, order_id, qty):
        return order_id

    def orders(self):
        return list(self.startup_order_payloads)

    def trades(self):
        return list(self.trade_payloads)

    def positions(self):
        if self.signed_qty == 0:
            return []
        return [
            {
                "exchange": "NSE",
                "product": "MIS",
                "tradingsymbol": "INFY",
                "quantity": self.signed_qty,
            }
        ]


class FakeAIReviewer:
    def __init__(self, decision):
        self.decision = decision
        self.review_calls: list[object] = []
        self.last_response_model = "fake-model"
        self.last_response_id = "fake-response"
        self.last_latency_ms = 1
        self.last_error = ""
        self.last_status = "OK"
        self.last_decision_id = "fake-decision"
        self.last_input_sha256 = "fake-input"
        self.last_input_tokens = 10
        self.last_output_tokens = 5
        self.last_total_tokens = 15

    def review(self, setup):
        self.review_calls.append(setup)
        return self.decision


class IsolatedBotTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.log_directory = root / "logs"
        self.log_directory.mkdir()

        self.patches = [
            mock.patch.object(bot, "STATE_FILE", root / "state.json"),
            mock.patch.object(bot, "LOG_DIR", self.log_directory),
            mock.patch.object(bot, "LIVE_TRADING", False),
            mock.patch.object(bot, "PAPER_SLIPPAGE_BPS", 5.0),
            mock.patch.object(bot, "now_ist", return_value=FIXED_NOW),
            mock.patch.object(bot, "log"),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.temporary_directory.cleanup)


class StrategyHelperSafetyTests(unittest.TestCase):
    def test_rsi_handles_gain_loss_and_flat_edge_cases(self) -> None:
        gains = bot.rsi(pd.Series(range(1, 22), dtype=float))
        losses = bot.rsi(pd.Series(range(22, 1, -1), dtype=float))
        flat = bot.rsi(pd.Series([7.0] * 21))

        self.assertTrue((gains.iloc[1:] == 100.0).all())
        self.assertTrue((losses.iloc[1:] == 0.0).all())
        self.assertTrue((flat == 50.0).all())

    def test_paper_fill_slippage_is_always_adverse(self) -> None:
        with mock.patch.object(bot, "PAPER_SLIPPAGE_BPS", 5.0):
            self.assertAlmostEqual(bot.paper_fill_price(100.0, "BUY"), 100.05)
            self.assertAlmostEqual(bot.paper_fill_price(100.0, "SELL"), 99.95)

    def test_after_cost_payoff_gate_rejects_low_volatility_friction(self) -> None:
        broker = FakeBroker()
        low_volatility = replace(
            make_setup(),
            atr=0.25,
            atr_pct=0.0025,
            lower_circuit_limit=80.0,
            upper_circuit_limit=120.0,
        )
        ordinary_volatility = replace(
            make_setup(),
            atr=1.0,
            atr_pct=0.01,
            lower_circuit_limit=80.0,
            upper_circuit_limit=120.0,
        )

        rejected = bot.build_trade_result(broker, low_volatility)
        accepted = bot.build_trade_result(broker, ordinary_volatility)

        self.assertIsNone(rejected.trade)
        self.assertIn("AFTER_COST_PAYOFF", rejected.reason)
        self.assertIsNotNone(accepted.trade)
        self.assertGreaterEqual(
            accepted.trade.planned_after_cost_payoff,
            bot.MIN_AFTER_COST_PAYOFF_RATIO,
        )
        self.assertLessEqual(
            accepted.trade.planned_risk_amount,
            bot.CAPITAL_LIMIT * bot.RISK_PER_TRADE_PCT,
        )

    def test_directional_circuit_geometry_rejects_unfillable_stop(self) -> None:
        broker = FakeBroker()
        setup = replace(
            make_setup(),
            atr=1.0,
            atr_pct=0.01,
            lower_circuit_limit=99.0,
            upper_circuit_limit=110.0,
        )

        result = bot.build_trade_result(broker, setup)

        self.assertIsNone(result.trade)
        self.assertEqual(result.reason, "LONG_STOP_OUTSIDE_PRICE_BAND")

    def test_modeled_adverse_fill_uses_exact_execution_risk_geometry(self) -> None:
        broker = FakeBroker()
        long_setup = replace(
            make_setup("LONG"),
            price=50.0,
            atr=0.305,
            atr_pct=0.0061,
            lower_circuit_limit=40.0,
            upper_circuit_limit=50.76,
        )
        short_setup = replace(
            make_setup("SHORT"),
            price=50.0,
            atr=0.305,
            atr_pct=0.0061,
            lower_circuit_limit=49.24,
            upper_circuit_limit=60.0,
        )

        long_result = bot.build_trade_result(broker, long_setup)
        short_result = bot.build_trade_result(broker, short_setup)

        self.assertEqual(long_result.reason, "MODELED_LONG_TARGET_OUTSIDE_PRICE_BAND")
        self.assertEqual(short_result.reason, "MODELED_SHORT_TARGET_OUTSIDE_PRICE_BAND")

    def test_malformed_matching_fill_cannot_produce_partial_average(self) -> None:
        broker = FakeBroker()
        broker.trade_payloads = [
            {"order_id": "EXIT-1", "quantity": 2, "average_price": 100.0},
            {"order_id": "EXIT-1", "quantity": 1.5, "average_price": 101.0},
        ]

        with self.assertRaises(ValueError):
            bot.broker_fill_average(broker, ["EXIT-1"])

    def test_remaining_daily_loss_and_open_risk_reduce_entry_budget(self) -> None:
        state = bot.fresh_state()
        state["realized_pnl"] = -700.0

        capacity = bot.entry_capacity(state)
        self.assertAlmostEqual(capacity.candidate_risk_budget, 100.0)

        open_trade = make_trade(status="OPEN_PROTECTED")
        open_trade.execution_mode = "paper"
        open_trade.planned_risk_amount = 100.0
        open_trade.reserved_risk_amount = 100.0
        state["trades"][open_trade.symbol] = asdict(open_trade)

        exhausted = bot.entry_capacity(state)
        self.assertFalse(exhausted.allowed)
        self.assertEqual(exhausted.candidate_risk_budget, 0.0)

    def test_positive_realized_pnl_does_not_expand_per_trade_budget(self) -> None:
        state = bot.fresh_state()
        state["realized_pnl"] = 10_000.0

        capacity = bot.entry_capacity(state)

        self.assertEqual(
            capacity.candidate_risk_budget,
            bot.CAPITAL_LIMIT * bot.RISK_PER_TRADE_PCT,
        )

    def test_adverse_entry_notional_is_reserved_below_position_cap(self) -> None:
        broker = FakeBroker(price=50.40)
        setup = replace(
            make_setup(),
            price=50.40,
            atr=0.23184,
            atr_pct=0.0046,
            lower_circuit_limit=40.0,
            upper_circuit_limit=60.0,
        )

        result = bot.build_trade_result(broker, setup)

        self.assertEqual(result.reason, "OK")
        self.assertIsNotNone(result.trade)
        cap = bot.CAPITAL_LIMIT * bot.MAX_POSITION_PCT
        self.assertLessEqual(result.trade.reserved_notional_amount, cap)
        self.assertLessEqual(result.outcome.entry_fill * result.trade.qty, cap)

    def test_gross_capacity_counts_reserved_adverse_entry_notional(self) -> None:
        state = bot.fresh_state()
        first = make_trade(status="OPEN_PROTECTED")
        first.symbol = "INFY"
        first.execution_mode = "paper"
        first.reserved_risk_amount = 100.0
        first.reserved_notional_amount = 25_000.0
        second = replace(first, symbol="TCS", token=456)
        state["trades"] = {
            first.symbol: asdict(first),
            second.symbol: asdict(second),
        }

        capacity = bot.entry_capacity(state)

        self.assertFalse(capacity.allowed)
        self.assertEqual(capacity.gross_notional_remaining, 0.0)

    def test_live_account_preflight_rejects_untracked_position(self) -> None:
        broker = FakeBroker(signed_qty=10)

        with self.assertRaisesRegex(RuntimeError, "do not match"):
            bot.verify_live_account_matches_state(broker, bot.fresh_state())

    def test_protected_limit_order_types_retain_role_identity(self) -> None:
        trade = make_trade(status="OPEN_PROTECTED")
        trade.entry_order_id = "ENTRY-1"
        trade.entry_tag = "AIENT000000000001"
        trade.stop_order_id = "STOP-1"
        trade.stop_tag = "AISTP000000000001"
        trade.exit_order_id = "EXIT-1"
        trade.exit_order_ids = ["EXIT-1"]
        trade.exit_tags = ["AIEXT000000000001"]
        entry = bot.OrderSnapshot.from_payload(
            {
                "order_id": "ENTRY-1",
                "status": "COMPLETE",
                "quantity": 10,
                "filled_quantity": 10,
                "order_type": "LIMIT",
                "transaction_type": "BUY",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "tag": trade.entry_tag,
            }
        )
        stop = bot.OrderSnapshot.from_payload(
            {
                "order_id": "STOP-1",
                "status": "TRIGGER PENDING",
                "quantity": 10,
                "pending_quantity": 10,
                "order_type": "LIMIT",
                "transaction_type": "SELL",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "tag": trade.stop_tag,
                "trigger_price": 98.0,
            }
        )
        exit_order = bot.OrderSnapshot.from_payload(
            {
                "order_id": "EXIT-1",
                "status": "COMPLETE",
                "quantity": 10,
                "filled_quantity": 10,
                "order_type": "LIMIT",
                "transaction_type": "SELL",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "tag": trade.exit_tags[0],
            }
        )
        instrument = bot.Instrument("INFY", "Infosys", 123, 0.05)

        self.assertTrue(bot.entry_identity_matches(entry, trade))
        self.assertTrue(bot.stop_identity_matches(stop, trade, 10, instrument))
        self.assertTrue(bot.exit_identity_matches(exit_order, trade))
        self.assertFalse(
            bot.exit_identity_matches(
                replace(exit_order, transaction_type="BUY"),
                trade,
            )
        )

    def test_invalid_position_payload_never_means_flat(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "quantity"):
            bot.parse_mis_position_quantities(
                [
                    {
                        "exchange": "NSE",
                        "product": "MIS",
                        "tradingsymbol": "INFY",
                    }
                ]
            )
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            bot.parse_mis_position_quantities(
                [
                    {
                        "exchange": "NSE",
                        "product": "MIS",
                        "tradingsymbol": "INFY",
                        "quantity": 1,
                    },
                    {
                        "exchange": "NSE",
                        "product": "MIS",
                        "tradingsymbol": "INFY",
                        "quantity": 0,
                    },
                ]
            )


class ConfigurationSafetyTests(unittest.TestCase):
    def test_live_mode_requires_exact_confirmation(self) -> None:
        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "LIVE_TRADING_CONFIRM", ""),
            mock.patch.object(bot, "AI_MODE", "off"),
        ):
            with self.assertRaisesRegex(RuntimeError, "I_UNDERSTAND_REAL_MONEY"):
                bot.validate_configuration()

    def test_live_mode_rejects_research_shadow_ai(self) -> None:
        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(
                bot,
                "LIVE_TRADING_CONFIRM",
                "I_UNDERSTAND_REAL_MONEY",
            ),
            mock.patch.object(bot, "AI_MODE", "shadow"),
        ):
            with self.assertRaisesRegex(RuntimeError, "AI_MODE=off or AI_MODE=gate"):
                bot.validate_configuration()

    def test_risk_slippage_must_cover_paper_slippage(self) -> None:
        with (
            mock.patch.object(bot, "PAPER_SLIPPAGE_BPS", 50.0),
            mock.patch.object(bot, "RISK_SLIPPAGE_BPS", 0.0),
        ):
            with self.assertRaisesRegex(RuntimeError, "at least PAPER_SLIPPAGE"):
                bot.validate_configuration()


class StateSafetyTests(IsolatedBotTestCase):
    def test_bot_state_round_trip_preserves_active_trade(self) -> None:
        state = bot.fresh_state()
        trade = make_trade(status="OPEN_PROTECTED")
        trade.execution_mode = "paper"
        state["trades"]["INFY"] = asdict(trade)
        bot.save_state(state)

        loaded = bot.load_state()

        self.assertEqual(loaded, state)
        self.assertEqual(bot.open_trade_count(loaded), 1)
        self.assertEqual(bot.STATE_FILE.stat().st_mode & 0o777, 0o600)

    def test_corrupt_state_refuses_to_start(self) -> None:
        bot.STATE_FILE.write_text("{", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "corrupt or unreadable"):
            bot.load_state()

    def test_active_live_state_cannot_restart_as_paper(self) -> None:
        state = bot.fresh_state()
        state["execution_mode"] = "live"
        trade = make_trade(status="OPEN_PROTECTED")
        trade.execution_mode = "live"
        trade.entry_order_id = "REAL-ENTRY"
        state["trades"][trade.symbol] = asdict(trade)
        bot.save_state(state)

        with self.assertRaisesRegex(RuntimeError, "cannot start in paper mode"):
            bot.load_state()

    def test_previous_date_active_state_is_not_silently_discarded(self) -> None:
        state = bot.fresh_state()
        state["date"] = "2026-08-10"
        trade = make_trade(status="OPEN_PROTECTED")
        trade.execution_mode = "paper"
        trade.entry_order_id = "DRY-ENTRY-INFY"
        state["trades"][trade.symbol] = asdict(trade)
        bot.save_state(state)

        with self.assertRaisesRegex(RuntimeError, "Previous-date active state"):
            bot.load_state()

    def test_explicit_paper_label_cannot_hide_real_order_ids(self) -> None:
        state = bot.fresh_state()
        trade = make_trade(status="OPEN_PROTECTED")
        trade.execution_mode = "paper"
        trade.entry_order_id = "REAL-ENTRY"
        trade.stop_order_id = "REAL-STOP"
        state["trades"][trade.symbol] = asdict(trade)
        bot.save_state(state)

        with self.assertRaisesRegex(RuntimeError, "labelled paper"):
            bot.load_state()

    def test_failed_persist_restores_previous_in_memory_trade(self) -> None:
        state = bot.fresh_state()
        existing = make_trade(status="OPEN_PROTECTED")
        existing.execution_mode = "paper"
        state["trades"][existing.symbol] = asdict(existing)
        changed = replace(existing, status="EXIT_PENDING")

        with mock.patch.object(bot, "save_state", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                bot.persist_trade(state, changed)

        self.assertEqual(state["trades"][existing.symbol]["status"], "OPEN_PROTECTED")


class PaperExecutionLifecycleTests(IsolatedBotTestCase):
    def test_post_fill_notional_breach_invokes_fail_safe(self) -> None:
        broker = FakeBroker(price=100.0)
        state = bot.fresh_state()
        trade = make_trade()
        trade.qty = 250
        trade.requested_qty = 250
        trade.initial_risk_per_share = 0.30
        trade.stop_price = 99.70
        trade.target_price = 100.55
        trade.planned_risk_amount = 100.0
        trade.reserved_risk_amount = 100.0
        trade.reserved_notional_amount = 25_000.0

        with mock.patch.object(bot, "fail_safe_trade_lifecycle") as fail_safe:
            bot.execute_trade(
                broker,
                trade,
                make_setup(),
                approved_decision(),
                state,
            )

        fail_safe.assert_called_once()
        self.assertIn(
            "REMAINING_GROSS_EXPOSURE_EXCEEDED",
            fail_safe.call_args.args[3],
        )

    def _open_paper_trade(self):
        broker = FakeBroker(price=100.0)
        state = bot.fresh_state()
        trade = make_trade()
        bot.execute_trade(
            broker,
            trade,
            make_setup(),
            approved_decision(),
            state,
        )
        return broker, state, bot.trade_from_dict(state["trades"]["INFY"])

    def test_paper_entry_becomes_protected_with_adverse_fill(self) -> None:
        broker, state, trade = self._open_paper_trade()

        self.assertEqual(trade.status, "OPEN_PROTECTED")
        self.assertGreater(trade.entry_price, 100.0)
        self.assertEqual(trade.qty, 10)
        self.assertTrue(trade.stop_order_id)
        self.assertEqual(state["trades_today"], 1)
        self.assertEqual(bot.open_trade_count(state), 1)
        self.assertEqual(broker.entry_calls, [])

    def test_paper_stop_closes_and_accounts_for_costs(self) -> None:
        broker, state, opened = self._open_paper_trade()
        broker.current_price = opened.stop_price - 1.0

        bot.monitor_open_trades(broker, state)
        closed = bot.trade_from_dict(state["trades"]["INFY"])

        self.assertTrue(closed.status.startswith("CLOSED"))
        self.assertIn("STOP", closed.exit_reason)
        self.assertLess(closed.gross_pnl, 0)
        self.assertGreater(closed.fees, 0)
        self.assertAlmostEqual(closed.net_pnl, closed.gross_pnl - closed.fees)
        self.assertAlmostEqual(state["realized_pnl"], closed.net_pnl)
        self.assertAlmostEqual(state["fees_paid"], closed.fees)
        self.assertEqual(state["consecutive_losses"], 1)
        self.assertEqual(bot.open_trade_count(state), 0)

    def test_paper_target_closes_and_accounts_for_costs(self) -> None:
        broker, state, opened = self._open_paper_trade()
        broker.current_price = opened.target_price + 1.0

        bot.monitor_open_trades(broker, state)
        closed = bot.trade_from_dict(state["trades"]["INFY"])

        self.assertTrue(closed.status.startswith("CLOSED"))
        self.assertEqual(closed.exit_reason, "TARGET")
        self.assertGreater(closed.gross_pnl, 0)
        self.assertGreater(closed.fees, 0)
        self.assertAlmostEqual(closed.net_pnl, closed.gross_pnl - closed.fees)
        self.assertAlmostEqual(state["realized_pnl"], closed.net_pnl)
        self.assertEqual(state["consecutive_losses"], 0)
        self.assertEqual(bot.open_trade_count(state), 0)

    def test_quote_failure_closes_unpriced_instead_of_using_target(self) -> None:
        broker, state, _ = self._open_paper_trade()

        with mock.patch.object(broker, "ltp", side_effect=TimeoutError("stale")):
            result = bot.close_trade_market(
                broker,
                state,
                "INFY",
                "FORCE_EXIT_1510",
            )

        closed = bot.trade_from_dict(state["trades"]["INFY"])
        self.assertTrue(result)
        self.assertEqual(closed.status, "CLOSED_UNPRICED")
        self.assertEqual(closed.net_pnl, 0.0)
        self.assertEqual(state["realized_pnl"], 0.0)
        self.assertTrue(state["kill_switch"])

    def test_entry_cutoff_blocks_direct_execution(self) -> None:
        broker = FakeBroker()
        state = bot.fresh_state()
        after_cutoff = FIXED_NOW.replace(hour=14, minute=30)

        with mock.patch.object(bot, "now_ist", return_value=after_cutoff):
            bot.execute_trade(
                broker,
                make_trade(),
                make_setup(),
                approved_decision(),
                state,
            )

        self.assertEqual(broker.entry_calls, [])
        self.assertNotIn("INFY", state["trades"])

    def test_cutoff_crossed_while_persisting_intent_aborts_before_entry(self) -> None:
        broker = FakeBroker()
        state = bot.fresh_state()

        with mock.patch.object(
            bot,
            "entry_window_open",
            side_effect=[True, True, False],
        ):
            bot.execute_trade(
                broker,
                make_trade(),
                make_setup(),
                approved_decision(),
                state,
            )

        trade = bot.trade_from_dict(state["trades"]["INFY"])
        self.assertEqual(trade.status, "ABORTED")
        self.assertEqual(trade.reserved_risk_amount, 0.0)
        self.assertEqual(state["trades_today"], 0)

    def test_post_fill_journal_failure_does_not_block_protection(self) -> None:
        broker = FakeBroker()
        state = bot.fresh_state()

        def flaky_journal(event, **fields):
            if event == "ENTRY_FILLED":
                raise OSError("disk full")

        with mock.patch.object(bot, "journal", side_effect=flaky_journal):
            bot.execute_trade(
                broker,
                make_trade(),
                make_setup(),
                approved_decision(),
                state,
            )

        trade = bot.trade_from_dict(state["trades"]["INFY"])
        self.assertEqual(trade.status, "OPEN_PROTECTED")
        self.assertTrue(trade.stop_order_id.startswith("DRY-STOP-"))


class LiveExecutionInvariantTests(IsolatedBotTestCase):
    def test_unresolved_stop_cancellation_blocks_separate_emergency_exit(self) -> None:
        broker = FakeBroker(price=100.0, signed_qty=10)
        trade = make_trade(status="HALTED_UNCERTAIN")
        trade.execution_mode = "live"
        trade.entry_order_id = "ENTRY-1"
        trade.entry_tag = "AIENT000000000001"
        trade.stop_order_id = "STOP-1"
        trade.stop_tag = "AISTP000000000001"
        broker.order_snapshots["ENTRY-1"] = bot.OrderSnapshot.from_payload(
            {
                "order_id": "ENTRY-1",
                "status": "COMPLETE",
                "quantity": 10,
                "filled_quantity": 10,
                "transaction_type": "BUY",
                "order_type": "LIMIT",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "tag": trade.entry_tag,
            }
        )
        broker.order_snapshots["STOP-1"] = bot.OrderSnapshot.from_payload(
            {
                "order_id": "STOP-1",
                "status": "TRIGGER PENDING",
                "quantity": 10,
                "pending_quantity": 10,
                "transaction_type": "SELL",
                "order_type": "SL-M",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "tag": trade.stop_tag,
                "trigger_price": 98.0,
            }
        )

        real_cancel = broker.cancel_order_confirmed

        def cancel(order_id, *, timeout_seconds=8):
            if order_id == "STOP-1":
                raise TimeoutError("stop cancellation unknown")
            return real_cancel(order_id, timeout_seconds=timeout_seconds)

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(broker, "cancel_order_confirmed", side_effect=cancel),
        ):
            flattened = bot.emergency_flatten_without_state(
                broker,
                trade,
                "TEST",
            )

        self.assertFalse(flattened)
        self.assertEqual(broker.exit_calls, [])

    def test_unknown_authoritative_pnl_halts_and_flattens(self) -> None:
        broker = FakeBroker(signed_qty=10)
        broker.current_intraday_pnl = mock.Mock(side_effect=ValueError("missing pnl"))
        state = bot.fresh_state()
        trade = make_trade(status="OPEN_PROTECTED")
        trade.execution_mode = "live"
        state["trades"][trade.symbol] = asdict(trade)

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "flatten_all", return_value=True) as flatten,
        ):
            bot.enforce_daily_pnl_limit(broker, state)

        self.assertTrue(state["kill_switch"])
        flatten.assert_called_once_with(broker, state, "PNL_DATA_UNAVAILABLE")

    def test_completed_exit_is_not_closed_until_position_is_flat(self) -> None:
        broker = FakeBroker(price=104.0, signed_qty=10)
        broker.flat_confirmation = False
        state = bot.fresh_state()
        trade = make_trade(status="OPEN_PROTECTED")
        trade.entry_order_id = "ENTRY-1"
        trade.stop_order_id = "STOP-1"
        trade.entry_tag = "AIENT000000000001"
        trade.stop_tag = "AISTP000000000001"
        trade.execution_mode = "live"
        trade.entry_status = "COMPLETE"
        trade.stop_status = "CANCELLED"
        broker.order_snapshots["ENTRY-1"] = bot.OrderSnapshot.from_payload(
            {
                "order_id": "ENTRY-1",
                "status": "COMPLETE",
                "quantity": 10,
                "filled_quantity": 10,
                "average_price": 100.0,
                "transaction_type": "BUY",
                "order_type": "LIMIT",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "tag": trade.entry_tag,
            }
        )
        broker.order_snapshots["STOP-1"] = bot.OrderSnapshot.from_payload(
            {
                "order_id": "STOP-1",
                "status": "CANCELLED",
                "quantity": 10,
                "filled_quantity": 0,
                "transaction_type": "SELL",
                "order_type": "SL-M",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "tag": trade.stop_tag,
                "trigger_price": 98.0,
            }
        )
        state["trades"][trade.symbol] = asdict(trade)

        with mock.patch.object(bot, "LIVE_TRADING", True):
            result = bot.close_trade_market(
                broker,
                state,
                trade.symbol,
                "TARGET",
            )

        persisted = bot.trade_from_dict(state["trades"][trade.symbol])
        self.assertFalse(result)
        self.assertFalse(persisted.status.startswith("CLOSED"))
        self.assertEqual(len(broker.exit_calls), 1)
        self.assertIn((trade.symbol, 0), broker.wait_position_calls)
        self.assertTrue(state["kill_switch"])


class LiveFaultRecoveryTests(IsolatedBotTestCase):
    @staticmethod
    def _armed_stop(qty: int):
        return bot.OrderSnapshot.from_payload(
            {
                "order_id": "STOP-1",
                "status": "TRIGGER PENDING",
                "quantity": qty,
                "filled_quantity": 0,
                "pending_quantity": qty,
                "transaction_type": "SELL",
                "order_type": "SL-M",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "trigger_price": 98.0,
            }
        )

    def test_ambiguous_entry_response_recovers_without_duplicate_submission(self) -> None:
        broker = FakeBroker(price=100.25)
        broker.raise_after_accept_entry = True
        broker.order_snapshots["STOP-1"] = self._armed_stop(10)
        state = bot.fresh_state()

        with mock.patch.object(bot, "LIVE_TRADING", True):
            bot.execute_trade(
                broker,
                make_trade(),
                make_setup(),
                approved_decision(),
                state,
            )

        trade = bot.trade_from_dict(state["trades"]["INFY"])
        self.assertEqual(len(broker.entry_calls), 1)
        self.assertEqual(trade.entry_order_id, "ENTRY-RECOVERED")
        self.assertEqual(trade.status, "OPEN_PROTECTED")
        self.assertEqual(len(broker.stop_calls), 1)
        self.assertFalse(state["kill_switch"])

    def test_partial_entry_is_cancelled_then_only_filled_quantity_is_protected(self) -> None:
        broker = FakeBroker(price=100.20)
        broker.order_snapshots["ENTRY-1"] = bot.OrderSnapshot.from_payload(
            {
                "order_id": "ENTRY-1",
                "status": "OPEN",
                "quantity": 10,
                "filled_quantity": 4,
                "pending_quantity": 6,
                "average_price": 100.20,
                "transaction_type": "BUY",
            }
        )
        broker.cancel_snapshots["ENTRY-1"] = bot.OrderSnapshot.from_payload(
            {
                "order_id": "ENTRY-1",
                "status": "CANCELLED",
                "quantity": 10,
                "filled_quantity": 4,
                "pending_quantity": 0,
                "average_price": 100.20,
                "transaction_type": "BUY",
            }
        )
        broker.order_snapshots["STOP-1"] = self._armed_stop(4)
        state = bot.fresh_state()

        with mock.patch.object(bot, "LIVE_TRADING", True):
            bot.execute_trade(
                broker,
                make_trade(),
                make_setup(),
                approved_decision(),
                state,
            )

        trade = bot.trade_from_dict(state["trades"]["INFY"])
        self.assertEqual(broker.cancel_calls.count("ENTRY-1"), 1)
        self.assertEqual(trade.qty, 4)
        self.assertEqual(trade.entry_status, "CANCELLED")
        self.assertEqual(trade.status, "OPEN_PROTECTED")
        self.assertEqual(len(broker.stop_calls), 1)
        self.assertEqual(broker.stop_calls[0][2], 4)
        self.assertEqual(broker.order_snapshots["STOP-1"].pending, 4)
        self.assertTrue(
            any(
                option["order_id"] == "ENTRY-1"
                and option["return_on_partial"]
                for option in broker.wait_options
            )
        )
        expected = bot.estimate_after_cost_outcome(
            trade.side,
            trade.entry_price,
            trade.initial_risk_per_share,
            trade.initial_risk_per_share * bot.TARGET_R_MULTIPLE,
            trade.qty,
            broker._instrument.tick_size,
            include_entry_slippage=False,
        )
        self.assertAlmostEqual(trade.reserved_risk_amount, expected.stop_loss)

    def test_state_write_failure_after_fill_forces_flat_account(self) -> None:
        broker = FakeBroker(price=100.20)
        calls = 0
        real_save_state = bot.save_state

        def fail_third_save(state):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("disk full after fill")
            return real_save_state(state)

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "save_state", side_effect=fail_third_save),
        ):
            state = bot.fresh_state()
            bot.execute_trade(
                broker,
                make_trade(),
                make_setup(),
                approved_decision(),
                state,
            )

        self.assertEqual(broker.signed_qty, 0)
        self.assertTrue(state["kill_switch"])
        self.assertGreaterEqual(len(broker.exit_calls), 1)

    def test_ambiguous_exit_response_recovers_and_second_close_is_idempotent(self) -> None:
        broker = FakeBroker(price=104.0, signed_qty=10)
        broker.raise_after_accept_exit = True
        broker.order_snapshots["ENTRY-1"] = bot.OrderSnapshot.from_payload(
            {
                "order_id": "ENTRY-1",
                "status": "COMPLETE",
                "quantity": 10,
                "filled_quantity": 10,
                "pending_quantity": 0,
                "average_price": 100.0,
                "transaction_type": "BUY",
            }
        )
        broker.order_snapshots["STOP-1"] = bot.OrderSnapshot.from_payload(
            {
                "order_id": "STOP-1",
                "status": "CANCELLED",
                "quantity": 10,
                "filled_quantity": 0,
                "pending_quantity": 0,
                "transaction_type": "SELL",
            }
        )
        state = bot.fresh_state()
        trade = make_trade(status="OPEN_PROTECTED")
        trade.entry_order_id = "ENTRY-1"
        trade.stop_order_id = "STOP-1"
        trade.entry_tag = "AIENT000000000001"
        trade.stop_tag = "AISTP000000000001"
        trade.execution_mode = "live"
        trade.entry_status = "COMPLETE"
        trade.stop_status = "CANCELLED"
        state["trades"][trade.symbol] = asdict(trade)
        broker.order_snapshots["ENTRY-1"] = replace(
            broker.order_snapshots["ENTRY-1"],
            order_type="LIMIT",
            symbol="INFY",
            exchange="NSE",
            product="MIS",
            tag=trade.entry_tag,
        )
        broker.order_snapshots["STOP-1"] = replace(
            broker.order_snapshots["STOP-1"],
            order_type="SL-M",
            symbol="INFY",
            exchange="NSE",
            product="MIS",
            tag=trade.stop_tag,
            trigger_price=98.0,
        )

        with mock.patch.object(bot, "LIVE_TRADING", True):
            first = bot.close_trade_market(broker, state, trade.symbol, "TARGET")
            second = bot.close_trade_market(broker, state, trade.symbol, "TARGET")

        closed = bot.trade_from_dict(state["trades"][trade.symbol])
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(len(broker.exit_calls), 1)
        self.assertEqual(closed.exit_order_id, "EXIT-RECOVERED")
        self.assertEqual(closed.status, "CLOSED_TARGET")
        self.assertEqual(broker.signed_qty, 0)

    def test_startup_cancels_bot_orphan_but_leaves_unowned_order(self) -> None:
        broker = FakeBroker()
        broker.startup_order_payloads = [
            {
                "order_id": "ORPHAN-AI",
                "status": "OPEN",
                "quantity": 10,
                "pending_quantity": 10,
                "tag": "AISTP000000000001",
            },
            {
                "order_id": "USER-ORDER",
                "status": "OPEN",
                "quantity": 2,
                "pending_quantity": 2,
                "tag": "MANUAL",
            },
        ]
        state = bot.fresh_state()

        with mock.patch.object(bot, "LIVE_TRADING", True):
            bot.reconcile_startup(broker, state)

        self.assertEqual(broker.cancel_calls, ["ORPHAN-AI"])
        self.assertEqual(broker.entry_calls, [])
        self.assertEqual(broker.exit_calls, [])
        self.assertFalse(state["kill_switch"])


class AIModeScanSemanticsTests(IsolatedBotTestCase):
    def _run_scan(self, mode: str, decision):
        broker = FakeBroker()
        broker.strategy_candles = mock.Mock(return_value=pd.DataFrame())
        state = bot.fresh_state()
        setup = make_setup()
        reviewer = FakeAIReviewer(decision)
        candidate = mock.Mock(symbol="INFY", token=123)

        with (
            mock.patch.object(bot, "AI_MODE", mode),
            mock.patch.object(
                bot,
                "select_stocks_in_play",
                return_value=[candidate],
            ),
            mock.patch.object(
                bot,
                "get_nifty_regime",
                return_value=("BULL", 0.3),
            ),
            mock.patch.object(bot, "detect_setup", return_value=setup),
            mock.patch.object(
                bot,
                "revalidate_live_setup",
                return_value=(setup, ""),
            ),
            mock.patch.object(
                bot,
                "build_trade_result",
                return_value=bot.TradeBuildResult(make_trade(), "OK"),
            ),
            mock.patch.object(bot, "execute_trade") as execute,
            mock.patch.object(bot, "journal"),
            mock.patch.object(bot.time, "sleep"),
        ):
            bot.scan_for_new_trades(broker, reviewer, state)

        return reviewer, execute

    def test_off_mode_skips_ai_and_executes_deterministic_approval(self) -> None:
        reviewer, execute = self._run_scan(
            "off",
            bot.AIDecision(
                decision="REJECT",
                confidence=100,
                quality_score=0,
                reason="must be ignored",
                risk_flags=[],
            ),
        )

        self.assertEqual(reviewer.review_calls, [])
        execute.assert_called_once()
        passed_decision = execute.call_args.args[3]
        self.assertEqual(passed_decision.decision, "APPROVE")
        self.assertEqual(passed_decision.confidence, 100)

    def test_shadow_mode_records_rejection_but_does_not_gate_execution(self) -> None:
        rejection = bot.AIDecision(
            decision="REJECT",
            confidence=99,
            quality_score=10,
            reason="shadow rejection",
            risk_flags=["TEST"],
        )
        reviewer, execute = self._run_scan("shadow", rejection)

        self.assertEqual(len(reviewer.review_calls), 1)
        model_payload = reviewer.review_calls[0]
        rendered = bot.stable_json_sha256(model_payload)
        self.assertNotIn("symbol", model_payload["setup"])
        self.assertNotIn("token", model_payload["setup"])
        self.assertNotIn("technical_score", model_payload["setup"])
        self.assertEqual(len(rendered), 64)
        execute.assert_called_once()
        self.assertIs(execute.call_args.args[3], rejection)

    def test_gate_mode_blocks_rejection_and_allows_strong_approval(self) -> None:
        rejection = bot.AIDecision(
            decision="REJECT",
            confidence=99,
            quality_score=99,
            reason="gate rejection",
            risk_flags=[],
        )
        rejected_reviewer, rejected_execute = self._run_scan("gate", rejection)
        self.assertEqual(len(rejected_reviewer.review_calls), 1)
        rejected_execute.assert_not_called()

        approval = bot.AIDecision(
            decision="APPROVE",
            confidence=90,
            quality_score=90,
            reason="gate approval",
            risk_flags=[],
        )
        approved_reviewer, approved_execute = self._run_scan("gate", approval)
        self.assertEqual(len(approved_reviewer.review_calls), 1)
        approved_execute.assert_called_once()
        self.assertIs(approved_execute.call_args.args[3], approval)


if __name__ == "__main__":
    unittest.main()


class ProductionOrderRecoveryRegressionTests(IsolatedBotTestCase):
    def test_aeroflex_pattern_trigger_pending_is_valid_when_exact_order_id_known(self) -> None:
        trade = make_trade(status="OPEN_PROTECTED")
        trade.symbol = "AEROFLEX"
        trade.qty = 44
        trade.requested_qty = 44
        trade.stop_price = 489.30
        trade.stop_order_id = "260821190570620"
        trade.stop_tag = "AISTPLOCALTAG01"
        instrument = bot.Instrument("AEROFLEX", "Aeroflex", 123, 0.05)
        # Reproduces the production symptom: broker order is armed and exactly
        # covers the position, but an authoritative payload can omit the tag.
        stop = bot.OrderSnapshot.from_payload(
            {
                "order_id": "260821190570620",
                "status": "TRIGGER PENDING",
                "quantity": 44,
                "filled_quantity": 0,
                "pending_quantity": 44,
                "transaction_type": "SELL",
                "order_type": "SL-M",
                "tradingsymbol": "AEROFLEX",
                "exchange": "NSE",
                "product": "MIS",
                "tag": "",
                "trigger_price": 489.30,
            }
        )

        self.assertTrue(stop.stop_armed)
        self.assertTrue(bot.stop_identity_matches(stop, trade, 44, instrument))
        self.assertTrue(bot.stop_exactly_protects(stop, trade, 44, instrument))
        self.assertEqual(bot.stop_identity_mismatches(stop, trade, 44, instrument), [])

    def test_duplicate_websocket_order_updates_are_idempotent(self) -> None:
        broker = object.__new__(bot.KiteBroker)
        broker._order_condition = threading.Condition()
        broker._order_updates = {}
        payload = {
            "order_id": "OID-1",
            "status": "TRIGGER PENDING",
            "quantity": 10,
            "filled_quantity": 0,
            "pending_quantity": 10,
            "transaction_type": "SELL",
            "order_type": "SL-M",
            "tradingsymbol": "INFY",
            "exchange": "NSE",
            "product": "MIS",
            "tag": "AISTP000000000001",
            "trigger_price": 98.0,
        }

        self.assertTrue(broker._record_order_update(payload))
        self.assertFalse(broker._record_order_update(dict(payload)))
        self.assertEqual(len(broker._order_updates), 1)

    def test_latest_order_prefers_full_orderbook_over_sparse_history(self) -> None:
        broker = object.__new__(bot.KiteBroker)
        broker._order_condition = threading.Condition()
        broker._order_updates = {}
        broker.kite = mock.Mock()
        broker.kite.orders.return_value = [
            {
                "order_id": "STOP-1",
                "status": "TRIGGER PENDING",
                "quantity": 10,
                "filled_quantity": 0,
                "pending_quantity": 10,
                "transaction_type": "SELL",
                "order_type": "SL-M",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "tag": "AISTP000000000001",
                "trigger_price": 98.0,
            }
        ]
        broker.kite.order_history.return_value = [
            {
                "order_id": "STOP-1",
                "status": "TRIGGER PENDING",
                "quantity": 10,
                "pending_quantity": 10,
                "tag": "",
            }
        ]

        snapshot = broker.latest_order("STOP-1")

        self.assertEqual(snapshot.tag, "AISTP000000000001")
        self.assertEqual(snapshot.symbol, "INFY")
        self.assertTrue(snapshot.stop_armed)
        broker.kite.order_history.assert_not_called()

    def _dedicated_broker(self, *, stop_fills_on_cancel: bool = False):
        broker = FakeBroker(signed_qty=10)
        broker._active_orders = [
            {
                "order_id": "UNKNOWN-STOP",
                "status": "TRIGGER PENDING",
                "quantity": 10,
                "filled_quantity": 0,
                "pending_quantity": 10,
                "transaction_type": "SELL",
                "order_type": "SL-M",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "tag": "",
                "trigger_price": 98.0,
            }
        ]
        broker.orders = mock.Mock(side_effect=lambda: list(broker._active_orders))
        broker.latest_order = mock.Mock(
            side_effect=lambda order_id: bot.OrderSnapshot.from_payload(
                next(row for row in broker._active_orders if row["order_id"] == order_id)
            )
        )

        def cancel(order_id, *, timeout_seconds=8):
            row = next(row for row in broker._active_orders if row["order_id"] == order_id)
            broker.cancel_calls.append(order_id)
            if stop_fills_on_cancel:
                broker.signed_qty = 0
                terminal_status = "COMPLETE"
                filled = row["quantity"]
            else:
                terminal_status = "CANCELLED"
                filled = 0
            broker._active_orders[:] = [
                item for item in broker._active_orders if item["order_id"] != order_id
            ]
            return bot.OrderSnapshot.from_payload(
                {
                    **row,
                    "status": terminal_status,
                    "filled_quantity": filled,
                    "pending_quantity": 0,
                    "average_price": 99.0 if filled else 0.0,
                }
            )

        broker.cancel_order_confirmed = mock.Mock(side_effect=cancel)

        def exit_market(inst, signed_qty, tag):
            broker.exit_calls.append((inst.symbol, signed_qty, tag))
            broker.signed_qty = 0
            broker.order_metadata["EXIT-RECOVERY"] = {
                "symbol": inst.symbol,
                "exchange": "NSE",
                "product": "MIS",
                "transaction_type": "SELL" if signed_qty > 0 else "BUY",
                "order_type": "MARKET",
                "tag": tag,
            }
            broker.order_snapshots["EXIT-RECOVERY"] = bot.OrderSnapshot.from_payload(
                {
                    "order_id": "EXIT-RECOVERY",
                    "status": "COMPLETE",
                    "quantity": abs(signed_qty),
                    "filled_quantity": abs(signed_qty),
                    "pending_quantity": 0,
                    "average_price": 99.0,
                    "transaction_type": "SELL" if signed_qty > 0 else "BUY",
                    "order_type": "MARKET",
                    "tradingsymbol": inst.symbol,
                    "exchange": "NSE",
                    "product": "MIS",
                    "tag": tag,
                }
            )
            return "EXIT-RECOVERY"

        broker.exit_market = mock.Mock(side_effect=exit_market)
        broker.wait_for_order = mock.Mock(
            side_effect=lambda order_id, **kwargs: broker.order_snapshots[order_id]
        )
        broker.wait_for_position_qty = mock.Mock(
            side_effect=lambda symbol, expected_qty, **kwargs: broker.signed_qty == expected_qty
        )
        return broker

    def test_dedicated_recovery_cancels_unknown_stop_then_flattens_once(self) -> None:
        broker = self._dedicated_broker()
        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot.time, "sleep"),
        ):
            flat, ids = bot.dedicated_force_flatten_symbol(broker, "INFY", "TEST")

        self.assertTrue(flat)
        self.assertEqual(ids, ["EXIT-RECOVERY"])
        self.assertEqual(broker.cancel_calls, ["UNKNOWN-STOP"])
        self.assertEqual(len(broker.exit_calls), 1)
        self.assertEqual(broker.signed_qty, 0)

    def test_dedicated_recovery_does_not_double_exit_if_stop_fills_while_canceling(self) -> None:
        broker = self._dedicated_broker(stop_fills_on_cancel=True)
        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot.time, "sleep"),
        ):
            flat, ids = bot.dedicated_force_flatten_symbol(broker, "INFY", "TEST")

        self.assertTrue(flat)
        self.assertEqual(ids, [])
        self.assertEqual(broker.exit_calls, [])
        self.assertEqual(broker.signed_qty, 0)

    def test_dedicated_startup_flattens_untracked_mis_position(self) -> None:
        broker = self._dedicated_broker()
        state = bot.fresh_state()
        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot.time, "sleep"),
        ):
            bot.reconcile_startup(broker, state)

        self.assertEqual(broker.signed_qty, 0)
        self.assertEqual(len(broker.exit_calls), 1)

    def test_kill_switch_recovery_stops_mutating_after_retry_cap(self) -> None:
        broker = FakeBroker(signed_qty=10)
        state = bot.fresh_state()
        trade = make_trade(status="OPEN_PROTECTED")
        trade.execution_mode = "live"
        state["trades"][trade.symbol] = asdict(trade)
        state["kill_switch"] = True
        state["halt_reason"] = "test halt"

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "MAX_KILL_SWITCH_FLATTEN_ATTEMPTS", 2),
            mock.patch.object(bot, "flatten_all", return_value=False) as flatten,
        ):
            self.assertFalse(bot.handle_kill_switch_recovery(broker, state))
            self.assertFalse(bot.handle_kill_switch_recovery(broker, state))
            self.assertFalse(bot.handle_kill_switch_recovery(broker, state))

        self.assertEqual(flatten.call_count, 2)
        self.assertTrue(state["manual_intervention_required"])
        self.assertIn("automatic flatten recovery exhausted", state["halt_reason"])


class DedicatedAccountAdditionalSafetyTests(IsolatedBotTestCase):
    def test_non_dedicated_startup_still_rejects_unowned_nse_mis_order(self) -> None:
        broker = FakeBroker()
        broker.startup_order_payloads = [
            {
                "order_id": "MANUAL-1",
                "status": "TRIGGER PENDING",
                "quantity": 10,
                "pending_quantity": 10,
                "transaction_type": "SELL",
                "order_type": "SL-M",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "tag": "MANUAL",
                "trigger_price": 98.0,
            }
        ]
        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", False),
        ):
            with self.assertRaisesRegex(RuntimeError, "Unowned active NSE/MIS order"):
                bot.reconcile_startup(broker, bot.fresh_state())

    def test_dedicated_startup_cancels_unowned_nse_mis_order_when_flat(self) -> None:
        broker = FakeBroker(signed_qty=0)
        active = {
            "order_id": "MANUAL-1",
            "status": "TRIGGER PENDING",
            "quantity": 10,
            "pending_quantity": 10,
            "transaction_type": "SELL",
            "order_type": "SL-M",
            "tradingsymbol": "INFY",
            "exchange": "NSE",
            "product": "MIS",
            "tag": "",
            "trigger_price": 98.0,
        }
        broker.startup_order_payloads = [active]
        broker.cancel_order_confirmed = mock.Mock(
            return_value=bot.OrderSnapshot.from_payload(
                {**active, "status": "CANCELLED", "pending_quantity": 0}
            )
        )
        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
        ):
            bot.reconcile_startup(broker, bot.fresh_state())

        broker.cancel_order_confirmed.assert_called_once()

    def test_ambiguous_dedicated_recovery_is_never_resubmitted_in_same_process(self) -> None:
        broker = FakeBroker(signed_qty=10)
        broker.orders = mock.Mock(return_value=[])
        broker._recovery_ambiguous_symbols = set()
        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot.time, "sleep"),
            mock.patch.object(
                bot,
                "submit_or_recover_order",
                side_effect=RuntimeError("UNKNOWN broker submission for tag X; do not retry"),
            ) as submit,
        ):
            first, _ = bot.dedicated_force_flatten_symbol(broker, "INFY", "TEST")
            second, _ = bot.dedicated_force_flatten_symbol(broker, "INFY", "TEST")

        self.assertFalse(first)
        self.assertFalse(second)
        self.assertEqual(submit.call_count, 1)
        self.assertIn("INFY", broker._recovery_ambiguous_symbols)

class DedicatedRecoveryInstrumentFallbackTests(IsolatedBotTestCase):
    def test_dedicated_recovery_can_flatten_symbol_missing_from_scanner_universe(self) -> None:
        broker = FakeBroker(signed_qty=10)
        broker.instrument = mock.Mock(return_value=None)
        broker.orders = mock.Mock(return_value=[])
        broker.exit_market = mock.Mock(side_effect=lambda inst, signed_qty, tag: "EXIT-FALLBACK")
        broker.order_snapshots["EXIT-FALLBACK"] = bot.OrderSnapshot.from_payload(
            {
                "order_id": "EXIT-FALLBACK",
                "status": "COMPLETE",
                "quantity": 10,
                "filled_quantity": 10,
                "pending_quantity": 0,
                "average_price": 99.0,
                "transaction_type": "SELL",
                "order_type": "MARKET",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "tag": "AIRCVA123456789",
            }
        )
        def wait(order_id, **kwargs):
            broker.signed_qty = 0
            return broker.order_snapshots[order_id]
        broker.wait_for_order = mock.Mock(side_effect=wait)
        broker.wait_for_position_qty = mock.Mock(
            side_effect=lambda symbol, expected_qty, **kwargs: broker.signed_qty == expected_qty
        )
        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot.time, "sleep"),
        ):
            flat, ids = bot.dedicated_force_flatten_symbol(broker, "INFY", "TEST")

        self.assertTrue(flat)
        self.assertEqual(ids, ["EXIT-FALLBACK"])
        called_inst = broker.exit_market.call_args.args[0]
        self.assertEqual(called_inst.symbol, "INFY")
