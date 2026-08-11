from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
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
        return "STOP-1"

    def cancel_order_confirmed(self, order_id, *, timeout_seconds=8):
        self.cancel_calls.append(order_id)
        if order_id in self.cancel_snapshots:
            snapshot = self.cancel_snapshots[order_id]
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
    ):
        if order_id in self.order_snapshots:
            return self.order_snapshots[order_id]
        if order_id == "STOP-1" or require_stop_armed:
            return bot.OrderSnapshot.from_payload(
                {
                    "order_id": order_id,
                    "status": "TRIGGER PENDING",
                    "quantity": 10,
                    "filled_quantity": 0,
                    "pending_quantity": 10,
                    "trigger_price": 98.0,
                }
            )
        return bot.OrderSnapshot.from_payload(
            {
                "order_id": order_id,
                "status": "COMPLETE",
                "quantity": 10,
                "filled_quantity": 10,
                "pending_quantity": 0,
                "average_price": self.current_price,
            }
        )

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


class StateSafetyTests(IsolatedBotTestCase):
    def test_bot_state_round_trip_preserves_active_trade(self) -> None:
        state = bot.fresh_state()
        state["trades"]["INFY"] = asdict(
            make_trade(status="OPEN_PROTECTED")
        )
        bot.save_state(state)

        loaded = bot.load_state()

        self.assertEqual(loaded, state)
        self.assertEqual(bot.open_trade_count(loaded), 1)
        self.assertEqual(bot.STATE_FILE.stat().st_mode & 0o777, 0o600)

    def test_corrupt_state_refuses_to_start(self) -> None:
        bot.STATE_FILE.write_text("{", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "corrupt or unreadable"):
            bot.load_state()


class PaperExecutionLifecycleTests(IsolatedBotTestCase):
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


class LiveExecutionInvariantTests(IsolatedBotTestCase):
    def test_completed_exit_is_not_closed_until_position_is_flat(self) -> None:
        broker = FakeBroker(price=104.0, signed_qty=10)
        broker.flat_confirmation = False
        state = bot.fresh_state()
        trade = make_trade(status="OPEN_PROTECTED")
        trade.entry_order_id = "ENTRY-1"
        trade.stop_order_id = "STOP-1"
        trade.entry_status = "COMPLETE"
        trade.stop_status = "TRIGGER PENDING"
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
        trade.entry_status = "COMPLETE"
        trade.stop_status = "CANCELLED"
        state["trades"][trade.symbol] = asdict(trade)

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
            mock.patch.object(bot, "build_trade", return_value=make_trade()),
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
