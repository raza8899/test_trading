from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from trading_core import (
    InstanceAlreadyRunningError,
    NSEEquityIntradayRates,
    OrderSnapshot,
    SingleInstanceLock,
    StateFileError,
    atomic_write_json,
    directional_slippage_bps,
    directional_slippage_per_share,
    estimate_nse_equity_intraday_cost,
    gross_pnl,
    load_json_strict,
    strict_finite_float,
    strict_integral,
)


class OrderSnapshotTests(unittest.TestCase):
    def test_payload_is_normalised_and_snapshot_is_immutable(self) -> None:
        snapshot = OrderSnapshot.from_payload(
            {
                "order_id": 12345,
                "status": " trigger_pending ",
                "quantity": "10",
                "filled_quantity": "2.0",
                "unfilled_quantity": 8,
                "average_price": "101.25",
                "status_message": "  accepted  ",
                "order_type": "sl-m",
                "transaction_type": "sell",
                "tradingsymbol": "infy",
                "exchange": "nse",
                "product": "mis",
                "tag": "AIB260811E01",
                "trigger_price": "99.50",
            }
        )

        self.assertEqual(snapshot.order_id, "12345")
        self.assertEqual(snapshot.status, "TRIGGER PENDING")
        self.assertEqual((snapshot.qty, snapshot.filled, snapshot.pending), (10, 2, 8))
        self.assertEqual(snapshot.avg, 101.25)
        self.assertEqual(snapshot.message, "accepted")
        self.assertEqual(snapshot.symbol, "INFY")
        self.assertEqual(snapshot.order_type, "SL-M")
        self.assertTrue(snapshot.stop_armed)
        self.assertTrue(snapshot.is_stop_armed())
        self.assertFalse(snapshot.terminal)

        with self.assertRaises(FrozenInstanceError):
            snapshot.status = "COMPLETE"  # type: ignore[misc]

    def test_terminal_statuses_and_cancelled_spelling(self) -> None:
        for status in ("complete", "cancelled", "canceled", "REJECTED"):
            with self.subTest(status=status):
                snapshot = OrderSnapshot.from_payload(
                    {"status": status, "quantity": 1, "filled_quantity": 0}
                )
                self.assertTrue(snapshot.terminal)
                self.assertTrue(snapshot.is_terminal())

        working = OrderSnapshot.from_payload(
            {"status": "OPEN", "quantity": 1, "pending_quantity": 1}
        )
        self.assertFalse(working.terminal)

    def test_pending_is_derived_when_broker_omits_it(self) -> None:
        snapshot = OrderSnapshot.from_payload(
            {
                "status": "OPEN",
                "quantity": 10,
                "filled_quantity": 3,
                "cancelled_quantity": 2,
            }
        )
        self.assertEqual(snapshot.pending, 5)

    def test_stop_requires_pending_quantity(self) -> None:
        snapshot = OrderSnapshot.from_payload(
            {
                "status": "TRIGGER PENDING",
                "quantity": 5,
                "filled_quantity": 5,
                "pending_quantity": 0,
            }
        )
        self.assertFalse(snapshot.stop_armed)

    def test_rejects_invalid_quantities_and_prices(self) -> None:
        bad_payloads = (
            {"quantity": -1},
            {"quantity": 1.5},
            {"quantity": 1, "filled_quantity": 2},
            {"quantity": 1, "average_price": float("nan")},
        )
        for payload in bad_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                OrderSnapshot.from_payload(payload)


class StrictBrokerValueTests(unittest.TestCase):
    def test_signed_integral_parser_does_not_round_or_default(self) -> None:
        self.assertEqual(strict_integral("-7", field="quantity"), -7)
        self.assertEqual(strict_integral("4.0", field="quantity"), 4)
        for value in (None, "", True, 1.5, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                strict_integral(value, field="quantity")

    def test_finite_float_parser_rejects_missing_and_nonfinite(self) -> None:
        self.assertEqual(strict_finite_float("-12.5", field="pnl"), -12.5)
        for value in (None, "", False, "bad", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                strict_finite_float(value, field="pnl")


class JsonStateTests(unittest.TestCase):
    def test_atomic_write_round_trips_and_uses_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "state.json"
            atomic_write_json(path, {"revision": 3, "kill_switch": True})

            self.assertEqual(
                load_json_strict(path),
                {"revision": 3, "kill_switch": True},
            )
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_failed_replace_preserves_previous_state_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text('{"revision": 1}\n', encoding="utf-8")

            with mock.patch("trading_core.os.replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    atomic_write_json(path, {"revision": 2})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"revision": 1})
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_strict_load_rejects_missing_malformed_duplicate_and_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "missing.json": None,
                "malformed.json": "{",
                "duplicate.json": '{"revision": 1, "revision": 2}',
                "nonfinite.json": '{"pnl": NaN}',
                "wrong_type.json": "[]",
            }
            for name, content in cases.items():
                path = root / name
                if content is not None:
                    path.write_text(content, encoding="utf-8")
                with self.subTest(name=name), self.assertRaises(StateFileError):
                    load_json_strict(path)

    def test_expected_type_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            path.write_text("[]", encoding="utf-8")
            self.assertEqual(load_json_strict(path, expected_type=list), [])


class SingleInstanceLockTests(unittest.TestCase):
    def test_context_manager_excludes_second_owner_then_releases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bot.lock"
            first = SingleInstanceLock(path)
            second = SingleInstanceLock(path)

            with first:
                self.assertTrue(first.acquired)
                self.assertEqual(path.read_text(encoding="ascii"), f"{os.getpid()}\n")
                with self.assertRaises(InstanceAlreadyRunningError):
                    second.acquire()

            self.assertFalse(first.acquired)
            with second:
                self.assertTrue(second.acquired)

    def test_acquire_and_release_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = SingleInstanceLock(Path(directory) / "bot.lock")
            self.assertIs(lock.acquire(), lock)
            self.assertIs(lock.acquire(), lock)
            lock.release()
            lock.release()
            self.assertFalse(lock.acquired)


class KiteBrokerAdapterTests(unittest.TestCase):
    @staticmethod
    def make_broker():
        import bot

        broker = bot.KiteBroker.__new__(bot.KiteBroker)
        broker._order_condition = threading.Condition()
        broker._order_updates = {}
        return bot, broker

    def test_terminal_websocket_state_cannot_regress_to_open(self) -> None:
        _, broker = self.make_broker()
        broker._record_order_update(
            {
                "order_id": "OID1",
                "status": "COMPLETE",
                "quantity": 5,
                "filled_quantity": 5,
            }
        )
        broker._record_order_update(
            {
                "order_id": "OID1",
                "status": "OPEN",
                "quantity": 5,
                "filled_quantity": 5,
            }
        )
        self.assertEqual(broker._order_updates["OID1"].status, "COMPLETE")

    def test_converted_stop_postback_cannot_regress_to_stale_slm_state(self) -> None:
        _, broker = self.make_broker()
        common = {
            "order_id": "OID-STOP",
            "quantity": 10,
            "filled_quantity": 0,
            "pending_quantity": 10,
            "transaction_type": "SELL",
            "tradingsymbol": "INFY",
            "exchange": "NSE",
            "product": "MIS",
            "tag": "AISTP000000000001",
            "trigger_price": 98.0,
        }
        broker._record_order_update(
            {**common, "status": "OPEN", "order_type": "LIMIT"}
        )
        broker._record_order_update(
            {**common, "status": "TRIGGER PENDING", "order_type": "SL-M"}
        )

        current = broker._order_updates["OID-STOP"]
        self.assertEqual(current.status, "OPEN")
        self.assertEqual(current.order_type, "LIMIT")

    def test_latest_order_falls_back_to_websocket_cache_on_rest_error(self) -> None:
        _, broker = self.make_broker()
        cached = OrderSnapshot.from_payload(
            {
                "order_id": "OID2",
                "status": "CANCELLED",
                "quantity": 3,
                "filled_quantity": 1,
            }
        )
        broker._order_updates["OID2"] = cached
        broker.kite = SimpleNamespace(
            order_history=mock.Mock(side_effect=TimeoutError("REST unavailable"))
        )

        self.assertIs(broker.latest_order("OID2"), cached)

    def test_cancel_confirmed_rejects_nonterminal_timeout_result(self) -> None:
        bot, broker = self.make_broker()
        working = OrderSnapshot.from_payload(
            {
                "order_id": "OID3",
                "status": "CANCEL PENDING",
                "quantity": 2,
                "pending_quantity": 2,
            }
        )
        broker.kite = SimpleNamespace(
            VARIETY_REGULAR="regular",
            cancel_order=mock.Mock(return_value="OID3"),
        )
        broker.latest_order = mock.Mock(return_value=working)
        broker.wait_for_order = mock.Mock(return_value=working)

        with mock.patch.object(bot, "LIVE_TRADING", True):
            with self.assertRaises(TimeoutError):
                broker.cancel_order_confirmed("OID3", timeout_seconds=1)

    def test_dry_exit_logs_absolute_quantity_and_uses_position_sign(self) -> None:
        bot, broker = self.make_broker()
        broker.kite = SimpleNamespace(
            TRANSACTION_TYPE_BUY="BUY",
            TRANSACTION_TYPE_SELL="SELL",
        )
        instrument = bot.Instrument("INFY", "Infosys", 408065, 0.05)

        with (
            mock.patch.object(bot, "LIVE_TRADING", False),
            mock.patch.object(bot, "log") as log_mock,
        ):
            order_id = broker.exit_market(instrument, -7, "AIEXIT123")

        self.assertTrue(order_id.startswith("DRY-EXIT-INFY-"))
        log_mock.assert_called_once_with("DRY RUN: EXIT BUY 7 INFY MARKET")
        with self.assertRaises(ValueError):
            broker.exit_market(instrument, 0, "AIEXIT123")

    def test_order_adapter_rejects_invalid_side_quantity_tag_and_trigger(self) -> None:
        bot, broker = self.make_broker()
        broker.kite = SimpleNamespace(
            TRANSACTION_TYPE_BUY="BUY",
            TRANSACTION_TYPE_SELL="SELL",
        )
        instrument = bot.Instrument("INFY", "Infosys", 408065, 0.05)

        with mock.patch.object(bot, "LIVE_TRADING", False):
            with self.assertRaises(ValueError):
                broker.place_market_entry(instrument, "SIDEWAYS", 1, "AIENTRY1")
            with self.assertRaises(ValueError):
                broker.place_market_entry(instrument, "LONG", 0, "AIENTRY1")
            with self.assertRaises(ValueError):
                broker.place_market_entry(instrument, "LONG", 1, "bad-tag")
            with self.assertRaises(ValueError):
                broker.place_protective_stop(
                    instrument, "LONG", 1, float("nan"), "AISTOP1"
                )


class TradingAccountingTests(unittest.TestCase):
    def test_current_default_nse_intraday_costs(self) -> None:
        costs = estimate_nse_equity_intraday_cost(Decimal("50000"), Decimal("52500"))

        self.assertEqual(costs.turnover, Decimal("102500"))
        self.assertEqual(costs.brokerage, Decimal("30.75"))
        self.assertEqual(costs.stt, Decimal("13"))
        self.assertEqual(costs.exchange_transaction_charges, Decimal("3.15"))
        self.assertEqual(costs.sebi_charges, Decimal("0.10"))
        self.assertEqual(costs.stamp_duty, Decimal("1.50"))
        self.assertEqual(costs.gst, Decimal("6.12"))
        self.assertEqual(costs.total, Decimal("54.62"))

    def test_brokerage_cap_is_applied_per_executed_order(self) -> None:
        one_order = estimate_nse_equity_intraday_cost(100_000, 0)
        two_orders = estimate_nse_equity_intraday_cost([50_000, 50_000], [])

        self.assertEqual(one_order.brokerage, Decimal("20.00"))
        self.assertEqual(two_orders.brokerage, Decimal("30.00"))

    def test_nse_transaction_charge_and_ipft_components_do_not_overlap(self) -> None:
        costs = estimate_nse_equity_intraday_cost(5_000_000, 5_000_000)

        self.assertEqual(costs.exchange_transaction_charges, Decimal("306.99"))
        self.assertEqual(costs.ipft_charges, Decimal("0.01"))

    def test_rates_are_configurable(self) -> None:
        rates = NSEEquityIntradayRates(
            brokerage_rate=0,
            brokerage_cap_per_order=0,
            stt_sell_rate=0,
            exchange_transaction_rate=0,
            sebi_rate=0,
            stamp_buy_rate=0,
            ipft_rate=0,
            gst_rate=0,
        )
        costs = estimate_nse_equity_intraday_cost(100_000, 110_000, rates=rates)
        self.assertEqual(costs.total, Decimal("0.00"))

    def test_gross_pnl_for_long_and_short(self) -> None:
        self.assertEqual(gross_pnl("LONG", 100, 103, 10), Decimal("30"))
        self.assertEqual(gross_pnl("SHORT", 100, 97, 10), Decimal("30"))
        self.assertEqual(gross_pnl("LONG", 100, 97, 10), Decimal("-30"))
        with self.assertRaises(ValueError):
            gross_pnl("FLAT", 100, 101, 1)

    def test_directional_slippage_positive_means_adverse(self) -> None:
        self.assertEqual(
            directional_slippage_per_share(100, 100.25, "BUY"),
            Decimal("0.25"),
        )
        self.assertEqual(
            directional_slippage_per_share(100, 99.75, "SELL"),
            Decimal("0.25"),
        )
        self.assertEqual(
            directional_slippage_per_share(100, 99.75, "BUY"),
            Decimal("-0.25"),
        )
        self.assertEqual(
            directional_slippage_bps(100, 100.25, "BUY"),
            Decimal("25.0000"),
        )

    def test_invalid_slippage_direction_and_zero_expected_price_fail(self) -> None:
        with self.assertRaises(ValueError):
            directional_slippage_bps(0, 1, "BUY")
        with self.assertRaises(ValueError):
            directional_slippage_per_share(100, 101, "LONG")


if __name__ == "__main__":
    unittest.main()
