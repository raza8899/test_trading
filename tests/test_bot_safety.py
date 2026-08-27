from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import tempfile
import unittest
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo

import dotenv
import pandas as pd
from kiteconnect.exceptions import InputException


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
        signal_price=100.0,
        signal_bar_closed_at=FIXED_NOW.isoformat(),
        quote_observed_at=FIXED_NOW.isoformat(),
        setup_detected_at=FIXED_NOW.isoformat(),
        last_validated_at=FIXED_NOW.isoformat(),
        opening_rvol=2.0,
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
        self.review_keys: list[object] = []
        self.last_response_model = "fake-model"
        self.last_response_id = "fake-response"
        self.last_latency_ms = 1
        self.last_error = ""
        self.last_status = "OK"
        self.last_decision_id = "fake-decision"
        self.last_input_sha256 = "fake-input"
        self.last_input_tokens = 10
        self.last_cached_input_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_output_tokens = 5
        self.last_reasoning_tokens = 0
        self.last_total_tokens = 15
        self.last_cache_hit_ratio = 0.0
        self.last_api_called = True
        self.last_duplicate_review_suppressed = False
        self.last_source_input_sha256 = "fake-input"
        self.last_prompt_cache_key = "fake-cache-key"
        self.last_prompt_cache_options_sent = True
        self.last_prompt_cache_mode = "explicit"
        self.last_prompt_cache_ttl = "30m"

    def has_cached_review(self, review_key, candidate_payload):
        return False

    def review(self, setup, *, review_key=None):
        self.review_calls.append(setup)
        self.review_keys.append(review_key)
        return self.decision


class AIFilterContractTests(unittest.TestCase):
    def _reviewer(self, responses):
        client = mock.Mock()
        client.responses.parse.side_effect = list(responses)
        with (
            mock.patch.object(bot, "OPENAI_API_KEY", "test-key"),
            mock.patch.object(bot, "OpenAI", return_value=client),
        ):
            reviewer = bot.AIFilter()
        return reviewer, client

    @staticmethod
    def _response(decision=None, *, usage=None):
        return SimpleNamespace(
            output_parsed=decision or approved_decision(),
            model="gpt-5.6-sol-2026-08-01",
            id="resp-test",
            usage=usage,
        )

    def test_detailed_usage_is_recorded_without_changing_decision(self) -> None:
        usage = SimpleNamespace(
            input_tokens=1200,
            input_tokens_details=SimpleNamespace(
                cached_tokens=896,
                cache_write_tokens=128,
            ),
            output_tokens=180,
            output_tokens_details=SimpleNamespace(reasoning_tokens=96),
            total_tokens=1380,
        )
        reviewer, client = self._reviewer([self._response(usage=usage)])

        decision = reviewer.review({"setup": {"rvol": 2.0}})

        self.assertEqual(decision.decision, "APPROVE")
        self.assertEqual(reviewer.last_cached_input_tokens, 896)
        self.assertEqual(reviewer.last_cache_write_tokens, 128)
        self.assertEqual(reviewer.last_reasoning_tokens, 96)
        self.assertAlmostEqual(reviewer.last_cache_hit_ratio, 896 / 1200)
        self.assertEqual(reviewer.usage_totals.calls, 1)
        request = client.responses.parse.call_args.kwargs
        self.assertEqual(request["max_output_tokens"], 400)
        self.assertEqual(request["text"], {"verbosity": "low"})
        self.assertEqual(
            request["prompt_cache_options"],
            {"mode": "explicit", "ttl": "30m"},
        )

    def test_usage_property_failure_never_invalidates_valid_review(self) -> None:
        class HostileUsageResponse:
            output_parsed = approved_decision()
            model = "gpt-5.6-sol"
            id = "resp-hostile"

            @property
            def usage(self):
                raise RuntimeError("metrics failed")

        reviewer, _ = self._reviewer([HostileUsageResponse()])

        result = reviewer.review({"setup": {"rvol": 2.0}})

        self.assertEqual(result.decision, "APPROVE")
        self.assertEqual(reviewer.last_status, "OK")
        self.assertEqual(reviewer.last_total_tokens, 0)
        self.assertEqual(reviewer.usage_totals.successful_calls, 1)

    def test_malformed_output_is_internal_error_and_fail_closed(self) -> None:
        reviewer, _ = self._reviewer(
            [self._response(decision=SimpleNamespace(decision="ERROR"))]
        )

        with mock.patch.object(bot, "log"):
            result = reviewer.review({"setup": {"rvol": 2.0}})

        self.assertIsInstance(result, bot.AIFailureDecision)
        self.assertEqual(result.decision, "ERROR")
        self.assertEqual(result.confidence, 0)
        self.assertEqual(reviewer.last_status, "ERROR")
        self.assertEqual(reviewer.usage_totals.failed_calls, 1)

    def test_exact_duplicate_is_reused_but_new_signal_calls_provider(self) -> None:
        reviewer, client = self._reviewer(
            [self._response(), self._response()]
        )
        payload = {"setup": {"rvol": 2.0}}
        first_key = ("strategy", "INFY", "LONG", "2026-08-27T10:00:00+05:30")
        next_key = ("strategy", "INFY", "LONG", "2026-08-27T10:05:00+05:30")

        first = reviewer.review(payload, review_key=first_key)
        duplicate = reviewer.review(payload, review_key=first_key)
        self.assertEqual(client.responses.parse.call_count, 1)
        self.assertIs(first, duplicate)
        self.assertEqual(reviewer.last_status, "DUPLICATE_REUSED")
        self.assertFalse(reviewer.last_api_called)

        reviewer.review(payload, review_key=next_key)
        self.assertEqual(client.responses.parse.call_count, 2)
        self.assertEqual(reviewer.usage_totals.calls, 2)
        self.assertEqual(reviewer.usage_totals.duplicate_reviews_suppressed, 1)

    def test_changed_payload_on_same_signal_is_not_reused(self) -> None:
        reviewer, client = self._reviewer(
            [self._response(), self._response()]
        )
        key = ("strategy", "INFY", "LONG", "2026-08-27T10:00:00+05:30")

        reviewer.review({"setup": {"spread_bps": 2.0}}, review_key=key)
        reviewer.review({"setup": {"spread_bps": 7.0}}, review_key=key)

        self.assertEqual(client.responses.parse.call_count, 2)
        self.assertEqual(reviewer.usage_totals.duplicate_reviews_suppressed, 0)


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
    @staticmethod
    def _current_candle_frame(
        *,
        end: str = "09:50",
    ) -> pd.DataFrame:
        session_date = FIXED_NOW.date().isoformat()
        dates = pd.date_range(
            f"{session_date} 09:15",
            f"{session_date} {end}",
            freq="5min",
            tz="Asia/Kolkata",
        )
        return pd.DataFrame(
            {
                "date": dates,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000.0,
            }
        )

    def test_rsi_handles_gain_loss_and_flat_edge_cases(self) -> None:
        gains = bot.rsi(pd.Series(range(1, 22), dtype=float))
        losses = bot.rsi(pd.Series(range(22, 1, -1), dtype=float))
        flat = bot.rsi(pd.Series([7.0] * 21))

        self.assertTrue((gains.iloc[1:] == 100.0).all())
        self.assertTrue((losses.iloc[1:] == 0.0).all())
        self.assertTrue((flat == 50.0).all())

    def test_strategy_candles_uses_one_causal_cutoff_across_boundary(self) -> None:
        raw = self._current_candle_frame(end="09:55")
        broker = bot.KiteBroker.__new__(bot.KiteBroker)
        broker.historical_candles = mock.Mock(return_value=raw)
        cutoff = FIXED_NOW.replace(second=1)

        with (
            mock.patch.object(bot, "CANDLE_CLOSE_GRACE_SECONDS", 2.0),
            mock.patch.object(bot, "now_ist", return_value=cutoff) as clock,
        ):
            candles = broker.strategy_candles(123)

        self.assertEqual(clock.call_count, 1)
        self.assertEqual(candles["date"].iloc[-1].strftime("%H:%M"), "09:50")
        self.assertEqual(candles.attrs["causal_as_of"], pd.Timestamp(cutoff).isoformat())

    def test_historical_candles_uses_kite_wire_dates_for_pandas_timestamps(self) -> None:
        broker = bot.KiteBroker.__new__(bot.KiteBroker)
        broker.kite = mock.Mock()
        broker.kite.historical_data.return_value = []
        start = pd.Timestamp("2026-08-06 11:11:04", tz="Asia/Kolkata")
        end = pd.Timestamp("2026-08-27 11:11:04.987654", tz="Asia/Kolkata")

        result = broker.historical_candles(256265, start, end)

        self.assertTrue(result.empty)
        broker.kite.historical_data.assert_called_once_with(
            instrument_token=256265,
            from_date="2026-08-06 11:11:04",
            to_date="2026-08-27 11:11:04",
            interval="5minute",
            continuous=False,
            oi=False,
        )

    def test_historical_request_dates_are_host_timezone_independent(self) -> None:
        start = datetime(
            2026,
            8,
            6,
            5,
            41,
            4,
            tzinfo=ZoneInfo("UTC"),
        )
        end = datetime(
            2026,
            8,
            27,
            5,
            41,
            4,
            tzinfo=ZoneInfo("UTC"),
        )

        self.assertEqual(
            bot.kite_historical_request_dates(start, end),
            (
                "2026-08-06 11:11:04",
                "2026-08-27 11:11:04",
            ),
        )

    def test_historical_candles_rejects_bad_range_before_broker_io(self) -> None:
        broker = bot.KiteBroker.__new__(bot.KiteBroker)
        broker.kite = mock.Mock()
        observed = pd.Timestamp("2026-08-27 11:11:04", tz="Asia/Kolkata")

        bad_ranges = (
            (observed, observed),
            (observed + pd.Timedelta(seconds=1), observed),
            (
                observed + pd.Timedelta(microseconds=100_000),
                observed + pd.Timedelta(microseconds=900_000),
            ),
        )
        for start, end in bad_ranges:
            with self.subTest(start=start, end=end):
                with self.assertRaisesRegex(ValueError, "earlier than to_date"):
                    broker.historical_candles(256265, start, end)
        with self.assertRaisesRegex(ValueError, "instrument_token must be positive"):
            broker.historical_candles(0, observed, observed + pd.Timedelta(days=1))

        broker.kite.historical_data.assert_not_called()

    def test_historical_candle_failure_is_contextual_and_fail_closed(self) -> None:
        broker = bot.KiteBroker.__new__(bot.KiteBroker)
        broker.kite = mock.Mock()
        broker.kite.historical_data.side_effect = InputException(
            "invalid from date"
        )
        start = pd.Timestamp("2026-08-06 11:11:04", tz="Asia/Kolkata")
        end = pd.Timestamp("2026-08-27 11:11:04", tz="Asia/Kolkata")

        with self.assertRaisesRegex(
            bot.HistoricalCandleError,
            (
                "token=256265 from=2026-08-06 11:11:04 "
                "to=2026-08-27 11:11:04: InputException: invalid from date"
            ),
        ) as raised:
            broker.historical_candles(256265, start, end)

        self.assertIsInstance(raised.exception.__cause__, InputException)

    def test_prior_session_only_candles_fail_closed_on_no_session_day(self) -> None:
        prior = self._current_candle_frame()
        prior["date"] = prior["date"] - pd.Timedelta(days=1)

        result = bot.validated_strategy_candles(prior, as_of=FIXED_NOW)

        self.assertTrue(result.empty)

    def test_full_scan_failure_delay_is_exponential_and_capped(self) -> None:
        with (
            mock.patch.object(bot, "FULL_SCAN_EVERY_SECONDS", 30),
            mock.patch.object(bot, "FULL_SCAN_ERROR_BACKOFF_MAX_SECONDS", 300),
        ):
            delays = [
                bot.full_scan_retry_delay_seconds(failures)
                for failures in range(1, 8)
            ]

        self.assertEqual(delays, [30, 60, 120, 240, 300, 300, 300])
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    bot.full_scan_retry_delay_seconds(invalid)

    def test_strategy_candle_validation_rejects_gaps_duplicates_and_partial_bars(self) -> None:
        frame = self._current_candle_frame()
        with mock.patch.object(bot, "CANDLE_CLOSE_GRACE_SECONDS", 2.0):
            accepted = bot.validated_strategy_candles(frame, as_of=FIXED_NOW)
            self.assertEqual(len(accepted), len(frame))

            missing = frame.drop(index=4).reset_index(drop=True)
            duplicate = pd.concat([frame, frame.iloc[[4]]], ignore_index=True)
            misaligned = frame.copy()
            misaligned.loc[4, "date"] += pd.Timedelta(seconds=1)
            invalid_ohlc = frame.copy()
            invalid_ohlc.loc[4, "high"] = 100.1
            partial = pd.concat(
                [frame, self._current_candle_frame(end="09:55").tail(1)],
                ignore_index=True,
            )

            for name, candidate in {
                "missing": missing,
                "duplicate": duplicate,
                "misaligned": misaligned,
                "invalid_ohlc": invalid_ohlc,
                "partial": partial,
            }.items():
                with self.subTest(name=name):
                    self.assertTrue(
                        bot.validated_strategy_candles(
                            candidate,
                            as_of=FIXED_NOW,
                        ).empty
                    )

    def test_opening_rvol_uses_only_prior_complete_opening_windows(self) -> None:
        rows = []
        sessions = pd.date_range("2026-08-03", "2026-08-11", freq="B")
        for session_index, session in enumerate(sessions):
            volume = 2_000.0 if session_index == len(sessions) - 1 else 1_000.0
            for minute in (15, 20, 25):
                rows.append(
                    {
                        "date": pd.Timestamp(
                            year=session.year,
                            month=session.month,
                            day=session.day,
                            hour=9,
                            minute=minute,
                            tz="Asia/Kolkata",
                        ),
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": volume,
                    }
                )

        enriched = bot.add_indicators(pd.DataFrame(rows))
        current = enriched[enriched["date"].dt.date == FIXED_NOW.date()]

        self.assertTrue((current["opening_rvol"] == 2.0).all())

    def test_detect_setup_requires_first_directional_close_and_records_provenance(self) -> None:
        detected_at = FIXED_NOW + pd.Timedelta(seconds=3)
        prior_base = self._current_candle_frame()
        prior = pd.concat(
            [prior_base, prior_base.iloc[:3]],
            ignore_index=True,
        )
        prior["date"] = pd.date_range(
            "2026-08-10 09:15",
            periods=len(prior),
            freq="5min",
            tz="Asia/Kolkata",
        )
        current = self._current_candle_frame(end="09:55")
        frame = pd.concat([prior, current], ignore_index=True)
        frame["ema9"] = 99.5
        frame["ema20"] = 99.0
        frame["rsi"] = 60.0
        frame["atr"] = 2.0
        frame["vwap"] = 99.0
        frame["rvol"] = 2.0
        frame["opening_rvol"] = 2.0
        frame["body_ratio"] = 0.8
        frame["close_location"] = 0.9
        current_start = len(prior)
        frame.loc[current_start:current_start + 2, "high"] = 99.0
        frame.loc[current_start + 3:len(frame) - 2, "close"] = 99.0
        frame.loc[len(frame) - 1, ["open", "high", "low", "close"]] = [
            99.5,
            100.2,
            99.4,
            100.0,
        ]
        quote = bot.Quote(
            symbol="INFY",
            token=123,
            ltp=100.0,
            open=98.0,
            high=100.2,
            low=98.0,
            prev_close=98.0,
            pct_change=2.0,
            trade_volume=1_000_000,
            turnover_crore=100.0,
            spread_bps=2.0,
            day_range_pct=2.0,
            gap_pct=0.0,
            circuit_buffer_pct=5.0,
            lower_circuit_limit=90.0,
            upper_circuit_limit=110.0,
            stock_in_play_score=90.0,
            observed_at=pd.Timestamp(detected_at).isoformat(),
        )

        def detect(candidate: pd.DataFrame):
            with (
                mock.patch.object(bot, "now_ist", return_value=detected_at),
                mock.patch.object(
                    bot,
                    "validated_strategy_candles",
                    side_effect=lambda data, **_: data,
                ),
                mock.patch.object(bot, "add_indicators", side_effect=lambda data: data),
            ):
                return bot.detect_setup(quote, candidate, "BULL", 0.3)

        setup = detect(frame.copy())
        self.assertIsNotNone(setup)
        self.assertEqual(setup.signal_price, 100.0)
        self.assertEqual(setup.signal_bar_closed_at, FIXED_NOW.isoformat())
        self.assertEqual(setup.quote_observed_at, pd.Timestamp(detected_at).isoformat())
        self.assertEqual(setup.setup_detected_at, pd.Timestamp(detected_at).isoformat())
        self.assertEqual(setup.opening_rvol, 2.0)

        already_broken = frame.copy()
        already_broken.loc[len(already_broken) - 2, "close"] = 99.1
        self.assertIsNone(detect(already_broken))

        failed_breakout_then_reentry = frame.copy()
        failed_breakout_then_reentry.loc[
            len(failed_breakout_then_reentry) - 3,
            "close",
        ] = 99.1
        self.assertIsNone(detect(failed_breakout_then_reentry))

        counter_direction = frame.copy()
        counter_direction.loc[len(counter_direction) - 1, "open"] = 100.1
        self.assertIsNone(detect(counter_direction))

    def test_revalidation_uses_executable_ask_and_immutable_signal_price(self) -> None:
        snapshots = [
            bot.ExecutionSnapshot(
                ltp=100.45,
                best_bid=100.40,
                best_ask=100.50,
                spread_bps=2.0,
                lower_circuit=90.0,
                upper_circuit=110.0,
                observed_at=FIXED_NOW.isoformat(),
            ),
            bot.ExecutionSnapshot(
                ltp=100.65,
                best_bid=100.60,
                best_ask=100.70,
                spread_bps=2.0,
                lower_circuit=90.0,
                upper_circuit=110.0,
                observed_at=FIXED_NOW.isoformat(),
            ),
        ]
        broker = mock.Mock()
        broker.execution_snapshot.side_effect = snapshots

        with mock.patch.object(bot, "now_ist", return_value=FIXED_NOW):
            first, first_reason = bot.revalidate_live_setup(broker, make_setup())
            second, second_reason = bot.revalidate_live_setup(broker, first)

        self.assertEqual(first_reason, "OK")
        self.assertEqual(first.price, 100.50)
        self.assertEqual(first.signal_price, 100.0)
        self.assertIsNone(second)
        self.assertEqual(second_reason, "PRICE_DRIFT_0.35ATR")

    def test_revalidation_checks_signal_and_exchange_quote_freshness_after_io(self) -> None:
        mutable_clock = [FIXED_NOW]

        class DelayedBroker:
            def execution_snapshot(self, symbol):
                mutable_clock[0] = FIXED_NOW + pd.Timedelta(
                    seconds=bot.MAX_SIGNAL_AGE_SECONDS + 1
                )
                return bot.ExecutionSnapshot(
                    ltp=100.0,
                    best_bid=99.95,
                    best_ask=100.05,
                    spread_bps=2.0,
                    lower_circuit=90.0,
                    upper_circuit=110.0,
                    observed_at=pd.Timestamp(mutable_clock[0]).isoformat(),
                )

        with mock.patch.object(bot, "now_ist", side_effect=lambda: mutable_clock[0]):
            delayed, delayed_reason = bot.revalidate_live_setup(
                DelayedBroker(),
                make_setup(),
            )

        self.assertIsNone(delayed)
        self.assertTrue(delayed_reason.startswith("STALE_SIGNAL_"))

        stale_snapshot = bot.ExecutionSnapshot(
            ltp=100.0,
            best_bid=99.95,
            best_ask=100.05,
            spread_bps=2.0,
            lower_circuit=90.0,
            upper_circuit=110.0,
            observed_at=(
                FIXED_NOW
                - pd.Timedelta(seconds=bot.MAX_EXECUTION_QUOTE_AGE_SECONDS + 1)
            ).isoformat(),
        )
        broker = mock.Mock()
        broker.execution_snapshot.return_value = stale_snapshot
        with mock.patch.object(bot, "now_ist", return_value=FIXED_NOW):
            stale, stale_reason = bot.revalidate_live_setup(broker, make_setup())

        self.assertIsNone(stale)
        self.assertTrue(stale_reason.startswith("STALE_EXECUTION_QUOTE_"))

    def test_revalidation_reapplies_vwap_extension_limit(self) -> None:
        setup = replace(make_setup(), vwap=96.6, vwap_distance_atr=1.7)
        broker = mock.Mock()
        broker.execution_snapshot.return_value = bot.ExecutionSnapshot(
            ltp=100.35,
            best_bid=100.30,
            best_ask=100.40,
            spread_bps=2.0,
            lower_circuit=90.0,
            upper_circuit=110.0,
            observed_at=FIXED_NOW.isoformat(),
        )

        with (
            mock.patch.object(bot, "MAX_VWAP_DISTANCE_ATR", 1.8),
            mock.patch.object(bot, "now_ist", return_value=FIXED_NOW),
        ):
            refreshed, reason = bot.revalidate_live_setup(broker, setup)

        self.assertIsNone(refreshed)
        self.assertEqual(reason, "LIVE_VWAP_DISTANCE_1.90ATR")

    def test_revalidation_reapplies_directional_day_change(self) -> None:
        cases = [
            (
                replace(
                    make_setup("LONG"),
                    signal_price=100.4,
                    price=100.4,
                    prev_close=100.0,
                    opening_range_high=99.8,
                    opening_range_low=99.0,
                    vwap=99.5,
                ),
                bot.ExecutionSnapshot(
                    ltp=100.15,
                    best_bid=100.10,
                    best_ask=100.20,
                    spread_bps=2.0,
                    lower_circuit=90.0,
                    upper_circuit=110.0,
                    observed_at=FIXED_NOW.isoformat(),
                ),
                "LONG_DAY_CHANGE_0.20PCT",
            ),
            (
                replace(
                    make_setup("SHORT"),
                    signal_price=99.6,
                    price=99.6,
                    prev_close=100.0,
                    opening_range_high=101.0,
                    opening_range_low=100.0,
                    vwap=100.2,
                ),
                bot.ExecutionSnapshot(
                    ltp=99.95,
                    best_bid=99.90,
                    best_ask=100.00,
                    spread_bps=2.0,
                    lower_circuit=90.0,
                    upper_circuit=110.0,
                    observed_at=FIXED_NOW.isoformat(),
                ),
                "SHORT_DAY_CHANGE_-0.10PCT",
            ),
        ]

        for setup, snapshot, expected in cases:
            with self.subTest(side=setup.side):
                broker = mock.Mock()
                broker.execution_snapshot.return_value = snapshot
                with mock.patch.object(bot, "now_ist", return_value=FIXED_NOW):
                    refreshed, reason = bot.revalidate_live_setup(broker, setup)
                self.assertIsNone(refreshed)
                self.assertEqual(reason, expected)

    def test_submission_rebuild_reprices_without_increasing_reviewed_quantity(self) -> None:
        broker = FakeBroker(price=100.5)
        original = make_trade()
        original.idea_id = "reviewed-idea"
        setup = replace(make_setup(), price=100.5)
        capacity = bot.entry_capacity(bot.fresh_state())

        result = bot.rebuild_trade_for_submission(
            broker,
            original,
            setup,
            capacity,
        )

        self.assertEqual(result.reason, "OK")
        self.assertIsNotNone(result.trade)
        self.assertEqual(result.trade.entry_price, 100.5)
        self.assertLessEqual(result.trade.qty, original.requested_qty)
        self.assertEqual(result.trade.idea_id, "reviewed-idea")
        self.assertLessEqual(
            result.trade.reserved_risk_amount,
            capacity.candidate_risk_budget,
        )

    def test_execution_snapshot_preserves_exchange_timestamp(self) -> None:
        exchange_time = datetime(2026, 8, 11, 9, 59, 58)
        broker = bot.KiteBroker.__new__(bot.KiteBroker)
        broker._rate_limited_quote = mock.Mock(
            return_value={
                "NSE:INFY": {
                    "timestamp": exchange_time,
                    "last_price": 100.0,
                    "depth": {
                        "buy": [{"price": 99.95}],
                        "sell": [{"price": 100.05}],
                    },
                    "lower_circuit_limit": 90.0,
                    "upper_circuit_limit": 110.0,
                }
            }
        )

        snapshot = broker.execution_snapshot("INFY")

        expected = pd.Timestamp(exchange_time, tz="Asia/Kolkata").isoformat()
        self.assertEqual(snapshot.observed_at, expected)

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
        self.assertFalse(bot.stop_exactly_protects(stop, trade, 10, instrument))
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
    def test_weekend_exits_before_state_or_broker_initialization(self) -> None:
        saturday = datetime(
            2026,
            8,
            15,
            10,
            0,
            tzinfo=ZoneInfo("Asia/Kolkata"),
        )
        with (
            mock.patch.object(bot, "now_ist", return_value=saturday),
            mock.patch.object(bot, "load_state") as load_state,
            mock.patch.object(bot, "KiteBroker") as broker_type,
            mock.patch.object(bot, "log"),
        ):
            bot._run_main()

        load_state.assert_not_called()
        broker_type.assert_not_called()

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

    def test_scan_backoff_and_historical_rate_must_be_safe(self) -> None:
        with (
            mock.patch.object(bot, "FULL_SCAN_EVERY_SECONDS", 30),
            mock.patch.object(bot, "FULL_SCAN_ERROR_BACKOFF_MAX_SECONDS", 20),
        ):
            with self.assertRaisesRegex(RuntimeError, "BACKOFF_MAX_SECONDS"):
                bot.validate_configuration()

        with mock.patch.object(bot, "CANDLE_DELAY_SECONDS", 0.1):
            with self.assertRaisesRegex(RuntimeError, "3 requests/second"):
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

    def test_previous_date_recovery_intent_is_not_silently_discarded(self) -> None:
        state = bot.fresh_state()
        state["date"] = "2026-08-10"
        state["execution_mode"] = "live"
        state["dedicated_recovery_intents"]["INFY"] = {
            "symbol": "INFY",
            "reason": "TEST",
            "status": "SUBMITTING",
            "signed_qty": 10,
            "transaction_type": "SELL",
            "tag": "AIRCV000000000001",
            "order_ids": [],
            "detail": "",
            "created_at": FIXED_NOW.isoformat(),
            "updated_at": FIXED_NOW.isoformat(),
        }
        bot.save_state(state)

        with self.assertRaisesRegex(RuntimeError, "Previous-date active state"):
            bot.load_state()

    def test_previous_date_malformed_trade_state_is_not_discarded(self) -> None:
        state = bot.fresh_state()
        state["date"] = "2026-08-10"
        state["trades"]["INFY"] = "invalid"
        bot.save_state(state)

        with self.assertRaisesRegex(RuntimeError, "State trades must map"):
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

    def test_negative_consecutive_losses_refuses_to_start(self) -> None:
        state = bot.fresh_state()
        state["consecutive_losses"] = -1
        bot.save_state(state)

        with self.assertRaisesRegex(RuntimeError, "consecutive_losses"):
            bot.load_state()

    def test_state_totals_must_match_persisted_closed_trades(self) -> None:
        state = bot.fresh_state()
        trade = make_trade(status="CLOSED_TARGET")
        trade.execution_mode = "paper"
        trade.gross_pnl = 30.0
        trade.fees = 5.0
        trade.net_pnl = 25.0
        state["trades"][trade.symbol] = asdict(trade)
        state["realized_pnl"] = 0.0
        state["fees_paid"] = 0.0
        bot.save_state(state)

        with self.assertRaisesRegex(RuntimeError, "realized_pnl"):
            bot.load_state()

    def test_previous_date_preserves_pending_journal_outbox(self) -> None:
        previous_day = FIXED_NOW.replace(day=10)
        with mock.patch.object(bot, "now_ist", return_value=previous_day):
            state = bot.fresh_state()
            state["journal_outbox"].append(
                bot.prepare_journal_event("CLOSE", symbol="INFY")
            )
        bot.save_state(state)

        loaded = bot.load_state()

        self.assertEqual(loaded["date"], str(FIXED_NOW.date()))
        self.assertEqual(len(loaded["journal_outbox"]), 1)
        self.assertTrue(bot.flush_journal_outbox(loaded))
        self.assertEqual(loaded["journal_outbox"], [])
        self.assertTrue((self.log_directory / "trades_20260810.jsonl").exists())


class JournalIntegrityTests(IsolatedBotTestCase):
    def test_journal_uses_one_timestamp_and_stable_session_sequence(self) -> None:
        first_time = FIXED_NOW.replace(hour=23, minute=59, second=59)
        second_time = first_time.replace(day=12, hour=0, minute=0, second=0)
        with mock.patch.object(
            bot,
            "now_ist",
            side_effect=[first_time, second_time],
        ) as clock:
            first_id = bot.journal("FIRST")
            second_id = bot.journal("SECOND")

        self.assertEqual(clock.call_count, 2)
        first_path = self.log_directory / "trades_20260811.jsonl"
        second_path = self.log_directory / "trades_20260812.jsonl"
        first = json.loads(first_path.read_text(encoding="utf-8"))
        second = json.loads(second_path.read_text(encoding="utf-8"))
        self.assertEqual(first["event_id"], first_id)
        self.assertEqual(second["event_id"], second_id)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(first["session_id"], second["session_id"])
        self.assertEqual(second["event_sequence"], first["event_sequence"] + 1)
        self.assertEqual(first_path.stat().st_mode & 0o777, 0o600)

    def test_repeated_halt_reasons_are_suppressed_but_details_are_kept(self) -> None:
        state = bot.fresh_state()

        bot.halt_trading(state, "root failure")
        bot.halt_trading(state, "root failure")
        bot.halt_trading(state, "follow-on failure")
        bot.halt_trading(state, "follow-on failure")

        self.assertEqual(state["halt_reason"], "root failure")
        self.assertEqual(
            [detail["reason"] for detail in state["halt_details"]],
            ["follow-on failure"],
        )
        path = self.log_directory / "trades_20260811.jsonl"
        events = [json.loads(line)["event"] for line in path.read_text().splitlines()]
        self.assertEqual(events, ["HALT", "HALT_DETAIL"])


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

        # Isolate the post-fill defence from the pre-submit repricer, which
        # would normally reduce this deliberately oversized synthetic plan.
        with (
            mock.patch.object(
                bot,
                "rebuild_trade_for_submission",
                return_value=bot.TradeBuildResult(trade, "OK"),
            ),
            mock.patch.object(bot, "fail_safe_trade_lifecycle") as fail_safe,
        ):
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

    def test_close_is_idempotent_and_failed_journal_replays_from_outbox(self) -> None:
        state = bot.fresh_state()
        trade = make_trade(status="OPEN_PROTECTED")
        trade.execution_mode = "paper"
        trade.ai_decision = "APPROVE"
        state["trades"][trade.symbol] = asdict(trade)

        with mock.patch.object(
            bot,
            "append_prepared_journal_event",
            side_effect=OSError("disk full"),
        ):
            self.assertTrue(
                bot.mark_trade_closed(
                    state,
                    trade.symbol,
                    "TARGET",
                    exit_price=103.0,
                )
            )

        first_realized = state["realized_pnl"]
        closed = bot.trade_from_dict(state["trades"][trade.symbol])
        self.assertEqual(len(state["journal_outbox"]), 1)
        self.assertEqual(closed.fees_source, "model_estimate")
        self.assertEqual(
            closed.fee_model_version,
            bot.NSE_EQUITY_INTRADAY_FEE_MODEL_VERSION,
        )
        self.assertTrue(closed.fee_breakdown)
        self.assertFalse(
            bot.mark_trade_closed(
                state,
                trade.symbol,
                "TARGET",
                exit_price=103.0,
            )
        )
        self.assertEqual(state["realized_pnl"], first_realized)

        self.assertTrue(bot.flush_journal_outbox(state))
        self.assertEqual(state["journal_outbox"], [])
        path = self.log_directory / "trades_20260811.jsonl"
        close_events = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["event"] == "CLOSE"
        ]
        self.assertEqual(len(close_events), 1)
        self.assertEqual(close_events[0]["fees_source"], "model_estimate")

    def test_close_refuses_corrupt_prior_accounting_instead_of_zeroing_it(self) -> None:
        state = bot.fresh_state()
        trade = make_trade(status="OPEN_PROTECTED")
        trade.execution_mode = "paper"
        state["trades"][trade.symbol] = asdict(trade)
        state["realized_pnl"] = "corrupt"

        with self.assertRaisesRegex(RuntimeError, "accounting refused"):
            bot.mark_trade_closed(
                state,
                trade.symbol,
                "TARGET",
                exit_price=103.0,
            )

        self.assertTrue(state["kill_switch"])
        self.assertEqual(
            state["trades"][trade.symbol]["status"],
            "OPEN_PROTECTED",
        )
        self.assertEqual(state["journal_outbox"], [])

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

    def test_direct_execution_rejects_stale_setup_before_intent(self) -> None:
        broker = FakeBroker()
        state = bot.fresh_state()

        with (
            mock.patch.object(
                bot,
                "revalidate_live_setup",
                return_value=(None, "STALE_SIGNAL_301s"),
            ),
            mock.patch.object(bot, "journal_best_effort", return_value=True) as journal,
        ):
            bot.execute_trade(
                broker,
                make_trade(),
                make_setup(),
                approved_decision(),
                state,
            )

        self.assertNotIn("INFY", state["trades"])
        self.assertEqual(broker.entry_calls, [])
        self.assertEqual(broker.stop_calls, [])
        self.assertTrue(
            any(
                call.kwargs.get("reason") == "PRE_SUBMIT_STALE_SIGNAL_301s"
                for call in journal.call_args_list
            )
        )

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
    def test_position_monitor_failure_halts_and_attempts_flatten(self) -> None:
        broker = FakeBroker(signed_qty=10)
        state = bot.fresh_state()
        trade = make_trade(status="OPEN_PROTECTED")
        trade.execution_mode = "live"
        state["trades"][trade.symbol] = asdict(trade)

        with mock.patch.object(bot, "flatten_all", return_value=True) as flatten:
            result = bot.handle_position_monitor_failure(
                broker,
                state,
                TimeoutError("positions unavailable"),
            )

        self.assertTrue(result)
        self.assertTrue(state["kill_switch"])
        self.assertIn("position monitor failure", state["halt_reason"])
        flatten.assert_called_once_with(
            broker,
            state,
            "POSITION_MONITOR_FAILURE",
        )

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

    def test_monitor_attributes_confirmed_stop_fill_and_persists_identity(self) -> None:
        broker = FakeBroker(price=98.0, signed_qty=0)
        state = bot.fresh_state()
        trade = make_trade(status="OPEN_PROTECTED")
        trade.execution_mode = "live"
        trade.entry_order_id = "ENTRY-1"
        trade.entry_tag = "AIENT000000000001"
        trade.entry_status = "COMPLETE"
        trade.stop_order_id = "STOP-1"
        trade.stop_tag = "AISTP000000000001"
        trade.stop_status = "TRIGGER PENDING"
        state["trades"][trade.symbol] = asdict(trade)
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
                "status": "COMPLETE",
                "quantity": 10,
                "filled_quantity": 10,
                "pending_quantity": 0,
                "average_price": 98.0,
                "transaction_type": "SELL",
                "order_type": "SL",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "tag": "",
                "trigger_price": 98.0,
            }
        )
        broker.trade_payloads = [
            {"order_id": "STOP-1", "quantity": 10, "average_price": 98.0}
        ]

        with mock.patch.object(bot, "LIVE_TRADING", True):
            bot.monitor_open_trades(broker, state)

        closed = bot.trade_from_dict(state["trades"][trade.symbol])
        self.assertEqual(closed.status, "CLOSED_STOP")
        self.assertEqual(closed.exit_reason, "STOP")
        self.assertEqual(closed.stop_status, "COMPLETE")
        self.assertEqual(closed.exit_order_id, "STOP-1")
        self.assertEqual(closed.exit_order_ids, ["STOP-1"])
        self.assertEqual(broker.exit_calls, [])


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

    def test_balramchin_sl_representation_completes_live_protection(self) -> None:
        broker = FakeBroker(price=100.25)
        broker.order_snapshots["STOP-1"] = replace(
            self._armed_stop(10),
            order_type="SL",
        )
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
        self.assertEqual(trade.status, "OPEN_PROTECTED")
        self.assertEqual(trade.stop_order_id, "STOP-1")
        self.assertEqual(trade.stop_status, "TRIGGER PENDING")
        self.assertFalse(state["kill_switch"])
        self.assertEqual(len(broker.exit_calls), 0)

    def test_post_fill_telemetry_runs_only_after_stop_submission(self) -> None:
        broker = FakeBroker(price=100.25)
        broker.order_snapshots["STOP-1"] = self._armed_stop(10)
        state = bot.fresh_state()
        sequence: list[str] = []
        real_place_stop = broker.place_protective_stop

        def place_stop(*args, **kwargs):
            sequence.append("STOP_SUBMITTED")
            return real_place_stop(*args, **kwargs)

        def record(event, **fields):
            if event == "ENTRY_FILLED":
                sequence.append("ENTRY_FILLED_TELEMETRY")
            elif event == "PROTECTION_CONFIRMED" and fields.get("role") == "STOP":
                sequence.append("STOP_TELEMETRY")
            return True

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(
                broker,
                "place_protective_stop",
                side_effect=place_stop,
            ),
            mock.patch.object(bot, "journal_best_effort", side_effect=record),
        ):
            bot.execute_trade(
                broker,
                make_trade(),
                make_setup(),
                approved_decision(),
                state,
            )

        self.assertEqual(
            sequence,
            ["STOP_SUBMITTED", "ENTRY_FILLED_TELEMETRY", "STOP_TELEMETRY"],
        )

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
    def test_scan_uses_one_candle_cutoff_across_five_minute_boundary(self) -> None:
        broker = FakeBroker()
        broker.nifty_token = 999
        cutoffs = []
        current = StrategyHelperSafetyTests._current_candle_frame()
        prior = pd.concat([current, current.iloc[:4]], ignore_index=True)
        prior["date"] = pd.date_range(
            "2026-08-10 09:15",
            periods=len(prior),
            freq="5min",
            tz="Asia/Kolkata",
        )
        nifty_frame = pd.concat([prior, current], ignore_index=True)

        def candles(token, *, as_of=None):
            cutoffs.append((token, pd.Timestamp(as_of)))
            result = nifty_frame.copy() if token == 999 else pd.DataFrame()
            result.attrs["causal_as_of"] = pd.Timestamp(as_of).isoformat()
            return result

        broker.strategy_candles = candles
        candidate = mock.Mock(symbol="INFY", token=123)
        crossed_boundary = FIXED_NOW + pd.Timedelta(minutes=5)
        state = bot.fresh_state()

        with (
            mock.patch.object(bot, "entry_window_open", return_value=True),
            mock.patch.object(
                bot,
                "select_stocks_in_play",
                return_value=[candidate],
            ),
            mock.patch.object(
                bot,
                "now_ist",
                side_effect=[FIXED_NOW, crossed_boundary, crossed_boundary],
            ),
            mock.patch.object(bot.time, "sleep"),
        ):
            bot.scan_for_new_trades(broker, None, state)

        self.assertEqual([token for token, _ in cutoffs], [999, 123])
        self.assertEqual(cutoffs[0][1], pd.Timestamp(FIXED_NOW))
        self.assertEqual(cutoffs[1][1], pd.Timestamp(FIXED_NOW))

    def test_unavailable_nifty_data_aborts_candidate_evaluation(self) -> None:
        broker = FakeBroker()
        broker.nifty_token = 999
        broker.strategy_candles = mock.Mock(return_value=pd.DataFrame())
        candidate = mock.Mock(symbol="INFY", token=123)

        with (
            mock.patch.object(bot, "entry_window_open", return_value=True),
            mock.patch.object(
                bot,
                "select_stocks_in_play",
                return_value=[candidate],
            ),
            mock.patch.object(bot, "detect_setup") as detect,
            mock.patch.object(bot.time, "sleep"),
        ):
            bot.scan_for_new_trades(broker, None, bot.fresh_state())

        detect.assert_not_called()
        broker.strategy_candles.assert_called_once_with(
            999,
            as_of=mock.ANY,
        )

    def test_candidate_historical_failure_aborts_remaining_scan_requests(self) -> None:
        broker = FakeBroker()
        candidates = [
            mock.Mock(symbol="INFY", token=123),
            mock.Mock(symbol="TCS", token=456),
        ]
        broker.strategy_candles = mock.Mock(
            side_effect=bot.HistoricalCandleError("historical transport failed")
        )
        state = bot.fresh_state()

        with (
            mock.patch.object(bot, "entry_window_open", return_value=True),
            mock.patch.object(
                bot,
                "select_stocks_in_play",
                return_value=candidates,
            ),
            mock.patch.object(
                bot,
                "get_nifty_regime",
                return_value=("BULL", 0.5),
            ),
            mock.patch.object(bot.time, "sleep"),
        ):
            with self.assertRaisesRegex(
                bot.HistoricalCandleError,
                "historical transport failed",
            ):
                bot.scan_for_new_trades(broker, None, state)

        broker.strategy_candles.assert_called_once_with(
            123,
            as_of=mock.ANY,
        )

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
            risk_flags=["VOLATILITY_RISK"],
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

    def test_gate_mode_never_turns_ai_error_into_approval(self) -> None:
        failure = bot.AIFailureDecision(
            reason="Malformed provider output; fail-closed.",
            risk_flags=["AI_FAILURE"],
        )

        reviewer, execute = self._run_scan("gate", failure)

        self.assertEqual(len(reviewer.review_calls), 1)
        execute.assert_not_called()


class ProductionOrderRecoveryRegressionTests(IsolatedBotTestCase):
    def test_balramchin_protected_sl_is_exactly_armed_with_persisted_id(self) -> None:
        trade = make_trade(status="OPEN_PROTECTED")
        trade.symbol = "BALRAMCHIN"
        trade.qty = 20
        trade.requested_qty = 20
        trade.stop_price = 737.15
        trade.stop_order_id = "260824190297527"
        trade.stop_tag = "AISTPLOCALTAG02"
        instrument = bot.Instrument("BALRAMCHIN", "Balrampur Chini", 456, 0.05)
        stop = bot.OrderSnapshot.from_payload(
            {
                "order_id": "260824190297527",
                "status": "TRIGGER PENDING",
                "quantity": 20,
                "filled_quantity": 0,
                "pending_quantity": 20,
                "transaction_type": "SELL",
                # Exact Aug-24 production representation: protected SL-M was
                # surfaced by the broker as SL while still trigger-pending.
                "order_type": "SL",
                "tradingsymbol": "BALRAMCHIN",
                "exchange": "NSE",
                "product": "MIS",
                "tag": "",
                "trigger_price": 737.15,
            }
        )

        self.assertTrue(stop.stop_armed)
        self.assertTrue(bot.stop_identity_matches(stop, trade, 20, instrument))
        self.assertTrue(bot.stop_exactly_protects(stop, trade, 20, instrument))
        self.assertEqual(bot.stop_identity_mismatches(stop, trade, 20, instrument), [])

        terminal = replace(
            stop,
            status="COMPLETE",
            filled=20,
            pending=0,
        )
        self.assertTrue(bot.stop_identity_matches(terminal, trade, 20, instrument))
        self.assertFalse(bot.stop_exactly_protects(terminal, trade, 20, instrument))

    def test_startup_preserves_position_protected_by_balramchin_sl_shape(self) -> None:
        broker = FakeBroker(signed_qty=10)
        state = bot.fresh_state()
        state["execution_mode"] = "live"
        trade = make_trade(status="OPEN_PROTECTED")
        trade.execution_mode = "live"
        trade.entry_order_id = "ENTRY-1"
        trade.entry_tag = "AIENT000000000001"
        trade.entry_status = "COMPLETE"
        trade.stop_order_id = "STOP-1"
        trade.stop_tag = "AISTP000000000001"
        trade.stop_status = "TRIGGER PENDING"
        state["trades"][trade.symbol] = asdict(trade)
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
        stop_payload = {
            "order_id": "STOP-1",
            "status": "TRIGGER PENDING",
            "quantity": 10,
            "filled_quantity": 0,
            "pending_quantity": 10,
            "transaction_type": "SELL",
            "order_type": "SL",
            "tradingsymbol": "INFY",
            "exchange": "NSE",
            "product": "MIS",
            "tag": "",
            "trigger_price": 98.0,
        }
        broker.order_snapshots["STOP-1"] = bot.OrderSnapshot.from_payload(stop_payload)
        broker.startup_order_payloads = [stop_payload]

        with mock.patch.object(bot, "LIVE_TRADING", True):
            bot.reconcile_startup(broker, state)

        recovered = bot.trade_from_dict(state["trades"][trade.symbol])
        self.assertEqual(recovered.status, "OPEN_PROTECTED")
        self.assertEqual(recovered.stop_status, "TRIGGER PENDING")
        self.assertFalse(state["kill_switch"])
        self.assertEqual(broker.exit_calls, [])

    def test_startup_attributes_exact_completed_stop_instead_of_generic_flat(self) -> None:
        broker = FakeBroker(signed_qty=0)
        state = bot.fresh_state()
        state["execution_mode"] = "live"
        trade = make_trade(status="OPEN_PROTECTED")
        trade.execution_mode = "live"
        trade.entry_order_id = "ENTRY-1"
        trade.entry_tag = "AIENT000000000001"
        trade.entry_status = "COMPLETE"
        trade.stop_order_id = "STOP-1"
        trade.stop_tag = "AISTP000000000001"
        trade.stop_status = "TRIGGER PENDING"
        state["trades"][trade.symbol] = asdict(trade)
        broker.order_snapshots["ENTRY-1"] = bot.OrderSnapshot.from_payload(
            {
                "order_id": "ENTRY-1",
                "status": "COMPLETE",
                "quantity": 10,
                "filled_quantity": 10,
                "pending_quantity": 0,
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
                "status": "COMPLETE",
                "quantity": 10,
                "filled_quantity": 10,
                "pending_quantity": 0,
                "average_price": 98.0,
                "transaction_type": "SELL",
                "order_type": "SL",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "tag": "",
                "trigger_price": 98.0,
            }
        )
        broker.trade_payloads = [
            {"order_id": "STOP-1", "quantity": 10, "average_price": 98.0}
        ]

        with mock.patch.object(bot, "LIVE_TRADING", True):
            bot.reconcile_startup(broker, state)

        closed = bot.trade_from_dict(state["trades"][trade.symbol])
        self.assertEqual(closed.status, "CLOSED_STOP")
        self.assertEqual(closed.exit_reason, "STOP")
        self.assertEqual(closed.stop_status, "COMPLETE")
        self.assertEqual(closed.exit_order_id, "STOP-1")
        self.assertEqual(closed.exit_order_ids, ["STOP-1"])
        self.assertEqual(broker.exit_calls, [])

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

    def test_market_data_start_is_explicit_and_idempotent(self) -> None:
        broker = object.__new__(bot.KiteBroker)
        broker._market_data_started = False
        broker._start_websockets = mock.Mock()

        broker.start_market_data()
        broker.start_market_data()

        broker._start_websockets.assert_called_once_with()
        self.assertTrue(broker._market_data_started)

    def test_latest_order_does_not_return_rejected_stale_orderbook_state(self) -> None:
        broker = object.__new__(bot.KiteBroker)
        broker._order_condition = threading.Condition()
        broker._order_updates = {}
        broker.kite = mock.Mock()
        converted = {
            "order_id": "STOP-1",
            "status": "OPEN",
            "quantity": 10,
            "filled_quantity": 0,
            "pending_quantity": 10,
            "transaction_type": "SELL",
            "order_type": "LIMIT",
            "tradingsymbol": "INFY",
            "exchange": "NSE",
            "product": "MIS",
            "tag": "AISTP000000000001",
            "trigger_price": 98.0,
        }
        stale = {
            **converted,
            "status": "TRIGGER PENDING",
            "order_type": "SL",
        }
        self.assertTrue(broker._record_order_update(converted))
        broker.kite.orders.return_value = [stale]

        snapshot = broker.latest_order("STOP-1")

        self.assertEqual(snapshot.status, "OPEN")
        self.assertEqual(snapshot.order_type, "LIMIT")
        broker.kite.order_history.assert_not_called()

        broker.kite.orders.return_value = []
        broker.kite.order_history.return_value = [stale]
        history_snapshot = broker.latest_order("STOP-1")
        self.assertEqual(history_snapshot.status, "OPEN")
        self.assertEqual(history_snapshot.order_type, "LIMIT")

    def test_latest_order_does_not_regress_terminal_cache(self) -> None:
        broker = object.__new__(bot.KiteBroker)
        broker._order_condition = threading.Condition()
        broker._order_updates = {}
        broker.kite = mock.Mock()
        complete = {
            "order_id": "EXIT-1",
            "status": "COMPLETE",
            "quantity": 10,
            "filled_quantity": 10,
            "pending_quantity": 0,
            "average_price": 99.0,
            "transaction_type": "SELL",
            "order_type": "LIMIT",
            "tradingsymbol": "INFY",
            "exchange": "NSE",
            "product": "MIS",
            "tag": "AIEXT000000000001",
        }
        stale_open = {
            **complete,
            "status": "OPEN",
            "filled_quantity": 0,
            "pending_quantity": 10,
            "average_price": 0.0,
        }
        self.assertTrue(broker._record_order_update(complete))
        broker.kite.orders.return_value = [stale_open]

        snapshot = broker.latest_order("EXIT-1")

        self.assertEqual(snapshot.status, "COMPLETE")
        self.assertEqual(snapshot.filled, 10)

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

    def test_dedicated_recovery_fails_closed_if_stop_fills_while_canceling(self) -> None:
        broker = self._dedicated_broker(stop_fills_on_cancel=True)
        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot.time, "sleep"),
        ):
            flat, ids = bot.dedicated_force_flatten_symbol(broker, "INFY", "TEST")
            second, _ = bot.dedicated_force_flatten_symbol(broker, "INFY", "TEST")

        self.assertFalse(flat)
        self.assertFalse(second)
        self.assertEqual(ids, [])
        self.assertEqual(broker.exit_calls, [])
        self.assertEqual(broker.signed_qty, 0)
        self.assertIn("INFY", broker._recovery_ambiguous_symbols)

    def test_dedicated_recovery_rejects_wrong_cancellation_identity(self) -> None:
        broker = self._dedicated_broker()
        broker.cancel_order_confirmed = mock.Mock(
            return_value=bot.OrderSnapshot.from_payload(
                {
                    "order_id": "OTHER-ORDER",
                    "status": "CANCELLED",
                    "quantity": 10,
                    "filled_quantity": 0,
                    "pending_quantity": 0,
                    "transaction_type": "SELL",
                    "order_type": "SL-M",
                    "tradingsymbol": "INFY",
                    "exchange": "NSE",
                    "product": "MIS",
                }
            )
        )

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot.time, "sleep"),
        ):
            flat, ids = bot.dedicated_force_flatten_symbol(
                broker,
                "INFY",
                "TEST",
            )

        self.assertFalse(flat)
        self.assertEqual(ids, [])
        self.assertEqual(broker.exit_calls, [])
        self.assertIn("INFY", broker._recovery_ambiguous_symbols)

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

    def test_recovery_intent_is_persisted_before_mutation_and_cleared_when_flat(self) -> None:
        broker = FakeBroker(signed_qty=10)
        state = bot.fresh_state()
        state["execution_mode"] = "live"
        observed_order_book_phases: list[str] = []
        observed_submission_phase: list[str] = []

        def orders():
            record = state["dedicated_recovery_intents"].get("INFY", {})
            observed_order_book_phases.append(record.get("status", "MISSING"))
            return []

        def exit_market(inst, signed_qty, tag):
            record = state["dedicated_recovery_intents"]["INFY"]
            observed_submission_phase.append(record["status"])
            broker.order_metadata["RECOVERY-1"] = {
                "symbol": inst.symbol,
                "exchange": "NSE",
                "product": "MIS",
                "transaction_type": "SELL",
                "order_type": "MARKET",
                "tag": tag,
            }
            return "RECOVERY-1"

        broker.orders = mock.Mock(side_effect=orders)
        broker.exit_market = mock.Mock(side_effect=exit_market)
        broker.order_snapshots["RECOVERY-1"] = bot.OrderSnapshot.from_payload(
            {
                "order_id": "RECOVERY-1",
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
            }
        )

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot.time, "sleep"),
        ):
            flat, ids = bot.dedicated_force_flatten_symbol(
                broker,
                "INFY",
                "TEST",
                state=state,
            )

        self.assertTrue(flat)
        self.assertEqual(ids, ["RECOVERY-1"])
        self.assertEqual(observed_order_book_phases[0], "PREPARED")
        self.assertEqual(observed_submission_phase, ["SUBMITTING"])
        self.assertIn("ORDER_SETTLED", observed_order_book_phases)
        self.assertEqual(state["dedicated_recovery_intents"], {})

    def test_recovery_does_not_mutate_when_initial_intent_persist_fails(self) -> None:
        broker = FakeBroker(signed_qty=10)
        broker.orders = mock.Mock(return_value=[])
        state = bot.fresh_state()
        state["execution_mode"] = "live"

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot, "save_state", side_effect=OSError("disk full")),
        ):
            flat, ids = bot.dedicated_force_flatten_symbol(
                broker,
                "INFY",
                "TEST",
                state=state,
            )

        self.assertFalse(flat)
        self.assertEqual(ids, [])
        broker.orders.assert_not_called()
        self.assertEqual(broker.exit_calls, [])
        self.assertEqual(broker.cancel_calls, [])
        self.assertEqual(state["dedicated_recovery_intents"], {})

    def test_known_recovery_order_is_durable_and_blocks_restart_resubmit(self) -> None:
        broker = FakeBroker(signed_qty=10)
        broker.orders = mock.Mock(return_value=[])
        broker.exit_market = mock.Mock(return_value="RECOVERY-1")
        broker.wait_for_order = mock.Mock(side_effect=TimeoutError("OMS timeout"))
        broker.latest_order = mock.Mock(side_effect=TimeoutError("still unknown"))
        state = bot.fresh_state()
        state["execution_mode"] = "live"

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot.time, "sleep"),
        ):
            first, first_ids = bot.dedicated_force_flatten_symbol(
                broker,
                "INFY",
                "TEST",
                state=state,
            )
            second, second_ids = bot.dedicated_force_flatten_symbol(
                broker,
                "INFY",
                "TEST",
                state=state,
            )

        record = state["dedicated_recovery_intents"]["INFY"]
        self.assertFalse(first)
        self.assertFalse(second)
        self.assertEqual(first_ids, ["RECOVERY-1"])
        self.assertEqual(second_ids, ["RECOVERY-1"])
        self.assertEqual(record["status"], "AMBIGUOUS")
        self.assertEqual(record["order_ids"], ["RECOVERY-1"])
        broker.exit_market.assert_called_once()

    def test_recovery_terminal_snapshot_identity_mismatch_blocks_retry(self) -> None:
        broker = FakeBroker(signed_qty=10)
        broker.orders = mock.Mock(return_value=[])
        broker.exit_market = mock.Mock(return_value="RECOVERY-1")
        broker.wait_for_order = mock.Mock(
            return_value=bot.OrderSnapshot.from_payload(
                {
                    "order_id": "RECOVERY-1",
                    "status": "CANCELLED",
                    "quantity": 10,
                    "filled_quantity": 0,
                    "pending_quantity": 0,
                    "transaction_type": "SELL",
                    "order_type": "MARKET",
                    "tradingsymbol": "OTHER",
                    "exchange": "NSE",
                    "product": "MIS",
                }
            )
        )

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot.time, "sleep"),
        ):
            flat, ids = bot.dedicated_force_flatten_symbol(
                broker,
                "INFY",
                "TEST",
            )

        self.assertFalse(flat)
        self.assertEqual(ids, ["RECOVERY-1"])
        broker.exit_market.assert_called_once()
        self.assertEqual(broker.wait_position_calls, [])
        self.assertIn("INFY", broker._recovery_ambiguous_symbols)

    def test_startup_refuses_unresolved_durable_recovery_before_broker_mutation(self) -> None:
        broker = FakeBroker(signed_qty=10)
        broker.positions = mock.Mock()
        broker.orders = mock.Mock()
        state = bot.fresh_state()
        state["dedicated_recovery_intents"]["INFY"] = {
            "symbol": "INFY",
            "status": "AMBIGUOUS",
            "order_ids": ["RECOVERY-1"],
        }

        with mock.patch.object(bot, "LIVE_TRADING", True):
            with self.assertRaisesRegex(RuntimeError, "reconcile.*manually"):
                bot.reconcile_startup(broker, state)

        broker.positions.assert_not_called()
        broker.orders.assert_not_called()

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

    def test_known_recovery_order_timeout_is_never_resubmitted(self) -> None:
        broker = FakeBroker(signed_qty=10)
        broker.orders = mock.Mock(return_value=[])
        broker.exit_market = mock.Mock(return_value="RECOVERY-1")
        broker.wait_for_order = mock.Mock(side_effect=TimeoutError("OMS timeout"))
        broker.latest_order = mock.Mock(side_effect=TimeoutError("still unknown"))

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot.time, "sleep"),
        ):
            first, ids = bot.dedicated_force_flatten_symbol(broker, "INFY", "TEST")
            second, _ = bot.dedicated_force_flatten_symbol(broker, "INFY", "TEST")

        self.assertFalse(first)
        self.assertFalse(second)
        self.assertEqual(ids, ["RECOVERY-1"])
        broker.exit_market.assert_called_once()
        self.assertIn("INFY", broker._recovery_ambiguous_symbols)

    def test_full_recovery_fill_with_unsettled_position_is_not_resubmitted(self) -> None:
        broker = FakeBroker(signed_qty=10)
        broker.flat_confirmation = False
        broker.orders = mock.Mock(return_value=[])
        broker.exit_market = mock.Mock(return_value="RECOVERY-1")
        broker.wait_for_order = mock.Mock(
            return_value=bot.OrderSnapshot.from_payload(
                {
                    "order_id": "RECOVERY-1",
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
                }
            )
        )

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot.time, "sleep"),
        ):
            flat, ids = bot.dedicated_force_flatten_symbol(broker, "INFY", "TEST")

        self.assertFalse(flat)
        self.assertEqual(ids, ["RECOVERY-1"])
        broker.exit_market.assert_called_once()
        self.assertEqual(broker.signed_qty, 10)
        self.assertIn("INFY", broker._recovery_ambiguous_symbols)

    def test_changing_recovery_position_blocks_automatic_exit(self) -> None:
        broker = FakeBroker(signed_qty=10)
        broker.orders = mock.Mock(return_value=[])
        broker.position_qty = mock.Mock(side_effect=[10, 0])

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot.time, "sleep"),
        ):
            flat, ids = bot.dedicated_force_flatten_symbol(
                broker,
                "INFY",
                "TEST",
            )

        self.assertFalse(flat)
        self.assertEqual(ids, [])
        self.assertEqual(broker.exit_calls, [])
        self.assertIn("INFY", broker._recovery_ambiguous_symbols)

    def test_partial_terminal_recovery_retries_only_exact_residual(self) -> None:
        broker = FakeBroker(signed_qty=10)
        broker.orders = mock.Mock(return_value=[])
        broker.exit_market = mock.Mock(side_effect=["RECOVERY-1", "RECOVERY-2"])
        snapshots = {
            "RECOVERY-1": bot.OrderSnapshot.from_payload(
                {
                    "order_id": "RECOVERY-1",
                    "status": "CANCELLED",
                    "quantity": 10,
                    "filled_quantity": 4,
                    "pending_quantity": 0,
                    "average_price": 99.0,
                    "transaction_type": "SELL",
                    "order_type": "MARKET",
                    "tradingsymbol": "INFY",
                    "exchange": "NSE",
                    "product": "MIS",
                }
            ),
            "RECOVERY-2": bot.OrderSnapshot.from_payload(
                {
                    "order_id": "RECOVERY-2",
                    "status": "COMPLETE",
                    "quantity": 6,
                    "filled_quantity": 6,
                    "pending_quantity": 0,
                    "average_price": 98.9,
                    "transaction_type": "SELL",
                    "order_type": "MARKET",
                    "tradingsymbol": "INFY",
                    "exchange": "NSE",
                    "product": "MIS",
                }
            ),
        }
        broker.wait_for_order = mock.Mock(
            side_effect=lambda order_id, **kwargs: snapshots[order_id]
        )

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(bot.time, "sleep"),
        ):
            flat, ids = bot.dedicated_force_flatten_symbol(broker, "INFY", "TEST")

        self.assertTrue(flat)
        self.assertEqual(ids, ["RECOVERY-1", "RECOVERY-2"])
        submitted_quantities = [call.args[1] for call in broker.exit_market.call_args_list]
        self.assertEqual(submitted_quantities, [10, 6])
        self.assertEqual(broker.signed_qty, 0)

    def test_tracked_unresolved_recovery_id_is_persisted_for_restart(self) -> None:
        broker = FakeBroker(signed_qty=10)
        state = bot.fresh_state()
        state["execution_mode"] = "live"
        trade = make_trade(status="HALTED_UNCERTAIN")
        trade.execution_mode = "live"
        state["trades"][trade.symbol] = asdict(trade)

        with (
            mock.patch.object(bot, "LIVE_TRADING", True),
            mock.patch.object(bot, "DEDICATED_BOT_ACCOUNT", True),
            mock.patch.object(
                bot,
                "dedicated_force_flatten_symbol",
                return_value=(False, ["RECOVERY-1"]),
            ),
        ):
            result = bot.recover_trade_on_dedicated_account(
                broker,
                state,
                trade,
                "TEST",
                "forced test recovery",
            )

        persisted = bot.trade_from_dict(state["trades"][trade.symbol])
        self.assertFalse(result)
        self.assertEqual(persisted.status, "EXIT_PENDING")
        self.assertEqual(persisted.exit_order_id, "RECOVERY-1")
        self.assertEqual(persisted.exit_order_ids, ["RECOVERY-1"])
        self.assertEqual(
            persisted.exit_status,
            "RECOVERY_RECONCILIATION_PENDING",
        )

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


if __name__ == "__main__":
    unittest.main()
