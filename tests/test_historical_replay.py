from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import historical_replay as replay
from trading_core import estimate_nse_equity_intraday_cost, gross_pnl


IST = ZoneInfo("Asia/Kolkata")
SOURCE_DIGEST = hashlib.sha256(b"synthetic point-in-time source").hexdigest()


def _at(session: date, hour: int, minute: int, second: int = 0) -> datetime:
    return datetime.combine(session, time(hour, minute, second), IST)


def _rule_input(side: str, *, stock_score: float = 90.0) -> dict[str, object]:
    if side == "LONG":
        return {
            "price": 100.0,
            "candle_open": 99.5,
            "prev_close": 98.0,
            "opening_high": 99.0,
            "opening_low": 98.0,
            "ema9": 99.5,
            "ema20": 99.0,
            "vwap": 99.0,
            "rsi": 60.0,
            "atr": 2.0,
            "rvol": 2.5,
            "opening_rvol": 2.0,
            "body_ratio": 0.8,
            "close_location": 0.9,
            "nifty_regime": "BULL",
            "stock_in_play_score": stock_score,
            "spread_bps": 2.0,
            "prior_post_opening_max_close": 99.0,
            "prior_post_opening_min_close": 98.5,
        }
    if side == "SHORT":
        return {
            "price": 100.0,
            "candle_open": 100.5,
            "prev_close": 102.0,
            "opening_high": 102.0,
            "opening_low": 101.0,
            "ema9": 100.5,
            "ema20": 101.0,
            "vwap": 101.0,
            "rsi": 40.0,
            "atr": 2.0,
            "rvol": 2.5,
            "opening_rvol": 2.0,
            "body_ratio": 0.8,
            "close_location": 0.1,
            "nifty_regime": "BEAR",
            "stock_in_play_score": stock_score,
            "spread_bps": 2.0,
            "prior_post_opening_max_close": 101.5,
            "prior_post_opening_min_close": 101.0,
        }
    raise ValueError("side must be LONG or SHORT")


def _opportunity(
    session: date,
    symbol: str,
    *,
    side: str = "LONG",
    opportunity_id: str | None = None,
    scan_id: str = "scan-1000",
    decision_at: datetime | None = None,
    stock_score: float = 90.0,
    best_bid_qty: int = 1_000,
    best_ask_qty: int = 1_000,
) -> dict[str, object]:
    decision = decision_at or _at(session, 10, 0, 3)
    signal_close = _at(session, 10, 0, 0)
    return {
        "schema_version": replay.OPPORTUNITY_SCHEMA_VERSION,
        "opportunity_id": opportunity_id or f"{session.isoformat()}-{symbol}",
        "scan_id": scan_id,
        "session_date": session.isoformat(),
        "symbol": symbol,
        "token": 100_000 + sum(ord(character) for character in symbol),
        "decision_at": decision.isoformat(),
        "signal_bar_closed_at": signal_close.isoformat(),
        "features_available_at": (signal_close + timedelta(seconds=1)).isoformat(),
        "quote_available_at": (decision - timedelta(seconds=1)).isoformat(),
        "best_bid": 99.99,
        "best_ask": 100.01,
        "best_bid_qty": best_bid_qty,
        "best_ask_qty": best_ask_qty,
        "tick_size": 0.05,
        "lower_circuit": 80.0,
        "upper_circuit": 120.0,
        "source_data_sha256": SOURCE_DIGEST,
        "rule_input": _rule_input(side, stock_score=stock_score),
    }


def _bars(
    session: date,
    symbol: str,
    *,
    outcome: str = "TARGET_LONG",
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    cursor = _at(session, 10, 0)
    final_start = _at(session, 15, 5)
    while cursor <= final_start:
        open_price, high, low, close = 100.0, 100.4, 99.6, 100.0
        if cursor.time() == time(10, 5):
            if outcome == "TARGET_LONG":
                open_price, high, low, close = 100.0, 105.0, 99.5, 104.0
            elif outcome == "TARGET_SHORT":
                open_price, high, low, close = 100.0, 100.5, 95.0, 96.0
            elif outcome == "AMBIGUOUS_LONG":
                open_price, high, low, close = 100.0, 106.0, 96.0, 100.0
            elif outcome == "GAP_STOP_LONG":
                open_price, high, low, close = 95.0, 96.0, 94.0, 95.0
            elif outcome == "FORCE_EXIT":
                pass
            else:
                raise ValueError(f"unknown outcome: {outcome}")
        if cursor.time() == time(15, 5) and outcome == "FORCE_EXIT":
            open_price, high, low, close = 100.0, 101.5, 99.5, 101.0
        records.append(
            {
                "schema_version": replay.BAR_SCHEMA_VERSION,
                "session_date": session.isoformat(),
                "symbol": symbol,
                "bar_start": cursor.isoformat(),
                "available_at": (cursor + timedelta(minutes=5, seconds=2)).isoformat(),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 50_000.0,
                "source_data_sha256": SOURCE_DIGEST,
            }
        )
        cursor += timedelta(minutes=5)
    return records


def _jsonl(records: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )


def _write_dataset(
    root: Path,
    sessions: list[date],
    opportunities: list[dict[str, object]],
    bars: list[dict[str, object]],
    *,
    dataset_id: str = "synthetic-replay-fixture",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    opportunities_path = root / "opportunities.jsonl"
    bars_path = root / "bars.jsonl"
    opportunities_path.write_text(_jsonl(opportunities), encoding="utf-8")
    bars_path.write_text(_jsonl(bars), encoding="utf-8")
    manifest = {
        "schema_version": replay.DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "created_at": _at(max(sessions), 18, 0).isoformat(),
        "timezone": "Asia/Kolkata",
        "bar_interval_minutes": 5,
        "bar_timestamp_semantics": "start",
        "fidelity": replay.FIDELITY,
        "decision_scope": replay.DECISION_SCOPE,
        "point_in_time_universe": True,
        "survivorship_bias_free": True,
        "complete_decision_trace": True,
        "raw_as_traded_prices": True,
        "source_strategy_fingerprint": "synthetic-strategy-fixture-v1",
        "raw_tape_sha256": SOURCE_DIGEST,
        "session_rules_version": "nse-cash-cas-20260803",
        "fee_model_version": replay.NSE_EQUITY_INTRADAY_FEE_MODEL_VERSION,
        "sessions": [session.isoformat() for session in sessions],
        "files": {
            "opportunities": {
                "path": opportunities_path.name,
                "sha256": hashlib.sha256(opportunities_path.read_bytes()).hexdigest(),
            },
            "bars": {
                "path": bars_path.name,
                "sha256": hashlib.sha256(bars_path.read_bytes()).hexdigest(),
            },
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return root


def _single_dataset(
    root: Path,
    *,
    session: date = date(2026, 8, 11),
    symbol: str = "ALPHA",
    side: str = "LONG",
    outcome: str = "TARGET_LONG",
    **opportunity_fields: object,
) -> Path:
    opportunity = _opportunity(
        session,
        symbol,
        side=side,
        **opportunity_fields,
    )
    return _write_dataset(
        root,
        [session],
        [opportunity],
        _bars(session, symbol, outcome=outcome),
    )


class ReplayDatasetContractTests(unittest.TestCase):
    def test_checked_in_trial_registry_matches_current_schema(self) -> None:
        configs = replay.load_trial_registry(
            PROJECT_ROOT / "replay_trials.example.json"
        )

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0], replay.ReplayConfig())

    def test_valid_load_replay_and_exact_after_cost_arithmetic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _single_dataset(Path(directory))
            dataset = replay.load_replay_dataset(root)
            result = replay.run_replay(dataset, replay.ReplayConfig())

        self.assertEqual(dataset.manifest.dataset_id, "synthetic-replay-fixture")
        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade.side, "LONG")
        self.assertEqual(trade.exit_reason, "TARGET")
        expected_gross = float(
            gross_pnl("LONG", trade.entry_fill, trade.exit_fill, trade.qty)
        )
        expected_fees = float(
            estimate_nse_equity_intraday_cost(
                trade.entry_fill * trade.qty,
                trade.exit_fill * trade.qty,
            ).total
        )
        self.assertAlmostEqual(trade.gross_pnl, expected_gross, places=9)
        self.assertAlmostEqual(trade.fees, expected_fees, places=9)
        self.assertAlmostEqual(trade.net_pnl, expected_gross - expected_fees, places=9)
        self.assertAlmostEqual(result.summary["net_pnl"], trade.net_pnl, places=9)
        self.assertEqual(result.fidelity, replay.FIDELITY)

    def test_current_trade_logs_are_refused_as_replay_data(self) -> None:
        with self.assertRaisesRegex(
            replay.ReplayDataError,
            "trade journals are not replay data",
        ):
            replay.load_replay_dataset(PROJECT_ROOT / "logs")

    def test_checksum_duplicate_and_missing_path_fail_closed(self) -> None:
        session = date(2026, 8, 11)
        with tempfile.TemporaryDirectory() as directory:
            checksum_root = _single_dataset(Path(directory) / "checksum")
            path = checksum_root / "bars.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(replay.ReplayDataError, "checksum mismatch"):
                replay.load_replay_dataset(checksum_root)

            duplicate = _opportunity(
                session,
                "ALPHA",
                opportunity_id="duplicate-id",
            )
            duplicate_root = _write_dataset(
                Path(directory) / "duplicate",
                [session],
                [duplicate, dict(duplicate)],
                _bars(session, "ALPHA"),
            )
            with self.assertRaisesRegex(replay.ReplayDataError, "duplicate opportunity_id"):
                replay.load_replay_dataset(duplicate_root)

            incomplete_bars = [
                record
                for record in _bars(session, "ALPHA")
                if not str(record["bar_start"]).startswith("2026-08-11T12:00:00")
            ]
            gap_root = _write_dataset(
                Path(directory) / "gap",
                [session],
                [_opportunity(session, "ALPHA")],
                incomplete_bars,
            )
            with self.assertRaisesRegex(replay.ReplayDataError, "incomplete five-minute path"):
                replay.load_replay_dataset(gap_root)

    def test_noncausal_opportunity_and_bar_availability_fail_closed(self) -> None:
        session = date(2026, 8, 11)
        with tempfile.TemporaryDirectory() as directory:
            opportunity = _opportunity(session, "ALPHA")
            opportunity["features_available_at"] = (
                _at(session, 10, 0, 4).isoformat()
            )
            root = _write_dataset(
                Path(directory) / "future-feature",
                [session],
                [opportunity],
                _bars(session, "ALPHA"),
            )
            with self.assertRaisesRegex(replay.ReplayDataError, "non-causal"):
                replay.load_replay_dataset(root)

            bars = _bars(session, "ALPHA")
            bars[0]["available_at"] = _at(session, 10, 4, 59).isoformat()
            root = _write_dataset(
                Path(directory) / "early-bar",
                [session],
                [_opportunity(session, "ALPHA")],
                bars,
            )
            with self.assertRaisesRegex(replay.ReplayDataError, "before its bar closed"):
                replay.load_replay_dataset(root)

    def test_scan_and_instrument_identities_fail_closed(self) -> None:
        session = date(2026, 8, 11)
        with tempfile.TemporaryDirectory() as directory:
            duplicate_symbol = [
                _opportunity(session, "ALPHA", opportunity_id="alpha-1"),
                _opportunity(session, "ALPHA", opportunity_id="alpha-2"),
            ]
            root = _write_dataset(
                Path(directory) / "duplicate-scan-symbol",
                [session],
                duplicate_symbol,
                _bars(session, "ALPHA"),
            )
            with self.assertRaisesRegex(replay.ReplayDataError, "duplicate scan symbol"):
                replay.load_replay_dataset(root)

            token_aliases = [
                _opportunity(session, "ALPHA", opportunity_id="alpha"),
                _opportunity(session, "BETA", opportunity_id="beta"),
            ]
            token_aliases[1]["token"] = token_aliases[0]["token"]
            root = _write_dataset(
                Path(directory) / "token-alias",
                [session],
                token_aliases,
                [*_bars(session, "ALPHA"), *_bars(session, "BETA")],
            )
            with self.assertRaisesRegex(replay.ReplayDataError, "multiple symbols"):
                replay.load_replay_dataset(root)

            inconsistent_scan = [
                _opportunity(session, "ALPHA", opportunity_id="alpha"),
                _opportunity(
                    session,
                    "BETA",
                    opportunity_id="beta",
                    decision_at=_at(session, 10, 0, 4),
                ),
            ]
            root = _write_dataset(
                Path(directory) / "scan-time",
                [session],
                inconsistent_scan,
                [*_bars(session, "ALPHA"), *_bars(session, "BETA")],
            )
            with self.assertRaisesRegex(replay.ReplayDataError, "shared decision timestamp"):
                replay.load_replay_dataset(root)


class ReplayExecutionTests(unittest.TestCase):
    def test_long_and_short_use_executable_side_and_directional_depth(self) -> None:
        session = date(2026, 8, 11)
        opportunities = [
            _opportunity(
                session,
                "LONGNAME",
                side="LONG",
                best_bid_qty=1_000,
                best_ask_qty=12,
            ),
            _opportunity(
                session,
                "SHORTNAME",
                side="SHORT",
                best_bid_qty=16,
                best_ask_qty=1_000,
            ),
        ]
        bars = [
            *_bars(session, "LONGNAME", outcome="TARGET_LONG"),
            *_bars(session, "SHORTNAME", outcome="TARGET_SHORT"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            dataset = replay.load_replay_dataset(
                _write_dataset(Path(directory), [session], opportunities, bars)
            )
            result = replay.run_replay(dataset, replay.ReplayConfig())

        trades = {trade.side: trade for trade in result.trades}
        self.assertEqual(set(trades), {"LONG", "SHORT"})
        self.assertEqual(trades["LONG"].entry_reference, 100.01)
        self.assertEqual(trades["SHORT"].entry_reference, 99.99)
        self.assertGreater(trades["LONG"].entry_fill, trades["LONG"].entry_reference)
        self.assertLess(trades["SHORT"].entry_fill, trades["SHORT"].entry_reference)
        self.assertEqual(trades["LONG"].qty, 3)
        self.assertEqual(trades["SHORT"].qty, 4)

    def test_ambiguous_bar_is_stop_first_and_gap_stop_uses_worse_open(self) -> None:
        session = date(2026, 8, 11)
        opportunities = [
            _opportunity(session, "AMBIGUOUS", side="LONG"),
            _opportunity(session, "GAPPED", side="LONG"),
        ]
        bars = [
            *_bars(session, "AMBIGUOUS", outcome="AMBIGUOUS_LONG"),
            *_bars(session, "GAPPED", outcome="GAP_STOP_LONG"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            dataset = replay.load_replay_dataset(
                _write_dataset(Path(directory), [session], opportunities, bars)
            )
            result = replay.run_replay(dataset, replay.ReplayConfig())

        trades = {trade.symbol: trade for trade in result.trades}
        self.assertEqual(trades["AMBIGUOUS"].exit_reason, "AMBIGUOUS_BAR_STOP_FIRST")
        self.assertEqual(trades["AMBIGUOUS"].exit_reference, trades["AMBIGUOUS"].stop_price)
        self.assertEqual(trades["GAPPED"].exit_reason, "STOP")
        self.assertEqual(trades["GAPPED"].exit_reference, 95.0)
        self.assertLess(trades["GAPPED"].exit_reference, trades["GAPPED"].stop_price)

    def test_simultaneous_candidates_have_deterministic_capacity_competition(self) -> None:
        session = date(2026, 8, 11)
        opportunities = [
            _opportunity(session, "ZETA", side="LONG", opportunity_id="zeta"),
            _opportunity(session, "ALPHA", side="LONG", opportunity_id="alpha"),
        ]
        bars = [
            *_bars(session, "ZETA", outcome="TARGET_LONG"),
            *_bars(session, "ALPHA", outcome="TARGET_LONG"),
        ]
        config = replay.ReplayConfig(max_open_positions=1)
        with tempfile.TemporaryDirectory() as directory:
            dataset = replay.load_replay_dataset(
                _write_dataset(Path(directory), [session], opportunities, bars)
            )
            forward = replay.run_replay(dataset, config)
            reversed_dataset = replace(
                dataset,
                opportunities=tuple(reversed(dataset.opportunities)),
            )
            backward = replay.run_replay(reversed_dataset, config)

        self.assertEqual([trade.symbol for trade in forward.trades], ["ALPHA"])
        self.assertEqual([trade.symbol for trade in backward.trades], ["ALPHA"])
        self.assertEqual(forward.summary, backward.summary)
        self.assertEqual(forward.rejection_reasons.get("MAX_OPEN_POSITIONS"), 1)

    def test_live_cutoff_and_circuit_buffer_are_replayed(self) -> None:
        session = date(2026, 8, 11)
        with tempfile.TemporaryDirectory() as directory:
            cutoff_dataset = replay.load_replay_dataset(
                _single_dataset(
                    Path(directory) / "cutoff",
                    decision_at=_at(session, 14, 29, 55),
                )
            )
            cutoff = replay.run_replay(cutoff_dataset, replay.ReplayConfig())

            circuit_opportunity = _opportunity(session, "ALPHA")
            circuit_opportunity["lower_circuit"] = 99.20
            circuit_dataset = replay.load_replay_dataset(
                _write_dataset(
                    Path(directory) / "circuit",
                    [session],
                    [circuit_opportunity],
                    _bars(session, "ALPHA"),
                )
            )
            circuit = replay.run_replay(circuit_dataset, replay.ReplayConfig())

        self.assertEqual(cutoff.trades, ())
        self.assertEqual(cutoff.rejection_reasons.get("ENTRY_CUTOFF_GUARD"), 1)
        self.assertEqual(circuit.trades, ())
        self.assertEqual(
            circuit.rejection_reasons.get(
                "EXECUTION_CIRCUIT_BUFFER_BELOW_MINIMUM"
            ),
            1,
        )

    def test_entry_bar_stop_is_an_explicit_worst_case_assumption(self) -> None:
        session = date(2026, 8, 11)
        bars = _bars(session, "ALPHA", outcome="TARGET_LONG")
        bars[0]["low"] = 95.0
        with tempfile.TemporaryDirectory() as directory:
            dataset = replay.load_replay_dataset(
                _write_dataset(
                    Path(directory),
                    [session],
                    [_opportunity(session, "ALPHA")],
                    bars,
                )
            )
            result = replay.run_replay(dataset, replay.ReplayConfig())

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(
            result.trades[0].exit_reason,
            "ENTRY_BAR_CONSERVATIVE_STOP_ASSUMPTION",
        )
        self.assertEqual(
            result.summary["entry_bar_conservative_stop_assumptions"],
            1,
        )

    def test_exact_tick_target_touch_is_not_lost_to_float_residue(self) -> None:
        session = date(2026, 8, 11)
        opportunity = _opportunity(session, "ALPHA")
        opportunity["rule_input"]["atr"] = 1.98
        bars = _bars(session, "ALPHA", outcome="TARGET_LONG")
        bars[1]["high"] = 104.35
        bars[1]["close"] = 104.0
        with tempfile.TemporaryDirectory() as directory:
            dataset = replay.load_replay_dataset(
                _write_dataset(
                    Path(directory),
                    [session],
                    [opportunity],
                    bars,
                )
            )
            result = replay.run_replay(dataset, replay.ReplayConfig())

        trade = result.trades[0]
        self.assertEqual(trade.target_price, 104.35)
        self.assertEqual(trade.exit_reference, 104.35)
        self.assertEqual(trade.exit_reason, "TARGET")

    def test_short_notional_sizing_uses_conservative_buy_side_price(self) -> None:
        session = date(2026, 8, 11)
        config = replay.ReplayConfig(
            capital_limit=1_000.0,
            risk_per_trade_pct=0.02,
            max_position_pct=0.10,
            max_daily_loss_pct=0.05,
            max_portfolio_stop_risk_pct=0.02,
            max_gross_exposure_pct=0.10,
        )
        with tempfile.TemporaryDirectory() as directory:
            dataset = replay.load_replay_dataset(
                _single_dataset(
                    Path(directory),
                    side="SHORT",
                    outcome="TARGET_SHORT",
                )
            )
            result = replay.run_replay(dataset, config)

        self.assertEqual(result.trades, ())
        self.assertEqual(
            result.rejection_reasons.get("NO_EXECUTABLE_QUANTITY"),
            1,
        )

    def test_equal_time_exit_state_and_drawdown_ignore_opportunity_ids(self) -> None:
        session = date(2026, 8, 11)
        winner = _opportunity(
            session,
            "WINNER",
            opportunity_id="winner-original",
        )
        loser = _opportunity(
            session,
            "LOSER",
            opportunity_id="loser-original",
        )
        later = _opportunity(
            session,
            "LATER",
            opportunity_id="later",
            scan_id="scan-1010",
            decision_at=_at(session, 10, 10, 3),
        )
        later["signal_bar_closed_at"] = _at(session, 10, 10).isoformat()
        later["features_available_at"] = _at(session, 10, 10, 1).isoformat()
        bars = [
            *_bars(session, "WINNER", outcome="TARGET_LONG"),
            *_bars(session, "LOSER", outcome="GAP_STOP_LONG"),
            *_bars(session, "LATER", outcome="TARGET_LONG"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            dataset = replay.load_replay_dataset(
                _write_dataset(
                    Path(directory),
                    [session],
                    [winner, loser, later],
                    bars,
                )
            )
            renamed_a = replace(
                dataset,
                opportunities=tuple(
                    replace(
                        opportunity,
                        opportunity_id=(
                            "a-loss" if opportunity.symbol == "LOSER"
                            else "z-win" if opportunity.symbol == "WINNER"
                            else opportunity.opportunity_id
                        ),
                    )
                    for opportunity in dataset.opportunities
                ),
            )
            renamed_b = replace(
                dataset,
                opportunities=tuple(
                    replace(
                        opportunity,
                        opportunity_id=(
                            "z-loss" if opportunity.symbol == "LOSER"
                            else "a-win" if opportunity.symbol == "WINNER"
                            else opportunity.opportunity_id
                        ),
                    )
                    for opportunity in dataset.opportunities
                ),
            )
            config = replay.ReplayConfig(max_consecutive_losses=1)
            first = replay.run_replay(renamed_a, config)
            second = replay.run_replay(renamed_b, config)

        self.assertEqual({trade.symbol for trade in first.trades}, {"LOSER", "WINNER"})
        self.assertEqual({trade.symbol for trade in second.trades}, {"LOSER", "WINNER"})
        self.assertEqual(first.summary, second.summary)
        self.assertEqual(first.rejection_reasons, second.rejection_reasons)
        self.assertEqual(first.rejection_reasons.get("MAX_CONSECUTIVE_LOSSES"), 1)

    def test_limit_breach_with_overlap_refuses_unpriced_live_flatten(self) -> None:
        session = date(2026, 8, 11)
        winner_bars = _bars(session, "LATEWIN", outcome="TARGET_LONG")
        winner_bars[1].update(
            {"open": 100.0, "high": 100.4, "low": 99.6, "close": 100.0}
        )
        winner_bars[3].update(
            {"open": 100.0, "high": 105.0, "low": 99.5, "close": 104.0}
        )
        with tempfile.TemporaryDirectory() as directory:
            dataset = replay.load_replay_dataset(
                _write_dataset(
                    Path(directory),
                    [session],
                    [
                        _opportunity(session, "LOSER"),
                        _opportunity(session, "LATEWIN"),
                    ],
                    [
                        *_bars(session, "LOSER", outcome="GAP_STOP_LONG"),
                        *winner_bars,
                    ],
                )
            )
            with self.assertRaisesRegex(
                replay.ReplayDataError,
                "kill-switch limit breached with overlapping exposure",
            ):
                replay.run_replay(
                    dataset,
                    replay.ReplayConfig(max_consecutive_losses=1),
                )


class ReplayWalkForwardTests(unittest.TestCase):
    @staticmethod
    def _many_session_records(
        sessions: list[date],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        opportunities: list[dict[str, object]] = []
        bars: list[dict[str, object]] = []
        for index, session in enumerate(sessions):
            symbol = f"S{index:02d}"
            opportunities.append(_opportunity(session, symbol, side="LONG"))
            bars.extend(_bars(session, symbol, outcome="TARGET_LONG"))
        return opportunities, bars

    def test_walk_forward_is_chronological_and_excludes_reserved_holdout(self) -> None:
        sessions = [
            date(2026, 8, 3),
            date(2026, 8, 4),
            date(2026, 8, 5),
            date(2026, 8, 6),
            date(2026, 8, 7),
            date(2026, 8, 10),
            date(2026, 8, 11),
        ]
        opportunities, bars = self._many_session_records(sessions)
        with tempfile.TemporaryDirectory() as directory:
            dataset = replay.load_replay_dataset(
                _write_dataset(Path(directory), sessions, opportunities, bars)
            )
            plan = replay.build_walk_forward_plan(
                sessions,
                min_train_sessions=2,
                test_sessions=1,
                purge_sessions=1,
                final_holdout_sessions=1,
            )
            report = replay.run_walk_forward(
                dataset,
                [replay.ReplayConfig()],
                plan,
                minimum_training_trades=1,
                evaluate_holdout=False,
            )

        holdout = sessions[-1]
        self.assertEqual(plan.holdout_sessions, (holdout,))
        self.assertEqual(report["holdout"]["status"], "RESERVED_NOT_EVALUATED")
        self.assertEqual(report["holdout"]["sessions"], [holdout.isoformat()])
        for fold in plan.folds:
            self.assertLess(max(fold.train_sessions), min(fold.test_sessions))
            self.assertNotIn(holdout, fold.train_sessions)
            self.assertNotIn(holdout, fold.purged_sessions)
            self.assertNotIn(holdout, fold.test_sessions)
        self.assertEqual(report["combined_oos_summary"]["trades"], len(plan.folds))

    def test_replay_prefix_is_invariant_when_future_sessions_are_appended(self) -> None:
        prefix_sessions = [
            date(2026, 8, 3),
            date(2026, 8, 4),
            date(2026, 8, 5),
        ]
        future_sessions = [date(2026, 8, 6), date(2026, 8, 7)]
        prefix_opportunities, prefix_bars = self._many_session_records(prefix_sessions)
        all_opportunities, all_bars = self._many_session_records(
            [*prefix_sessions, *future_sessions]
        )
        with tempfile.TemporaryDirectory() as directory:
            prefix = replay.load_replay_dataset(
                _write_dataset(
                    Path(directory) / "prefix",
                    prefix_sessions,
                    prefix_opportunities,
                    prefix_bars,
                    dataset_id="same-logical-dataset",
                )
            )
            extended = replay.load_replay_dataset(
                _write_dataset(
                    Path(directory) / "extended",
                    [*prefix_sessions, *future_sessions],
                    all_opportunities,
                    all_bars,
                    dataset_id="same-logical-dataset",
                )
            )
            config = replay.ReplayConfig()
            before = replay.run_replay(prefix, config, sessions=prefix_sessions)
            after = replay.run_replay(extended, config, sessions=prefix_sessions)

        self.assertEqual(before.trades, after.trades)
        self.assertEqual(before.summary, after.summary)
        self.assertEqual(before.rejection_reasons, after.rejection_reasons)

    def test_public_walk_forward_rejects_malformed_anti_leakage_plan(self) -> None:
        sessions = [
            date(2026, 8, 3),
            date(2026, 8, 4),
            date(2026, 8, 5),
            date(2026, 8, 6),
            date(2026, 8, 7),
            date(2026, 8, 10),
            date(2026, 8, 11),
        ]
        opportunities, bars = self._many_session_records(sessions)
        with tempfile.TemporaryDirectory() as directory:
            dataset = replay.load_replay_dataset(
                _write_dataset(Path(directory), sessions, opportunities, bars)
            )
            valid = replay.build_walk_forward_plan(
                sessions,
                min_train_sessions=2,
                test_sessions=1,
                purge_sessions=1,
                final_holdout_sessions=1,
            )
            malformed = (
                replay.WalkForwardPlan((), valid.holdout_sessions),
                replace(valid, folds=tuple(reversed(valid.folds))),
                replace(valid, holdout_sessions=(sessions[-2],)),
                replace(
                    valid,
                    folds=(
                        replace(
                            valid.folds[0],
                            purged_sessions=(sessions[4],),
                        ),
                        *valid.folds[1:],
                    ),
                ),
            )
            for plan in malformed:
                with self.subTest(plan=plan):
                    with self.assertRaises(ValueError):
                        replay.run_walk_forward(
                            dataset,
                            [replay.ReplayConfig()],
                            plan,
                            minimum_training_trades=1,
                        )


if __name__ == "__main__":
    unittest.main()
