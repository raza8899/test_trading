from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import performance_report


def write_jsonl(path: Path, records: list[object], raw_lines: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        for raw_line in raw_lines or []:
            handle.write(raw_line + "\n")


class PerformanceReportTests(unittest.TestCase):
    def test_ai_review_usage_includes_rejects_errors_cache_and_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades_20260811.jsonl"
            first = {
                "event": "AI_REVIEW",
                "event_id": "review-1",
                "review_status": "OK",
                "decision": "APPROVE",
                "api_called": True,
                "ai_input_tokens": 100,
                "ai_cached_input_tokens": 80,
                "ai_cache_write_tokens": 0,
                "ai_output_tokens": 20,
                "ai_reasoning_tokens": 8,
                "ai_total_tokens": 120,
                "latency_ms": 1000,
                "prompt_version": "prompt-v5",
                "response_model": "gpt-test",
            }
            write_jsonl(
                path,
                [
                    first,
                    first,
                    {
                        "event": "AI_REVIEW",
                        "event_id": "review-2",
                        "review_status": "ERROR",
                        "decision": "ERROR",
                        "api_called": True,
                        "ai_input_tokens": 50,
                        "ai_cached_input_tokens": 0,
                        "ai_cache_write_tokens": 50,
                        "ai_output_tokens": 10,
                        "ai_reasoning_tokens": 4,
                        "ai_total_tokens": 60,
                        "latency_ms": 2000,
                        "prompt_version": "prompt-v5",
                        "response_model": "gpt-test",
                    },
                    {
                        "event": "AI_REVIEW",
                        "event_id": "review-3",
                        "review_status": "DUPLICATE_REUSED",
                        "decision": "APPROVE",
                        "api_called": False,
                        "duplicate_review_suppressed": True,
                        "ai_input_tokens": 0,
                        "ai_cached_input_tokens": 0,
                        "ai_cache_write_tokens": 0,
                        "ai_output_tokens": 0,
                        "ai_reasoning_tokens": 0,
                        "ai_total_tokens": 0,
                    },
                    {
                        "event": "AI_REVIEW",
                        "event_id": "review-legacy",
                        "review_status": "OK",
                        "decision": "REJECT",
                        "input_tokens": 200,
                        "output_tokens": 40,
                        "total_tokens": 240,
                        "latency_ms": 3000,
                        "prompt_version": "prompt-v4",
                        "response_model": "gpt-test-old",
                    },
                ],
            )

            report = performance_report.build_report([path])
            usage = report["ai_usage"]

            self.assertEqual(usage["review_events"], 4)
            self.assertEqual(usage["api_calls"], 3)
            self.assertEqual(usage["successful_calls"], 2)
            self.assertEqual(usage["failed_calls"], 1)
            self.assertEqual(usage["duplicate_reviews_suppressed"], 1)
            self.assertEqual(usage["input_tokens"], 350)
            self.assertEqual(usage["cached_input_tokens"], 80)
            self.assertEqual(usage["cache_write_tokens"], 50)
            self.assertEqual(usage["output_tokens"], 70)
            self.assertEqual(usage["reasoning_tokens"], 12)
            self.assertAlmostEqual(usage["cache_hit_rate"], 80 / 150)
            self.assertTrue(usage["cache_backend_verified"])
            self.assertEqual(
                usage["diagnostics"]["duplicate_review_events"],
                1,
            )
            self.assertIn("AI cache hit rate: 53.3%", performance_report.format_text_report(report))

    def test_malformed_ai_usage_never_changes_valid_pnl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades_20260811.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "event": "AI_REVIEW",
                        "event_id": "review-bad",
                        "review_status": "OK",
                        "decision": "APPROVE",
                        "api_called": True,
                        "ai_input_tokens": "100",
                    },
                    {
                        "event": "CLOSE",
                        "net_pnl": 8,
                        "gross_pnl": 10,
                        "fees": 2,
                        "r_multiple": 0.4,
                        "ai_decision": "APPROVE",
                    },
                ],
            )

            report = performance_report.build_report([path])

            self.assertEqual(report["summary"]["net_pnl"], 8)
            self.assertIsNone(report["ai_usage"]["input_tokens"])
            self.assertEqual(
                report["ai_usage"]["diagnostics"]["malformed_fields"],
                {"invalid_field:ai_input_tokens": 1},
            )

    def test_summary_drawdown_and_ai_shadow_cohorts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            later = root / "trades_20260811.jsonl"
            earlier = root / "trades_20260810.jsonl"

            write_jsonl(
                earlier,
                [
                    {"event": "OPEN", "symbol": "AAA"},
                    {
                        "event": "CLOSE",
                        "net_pnl": 80,
                        "gross_pnl": 100,
                        "fees": 20,
                        "r_multiple": 1.0,
                        "ai_decision": "approve",
                    },
                    {
                        "event": "CLOSE",
                        "net_pnl": -50,
                        "gross_pnl": -40,
                        "fees": 10,
                        "r_multiple": -0.5,
                        "ai_decision": "REJECT",
                    },
                ],
            )
            write_jsonl(
                later,
                [
                    {
                        "event": "CLOSE",
                        "net_pnl": 20,
                        "gross_pnl": 25,
                        "fees": 5,
                        "r_multiple": 0.25,
                        "ai_decision": "APPROVE",
                    }
                ],
            )

            report = performance_report.build_report([later, earlier])
            summary = report["summary"]

            self.assertEqual(summary["trades"], 3)
            self.assertEqual(summary["wins"], 2)
            self.assertEqual(summary["losses"], 1)
            self.assertAlmostEqual(summary["win_rate_pct"], 200 / 3)
            self.assertAlmostEqual(summary["net_pnl"], 50)
            self.assertAlmostEqual(summary["gross_pnl"], 85)
            self.assertAlmostEqual(summary["fees"], 35)
            self.assertAlmostEqual(summary["expectancy"], 50 / 3)
            self.assertAlmostEqual(summary["profit_factor"], 2.0)
            self.assertEqual(summary["profit_factor_status"], "finite")
            self.assertAlmostEqual(summary["max_drawdown"], 50)
            self.assertAlmostEqual(summary["average_r"], 0.25)

            approve = report["ai_shadow_cohorts"]["APPROVE"]
            reject = report["ai_shadow_cohorts"]["REJECT"]
            self.assertEqual(approve["trades"], 2)
            self.assertAlmostEqual(approve["net_pnl"], 100)
            self.assertEqual(approve["profit_factor_status"], "infinite_no_losses")
            self.assertEqual(reject["trades"], 1)
            self.assertAlmostEqual(reject["net_pnl"], -50)
            self.assertAlmostEqual(reject["profit_factor"], 0.0)

    def test_incomplete_legacy_close_records_are_explicitly_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades_20260811.jsonl"
            write_jsonl(
                path,
                [
                    {"event": "CLOSE", "symbol": "LEGACY", "reason": "TARGET"},
                    {
                        "event": "CLOSE",
                        "net_pnl": 999,
                        "gross_pnl": 1000,
                        "fees": 1,
                        "r_multiple": 10,
                    },
                    {
                        "event": "CLOSE",
                        "net_pnl": "12.50",
                        "gross_pnl": 15,
                        "fees": 2.5,
                        "r_multiple": 1,
                        "ai_decision": "APPROVE",
                    },
                    {
                        "event": "CLOSE",
                        "net_pnl": 12.5,
                        "gross_pnl": 15,
                        "fees": 2.5,
                        "r_multiple": 1,
                        "ai_decision": "APPROVE",
                    },
                ],
            )

            report = performance_report.build_report([path])
            diagnostics = report["diagnostics"]

            self.assertEqual(diagnostics["close_events"], 4)
            self.assertEqual(diagnostics["complete_close_events"], 1)
            self.assertEqual(diagnostics["incomplete_close_events"], 3)
            self.assertEqual(diagnostics["incomplete_reasons"]["missing_field:ai_decision"], 2)
            self.assertEqual(diagnostics["incomplete_reasons"]["invalid_field:net_pnl"], 1)
            self.assertEqual(report["summary"]["trades"], 1)
            self.assertAlmostEqual(report["summary"]["net_pnl"], 12.5)
            self.assertTrue(any("no P&L values were inferred" in item for item in report["warnings"]))

    def test_off_mode_trades_count_overall_but_not_in_ai_cohorts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades_20260811.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "event": "CLOSE",
                        "net_pnl": 10,
                        "gross_pnl": 12,
                        "fees": 2,
                        "r_multiple": 0.5,
                        "ai_mode": "off",
                    },
                    {
                        "event": "CLOSE",
                        "net_pnl": -4,
                        "gross_pnl": -3,
                        "fees": 1,
                        "r_multiple": -0.2,
                        "ai_decision": "off",
                    },
                    {
                        "event": "CLOSE",
                        "net_pnl": 6,
                        "gross_pnl": 7,
                        "fees": 1,
                        "r_multiple": 0.3,
                        "ai_mode": "OFF",
                        "ai_decision": None,
                    },
                ],
            )

            report = performance_report.build_report([path])

            self.assertEqual(report["diagnostics"]["complete_close_events"], 3)
            self.assertEqual(report["diagnostics"]["incomplete_close_events"], 0)
            self.assertEqual(report["summary"]["trades"], 3)
            self.assertAlmostEqual(report["summary"]["net_pnl"], 12)
            self.assertEqual(report["ai_shadow_cohorts"]["APPROVE"]["trades"], 0)
            self.assertEqual(report["ai_shadow_cohorts"]["REJECT"]["trades"], 0)

    def test_ai_errors_are_reported_separately_from_model_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades_20260811.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "event": "CLOSE",
                        "net_pnl": -7,
                        "gross_pnl": -5,
                        "fees": 2,
                        "r_multiple": -0.1,
                        "ai_decision": "ERROR",
                    },
                    {
                        "event": "CLOSE",
                        "net_pnl": 3,
                        "gross_pnl": 4,
                        "fees": 1,
                        "r_multiple": 0.05,
                        "ai_decision": "REJECT",
                    },
                ],
            )

            report = performance_report.build_report([path])

            self.assertEqual(report["summary"]["trades"], 2)
            self.assertEqual(report["ai_shadow_cohorts"]["ERROR"]["trades"], 1)
            self.assertEqual(report["ai_shadow_cohorts"]["ERROR"]["net_pnl"], -7)
            self.assertEqual(report["ai_shadow_cohorts"]["REJECT"]["trades"], 1)

    def test_complete_provenance_is_parsed_and_marked_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades_20260811.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "event": "CLOSE",
                        "net_pnl": 12,
                        "gross_pnl": 14,
                        "fees": 2,
                        "r_multiple": 0.6,
                        "ai_decision": "APPROVE",
                        "ai_mode": "SHADOW",
                        "execution_mode": "PAPER",
                        "config_fingerprint": "config-a",
                        "ai_response_model": "gpt-test-1",
                        "ai_prompt_version": "prompt-v1",
                        "fees_source": "contract_note",
                        "fee_model_version": "broker-20260811",
                    }
                ],
            )

            records, diagnostics = performance_report.parse_trade_files([path])
            report = performance_report.build_report([path])

            self.assertEqual(diagnostics.complete_close_events, 1)
            self.assertEqual(records[0].execution_mode, "paper")
            self.assertEqual(records[0].config_fingerprint, "config-a")
            self.assertEqual(records[0].ai_mode, "shadow")
            self.assertEqual(records[0].ai_response_model, "gpt-test-1")
            self.assertEqual(records[0].ai_prompt_version, "prompt-v1")
            self.assertEqual(report["provenance"]["compatibility_status"], "compatible")
            self.assertEqual(report["provenance"]["ai"]["compatibility_status"], "compatible")
            self.assertEqual(report["warnings"], [])

    def test_mixed_experiment_provenance_is_not_presented_as_comparable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades_20260811.jsonl"
            base = {
                "event": "CLOSE",
                "gross_pnl": 11,
                "fees": 1,
                "r_multiple": 0.5,
                "ai_decision": "APPROVE",
            }
            write_jsonl(
                path,
                [
                    {
                        **base,
                        "net_pnl": 10,
                        "execution_mode": "paper",
                        "config_fingerprint": "config-a",
                        "ai_response_model": "gpt-test-1",
                        "ai_prompt_version": "prompt-v1",
                    },
                    {
                        **base,
                        "net_pnl": -5,
                        "gross_pnl": -4,
                        "execution_mode": "live",
                        "config_fingerprint": "config-b",
                        "ai_response_model": "gpt-test-2",
                        "ai_prompt_version": "prompt-v2",
                    },
                ],
            )

            report = performance_report.build_report([path])
            provenance = report["provenance"]

            self.assertEqual(report["summary"]["trades"], 2)
            self.assertEqual(
                report["summary_scope"],
                "cross_experiment_aggregate_not_comparable",
            )
            self.assertEqual(len(report["experiment_cohorts"]), 2)
            self.assertEqual(provenance["compatibility_status"], "incompatible_mixed")
            self.assertFalse(provenance["compatible"])
            self.assertEqual(provenance["execution_modes"], {"live": 1, "paper": 1})
            self.assertEqual(
                provenance["config_fingerprints"],
                {"config-a": 1, "config-b": 1},
            )
            self.assertEqual(
                provenance["ai"]["compatibility_status"],
                "incompatible_mixed",
            )
            self.assertTrue(
                any("mixed execution modes" in warning for warning in report["warnings"])
            )
            self.assertTrue(
                any("mixed config fingerprints" in warning for warning in report["warnings"])
            )
            self.assertTrue(
                any("mixed response models" in warning for warning in report["warnings"])
            )
            self.assertIn("Provenance compatibility: incompatible_mixed", performance_report.format_text_report(report))

    def test_invalid_pnl_arithmetic_and_negative_fees_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades_20260811.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "event": "CLOSE",
                        "net_pnl": 10,
                        "gross_pnl": 12,
                        "fees": -2,
                        "r_multiple": 0.5,
                        "ai_decision": "APPROVE",
                    },
                    {
                        "event": "CLOSE",
                        "net_pnl": 50,
                        "gross_pnl": 12,
                        "fees": 2,
                        "r_multiple": 0.5,
                        "ai_decision": "APPROVE",
                    },
                ],
            )

            report = performance_report.build_report([path])
            reasons = report["diagnostics"]["incomplete_reasons"]

            self.assertEqual(report["summary"]["trades"], 0)
            self.assertEqual(reasons["invalid_field:fees_negative"], 1)
            self.assertEqual(reasons["inconsistent_pnl_arithmetic"], 2)

    def test_replayed_close_event_is_deduplicated_by_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades_20260811.jsonl"
            close = {
                "event": "CLOSE",
                "event_id": "close-1",
                "net_pnl": 10,
                "gross_pnl": 12,
                "fees": 2,
                "r_multiple": 0.5,
                "ai_decision": "APPROVE",
            }
            write_jsonl(path, [close, close])

            report = performance_report.build_report([path])

            self.assertEqual(report["diagnostics"]["close_events"], 2)
            self.assertEqual(report["diagnostics"]["complete_close_events"], 1)
            self.assertEqual(report["diagnostics"]["duplicate_close_events"], 1)
            self.assertEqual(report["summary"]["trades"], 1)
            self.assertTrue(
                any("Deduplicated 1" in warning for warning in report["warnings"])
            )

    def test_complete_legacy_records_remain_calculable_but_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades_legacy.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "event": "CLOSE",
                        "net_pnl": 8,
                        "gross_pnl": 10,
                        "fees": 2,
                        "r_multiple": 0.4,
                        "ai_decision": "APPROVE",
                    }
                ],
            )

            report = performance_report.build_report([path])
            provenance = report["provenance"]

            self.assertEqual(report["summary"]["trades"], 1)
            self.assertEqual(report["summary"]["net_pnl"], 8)
            self.assertEqual(provenance["compatibility_status"], "unverified")
            self.assertEqual(provenance["missing_execution_mode"], 1)
            self.assertEqual(provenance["missing_config_fingerprint"], 1)
            self.assertTrue(
                any("legacy P&L remains included" in warning for warning in report["warnings"])
            )

    def test_missing_or_unknown_ai_decision_is_not_inferred_outside_off_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades_20260811.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "event": "CLOSE",
                        "net_pnl": 10,
                        "gross_pnl": 12,
                        "fees": 2,
                        "r_multiple": 0.5,
                        "ai_mode": "shadow",
                    },
                    {
                        "event": "CLOSE",
                        "net_pnl": 10,
                        "gross_pnl": 12,
                        "fees": 2,
                        "r_multiple": 0.5,
                        "ai_mode": "off",
                        "ai_decision": "SKIPPED",
                    },
                ],
            )

            report = performance_report.build_report([path])
            reasons = report["diagnostics"]["incomplete_reasons"]

            self.assertEqual(report["summary"]["trades"], 0)
            self.assertEqual(reasons["missing_field:ai_decision"], 1)
            self.assertEqual(reasons["invalid_field:ai_decision"], 1)

    def test_malformed_non_object_blank_and_non_close_lines_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades_20260811.jsonl"
            write_jsonl(
                path,
                [
                    ["not", "an", "object"],
                    {"event": "AI_REVIEW", "decision": "APPROVE"},
                    {
                        "event": "close",
                        "net_pnl": 0,
                        "gross_pnl": 3,
                        "fees": 3,
                        "r_multiple": 0,
                        "ai_decision": "reject",
                    },
                ],
                raw_lines=["", "{not-json"],
            )

            report = performance_report.build_report([path])
            diagnostics = report["diagnostics"]

            self.assertEqual(diagnostics["total_lines"], 5)
            self.assertEqual(diagnostics["blank_lines"], 1)
            self.assertEqual(diagnostics["malformed_json_lines"], 1)
            self.assertEqual(diagnostics["non_object_lines"], 1)
            self.assertEqual(diagnostics["ignored_non_close_events"], 1)
            self.assertEqual(report["summary"]["breakeven"], 1)
            self.assertEqual(report["summary"]["profit_factor_status"], "undefined_no_profit_or_loss")

    def test_directory_resolution_is_sorted_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "trades_20260810.jsonl"
            second = root / "trades_20260811.jsonl"
            other = root / "unrelated.jsonl"
            write_jsonl(first, [])
            write_jsonl(second, [])
            write_jsonl(other, [])

            paths, errors = performance_report.resolve_input_paths([root, first])

            self.assertEqual(errors, [])
            self.assertEqual(paths, [first.resolve(), second.resolve()])

    def test_json_cli_and_nonzero_exit_without_complete_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "trades_complete.jsonl"
            incomplete = root / "trades_incomplete.jsonl"
            write_jsonl(
                complete,
                [
                    {
                        "event": "CLOSE",
                        "net_pnl": 5,
                        "gross_pnl": 6,
                        "fees": 1,
                        "r_multiple": 0.5,
                        "ai_decision": "APPROVE",
                    }
                ],
            )
            write_jsonl(incomplete, [{"event": "CLOSE", "reason": "LEGACY"}])

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                success_code = performance_report.main([str(complete), "--json"])
            payload = json.loads(stdout.getvalue())
            self.assertEqual(success_code, 0)
            self.assertEqual(payload["summary"]["trades"], 1)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                failure_code = performance_report.main([str(incomplete)])
            self.assertEqual(failure_code, 1)
            self.assertIn("profitability is unknown", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
