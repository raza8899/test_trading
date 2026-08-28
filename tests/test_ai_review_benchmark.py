from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import ast
import io
import json
from pathlib import Path
import tempfile
import unittest

import ai_review_benchmark as benchmark


def case_payload(case: benchmark.BenchmarkCase) -> dict:
    return {
        "schema_version": benchmark.CASE_SCHEMA_VERSION,
        "case_id": case.case_id,
        "source_kind": case.source_kind,
        "scenario_tags": list(case.scenario_tags),
        "input_sha256": case.input_sha256,
        "candidate": case.candidate,
    }


def compact_payload(
    case: benchmark.BenchmarkCase,
    sequence: int,
    *,
    decision: str,
    confidence: int,
    quality_score: int,
    status: str = "OK",
) -> dict:
    successful = status == "OK"
    return {
        "schema_version": benchmark.RESULT_SCHEMA_VERSION,
        "event": benchmark.RESULT_EVENT,
        "sequence": sequence,
        "case_id": case.case_id,
        "input_sha256": case.input_sha256,
        "variant": "compact",
        "provenance": "synthetic_test",
        "status": status,
        "request": {
            "prompt_version": "compact-test-v1",
            "prompt_sha256": "a" * 64,
            "schema_sha256": "b" * 64,
            "requested_model": "test-model",
            "reasoning_effort": "low",
            "max_output_tokens": 350,
        },
        "response": {
            "actual_model": "test-model-2026" if successful else None,
            "decision": decision if successful else None,
            "confidence": confidence if successful else None,
            "quality_score": quality_score if successful else None,
            "reason": "Concise recorded decision summary." if successful else None,
            "risk_flags": ["TEST_FLAG"] if successful else None,
        },
        "usage": {
            "input_tokens": 1000 if successful else None,
            "cached_input_tokens": 800 if successful else None,
            "cache_write_tokens": 0 if successful else None,
            "output_tokens": 100 if successful else None,
            "reasoning_tokens": 20 if successful else None,
            "total_tokens": 1100 if successful else None,
        },
        "latency_ms": 100 if successful else None,
        "error_type": None if successful else "StructuredOutputError",
    }


def result_payload(result: benchmark.BenchmarkResult) -> dict:
    return {
        "schema_version": benchmark.RESULT_SCHEMA_VERSION,
        "event": benchmark.RESULT_EVENT,
        "sequence": result.sequence,
        "case_id": result.case_id,
        "input_sha256": result.input_sha256,
        "variant": result.variant,
        "provenance": result.provenance,
        "status": result.status,
        "request": result.request,
        "response": result.response,
        "usage": result.usage,
        "latency_ms": result.latency_ms,
        "error_type": result.error_type,
    }


def compact_results(
    cases: list[benchmark.BenchmarkCase],
    legacy: list[benchmark.BenchmarkResult],
) -> list[benchmark.BenchmarkResult]:
    legacy_by_id = {result.case_id: result for result in legacy}
    values: list[benchmark.BenchmarkResult] = []
    for sequence, case in enumerate(cases, start=1):
        prior = legacy_by_id.get(case.case_id)
        decision = prior.decision if prior is not None else "REJECT"
        confidence = prior.confidence if prior is not None else 95
        quality = prior.quality_score if prior is not None else 30
        assert decision is not None and confidence is not None and quality is not None
        values.append(
            benchmark.validate_result_payload(
                compact_payload(
                    case,
                    sequence,
                    decision=decision,
                    confidence=confidence,
                    quality_score=quality,
                ),
                source="synthetic-test",
                line_number=sequence,
            )
        )
    return values


def complete_legacy_results(
    cases: list[benchmark.BenchmarkCase],
    legacy: list[benchmark.BenchmarkResult],
) -> list[benchmark.BenchmarkResult]:
    values = list(legacy)
    trend_case = next(case for case in cases if case.case_id == "trend-conflict")
    payload = compact_payload(
        trend_case,
        sequence=max(result.sequence for result in legacy) + 1,
        decision="REJECT",
        confidence=95,
        quality_score=30,
    )
    payload["variant"] = "legacy"
    values.append(
        benchmark.validate_result_payload(
            payload,
            source="synthetic-test",
            line_number=1,
        )
    )
    return sorted(values, key=lambda item: item.case_id)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(benchmark.canonical_json(record) + "\n" for record in records),
        encoding="utf-8",
    )


class AIReviewBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = benchmark.load_cases()
        cls.legacy = benchmark.load_results(
            benchmark.DEFAULT_LEGACY_RESULTS_PATH,
            expected_variant="legacy",
        )

    def test_checked_in_cases_are_deidentified_hashed_and_cover_scenarios(self) -> None:
        self.assertEqual(len(self.cases), 8)
        self.assertEqual(
            sum(case.source_kind == "historical_journal" for case in self.cases),
            7,
        )
        self.assertEqual(
            sum(case.source_kind == "synthetic_perturbation" for case in self.cases),
            1,
        )
        covered = {
            tag for case in self.cases for tag in case.scenario_tags
        }
        self.assertEqual(covered, benchmark.REQUIRED_SCENARIOS)
        for case in self.cases:
            self.assertEqual(case.input_sha256, benchmark.sha256_json(case.candidate))
            rendered = benchmark.canonical_json(case.candidate)
            self.assertNotIn('"symbol"', rendered)
            self.assertNotIn('"token"', rendered)
            self.assertNotIn("NEPHROPLUS", rendered)
            self.assertNotIn("2026-08", rendered)

    def test_module_has_no_duplicate_literal_dictionary_keys(self) -> None:
        source = Path(benchmark.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            self.assertEqual(
                len(keys),
                len(set(keys)),
                f"duplicate literal key at line {node.lineno}",
            )

    def test_checked_in_legacy_metrics_keep_unknown_cache_fields_null(self) -> None:
        report = benchmark.build_report(self.cases, self.legacy, [])

        self.assertEqual(report["status"], "INSUFFICIENT_COMPACT_RESULTS")
        self.assertEqual(report["legacy"]["result_records"], 7)
        self.assertEqual(report["legacy"]["parse_success_rate"], 1.0)
        self.assertAlmostEqual(
            report["legacy"]["usage"]["output_tokens"]["average"],
            2903 / 7,
        )
        self.assertIsNone(report["legacy"]["cache_hit_ratio"])
        self.assertIsNone(
            report["legacy"]["usage"]["cached_input_tokens"]["total"]
        )
        self.assertIsNone(
            report["legacy"]["usage"]["reasoning_tokens"]["average"]
        )
        self.assertIsNone(report["compact"]["parse_success_rate"])
        self.assertIsNone(report["compact"]["latency_ms"]["average"])

    def test_default_cli_reports_insufficient_without_fabricating_compact_data(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = benchmark.main(["--json"])

        self.assertEqual(exit_code, 2)
        self.assertEqual(stderr.getvalue(), "")
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["status"], "INSUFFICIENT_COMPACT_RESULTS")
        self.assertEqual(report["compact"]["result_records"], 0)
        self.assertIsNone(report["compact"]["cache_hit_ratio"])
        self.assertIsNone(
            report["compact"]["usage"]["output_tokens"]["average"]
        )
        self.assertIsNone(report["compact"]["estimated_cost"]["total"])
        self.assertIsNone(report["comparison"]["raw_decision_agreement_rate"])

    def test_case_validation_rejects_identity_fields_and_hash_mismatch(self) -> None:
        payload = case_payload(self.cases[0])
        payload["candidate"] = json.loads(json.dumps(payload["candidate"]))
        payload["candidate"]["setup"]["symbol"] = "SECRET"
        payload["input_sha256"] = benchmark.sha256_json(payload["candidate"])

        with self.assertRaisesRegex(benchmark.BenchmarkDataError, "forbidden"):
            benchmark.validate_case_payload(
                payload,
                source="test",
                line_number=1,
            )

        payload = case_payload(self.cases[0])
        payload["input_sha256"] = "0" * 64
        with self.assertRaisesRegex(benchmark.BenchmarkDataError, "hash mismatch"):
            benchmark.validate_case_payload(
                payload,
                source="test",
                line_number=1,
            )

    def test_case_validation_rejects_sensitive_time_and_oversized_data(self) -> None:
        for forbidden_key in (
            "OPENAI_API_KEY",
            "broker_access_token",
            "observed_timestamp",
        ):
            with self.subTest(forbidden_key=forbidden_key):
                payload = case_payload(self.cases[0])
                payload["candidate"] = json.loads(
                    json.dumps(payload["candidate"])
                )
                payload["candidate"]["context"][forbidden_key] = "sensitive"
                payload["input_sha256"] = benchmark.sha256_json(
                    payload["candidate"]
                )
                with self.assertRaisesRegex(
                    benchmark.BenchmarkDataError,
                    "forbidden",
                ):
                    benchmark.validate_case_payload(
                        payload,
                        source="test",
                        line_number=1,
                    )

        payload = case_payload(self.cases[0])
        payload["candidate"] = json.loads(json.dumps(payload["candidate"]))
        payload["candidate"]["context"]["padding"] = (
            "x" * benchmark.MAX_CANDIDATE_JSON_BYTES
        )
        payload["input_sha256"] = benchmark.sha256_json(payload["candidate"])
        with self.assertRaisesRegex(benchmark.BenchmarkDataError, "exceeds"):
            benchmark.validate_case_payload(
                payload,
                source="test",
                line_number=1,
            )

    def test_loaders_reject_duplicate_json_keys_and_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.jsonl"
            duplicate.write_text(
                '{"schema_version":"ai-review-case-v1",'
                '"schema_version":"ai-review-case-v1"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(benchmark.BenchmarkDataError, "duplicate"):
                benchmark.load_cases(duplicate)

            unknown = case_payload(self.cases[0])
            unknown["unexpected"] = True
            path = root / "unknown.jsonl"
            write_jsonl(path, [unknown])
            with self.assertRaisesRegex(benchmark.BenchmarkDataError, "unknown"):
                benchmark.load_cases(path)

    def test_result_must_join_case_by_exact_input_hash(self) -> None:
        invalid = replace(self.legacy[0], input_sha256="0" * 64)
        with self.assertRaisesRegex(benchmark.BenchmarkDataError, "hash"):
            benchmark.build_report(self.cases, [invalid], [])

    def test_complete_comparison_reports_agreement_cache_latency_and_cost(self) -> None:
        compact = compact_results(self.cases, self.legacy)
        complete_legacy = complete_legacy_results(self.cases, self.legacy)
        pricing = benchmark.Pricing(
            currency="USD",
            source_url="https://example.test/effective-pricing",
            effective_at="2026-08-27",
            rates_per_million_tokens={
                "uncached_input": 1.0,
                "cached_input": 1.0,
                "cache_write_input": 1.0,
                "output": 1.0,
            },
        )

        report = benchmark.build_report(
            self.cases,
            complete_legacy,
            compact,
            pricing=pricing,
        )

        self.assertEqual(report["status"], "COMPARISON_COMPLETE")
        self.assertEqual(report["compact"]["result_records"], 8)
        self.assertEqual(report["compact"]["parse_success_rate"], 1.0)
        self.assertEqual(report["compact"]["cache_hit_ratio"], 0.8)
        self.assertEqual(
            report["compact"]["usage"]["output_tokens"]["average"], 100
        )
        self.assertEqual(report["compact"]["latency_ms"]["average"], 100)
        self.assertEqual(
            report["comparison"]["raw_decision_agreement_rate"], 1.0
        )
        self.assertEqual(
            report["comparison"]["effective_gate_agreement_rate"], 1.0
        )
        self.assertEqual(report["comparison"]["compact_without_legacy"], [])
        self.assertEqual(
            report["compact"]["estimated_cost"]["status"],
            "ESTIMATE_AVAILABLE",
        )
        self.assertAlmostEqual(
            report["compact"]["estimated_cost"]["total"], 0.0088
        )
        # Legacy cache-read/write fields were not captured, so even complete
        # pricing must not invent its historical cost.
        self.assertEqual(
            report["legacy"]["estimated_cost"]["status"], "USAGE_INCOMPLETE"
        )
        self.assertIsNone(report["legacy"]["estimated_cost"]["total"])

    def test_full_compact_capture_still_refuses_missing_legacy_comparator(self) -> None:
        compact = compact_results(self.cases, self.legacy)
        report = benchmark.build_report(self.cases, self.legacy, compact)

        self.assertEqual(report["status"], "INSUFFICIENT_LEGACY_RESULTS")
        self.assertEqual(
            report["cases"]["missing_legacy_case_ids"], ["trend-conflict"]
        )
        self.assertEqual(report["cases"]["missing_compact_case_ids"], [])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compact.jsonl"
            write_jsonl(path, [result_payload(result) for result in compact])
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = benchmark.main(
                    ["--compact-results", str(path), "--json"]
                )
        self.assertEqual(exit_code, 2)
        self.assertEqual(
            json.loads(stdout.getvalue())["status"],
            "INSUFFICIENT_LEGACY_RESULTS",
        )

    def test_decision_and_effective_gate_agreement_are_distinct(self) -> None:
        compact = compact_results(self.cases, self.legacy)
        complete_legacy = complete_legacy_results(self.cases, self.legacy)
        index = next(
            i for i, result in enumerate(compact) if result.case_id == "obvious-approve"
        )
        changed_response = dict(compact[index].response)
        changed_response.update(
            {"decision": "REJECT", "confidence": 80, "quality_score": 60}
        )
        compact[index] = replace(compact[index], response=changed_response)

        report = benchmark.build_report(self.cases, complete_legacy, compact)

        self.assertAlmostEqual(
            report["comparison"]["raw_decision_agreement_rate"], 7 / 8
        )
        self.assertAlmostEqual(
            report["comparison"]["effective_gate_agreement_rate"], 7 / 8
        )
        self.assertEqual(
            report["comparison"]["quality_score"]["max_absolute_delta"], 30
        )

    def test_recorded_parse_failure_is_counted_and_keeps_usage_unknown(self) -> None:
        compact = compact_results(self.cases, self.legacy)
        complete_legacy = complete_legacy_results(self.cases, self.legacy)
        index = next(
            i for i, result in enumerate(compact) if result.case_id == "wide-spread"
        )
        failed_payload = compact_payload(
            self.cases[index],
            compact[index].sequence,
            decision="REJECT",
            confidence=0,
            quality_score=0,
            status="ERROR",
        )
        # Case sorting and result sorting are both by case_id, so use the
        # failed result's own case identity rather than relying on source order.
        failed_case = next(case for case in self.cases if case.case_id == "wide-spread")
        failed_payload["case_id"] = failed_case.case_id
        failed_payload["input_sha256"] = failed_case.input_sha256
        compact[index] = benchmark.validate_result_payload(
            failed_payload,
            source="synthetic-test",
            line_number=1,
        )

        report = benchmark.build_report(self.cases, complete_legacy, compact)

        self.assertEqual(report["status"], "COMPARISON_COMPLETE")
        self.assertEqual(report["compact"]["failed_calls"], 1)
        self.assertEqual(report["compact"]["parse_success_rate"], 7 / 8)
        self.assertIsNone(
            report["compact"]["usage"]["output_tokens"]["average"]
        )
        self.assertEqual(
            report["compact"]["usage"]["output_tokens"]["reported_records"], 7
        )

    def test_example_pricing_is_source_linked_but_intentionally_incomplete(self) -> None:
        pricing = benchmark.load_pricing(
            benchmark.PROJECT_ROOT / "benchmarks" / "openai_pricing.example.json"
        )
        self.assertFalse(pricing.complete)
        self.assertTrue(pricing.source_url.startswith("https://developers.openai.com/"))
        self.assertIsNone(pricing.effective_at)
        self.assertTrue(
            all(value is None for value in pricing.rates_per_million_tokens.values())
        )

        report = benchmark.build_report(
            self.cases,
            self.legacy,
            [],
            pricing=pricing,
        )
        self.assertEqual(
            report["legacy"]["estimated_cost"]["status"],
            "PRICING_INCOMPLETE",
        )
        self.assertIsNone(report["legacy"]["estimated_cost"]["total"])

    def test_compact_schema_rejects_more_than_four_flags(self) -> None:
        payload = compact_payload(
            self.cases[0],
            1,
            decision="APPROVE",
            confidence=90,
            quality_score=90,
        )
        payload["response"]["risk_flags"] = [f"FLAG_{index}" for index in range(5)]
        with self.assertRaisesRegex(benchmark.BenchmarkDataError, "exceeds four"):
            benchmark.validate_result_payload(
                payload,
                source="test",
                line_number=1,
            )


if __name__ == "__main__":
    unittest.main()
