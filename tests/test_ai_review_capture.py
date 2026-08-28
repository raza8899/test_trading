from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import ai_review
import ai_review_benchmark
import capture_ai_review_benchmark as capture


SYSTEM_PROMPT = "Stable compact reviewer instructions. " * 80
SCENARIOS = (
    "OBVIOUS_APPROVE",
    "OBVIOUS_REJECT",
    "EXHAUSTED_LONG",
    "EXHAUSTED_SHORT",
    "NEUTRAL_MARKET",
    "EXTREME_RVOL",
    "WIDE_SPREAD",
    "TREND_CONFLICT",
    "BORDERLINE_SETUP",
)


def candidate(index: int) -> dict:
    return {
        "setup": {
            "side": "LONG" if index % 2 else "SHORT",
            "rvol": 1.5 + index,
            "vwap_distance_atr": 0.4,
        },
        "context": {"signal_age_seconds": 20.0 + index},
        "economics": {"after_cost_payoff": 1.6},
    }


def write_cases(path: Path, count: int = len(SCENARIOS)) -> list[dict]:
    rows = []
    for index in range(1, count + 1):
        payload = candidate(index)
        rows.append(
            {
                "schema_version": ai_review_benchmark.CASE_SCHEMA_VERSION,
                "case_id": f"case-{index:02d}",
                "source_kind": (
                    "historical_journal"
                    if index == 1
                    else "synthetic_perturbation"
                ),
                "scenario_tags": [SCENARIOS[(index - 1) % len(SCENARIOS)]],
                "input_sha256": ai_review_benchmark.sha256_json(payload),
                "candidate": payload,
            }
        )
    path.write_text(
        "".join(
            ai_review_benchmark.canonical_json(row) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return rows


def response_for(index: int, *, cached_tokens: int | None = 0):
    input_details = None
    if cached_tokens is not None:
        input_details = SimpleNamespace(
            cached_tokens=cached_tokens,
            cache_write_tokens=1024 if index == 1 else 0,
        )
    usage = SimpleNamespace(
        input_tokens=1400,
        input_tokens_details=input_details,
        output_tokens=120,
        output_tokens_details=SimpleNamespace(reasoning_tokens=60),
        total_tokens=1520,
    )
    return SimpleNamespace(
        id=f"resp-{index}",
        model="gpt-5.6-2026-08-01",
        output_parsed=ai_review.AIDecision(
            decision="APPROVE" if index % 2 else "REJECT",
            confidence=80 + index,
            quality_score=75 + index,
            reason="Compact material decision summary.",
            risk_flags=[] if index % 2 else ["VWAP_EXTENSION"],
        ),
        usage=usage,
    )


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []

    def parse(self, **request):
        self.requests.append(request)
        outcome = self.outcomes[len(self.requests) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.responses = FakeResponses(outcomes)


class LiveCaptureSafetyTests(unittest.TestCase):
    def test_production_prompt_is_read_as_stripped_literal_without_import(self) -> None:
        prompt = capture._literal_string_assignment(
            capture.BOT_SOURCE,
            "AI_SYSTEM_PROMPT",
        )

        self.assertTrue(prompt.startswith("You are the FINAL QUALITY"))
        self.assertTrue(prompt.endswith("structured fields."))

    def test_wrong_confirmation_stops_before_import_file_or_client(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "must-not-be-created" / "capture.jsonl"
            with mock.patch.object(
                capture.importlib,
                "import_module",
                side_effect=AssertionError("must not import"),
            ):
                with self.assertRaises(capture.CaptureSafetyError):
                    capture.capture_live_reviews(
                        cases_path=Path(raw) / "missing.jsonl",
                        output_path=output,
                        confirmation_token="wrong",
                    )
            self.assertFalse(output.exists())
            self.assertFalse(output.parent.exists())

    def test_existing_output_is_refused_before_client_creation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            cases = directory / "cases.jsonl"
            output = directory / "capture.jsonl"
            write_cases(cases)
            output.write_text("owned\n", encoding="utf-8")
            called = False

            def client_factory(**_kwargs):
                nonlocal called
                called = True
                raise AssertionError("client must not be created")

            with self.assertRaises(FileExistsError):
                capture.capture_live_reviews(
                    cases_path=cases,
                    output_path=output,
                    confirmation_token=capture.LIVE_CONFIRMATION_TOKEN,
                    _client_factory=client_factory,
                    _system_prompt=SYSTEM_PROMPT,
                )

            self.assertFalse(called)
            self.assertEqual(output.read_text(encoding="utf-8"), "owned\n")

    def test_default_cli_is_validation_only_and_creates_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            cases = directory / "cases.jsonl"
            output = directory / "must-not-be-created" / "capture.jsonl"
            write_cases(cases)
            stdout = io.StringIO()
            with mock.patch.object(
                capture,
                "_default_client_factory",
                side_effect=AssertionError("must not create OpenAI client"),
            ):
                with redirect_stdout(stdout):
                    status = capture.main(
                        ["--cases", str(cases), "--output", str(output)]
                    )

            self.assertEqual(status, 0)
            self.assertIn("no OpenAI call made", stdout.getvalue())
            self.assertFalse(output.exists())
            self.assertFalse(output.parent.exists())


class LiveCaptureResultTests(unittest.TestCase):
    def test_sequential_capture_is_private_contract_valid_and_cache_measured(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            cases_path = directory / "cases.jsonl"
            output = directory / "new-private-directory" / "capture.jsonl"
            source_rows = write_cases(cases_path)
            client = FakeClient(
                [
                    response_for(1, cached_tokens=0),
                    response_for(2, cached_tokens=1024),
                ]
            )

            with mock.patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-5.6"},
            ):
                summary = capture.capture_live_reviews(
                    cases_path=cases_path,
                    output_path=output,
                    confirmation_token=capture.LIVE_CONFIRMATION_TOKEN,
                    limit=2,
                    _client_factory=lambda **_kwargs: client,
                    _system_prompt=SYSTEM_PROMPT,
                )

            self.assertEqual(summary.api_calls, 2)
            self.assertEqual(summary.successful_calls, 2)
            self.assertEqual(summary.failed_calls, 0)
            self.assertTrue(summary.provider_cache_read_observed)
            self.assertEqual(summary.cached_input_tokens, 1024)
            self.assertEqual(len(client.responses.requests), 2)
            self.assertIn('"rvol":2.5', client.responses.requests[0]["input"][-1]["content"])
            self.assertIn('"rvol":3.5', client.responses.requests[1]["input"][-1]["content"])
            self.assertEqual(
                client.responses.requests[0]["prompt_cache_options"],
                {"mode": "explicit", "ttl": "30m"},
            )

            mode = stat.S_IMODE(output.stat().st_mode)
            self.assertEqual(mode, 0o600)
            self.assertTrue(output.parent.is_dir())
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[1]["usage"]["cached_input_tokens"], 1024)
            self.assertEqual(records[0]["input_sha256"], source_rows[0]["input_sha256"])
            self.assertTrue(all("candidate" not in record for record in records))
            for line_number, record in enumerate(records, start=1):
                ai_review_benchmark.validate_result_payload(
                    record,
                    source=str(output),
                    line_number=line_number,
                )

    def test_provider_errors_are_redacted_and_do_not_stop_later_cases(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            cases_path = directory / "cases.jsonl"
            output = directory / "capture.jsonl"
            write_cases(cases_path)
            secret_echo = "DO_NOT_PERSIST_CANDIDATE_ECHO"
            client = FakeClient(
                [
                    response_for(1),
                    RuntimeError(secret_echo),
                    response_for(3, cached_tokens=None),
                ]
            )

            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                summary = capture.capture_live_reviews(
                    cases_path=cases_path,
                    output_path=output,
                    confirmation_token=capture.LIVE_CONFIRMATION_TOKEN,
                    model="gpt-5.6",
                    limit=3,
                    _client_factory=lambda **_kwargs: client,
                    _system_prompt=SYSTEM_PROMPT,
                )

            self.assertEqual(summary.api_calls, 3)
            self.assertEqual(summary.successful_calls, 2)
            self.assertEqual(summary.failed_calls, 1)
            encoded = output.read_text(encoding="utf-8")
            self.assertNotIn(secret_echo, encoded)
            records = [json.loads(line) for line in encoded.splitlines()]
            self.assertEqual(records[1]["status"], "ERROR")
            self.assertEqual(records[1]["error_type"], "RuntimeError")
            self.assertTrue(
                all(value is None for value in records[1]["response"].values())
            )
            self.assertEqual(records[2]["status"], "OK")
            self.assertIsNone(records[2]["usage"]["cached_input_tokens"])
            self.assertIsNone(records[2]["usage"]["cache_write_tokens"])

    def test_execute_cli_with_wrong_token_returns_before_provider(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            cases = directory / "cases.jsonl"
            output = directory / "capture.jsonl"
            write_cases(cases)
            stderr = io.StringIO()
            with mock.patch.object(
                capture,
                "_default_client_factory",
                side_effect=AssertionError("must not create OpenAI client"),
            ):
                with redirect_stderr(stderr):
                    status = capture.main(
                        [
                            "--cases",
                            str(cases),
                            "--output",
                            str(output),
                            "--confirm",
                            "wrong",
                        ]
                    )

            self.assertEqual(status, 2)
            self.assertIn("Capture refused", stderr.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
