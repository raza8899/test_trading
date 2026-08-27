from __future__ import annotations

import inspect
import math
from types import SimpleNamespace
import unittest

from openai import OpenAI
from pydantic import ValidationError

import ai_review


SYSTEM_PROMPT = "Stable reviewer instructions " * 80


def candidate(symbol: str = "INFY", value: float = 1.25) -> dict:
    return {
        "setup": {"side": "LONG", "rvol": value},
        "context": {"signal_age_seconds": 30.0},
        "economics": {"after_cost_payoff": 1.8},
        "test_identity": symbol,
    }


class PromptCacheContractTests(unittest.TestCase):
    def _key(self, *, version: str = ai_review.AI_PROMPT_VERSION) -> str:
        return ai_review.build_prompt_cache_key(
            model="gpt-5.6",
            prompt_version=version,
            system_prompt=SYSTEM_PROMPT,
        )

    def _request(self, payload: dict, **overrides) -> dict:
        values = {
            "model": "gpt-5.6",
            "prompt_version": ai_review.AI_PROMPT_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "reasoning_effort": "low",
            "timeout_seconds": 15.0,
            "prompt_cache_enabled": True,
            "prompt_cache_ttl": "30m",
        }
        values.update(overrides)
        return ai_review.build_openai_review_request(payload, **values)

    def test_cache_key_is_stable_candidate_independent_and_bounded(self) -> None:
        first = self._key()
        second = self._key()

        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 64)
        self.assertNotIn("INFY", first)
        self.assertNotIn("TCS", first)
        # Candidate payload is intentionally not an argument to key creation.
        self.assertNotIn("candidate_payload", inspect.signature(
            ai_review.build_prompt_cache_key
        ).parameters)

    def test_cache_namespace_changes_with_prompt_version_model_or_prompt(self) -> None:
        base = self._key()
        changed_version = self._key(version="nse-orb-review-v6")
        changed_model = ai_review.build_prompt_cache_key(
            model="gpt-5.6-terra",
            prompt_version=ai_review.AI_PROMPT_VERSION,
            system_prompt=SYSTEM_PROMPT,
        )
        changed_prompt = ai_review.build_prompt_cache_key(
            model="gpt-5.6",
            prompt_version=ai_review.AI_PROMPT_VERSION,
            system_prompt=SYSTEM_PROMPT + " changed",
        )

        self.assertEqual(len({base, changed_version, changed_model, changed_prompt}), 4)

    def test_enabled_gpt56_request_has_explicit_static_breakpoint(self) -> None:
        request = self._request(candidate())

        self.assertEqual(
            request["prompt_cache_options"],
            {"mode": "explicit", "ttl": "30m"},
        )
        self.assertLessEqual(len(request["prompt_cache_key"]), 64)
        static_block = request["input"][0]["content"][0]
        self.assertEqual(
            static_block["prompt_cache_breakpoint"],
            {"mode": "explicit"},
        )
        self.assertEqual(request["text"], {"verbosity": "low"})
        self.assertIs(request["text_format"], ai_review.AIDecision)
        self.assertEqual(request["max_output_tokens"], 400)

    def test_disabled_cache_omits_key_options_and_breakpoint(self) -> None:
        request = self._request(candidate(), prompt_cache_enabled=False)

        self.assertNotIn("prompt_cache_key", request)
        self.assertNotIn("prompt_cache_options", request)
        self.assertIsInstance(request["input"][0]["content"], str)
        self.assertNotIn("prompt_cache_breakpoint", str(request["input"]))

    def test_unknown_model_uses_stable_key_without_unsupported_options(self) -> None:
        request = self._request(candidate(), model="future-or-custom-model")

        self.assertIn("prompt_cache_key", request)
        self.assertNotIn("prompt_cache_options", request)
        self.assertIsInstance(request["input"][0]["content"], str)

    def test_invalid_ttl_is_rejected_before_provider_call(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported.*TTL"):
            self._request(candidate(), prompt_cache_ttl="24h")

    def test_candidate_json_is_deterministic_finite_and_last(self) -> None:
        first = {"z": 1, "a": {"y": 2, "x": 3}}
        second = {"a": {"x": 3, "y": 2}, "z": 1}

        self.assertEqual(
            ai_review.canonical_candidate_json(first),
            ai_review.canonical_candidate_json(second),
        )
        request = self._request(first)
        self.assertEqual(request["input"][-1]["role"], "user")
        self.assertTrue(
            request["input"][-1]["content"].endswith(
                ai_review.canonical_candidate_json(first)
            )
        )
        with self.assertRaises(ValueError):
            ai_review.canonical_candidate_json({"bad": math.nan})

    def test_installed_parse_signature_supports_exact_request_parameters(self) -> None:
        parameters = inspect.signature(
            OpenAI(api_key="test-key").responses.parse
        ).parameters
        for name in (
            "prompt_cache_key",
            "prompt_cache_options",
            "text",
            "text_format",
        ):
            self.assertIn(name, parameters)


class CompactDecisionSchemaTests(unittest.TestCase):
    def test_valid_compact_approve_and_reject(self) -> None:
        approve = ai_review.AIDecision(
            decision="APPROVE",
            confidence=86,
            quality_score=82,
            reason="Fresh aligned breakout with no clear exhaustion.",
            risk_flags=["EXTREME_RVOL"],
        )
        reject = ai_review.AIDecision(
            decision="REJECT",
            confidence=91,
            quality_score=58,
            reason="Late extended short carries elevated mean-reversion risk.",
            risk_flags=["VWAP_EXTENSION", "LARGE_DAY_MOVE"],
        )

        self.assertEqual(approve.decision, "APPROVE")
        self.assertEqual(reject.decision, "REJECT")

    def test_score_bounds_are_enforced(self) -> None:
        base = {
            "decision": "REJECT",
            "confidence": 50,
            "quality_score": 50,
            "reason": "Material risk is present.",
            "risk_flags": [],
        }
        for field, value in (
            ("confidence", -1),
            ("confidence", 101),
            ("quality_score", -1),
            ("quality_score", 101),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValidationError):
                    ai_review.AIDecision(**{**base, field: value})

    def test_flags_reason_and_extra_fields_are_constrained(self) -> None:
        base = {
            "decision": "REJECT",
            "confidence": 80,
            "quality_score": 60,
            "reason": "Material risk is present.",
            "risk_flags": [],
        }
        with self.assertRaises(ValidationError):
            ai_review.AIDecision(
                **{
                    **base,
                    "risk_flags": [
                        "CHASE_RISK",
                        "VWAP_EXTENSION",
                        "RSI_STRETCHED",
                        "LARGE_DAY_MOVE",
                        "WIDE_SPREAD",
                    ],
                }
            )
        with self.assertRaises(ValidationError):
            ai_review.AIDecision(**{**base, "risk_flags": ["FREE_PROSE"]})
        with self.assertRaises(ValidationError):
            ai_review.AIDecision(**{**base, "reason": "x" * 251})
        with self.assertRaises(ValidationError):
            ai_review.AIDecision(**{**base, "analysis": "not allowed"})

    def test_provider_schema_excludes_error_but_internal_failure_remains_error(self) -> None:
        with self.assertRaises(ValidationError):
            ai_review.AIDecision(
                decision="ERROR",
                confidence=0,
                quality_score=0,
                reason="Provider failure.",
                risk_flags=[],
            )
        failure = ai_review.AIFailureDecision(
            reason="Provider failure; fail-closed.",
            risk_flags=["AI_FAILURE"],
        )
        self.assertEqual(failure.decision, "ERROR")
        self.assertNotEqual(failure.decision, "APPROVE")


class UsageExtractionTests(unittest.TestCase):
    def test_extracts_nested_dict_cache_and_reasoning_details(self) -> None:
        response = {
            "usage": {
                "input_tokens": 1200,
                "input_tokens_details": {
                    "cached_tokens": 896,
                    "cache_write_tokens": 128,
                },
                "output_tokens": 180,
                "output_tokens_details": {"reasoning_tokens": 96},
                "total_tokens": 1380,
            }
        }

        usage = ai_review.extract_openai_usage(response)

        self.assertEqual(usage.cached_input_tokens, 896)
        self.assertEqual(usage.cache_write_tokens, 128)
        self.assertEqual(usage.reasoning_tokens, 96)
        self.assertAlmostEqual(usage.cache_hit_ratio, 896 / 1200)

    def test_extracts_sdk_style_attributes(self) -> None:
        usage = SimpleNamespace(
            input_tokens=100,
            input_tokens_details=SimpleNamespace(
                cached_tokens=64,
                cache_write_tokens=32,
            ),
            output_tokens=20,
            output_tokens_details=SimpleNamespace(reasoning_tokens=8),
            total_tokens=120,
        )
        extracted = ai_review.extract_openai_usage(
            SimpleNamespace(usage=usage)
        )

        self.assertEqual(extracted.input_tokens, 100)
        self.assertEqual(extracted.cached_input_tokens, 64)
        self.assertEqual(extracted.cache_write_tokens, 32)
        self.assertEqual(extracted.reasoning_tokens, 8)

    def test_missing_details_or_usage_returns_safe_zeros(self) -> None:
        without_details = ai_review.extract_openai_usage(
            {"usage": {"input_tokens": 10, "output_tokens": 2}}
        )
        missing = ai_review.extract_openai_usage({})

        self.assertEqual(without_details.cached_input_tokens, 0)
        self.assertEqual(without_details.cache_write_tokens, 0)
        self.assertEqual(without_details.reasoning_tokens, 0)
        self.assertEqual(missing, ai_review.OpenAIUsage())

    def test_hostile_metrics_property_cannot_raise(self) -> None:
        class HostileResponse:
            @property
            def usage(self):
                raise RuntimeError("telemetry unavailable")

        self.assertEqual(
            ai_review.extract_openai_usage(HostileResponse()),
            ai_review.OpenAIUsage(),
        )

    def test_session_usage_is_weighted_and_counts_failures(self) -> None:
        totals = ai_review.AIUsageTotals()
        totals.record_call(
            ai_review.OpenAIUsage(
                input_tokens=100,
                cached_input_tokens=50,
                output_tokens=20,
                total_tokens=120,
            ),
            successful=True,
        )
        totals.record_call(
            ai_review.OpenAIUsage(
                input_tokens=300,
                cached_input_tokens=0,
                output_tokens=10,
                total_tokens=310,
            ),
            successful=False,
        )

        self.assertEqual(totals.calls, 2)
        self.assertEqual(totals.successful_calls, 1)
        self.assertEqual(totals.failed_calls, 1)
        self.assertEqual(totals.cache_hit_rate, 0.125)
        self.assertEqual(totals.average_output_tokens, 15.0)


if __name__ == "__main__":
    unittest.main()
