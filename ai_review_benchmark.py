#!/usr/bin/env python3
"""Deterministic, offline comparison of recorded AI trade reviews.

This module intentionally imports only the Python standard library.  It does
not import the trading bot, load ``.env``, contact OpenAI, contact Kite, or use
the network.  Actual API observations are written by a separate, explicitly
invoked capture process and are only *read* here.

Unknown token, latency, and pricing values remain ``None``.  In particular,
legacy journals did not record prompt-cache or reasoning-token details, so the
benchmark never guesses those values or silently treats them as zero.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = PROJECT_ROOT / "benchmarks" / "ai_review_cases.jsonl"
DEFAULT_LEGACY_RESULTS_PATH = (
    PROJECT_ROOT / "benchmarks" / "ai_review_legacy_results.jsonl"
)

CASE_SCHEMA_VERSION = "ai-review-case-v1"
RESULT_SCHEMA_VERSION = "ai-review-result-v1"
RESULT_EVENT = "AI_REVIEW_BENCHMARK_RESULT"
REPORT_SCHEMA_VERSION = "ai-review-benchmark-report-v1"
PRICING_SCHEMA_VERSION = "ai-review-pricing-v1"

REQUIRED_SCENARIOS = frozenset(
    {
        "OBVIOUS_APPROVE",
        "OBVIOUS_REJECT",
        "EXHAUSTED_LONG",
        "EXHAUSTED_SHORT",
        "NEUTRAL_MARKET",
        "EXTREME_RVOL",
        "WIDE_SPREAD",
        "TREND_CONFLICT",
        "BORDERLINE_SETUP",
    }
)
VALID_VARIANTS = frozenset({"legacy", "compact"})
VALID_PROVENANCE = frozenset({"recorded_api", "synthetic_test"})
VALID_STATUSES = frozenset({"OK", "ERROR"})
VALID_DECISIONS = frozenset({"APPROVE", "REJECT"})
VALID_SOURCE_KINDS = frozenset(
    {"historical_journal", "synthetic_perturbation"}
)
MAX_CANDIDATE_JSON_BYTES = 32 * 1024

_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_CANDIDATE_KEYS = frozenset(
    {
        "symbol",
        "token",
        "signal_at",
        "signal_bar_closed_at",
        "quote_observed_at",
        "setup_detected_at",
        "last_validated_at",
        "config_fingerprint",
        "account_state",
        "trade_id",
        "idea_id",
        "tradingsymbol",
        "instrument_token",
        "api_key",
        "api_secret",
        "access_token",
        "refresh_token",
        "authorization",
        "password",
        "credential",
        "secret",
        "cookie",
        "timestamp",
        "datetime",
        "date",
        "today",
        "current_time",
        "now",
    }
)
_FORBIDDEN_CANDIDATE_KEY_FRAGMENTS = (
    "api_key",
    "api_secret",
    "access_token",
    "refresh_token",
    "password",
    "credential",
)

_CASE_KEYS = frozenset(
    {
        "schema_version",
        "case_id",
        "source_kind",
        "scenario_tags",
        "input_sha256",
        "candidate",
    }
)
_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "event",
        "sequence",
        "case_id",
        "input_sha256",
        "variant",
        "provenance",
        "status",
        "request",
        "response",
        "usage",
        "latency_ms",
        "error_type",
    }
)
_REQUEST_KEYS = frozenset(
    {
        "prompt_version",
        "prompt_sha256",
        "schema_sha256",
        "requested_model",
        "reasoning_effort",
        "max_output_tokens",
    }
)
_RESPONSE_KEYS = frozenset(
    {
        "actual_model",
        "decision",
        "confidence",
        "quality_score",
        "reason",
        "risk_flags",
    }
)
_USAGE_KEYS = frozenset(
    {
        "input_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    }
)
_PRICING_KEYS = frozenset(
    {
        "schema_version",
        "currency",
        "source_url",
        "effective_at",
        "rates_per_million_tokens",
    }
)
_PRICING_RATE_KEYS = frozenset(
    {"uncached_input", "cached_input", "cache_write_input", "output"}
)


class BenchmarkDataError(ValueError):
    """A benchmark artifact violated its strict offline contract."""


class _DuplicateJSONKey(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    source_kind: str
    scenario_tags: tuple[str, ...]
    input_sha256: str
    candidate: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkResult:
    sequence: int
    case_id: str
    input_sha256: str
    variant: str
    provenance: str
    status: str
    request: dict[str, Any]
    response: dict[str, Any]
    usage: dict[str, int | None]
    latency_ms: int | None
    error_type: str | None

    @property
    def decision(self) -> str | None:
        return self.response["decision"]

    @property
    def confidence(self) -> int | None:
        return self.response["confidence"]

    @property
    def quality_score(self) -> int | None:
        return self.response["quality_score"]

    @property
    def reason(self) -> str | None:
        return self.response["reason"]


@dataclass(frozen=True)
class Pricing:
    currency: str
    source_url: str
    effective_at: str | None
    rates_per_million_tokens: dict[str, float | None]

    @property
    def complete(self) -> bool:
        return self.effective_at is not None and all(
            self.rates_per_million_tokens[name] is not None
            for name in _PRICING_RATE_KEYS
        )


def canonical_json(value: Any) -> str:
    """Serialize a JSON value exactly as input hashes are defined."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _load_json(raw: str, *, source: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise BenchmarkDataError(f"{source}: invalid JSON: {exc}") from exc


def _load_jsonl(path: Path) -> list[tuple[dict[str, Any], int]]:
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchmarkDataError(f"{path}: cannot read artifact: {exc}") from exc

    records: list[tuple[dict[str, Any], int]] = []
    for line_number, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            continue
        source = f"{path}:{line_number}"
        payload = _load_json(raw, source=source)
        if not isinstance(payload, dict):
            raise BenchmarkDataError(f"{source}: record must be a JSON object")
        records.append((payload, line_number))
    if not records:
        raise BenchmarkDataError(f"{path}: artifact contains no records")
    return records


def _exact_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    *,
    source: str,
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise BenchmarkDataError(f"{source}: invalid fields ({'; '.join(details)})")


def _text(value: Any, *, source: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkDataError(f"{source}: expected non-empty text")
    return value.strip()


def _sha256(value: Any, *, source: str, nullable: bool = False) -> str | None:
    text = _text(value, source=source, nullable=nullable)
    if text is None:
        return None
    if not _SHA256_RE.fullmatch(text):
        raise BenchmarkDataError(f"{source}: expected lowercase SHA-256")
    return text


def _integer(
    value: Any,
    *,
    source: str,
    nullable: bool = False,
    minimum: int = 0,
    maximum: int | None = None,
) -> int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkDataError(f"{source}: expected integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "" if maximum is None else f"..{maximum}"
        raise BenchmarkDataError(
            f"{source}: integer must be in range {minimum}{upper}"
        )
    return value


def _nullable_rate(value: Any, *, source: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkDataError(f"{source}: expected non-negative number or null")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise BenchmarkDataError(f"{source}: expected non-negative finite rate")
    return number


def _validate_finite_json(value: Any, *, source: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise BenchmarkDataError(f"{source}: contains non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, source=f"{source}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise BenchmarkDataError(f"{source}: contains non-text key")
            normalized_key = key.strip().lower()
            if (
                normalized_key in _FORBIDDEN_CANDIDATE_KEYS
                or normalized_key.endswith(
                    ("_timestamp", "_datetime", "_date", "_at")
                )
                or any(
                    fragment in normalized_key
                    for fragment in _FORBIDDEN_CANDIDATE_KEY_FRAGMENTS
                )
            ):
                raise BenchmarkDataError(
                    f"{source}: identity/time key {key!r} is forbidden"
                )
            _validate_finite_json(item, source=f"{source}.{key}")
        return
    raise BenchmarkDataError(
        f"{source}: unsupported JSON value {type(value).__name__}"
    )


def validate_case_payload(
    payload: Mapping[str, Any],
    *,
    source: str,
    line_number: int,
) -> BenchmarkCase:
    location = f"{source}:{line_number}"
    _exact_keys(payload, _CASE_KEYS, source=location)
    if payload["schema_version"] != CASE_SCHEMA_VERSION:
        raise BenchmarkDataError(f"{location}: unsupported case schema version")

    case_id = _text(payload["case_id"], source=f"{location}.case_id")
    assert case_id is not None
    if not _CASE_ID_RE.fullmatch(case_id):
        raise BenchmarkDataError(f"{location}.case_id: invalid identifier")

    source_kind = _text(
        payload["source_kind"], source=f"{location}.source_kind"
    )
    assert source_kind is not None
    if source_kind not in VALID_SOURCE_KINDS:
        raise BenchmarkDataError(f"{location}.source_kind: unsupported value")

    raw_tags = payload["scenario_tags"]
    if not isinstance(raw_tags, list) or not raw_tags:
        raise BenchmarkDataError(f"{location}.scenario_tags: expected non-empty list")
    tags: list[str] = []
    for index, raw_tag in enumerate(raw_tags):
        tag = _text(raw_tag, source=f"{location}.scenario_tags[{index}]")
        assert tag is not None
        if tag not in REQUIRED_SCENARIOS:
            raise BenchmarkDataError(
                f"{location}.scenario_tags[{index}]: unsupported scenario {tag!r}"
            )
        tags.append(tag)
    if len(tags) != len(set(tags)):
        raise BenchmarkDataError(f"{location}.scenario_tags: duplicate scenario")

    candidate = payload["candidate"]
    if not isinstance(candidate, dict):
        raise BenchmarkDataError(f"{location}.candidate: expected object")
    _exact_keys(
        candidate,
        frozenset({"setup", "context", "economics"}),
        source=f"{location}.candidate",
    )
    for section in ("setup", "context", "economics"):
        if not isinstance(candidate[section], dict) or not candidate[section]:
            raise BenchmarkDataError(
                f"{location}.candidate.{section}: expected non-empty object"
            )
    _validate_finite_json(candidate, source=f"{location}.candidate")
    candidate_size = len(canonical_json(candidate).encode("utf-8"))
    if candidate_size > MAX_CANDIDATE_JSON_BYTES:
        raise BenchmarkDataError(
            f"{location}.candidate: canonical JSON exceeds "
            f"{MAX_CANDIDATE_JSON_BYTES} bytes"
        )

    input_sha256 = _sha256(
        payload["input_sha256"], source=f"{location}.input_sha256"
    )
    assert input_sha256 is not None
    calculated = sha256_json(candidate)
    if input_sha256 != calculated:
        raise BenchmarkDataError(
            f"{location}.input_sha256: hash mismatch; calculated {calculated}"
        )
    return BenchmarkCase(
        case_id=case_id,
        source_kind=source_kind,
        scenario_tags=tuple(tags),
        input_sha256=input_sha256,
        candidate=candidate,
    )


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[BenchmarkCase]:
    cases = [
        validate_case_payload(payload, source=str(path), line_number=line_number)
        for payload, line_number in _load_jsonl(path)
    ]
    by_id = {case.case_id: case for case in cases}
    if len(by_id) != len(cases):
        raise BenchmarkDataError(f"{path}: duplicate case_id")
    covered = frozenset(
        tag for case in cases for tag in case.scenario_tags
    )
    missing = sorted(REQUIRED_SCENARIOS - covered)
    if missing:
        raise BenchmarkDataError(
            f"{path}: missing required scenarios: {', '.join(missing)}"
        )
    if not any(case.source_kind == "historical_journal" for case in cases):
        raise BenchmarkDataError(f"{path}: no historical cases")
    return sorted(cases, key=lambda item: item.case_id)


def _validate_request(value: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkDataError(f"{source}: expected object")
    _exact_keys(value, _REQUEST_KEYS, source=source)
    prompt_version = _text(
        value["prompt_version"], source=f"{source}.prompt_version"
    )
    assert prompt_version is not None
    return {
        "prompt_version": prompt_version,
        "prompt_sha256": _sha256(
            value["prompt_sha256"],
            source=f"{source}.prompt_sha256",
            nullable=True,
        ),
        "schema_sha256": _sha256(
            value["schema_sha256"],
            source=f"{source}.schema_sha256",
            nullable=True,
        ),
        "requested_model": _text(
            value["requested_model"],
            source=f"{source}.requested_model",
            nullable=True,
        ),
        "reasoning_effort": _text(
            value["reasoning_effort"],
            source=f"{source}.reasoning_effort",
            nullable=True,
        ),
        "max_output_tokens": _integer(
            value["max_output_tokens"],
            source=f"{source}.max_output_tokens",
            nullable=True,
            minimum=1,
        ),
    }


def _validate_response(
    value: Any,
    *,
    status: str,
    variant: str,
    source: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkDataError(f"{source}: expected object")
    _exact_keys(value, _RESPONSE_KEYS, source=source)
    if status == "ERROR":
        if any(value[name] is not None for name in _RESPONSE_KEYS):
            raise BenchmarkDataError(
                f"{source}: ERROR result must have null response fields"
            )
        return dict(value)

    actual_model = _text(value["actual_model"], source=f"{source}.actual_model")
    decision = _text(value["decision"], source=f"{source}.decision")
    assert actual_model is not None and decision is not None
    if decision not in VALID_DECISIONS:
        raise BenchmarkDataError(f"{source}.decision: unsupported decision")
    confidence = _integer(
        value["confidence"],
        source=f"{source}.confidence",
        maximum=100,
    )
    quality_score = _integer(
        value["quality_score"],
        source=f"{source}.quality_score",
        maximum=100,
    )
    reason = _text(value["reason"], source=f"{source}.reason")
    raw_flags = value["risk_flags"]
    if not isinstance(raw_flags, list):
        raise BenchmarkDataError(f"{source}.risk_flags: expected list")
    flags: list[str] = []
    for index, raw_flag in enumerate(raw_flags):
        flag = _text(raw_flag, source=f"{source}.risk_flags[{index}]")
        assert flag is not None
        flags.append(flag)
    if len(flags) != len(set(flags)):
        raise BenchmarkDataError(f"{source}.risk_flags: duplicate flag")
    if variant == "compact" and len(flags) > 4:
        raise BenchmarkDataError(f"{source}.risk_flags: compact result exceeds four")
    return {
        "actual_model": actual_model,
        "decision": decision,
        "confidence": confidence,
        "quality_score": quality_score,
        "reason": reason,
        "risk_flags": flags,
    }


def _validate_usage(value: Any, *, source: str) -> dict[str, int | None]:
    if not isinstance(value, dict):
        raise BenchmarkDataError(f"{source}: expected object")
    _exact_keys(value, _USAGE_KEYS, source=source)
    usage = {
        name: _integer(
            value[name],
            source=f"{source}.{name}",
            nullable=True,
        )
        for name in sorted(_USAGE_KEYS)
    }
    input_tokens = usage["input_tokens"]
    cached_tokens = usage["cached_input_tokens"]
    write_tokens = usage["cache_write_tokens"]
    output_tokens = usage["output_tokens"]
    reasoning_tokens = usage["reasoning_tokens"]
    if input_tokens is not None:
        if cached_tokens is not None and cached_tokens > input_tokens:
            raise BenchmarkDataError(f"{source}: cached tokens exceed input tokens")
        if write_tokens is not None and write_tokens > input_tokens:
            raise BenchmarkDataError(f"{source}: cache-write tokens exceed input tokens")
        if (
            cached_tokens is not None
            and write_tokens is not None
            and cached_tokens + write_tokens > input_tokens
        ):
            raise BenchmarkDataError(
                f"{source}: cached plus cache-write tokens exceed input tokens"
            )
    if (
        output_tokens is not None
        and reasoning_tokens is not None
        and reasoning_tokens > output_tokens
    ):
        raise BenchmarkDataError(f"{source}: reasoning tokens exceed output tokens")
    return usage


def validate_result_payload(
    payload: Mapping[str, Any],
    *,
    source: str,
    line_number: int,
) -> BenchmarkResult:
    location = f"{source}:{line_number}"
    _exact_keys(payload, _RESULT_KEYS, source=location)
    if payload["schema_version"] != RESULT_SCHEMA_VERSION:
        raise BenchmarkDataError(f"{location}: unsupported result schema version")
    if payload["event"] != RESULT_EVENT:
        raise BenchmarkDataError(f"{location}: unsupported event")
    sequence = _integer(
        payload["sequence"], source=f"{location}.sequence", minimum=1
    )
    assert sequence is not None
    case_id = _text(payload["case_id"], source=f"{location}.case_id")
    assert case_id is not None
    if not _CASE_ID_RE.fullmatch(case_id):
        raise BenchmarkDataError(f"{location}.case_id: invalid identifier")
    input_sha256 = _sha256(
        payload["input_sha256"], source=f"{location}.input_sha256"
    )
    assert input_sha256 is not None
    variant = _text(payload["variant"], source=f"{location}.variant")
    provenance = _text(payload["provenance"], source=f"{location}.provenance")
    status = _text(payload["status"], source=f"{location}.status")
    assert variant is not None and provenance is not None and status is not None
    if variant not in VALID_VARIANTS:
        raise BenchmarkDataError(f"{location}.variant: unsupported value")
    if provenance not in VALID_PROVENANCE:
        raise BenchmarkDataError(f"{location}.provenance: unsupported value")
    if status not in VALID_STATUSES:
        raise BenchmarkDataError(f"{location}.status: unsupported value")

    request = _validate_request(payload["request"], source=f"{location}.request")
    response = _validate_response(
        payload["response"],
        status=status,
        variant=variant,
        source=f"{location}.response",
    )
    usage = _validate_usage(payload["usage"], source=f"{location}.usage")
    latency_ms = _integer(
        payload["latency_ms"],
        source=f"{location}.latency_ms",
        nullable=True,
    )
    error_type = _text(
        payload["error_type"],
        source=f"{location}.error_type",
        nullable=True,
    )
    if status == "OK" and error_type is not None:
        raise BenchmarkDataError(f"{location}: OK result cannot have error_type")
    if status == "ERROR" and error_type is None:
        raise BenchmarkDataError(f"{location}: ERROR result requires error_type")
    return BenchmarkResult(
        sequence=sequence,
        case_id=case_id,
        input_sha256=input_sha256,
        variant=variant,
        provenance=provenance,
        status=status,
        request=request,
        response=response,
        usage=usage,
        latency_ms=latency_ms,
        error_type=error_type,
    )


def load_results(
    path: Path,
    *,
    expected_variant: str | None = None,
) -> list[BenchmarkResult]:
    results = [
        validate_result_payload(payload, source=str(path), line_number=line_number)
        for payload, line_number in _load_jsonl(path)
    ]
    if expected_variant is not None and expected_variant not in VALID_VARIANTS:
        raise ValueError(f"invalid expected variant {expected_variant!r}")
    if expected_variant is not None:
        wrong = sorted(
            result.case_id
            for result in results
            if result.variant != expected_variant
        )
        if wrong:
            raise BenchmarkDataError(
                f"{path}: expected {expected_variant} variant for: {', '.join(wrong)}"
            )
    case_ids = [result.case_id for result in results]
    if len(case_ids) != len(set(case_ids)):
        raise BenchmarkDataError(f"{path}: duplicate case result")
    sequences = [result.sequence for result in results]
    if len(sequences) != len(set(sequences)):
        raise BenchmarkDataError(f"{path}: duplicate sequence")
    return sorted(results, key=lambda item: item.case_id)


def load_pricing(path: Path) -> Pricing:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchmarkDataError(f"{path}: cannot read pricing: {exc}") from exc
    payload = _load_json(raw, source=str(path))
    if not isinstance(payload, dict):
        raise BenchmarkDataError(f"{path}: pricing must be a JSON object")
    _exact_keys(payload, _PRICING_KEYS, source=str(path))
    if payload["schema_version"] != PRICING_SCHEMA_VERSION:
        raise BenchmarkDataError(f"{path}: unsupported pricing schema version")
    currency = _text(payload["currency"], source=f"{path}.currency")
    source_url = _text(payload["source_url"], source=f"{path}.source_url")
    effective_at = _text(
        payload["effective_at"], source=f"{path}.effective_at", nullable=True
    )
    assert currency is not None and source_url is not None
    raw_rates = payload["rates_per_million_tokens"]
    if not isinstance(raw_rates, dict):
        raise BenchmarkDataError(f"{path}.rates_per_million_tokens: expected object")
    _exact_keys(raw_rates, _PRICING_RATE_KEYS, source=f"{path}.rates")
    rates = {
        name: _nullable_rate(raw_rates[name], source=f"{path}.rates.{name}")
        for name in sorted(_PRICING_RATE_KEYS)
    }
    return Pricing(
        currency=currency,
        source_url=source_url,
        effective_at=effective_at,
        rates_per_million_tokens=rates,
    )


def _validate_result_case_links(
    cases: Sequence[BenchmarkCase],
    results: Sequence[BenchmarkResult],
) -> None:
    by_id = {case.case_id: case for case in cases}
    for result in results:
        case = by_id.get(result.case_id)
        if case is None:
            raise BenchmarkDataError(
                f"result references unknown case_id {result.case_id!r}"
            )
        if result.input_sha256 != case.input_sha256:
            raise BenchmarkDataError(
                f"result {result.case_id!r} input hash does not match case"
            )


def _complete_numeric_summary(
    results: Sequence[BenchmarkResult],
    getter: Callable[[BenchmarkResult], int | float | None],
) -> dict[str, Any]:
    values = [value for result in results if (value := getter(result)) is not None]
    complete = bool(results) and len(values) == len(results)
    total = math.fsum(float(value) for value in values) if complete else None
    return {
        "expected_records": len(results),
        "reported_records": len(values),
        "total": total,
        "average": (total / len(values)) if total is not None else None,
    }


def _estimated_result_cost(
    result: BenchmarkResult,
    pricing: Pricing | None,
) -> float | None:
    if pricing is None or not pricing.complete:
        return None
    usage = result.usage
    required = (
        usage["input_tokens"],
        usage["cached_input_tokens"],
        usage["cache_write_tokens"],
        usage["output_tokens"],
    )
    if any(value is None for value in required):
        return None
    input_tokens, cached_tokens, write_tokens, output_tokens = required
    assert input_tokens is not None
    assert cached_tokens is not None
    assert write_tokens is not None
    assert output_tokens is not None
    uncached_tokens = input_tokens - cached_tokens - write_tokens
    if uncached_tokens < 0:
        return None
    rates = pricing.rates_per_million_tokens
    assert all(rates[name] is not None for name in _PRICING_RATE_KEYS)
    return math.fsum(
        (
            uncached_tokens * float(rates["uncached_input"]),
            cached_tokens * float(rates["cached_input"]),
            write_tokens * float(rates["cache_write_input"]),
            output_tokens * float(rates["output"]),
        )
    ) / 1_000_000


def summarize_variant(
    variant: str,
    cases: Sequence[BenchmarkCase],
    results: Sequence[BenchmarkResult],
    pricing: Pricing | None,
) -> dict[str, Any]:
    case_ids = {case.case_id for case in cases}
    result_ids = {result.case_id for result in results}
    successful = [result for result in results if result.status == "OK"]
    failed = [result for result in results if result.status == "ERROR"]
    reason_chars = _complete_numeric_summary(
        successful,
        lambda result: len(result.reason) if result.reason is not None else None,
    )
    reason_words = _complete_numeric_summary(
        successful,
        lambda result: len(result.reason.split()) if result.reason is not None else None,
    )
    usage = {
        name: _complete_numeric_summary(
            results, lambda result, field=name: result.usage[field]
        )
        for name in sorted(_USAGE_KEYS)
    }
    latency = _complete_numeric_summary(
        results, lambda result: result.latency_ms
    )

    input_complete = usage["input_tokens"]["total"]
    cached_complete = usage["cached_input_tokens"]["total"]
    cache_hit_ratio = None
    if (
        input_complete is not None
        and cached_complete is not None
        and input_complete > 0
    ):
        cache_hit_ratio = cached_complete / input_complete

    costs = [_estimated_result_cost(result, pricing) for result in results]
    cost_complete = bool(results) and all(value is not None for value in costs)
    total_cost = (
        math.fsum(float(value) for value in costs if value is not None)
        if cost_complete
        else None
    )
    if pricing is None:
        cost_status = "NOT_CONFIGURED"
    elif not pricing.complete:
        cost_status = "PRICING_INCOMPLETE"
    elif not cost_complete:
        cost_status = "USAGE_INCOMPLETE"
    else:
        cost_status = "ESTIMATE_AVAILABLE"

    return {
        "variant": variant,
        "case_count": len(cases),
        "result_records": len(results),
        "case_coverage_rate": (len(result_ids) / len(cases)) if cases else None,
        "missing_case_ids": sorted(case_ids - result_ids),
        "successful_calls": len(successful),
        "failed_calls": len(failed),
        "parse_success_rate": (
            len(successful) / len(results) if results else None
        ),
        "decision_counts": {
            decision: sum(result.decision == decision for result in successful)
            for decision in sorted(VALID_DECISIONS)
        },
        "reason_characters": reason_chars,
        "reason_words": reason_words,
        "usage": usage,
        "cache_hit_ratio": cache_hit_ratio,
        "latency_ms": latency,
        "estimated_cost": {
            "status": cost_status,
            "currency": pricing.currency if pricing is not None else None,
            "reported_records": sum(value is not None for value in costs),
            "total": total_cost,
            "average": (total_cost / len(results)) if total_cost is not None else None,
        },
        "prompt_versions": sorted(
            {str(result.request["prompt_version"]) for result in results}
        ),
        "actual_models": sorted(
            {
                str(result.response["actual_model"])
                for result in successful
                if result.response["actual_model"] is not None
            }
        ),
    }


def _effective_gate(
    result: BenchmarkResult,
    *,
    confidence_threshold: int,
    quality_threshold: int,
) -> str:
    if (
        result.status == "OK"
        and result.decision == "APPROVE"
        and result.confidence is not None
        and result.confidence >= confidence_threshold
        and result.quality_score is not None
        and result.quality_score >= quality_threshold
    ):
        return "APPROVE"
    return "BLOCK"


def _delta_summary(values: Sequence[int]) -> dict[str, Any]:
    return {
        "paired_successful_records": len(values),
        "mean_absolute_delta": (
            math.fsum(values) / len(values) if values else None
        ),
        "max_absolute_delta": max(values) if values else None,
    }


def compare_variants(
    legacy: Sequence[BenchmarkResult],
    compact: Sequence[BenchmarkResult],
    *,
    confidence_threshold: int,
    quality_threshold: int,
) -> dict[str, Any]:
    legacy_by_id = {result.case_id: result for result in legacy}
    compact_by_id = {result.case_id: result for result in compact}
    paired_ids = sorted(set(legacy_by_id) & set(compact_by_id))
    successful_pairs = [
        (legacy_by_id[case_id], compact_by_id[case_id])
        for case_id in paired_ids
        if legacy_by_id[case_id].status == "OK"
        and compact_by_id[case_id].status == "OK"
    ]

    decision_matches = sum(
        before.decision == after.decision for before, after in successful_pairs
    )
    effective_matches = sum(
        _effective_gate(
            before,
            confidence_threshold=confidence_threshold,
            quality_threshold=quality_threshold,
        )
        == _effective_gate(
            after,
            confidence_threshold=confidence_threshold,
            quality_threshold=quality_threshold,
        )
        for before, after in (
            (legacy_by_id[case_id], compact_by_id[case_id])
            for case_id in paired_ids
        )
    )
    confidence_deltas = [
        abs(int(before.confidence) - int(after.confidence))
        for before, after in successful_pairs
        if before.confidence is not None and after.confidence is not None
    ]
    quality_deltas = [
        abs(int(before.quality_score) - int(after.quality_score))
        for before, after in successful_pairs
        if before.quality_score is not None and after.quality_score is not None
    ]
    return {
        "paired_case_count": len(paired_ids),
        "paired_case_ids": paired_ids,
        "paired_successful_count": len(successful_pairs),
        "raw_decision_agreement_rate": (
            decision_matches / len(successful_pairs) if successful_pairs else None
        ),
        "effective_gate_agreement_rate": (
            effective_matches / len(paired_ids) if paired_ids else None
        ),
        "confidence": _delta_summary(confidence_deltas),
        "quality_score": _delta_summary(quality_deltas),
        "legacy_without_compact": sorted(set(legacy_by_id) - set(compact_by_id)),
        "compact_without_legacy": sorted(set(compact_by_id) - set(legacy_by_id)),
    }


def build_report(
    cases: Sequence[BenchmarkCase],
    legacy: Sequence[BenchmarkResult],
    compact: Sequence[BenchmarkResult],
    *,
    confidence_threshold: int = 80,
    quality_threshold: int = 75,
    pricing: Pricing | None = None,
) -> dict[str, Any]:
    for name, value in (
        ("confidence_threshold", confidence_threshold),
        ("quality_threshold", quality_threshold),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
            raise BenchmarkDataError(f"{name} must be an integer in 0..100")
    _validate_result_case_links(cases, [*legacy, *compact])
    if any(result.variant != "legacy" for result in legacy):
        raise BenchmarkDataError("legacy result collection contains another variant")
    if any(result.variant != "compact" for result in compact):
        raise BenchmarkDataError("compact result collection contains another variant")

    case_ids = {case.case_id for case in cases}
    legacy_ids = {result.case_id for result in legacy}
    compact_ids = {result.case_id for result in compact}
    missing_legacy = sorted(case_ids - legacy_ids)
    missing_compact = sorted(case_ids - compact_ids)
    if not compact:
        status = "INSUFFICIENT_COMPACT_RESULTS"
    elif missing_legacy and missing_compact:
        status = "INSUFFICIENT_PAIRED_RESULTS"
    elif missing_compact:
        status = "INSUFFICIENT_COMPACT_RESULTS"
    elif missing_legacy:
        status = "INSUFFICIENT_LEGACY_RESULTS"
    else:
        status = "COMPARISON_COMPLETE"
    scenario_counts = {
        scenario: sum(scenario in case.scenario_tags for case in cases)
        for scenario in sorted(REQUIRED_SCENARIOS)
    }
    warnings: list[str] = []
    if missing_compact:
        warnings.append(
            "No complete compact capture exists for every benchmark case; "
            "new-model token, cache, latency, cost, and agreement claims are unavailable."
        )
    if missing_legacy:
        warnings.append(
            "No recorded legacy result exists for every benchmark case; only exact "
            "case/hash pairs may be used for before/after agreement claims."
        )
    if any(case.source_kind == "synthetic_perturbation" for case in cases):
        warnings.append(
            "Synthetic perturbation cases test explicit edge conditions and are not "
            "historical API observations."
        )
    if pricing is None or not pricing.complete:
        warnings.append(
            "Estimated cost is unavailable until complete, externally verified pricing "
            "is supplied."
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "gate_thresholds": {
            "confidence": confidence_threshold,
            "quality_score": quality_threshold,
        },
        "cases": {
            "count": len(cases),
            "historical_count": sum(
                case.source_kind == "historical_journal" for case in cases
            ),
            "synthetic_perturbation_count": sum(
                case.source_kind == "synthetic_perturbation" for case in cases
            ),
            "scenario_counts": scenario_counts,
            "missing_legacy_case_ids": missing_legacy,
            "missing_compact_case_ids": missing_compact,
        },
        "legacy": summarize_variant("legacy", cases, legacy, pricing),
        "compact": summarize_variant("compact", cases, compact, pricing),
        "comparison": compare_variants(
            legacy,
            compact,
            confidence_threshold=confidence_threshold,
            quality_threshold=quality_threshold,
        ),
        "pricing": {
            "configured": pricing is not None,
            "complete": pricing.complete if pricing is not None else False,
            "currency": pricing.currency if pricing is not None else None,
            "source_url": pricing.source_url if pricing is not None else None,
            "effective_at": pricing.effective_at if pricing is not None else None,
        },
        "warnings": warnings,
    }


def format_text_report(report: Mapping[str, Any]) -> str:
    legacy = report["legacy"]
    compact = report["compact"]
    comparison = report["comparison"]

    def display(value: Any, *, percent: bool = False) -> str:
        if value is None:
            return "N/A"
        if percent:
            return f"{float(value) * 100:.1f}%"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    return "\n".join(
        [
            f"AI review benchmark: {report['status']}",
            f"Cases: {report['cases']['count']} "
            f"({report['cases']['historical_count']} historical, "
            f"{report['cases']['synthetic_perturbation_count']} synthetic edge)",
            f"Legacy: results={legacy['result_records']}, "
            f"parse={display(legacy['parse_success_rate'], percent=True)}, "
            f"avg_output={display(legacy['usage']['output_tokens']['average'])}, "
            f"cache_hit={display(legacy['cache_hit_ratio'], percent=True)}",
            f"Compact: results={compact['result_records']}, "
            f"parse={display(compact['parse_success_rate'], percent=True)}, "
            f"avg_output={display(compact['usage']['output_tokens']['average'])}, "
            f"cache_hit={display(compact['cache_hit_ratio'], percent=True)}",
            f"Paired cases: {comparison['paired_case_count']}",
            "Raw decision agreement: "
            + display(comparison["raw_decision_agreement_rate"], percent=True),
            "Effective gate agreement: "
            + display(comparison["effective_gate_agreement_rate"], percent=True),
            *[f"WARNING: {warning}" for warning in report["warnings"]],
        ]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare recorded legacy and compact AI-review results without "
            "calling OpenAI, Kite, or any network service."
        )
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--legacy-results", type=Path, default=DEFAULT_LEGACY_RESULTS_PATH
    )
    parser.add_argument(
        "--compact-results",
        type=Path,
        help="explicitly captured compact-result JSONL; omitted by default",
    )
    parser.add_argument(
        "--pricing",
        type=Path,
        help="optional externally verified token-pricing JSON",
    )
    parser.add_argument("--gate-confidence", type=int, default=80)
    parser.add_argument("--gate-quality", type=int, default=75)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cases = load_cases(args.cases)
        legacy = load_results(args.legacy_results, expected_variant="legacy")
        compact = (
            load_results(args.compact_results, expected_variant="compact")
            if args.compact_results is not None
            else []
        )
        pricing = load_pricing(args.pricing) if args.pricing is not None else None
        report = build_report(
            cases,
            legacy,
            compact,
            confidence_threshold=args.gate_confidence,
            quality_threshold=args.gate_quality,
            pricing=pricing,
        )
    except BenchmarkDataError as exc:
        print(f"BENCHMARK_DATA_ERROR: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(format_text_report(report))
    return 0 if report["status"] == "COMPARISON_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
