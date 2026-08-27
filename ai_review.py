"""Pure OpenAI trade-review schemas, request construction, and telemetry.

This module imports neither Kite nor the OpenAI client, loads no environment
file, and performs no network calls. It has no trading or execution authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


AI_REVIEW_WORKFLOW = "nse_intraday_review"
AI_PROMPT_VERSION = "nse-orb-review-v5-compact"
AI_MAX_OUTPUT_TOKENS = 400
AI_OUTPUT_VERBOSITY = "low"

# GPT-5.6 currently supports explicit prefix breakpoints and only the 30m TTL.
# Explicit-only mode avoids paying to write the changing candidate suffix.
PROMPT_CACHE_MODE = "explicit"
SUPPORTED_PROMPT_CACHE_TTLS = frozenset({"30m"})
PROMPT_CACHE_OPTIONS_MODEL_FAMILIES = frozenset({"gpt-5.6"})


RiskFlag: TypeAlias = Literal[
    "LATE_BREAKOUT",
    "CHASE_RISK",
    "VWAP_EXTENSION",
    "BREAKOUT_EXTENSION",
    "RSI_STRETCHED",
    "LARGE_DAY_MOVE",
    "EXTREME_RVOL",
    "WEAK_RVOL",
    "WEAK_CANDLE",
    "MARKET_CONFLICT",
    "TREND_CONFLICT",
    "WIDE_SPREAD",
    "LOW_LIQUIDITY",
    "GAP_RISK",
    "VOLATILITY_RISK",
    "INCONSISTENT_INPUT",
    "ECONOMICS_CONCERN",
]

OperationalRiskFlag: TypeAlias = Literal[
    "AI_FAILURE",
    "AI_UNAVAILABLE",
    "AI_BUDGET_SKIPPED",
]


class AIDecision(BaseModel):
    """Provider-visible Structured Output for one final trade review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["APPROVE", "REJECT"]
    confidence: int = Field(ge=0, le=100)
    quality_score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=250)
    risk_flags: list[RiskFlag] = Field(default_factory=list, max_length=4)


class AIFailureDecision(BaseModel):
    """Internal fail-closed sentinel; this schema is never sent to OpenAI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["ERROR"] = "ERROR"
    confidence: Literal[0] = 0
    quality_score: Literal[0] = 0
    reason: str = Field(min_length=1, max_length=250)
    risk_flags: list[OperationalRiskFlag] = Field(min_length=1, max_length=1)


AIReviewOutcome: TypeAlias = AIDecision | AIFailureDecision


def canonical_candidate_json(candidate_payload: Mapping[str, Any]) -> str:
    """Serialize only the changing suffix in a stable, finite form."""
    return json.dumps(
        candidate_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_prompt_cache_key(
    *,
    model: str,
    prompt_version: str,
    system_prompt: str,
) -> str:
    """Return a candidate-independent namespace of at most 64 characters."""
    readable = "".join(
        character if character.isalnum() else "-"
        for character in prompt_version.lower()
    )
    readable = "-".join(part for part in readable.split("-") if part)
    namespace = json.dumps(
        {
            "model": model.strip(),
            "prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "prompt_version": prompt_version,
            "schema_sha256": hashlib.sha256(
                json.dumps(
                    AIDecision.model_json_schema(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "workflow": AI_REVIEW_WORKFLOW,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]
    prefix = "nse-review-"
    max_readable = 64 - len(prefix) - len(digest) - 1
    key = f"{prefix}{readable[:max_readable]}-{digest}"
    if len(key) > 64:
        raise AssertionError("prompt cache key exceeds API limit")
    return key


def model_supports_prompt_cache_options(model: str) -> bool:
    """Use a small explicit capability registry; unknown models use defaults."""
    normalized = model.strip().lower()
    return any(
        normalized == family or normalized.startswith(f"{family}-")
        for family in PROMPT_CACHE_OPTIONS_MODEL_FAMILIES
    )


def build_openai_review_request(
    candidate_payload: Mapping[str, Any],
    *,
    model: str,
    prompt_version: str,
    system_prompt: str,
    reasoning_effort: str,
    timeout_seconds: float,
    prompt_cache_enabled: bool,
    prompt_cache_ttl: str,
) -> dict[str, Any]:
    """Build kwargs for ``client.responses.parse`` without making a call."""
    candidate_json = canonical_candidate_json(candidate_payload)
    cache_options_supported = model_supports_prompt_cache_options(model)
    system_content: str | list[dict[str, Any]] = system_prompt

    request: dict[str, Any] = {
        "model": model,
        "input": [],
        "text_format": AIDecision,
        "text": {"verbosity": AI_OUTPUT_VERBOSITY},
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": AI_MAX_OUTPUT_TOKENS,
        "store": False,
        "metadata": {
            "workflow": AI_REVIEW_WORKFLOW,
            "prompt_version": prompt_version,
        },
        "timeout": timeout_seconds,
    }

    if prompt_cache_enabled:
        if prompt_cache_ttl not in SUPPORTED_PROMPT_CACHE_TTLS:
            raise ValueError(
                f"unsupported OpenAI prompt-cache TTL: {prompt_cache_ttl}"
            )
        request["prompt_cache_key"] = build_prompt_cache_key(
            model=model,
            prompt_version=prompt_version,
            system_prompt=system_prompt,
        )
        if cache_options_supported:
            system_content = [
                {
                    "type": "input_text",
                    "text": system_prompt,
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ]
            request["prompt_cache_options"] = {
                "mode": PROMPT_CACHE_MODE,
                "ttl": prompt_cache_ttl,
            }

    request["input"] = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": "Review this candidate JSON:\n" + candidate_json,
        },
    ]
    return request


def _member(value: Any, name: str) -> Any:
    try:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)
    except Exception:
        return None


def _token_count(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, number)


@dataclass(frozen=True)
class OpenAIUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def cache_hit_ratio(self) -> float:
        if self.input_tokens <= 0:
            return 0.0
        return self.cached_input_tokens / self.input_tokens

    def journal_fields(self) -> dict[str, int | float]:
        return {
            "ai_input_tokens": self.input_tokens,
            "ai_output_tokens": self.output_tokens,
            "ai_total_tokens": self.total_tokens,
            "ai_cached_input_tokens": self.cached_input_tokens,
            "ai_cache_write_tokens": self.cache_write_tokens,
            "ai_reasoning_tokens": self.reasoning_tokens,
            "ai_cache_hit_ratio": self.cache_hit_ratio,
        }


def extract_openai_usage(response: Any) -> OpenAIUsage:
    """Extract SDK/dict usage defensively; telemetry must never break review."""
    try:
        usage = _member(response, "usage")
        if usage is None:
            return OpenAIUsage()
        input_details = _member(usage, "input_tokens_details")
        output_details = _member(usage, "output_tokens_details")
        cached = _member(input_details, "cached_tokens")
        cache_write = _member(input_details, "cache_write_tokens")
        reasoning = _member(output_details, "reasoning_tokens")
        return OpenAIUsage(
            input_tokens=_token_count(_member(usage, "input_tokens")),
            output_tokens=_token_count(_member(usage, "output_tokens")),
            total_tokens=_token_count(_member(usage, "total_tokens")),
            cached_input_tokens=_token_count(
                cached
                if cached is not None
                else _member(usage, "cached_input_tokens")
            ),
            cache_write_tokens=_token_count(
                cache_write
                if cache_write is not None
                else _member(usage, "cache_write_tokens")
            ),
            reasoning_tokens=_token_count(
                reasoning
                if reasoning is not None
                else _member(usage, "reasoning_tokens")
            ),
        )
    except Exception:
        return OpenAIUsage()


@dataclass
class AIUsageTotals:
    calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    duplicate_reviews_suppressed: int = 0

    def record_call(self, usage: OpenAIUsage, *, successful: bool) -> None:
        self.calls += 1
        self.successful_calls += int(successful)
        self.failed_calls += int(not successful)
        self.input_tokens += usage.input_tokens
        self.cached_input_tokens += usage.cached_input_tokens
        self.cache_write_tokens += usage.cache_write_tokens
        self.output_tokens += usage.output_tokens
        self.reasoning_tokens += usage.reasoning_tokens
        self.total_tokens += usage.total_tokens

    @property
    def cache_hit_rate(self) -> float:
        if self.input_tokens <= 0:
            return 0.0
        return self.cached_input_tokens / self.input_tokens

    @property
    def average_output_tokens(self) -> float:
        if self.calls <= 0:
            return 0.0
        return self.output_tokens / self.calls

    def journal_fields(self) -> dict[str, int | float]:
        return {
            "ai_calls": self.calls,
            "ai_successful_calls": self.successful_calls,
            "ai_failed_calls": self.failed_calls,
            "ai_input_tokens": self.input_tokens,
            "ai_cached_input_tokens": self.cached_input_tokens,
            "ai_cache_write_tokens": self.cache_write_tokens,
            "ai_output_tokens": self.output_tokens,
            "ai_reasoning_tokens": self.reasoning_tokens,
            "ai_total_tokens": self.total_tokens,
            "ai_cache_hit_rate": self.cache_hit_rate,
            "ai_average_output_tokens": self.average_output_tokens,
            "ai_duplicate_reviews_suppressed": (
                self.duplicate_reviews_suppressed
            ),
        }

    def concise_log_line(self) -> str:
        return (
            "AI USAGE: "
            f"calls={self.calls} successes={self.successful_calls} "
            f"errors={self.failed_calls} input={self.input_tokens} "
            f"cached_input={self.cached_input_tokens} "
            f"cache_write={self.cache_write_tokens} "
            f"cache_hit={self.cache_hit_rate:.1%} "
            f"output={self.output_tokens} "
            f"reasoning={self.reasoning_tokens} "
            f"total={self.total_tokens} "
            f"avg_output={self.average_output_tokens:.1f} "
            f"duplicates={self.duplicate_reviews_suppressed}"
        )
