#!/usr/bin/env python3
"""Generate structured, research-only AI ideas from journalled candidates.

This worker is intentionally outside the market execution loop.  It reads
immutable ``AI_CANDIDATE`` events, removes symbol/date/composite-score fields,
and asks OpenAI to rank only the supplied candidates.  Its output can be joined
back by ``idea_id`` for counterfactual analysis; it never imports Kite, sends an
order, changes state, or grants the model risk authority.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Literal, Sequence
import uuid

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
ENV_FILE = PROJECT_ROOT / ".env"
PROMPT_VERSION = "nse-shadow-trade-ideas-v1"
SCHEMA_VERSION = "shadow-trade-idea-v1"

load_dotenv(ENV_FILE)


class ShadowTradeIdea(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["shadow-trade-idea-v1"]
    candidate_id: str = Field(pattern=r"^C\d{3}$")
    verdict: Literal["TAKE", "PASS", "ABSTAIN"]
    confidence_band: Literal["LOW", "MEDIUM", "HIGH"]
    quality_band: Literal["WEAK", "MARGINAL", "GOOD", "STRONG"]
    primary_reason: Literal[
        "TREND_VOLUME_ALIGNMENT",
        "EXECUTION_QUALITY",
        "OVEREXTENDED",
        "WEAK_CONFIRMATION",
        "REGIME_CONFLICT",
        "COST_PAYOFF",
        "INCONSISTENT_INPUT",
    ]
    risk_flags: list[
        Literal[
            "LATE_BREAKOUT",
            "HIGH_SPREAD",
            "LOW_RVOL",
            "RSI_EXTREME",
            "VWAP_EXTENSION",
            "REGIME_MISMATCH",
            "LOW_NET_PAYOFF",
            "NONE",
        ]
    ] = Field(max_length=4)
    evidence_fields: list[str] = Field(min_length=1, max_length=5)
    rationale: str = Field(min_length=1, max_length=180)


class ShadowTradeIdeaBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["shadow-trade-idea-v1"]
    market_summary: str = Field(min_length=1, max_length=300)
    ideas: list[ShadowTradeIdea] = Field(min_length=1, max_length=25)


SYSTEM_PROMPT = """
You are a shadow-only classifier for a deterministic NSE cash intraday
research system.

You have no authority over orders, side, entry, stop, target, quantity, or
risk. Rank and classify only the anonymous, point-in-time numeric candidates
supplied. Use no news, company knowledge, fundamentals, or facts outside the
JSON. Never propose different trade parameters.

TAKE means the already-valid candidate has strong joint confirmation and is
not overextended. PASS means it has conflicting evidence or no clear
incremental quality. ABSTAIN means the input is missing, stale, non-finite, or
internally inconsistent. Confidence is a label-confidence band, not a win
probability or expected return. Cite only fields present in the candidate.
""".strip()


@dataclass(frozen=True)
class CandidateEnvelope:
    idea_id: str
    symbol: str
    side: str
    source: str
    line_number: int
    config_fingerprint: str
    anonymous_payload: dict[str, Any]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _finite_json(value: Any, *, path: str = "candidate") -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [
            _finite_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            str(key): _finite_json(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    raise ValueError(f"{path} contains unsupported {type(value).__name__}")


def anonymize_candidate(candidate: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    """Remove identity, timestamps and composite scores from model input."""
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be an object")
    setup = candidate.get("setup")
    economics = candidate.get("economics")
    capacity = candidate.get("capacity")
    if not all(isinstance(value, dict) for value in (setup, economics, capacity)):
        raise ValueError("candidate is missing setup/economics/capacity")

    omitted_setup = {
        "symbol",
        "token",
        "signal_at",
        "technical_score",
        "stock_in_play_score",
    }
    clean_setup = {
        key: value
        for key, value in setup.items()
        if key not in omitted_setup
    }
    # Capacity values reveal whether other positions already won portfolio
    # competition; give the model only the candidate's frozen economics.
    payload = {
        "candidate_id": candidate_id,
        "side": clean_setup.pop("side", ""),
        "features": clean_setup,
        "economics": economics,
    }
    return _finite_json(payload)


def resolve_paths(raw_paths: Sequence[str] | None) -> list[Path]:
    values = raw_paths or [str(LOG_DIR / "trades_*.jsonl")]
    found: dict[str, Path] = {}
    for raw in values:
        path = Path(raw).expanduser()
        matches: Iterable[str | Path]
        if glob.has_magic(str(path)):
            matches = glob.glob(str(path))
        elif path.is_dir():
            matches = path.glob("trades_*.jsonl")
        else:
            matches = [path]
        for match in matches:
            candidate = Path(match)
            if candidate.is_file():
                resolved = candidate.resolve()
                found[str(resolved)] = resolved
    return [found[key] for key in sorted(found)]


def load_candidates(paths: Sequence[Path], limit: int) -> list[CandidateEnvelope]:
    if limit < 1:
        raise ValueError("limit must be positive")
    selected: dict[str, CandidateEnvelope] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict) or payload.get("event") != "AI_CANDIDATE":
                    continue
                idea_id = str(payload.get("idea_id") or "").strip()
                symbol = str(payload.get("symbol") or "").strip().upper()
                side = str(payload.get("side") or "").strip().upper()
                candidate = payload.get("candidate")
                if not idea_id or not symbol or side not in {"LONG", "SHORT"}:
                    continue
                candidate_id = f"C{len(selected) + 1:03d}"
                anonymous = anonymize_candidate(candidate, candidate_id)
                selected.setdefault(
                    idea_id,
                    CandidateEnvelope(
                        idea_id=idea_id,
                        symbol=symbol,
                        side=side,
                        source=str(path),
                        line_number=line_number,
                        config_fingerprint=str(
                            payload.get("config_fingerprint") or ""
                        ),
                        anonymous_payload=anonymous,
                    ),
                )
                if len(selected) >= limit:
                    return list(selected.values())
    return list(selected.values())


def validate_batch(
    result: ShadowTradeIdeaBatch,
    candidates: Sequence[CandidateEnvelope],
) -> dict[str, CandidateEnvelope]:
    by_id = {
        candidate.anonymous_payload["candidate_id"]: candidate
        for candidate in candidates
    }
    seen: set[str] = set()
    for idea in result.ideas:
        if idea.candidate_id not in by_id:
            raise ValueError(f"model invented candidate {idea.candidate_id}")
        if idea.candidate_id in seen:
            raise ValueError(f"duplicate model idea {idea.candidate_id}")
        seen.add(idea.candidate_id)
    if seen != set(by_id):
        missing = ", ".join(sorted(set(by_id) - seen))
        raise ValueError(f"model omitted candidates: {missing}")
    return by_id


def _append_private_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(payload) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            fd = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)


def review_candidates(
    candidates: Sequence[CandidateEnvelope],
    *,
    output_path: Path,
) -> int:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    model = os.getenv("OPENAI_MODEL", "gpt-5.6").strip()
    timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
    effort = os.getenv("OPENAI_REASONING_EFFORT", "low").strip().lower()
    client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    request_payload = {
        "schema_version": SCHEMA_VERSION,
        "candidates": [candidate.anonymous_payload for candidate in candidates],
    }
    attempt_id = uuid.uuid4().hex
    started = time.monotonic()
    base_event: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "event": "AI_IDEA_REVIEW",
        "attempt_id": attempt_id,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "schema_sha256": sha256_json(ShadowTradeIdeaBatch.model_json_schema()),
        "input_sha256": sha256_json(request_payload),
        "requested_model": model,
        "reasoning_effort": effort,
        "idea_ids": [candidate.idea_id for candidate in candidates],
    }
    try:
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": canonical_json(request_payload)},
            ],
            text_format=ShadowTradeIdeaBatch,
            reasoning={"effort": effort},
            max_output_tokens=1600,
            store=False,
            metadata={
                "workflow": "nse_shadow_trade_ideas",
                "prompt_version": PROMPT_VERSION,
            },
            timeout=timeout,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no parsed idea batch")
        mapping = validate_batch(parsed, candidates)
        usage = getattr(response, "usage", None)
        ideas = []
        for idea in parsed.ideas:
            source = mapping[idea.candidate_id]
            ideas.append(
                {
                    "idea_id": source.idea_id,
                    "symbol": source.symbol,
                    "source_side": source.side,
                    **idea.model_dump(),
                }
            )
        event = {
            **base_event,
            "status": "OK",
            "actual_model": str(response.model),
            "response_id": str(response.id),
            "latency_ms": int((time.monotonic() - started) * 1000),
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            "market_summary": parsed.market_summary,
            "ideas": ideas,
            "output_sha256": sha256_json(parsed.model_dump()),
        }
        _append_private_jsonl(output_path, event)
        return len(ideas)
    except Exception as exc:
        _append_private_jsonl(
            output_path,
            {
                **base_event,
                "status": "ERROR",
                "latency_ms": int((time.monotonic() - started) * 1000),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    configured_limit = int(os.getenv("AI_IDEA_MAX_CANDIDATES", "8"))
    parser = argparse.ArgumentParser(
        description="Generate research-only AI ideas from AI_CANDIDATE journals."
    )
    parser.add_argument("paths", nargs="*", help="JSONL file, directory or glob")
    parser.add_argument(
        "--limit",
        type=int,
        default=configured_limit,
        help="maximum candidates (default: AI_IDEA_MAX_CANDIDATES or 8)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LOG_DIR / "ai_ideas.jsonl",
        help="private JSONL output path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print anonymous inputs without calling OpenAI",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = resolve_paths(args.paths)
    candidates = load_candidates(paths, args.limit)
    if not candidates:
        print("No valid AI_CANDIDATE records found.")
        return 1
    if args.dry_run:
        print(
            json.dumps(
                [candidate.anonymous_payload for candidate in candidates],
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    count = review_candidates(candidates, output_path=args.output)
    print(f"Wrote {count} structured shadow idea(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
