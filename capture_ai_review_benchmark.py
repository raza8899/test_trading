#!/usr/bin/env python3
"""Explicitly opt-in capture of compact OpenAI trade-review benchmark results.

Importing this module, or running it without the exact live confirmation,
does not import the OpenAI SDK, load ``.env``, initialize Kite, or make a
network call.  The production system prompt is read as a literal from
``bot.py``; that module is never imported or executed.

The output is one private JSONL record per case.  It intentionally excludes
the candidate payload and records prompt-cache evidence only from provider
usage counters.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
BOT_SOURCE = PROJECT_ROOT / "bot.py"
LIVE_CONFIRMATION_TOKEN = "OPENAI_COST_ACCEPTED"
MAX_CAPTURE_CASES = 25


class CaptureSafetyError(RuntimeError):
    """Raised before any provider call when an opt-in invariant is absent."""


@dataclass(frozen=True)
class CaptureSummary:
    output_path: Path
    cases: int
    api_calls: int
    successful_calls: int
    failed_calls: int
    cached_input_tokens: int
    cache_write_tokens: int

    @property
    def provider_cache_read_observed(self) -> bool:
        """True only when OpenAI reported a positive cached-token count."""
        return self.cached_input_tokens > 0


def _require_live_confirmation(
    *,
    confirmation_token: str,
    output_path: Path | None,
) -> Path:
    """Fail before imports, file creation, credentials, or network activity."""
    if not secrets.compare_digest(
        str(confirmation_token),
        LIVE_CONFIRMATION_TOKEN,
    ):
        raise CaptureSafetyError("exact live OpenAI confirmation token is required")
    if output_path is None or not str(output_path).strip():
        raise CaptureSafetyError("an explicit output path is required")
    return Path(output_path).expanduser().absolute()


def _path_lexists(path: Path) -> bool:
    """Return True for existing paths, including dangling symlinks."""
    return os.path.lexists(os.fspath(path))


def _prepare_output_destination(output_path: Path) -> None:
    """Create only the explicit parent, after confirmation, without a file."""
    if output_path.suffix.lower() != ".jsonl":
        raise CaptureSafetyError("capture output must use a .jsonl filename")
    if _path_lexists(output_path):
        raise FileExistsError(f"refusing to overwrite capture: {output_path}")
    parent = output_path.parent
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise CaptureSafetyError(
            f"cannot create explicit capture directory: {parent}"
        ) from exc
    if not parent.is_dir():
        raise CaptureSafetyError(f"capture parent is not a directory: {parent}")


def _literal_string_assignment(source_path: Path, variable_name: str) -> str:
    """Read one top-level literal string assignment without importing code."""
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(source_path))
    matches: list[str] = []
    for node in module.body:
        value_node: ast.expr | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            value_node = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == variable_name
        ):
            value_node = node.value
        if value_node is None:
            continue
        try:
            if (
                isinstance(value_node, ast.Call)
                and isinstance(value_node.func, ast.Attribute)
                and value_node.func.attr == "strip"
                and not value_node.args
                and not value_node.keywords
            ):
                # Support exactly: ``"literal".strip()``. Do not evaluate
                # names, arbitrary calls, arguments, or chained expressions.
                value = ast.literal_eval(value_node.func.value)
                if not isinstance(value, str):
                    raise ValueError("strip receiver is not a string")
                value = value.strip()
            else:
                value = ast.literal_eval(value_node)
        except (TypeError, ValueError, SyntaxError) as exc:
            raise CaptureSafetyError(
                f"{variable_name} must remain a literal string"
            ) from exc
        if not isinstance(value, str) or not value.strip():
            raise CaptureSafetyError(
                f"{variable_name} must be a non-empty literal string"
            )
        matches.append(value)
    if len(matches) != 1:
        raise CaptureSafetyError(
            f"expected exactly one literal {variable_name} assignment"
        )
    return matches[0]


def _select_cases(cases: Sequence[Any], limit: int | None) -> list[Any]:
    if not cases:
        raise CaptureSafetyError("benchmark case file contains no cases")
    if limit is not None:
        if isinstance(limit, bool) or limit < 1 or limit > MAX_CAPTURE_CASES:
            raise CaptureSafetyError(
                f"case limit must be between 1 and {MAX_CAPTURE_CASES}"
            )
        return list(cases[:limit])
    if len(cases) > MAX_CAPTURE_CASES:
        raise CaptureSafetyError(
            f"refusing {len(cases)} paid calls; use --limit with a value "
            f"no greater than {MAX_CAPTURE_CASES}"
        )
    return list(cases)


def _member(value: Any, name: str) -> Any:
    try:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)
    except Exception:
        return None


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    try:
        text = str(value)
    except Exception:
        return None
    return text if text else None


def _usage_payload(response: Any) -> dict[str, int | None]:
    """Preserve unavailable usage as null rather than manufacturing a hit."""
    usage = _member(response, "usage")
    input_details = _member(usage, "input_tokens_details")
    output_details = _member(usage, "output_tokens_details")
    return {
        "input_tokens": _optional_nonnegative_int(
            _member(usage, "input_tokens")
        ),
        "cached_input_tokens": _optional_nonnegative_int(
            _member(input_details, "cached_tokens")
        ),
        "cache_write_tokens": _optional_nonnegative_int(
            _member(input_details, "cache_write_tokens")
        ),
        "output_tokens": _optional_nonnegative_int(
            _member(usage, "output_tokens")
        ),
        "reasoning_tokens": _optional_nonnegative_int(
            _member(output_details, "reasoning_tokens")
        ),
        "total_tokens": _optional_nonnegative_int(
            _member(usage, "total_tokens")
        ),
    }


class _AtomicPrivateJSONL:
    """Build a mode-0600 sibling file, then publish it without overwrite."""

    def __init__(self, target: Path, canonical_json: Callable[[Any], str]):
        _prepare_output_destination(target)
        self.target = target
        self._canonical_json = canonical_json
        fd, raw_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        self._temporary = Path(raw_path)
        try:
            os.fchmod(fd, 0o600)
            self._handle = os.fdopen(fd, "w", encoding="utf-8")
        except Exception:
            os.close(fd)
            self._temporary.unlink(missing_ok=True)
            raise
        self._committed = False

    def append(self, payload: Mapping[str, Any]) -> None:
        self._handle.write(self._canonical_json(payload))
        self._handle.write("\n")

    def commit(self) -> None:
        if self._committed:
            raise RuntimeError("capture output was already committed")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        try:
            # A hard link in the same directory is atomic and fails if the
            # destination appeared after the initial no-overwrite check.
            os.link(self._temporary, self.target)
            self._committed = True
            try:
                self._temporary.unlink()
            except OSError:
                # Publication already succeeded.  A stale private temporary
                # link can be cleaned later and must not turn success into a
                # reported failure that could prompt a duplicate paid run.
                pass
            if hasattr(os, "O_DIRECTORY"):
                try:
                    directory_fd = os.open(
                        self.target.parent,
                        os.O_RDONLY | os.O_DIRECTORY,
                    )
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except OSError:
                    # The private, fully-fsynced file is already published.
                    # Some filesystems do not support directory fsync.
                    pass
        except Exception:
            self._temporary.unlink(missing_ok=True)
            raise

    def abort(self) -> None:
        try:
            if not self._handle.closed:
                self._handle.close()
        finally:
            self._temporary.unlink(missing_ok=True)


def _default_client_factory(*, api_key: str, timeout_seconds: float) -> Any:
    """Import the SDK only on the fully confirmed execution path."""
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        timeout=timeout_seconds,
        max_retries=0,
    )


def _error_response_payload() -> dict[str, Any]:
    return {
        "actual_model": None,
        "decision": None,
        "confidence": None,
        "quality_score": None,
        "reason": None,
        "risk_flags": None,
    }


def capture_live_reviews(
    *,
    cases_path: Path | None,
    output_path: Path,
    confirmation_token: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
    timeout_seconds: float | None = None,
    limit: int | None = None,
    _client_factory: Callable[..., Any] | None = None,
    _system_prompt: str | None = None,
    _clock: Callable[[], float] = time.monotonic,
) -> CaptureSummary:
    """Run sequential paid reviews only after every safety check succeeds."""
    destination = _require_live_confirmation(
        confirmation_token=confirmation_token,
        output_path=output_path,
    )
    _prepare_output_destination(destination)

    # Both modules are side-effect-free.  In particular, importing ai_review
    # loads neither .env nor the OpenAI SDK and has no broker authority.
    benchmark = importlib.import_module("ai_review_benchmark")
    review = importlib.import_module("ai_review")

    source = Path(cases_path) if cases_path is not None else benchmark.DEFAULT_CASES_PATH
    cases = _select_cases(benchmark.load_cases(source), limit)

    chosen_model = (model or os.getenv("OPENAI_MODEL", "gpt-5.6")).strip()
    chosen_effort = (
        reasoning_effort
        or os.getenv("OPENAI_REASONING_EFFORT", "low")
    ).strip().lower()
    chosen_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
    )
    if not chosen_model:
        raise CaptureSafetyError("OpenAI model must be non-empty")
    if chosen_effort not in {"none", "low", "medium", "high", "xhigh"}:
        raise CaptureSafetyError("unsupported reasoning effort")
    if not (0 < float(chosen_timeout) <= 300):
        raise CaptureSafetyError("timeout must be in (0, 300] seconds")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise CaptureSafetyError(
            "OPENAI_API_KEY must be exported for a confirmed capture"
        )

    system_prompt = _system_prompt or _literal_string_assignment(
        BOT_SOURCE,
        "AI_SYSTEM_PROMPT",
    )
    prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    schema_sha256 = benchmark.sha256_json(review.AIDecision.model_json_schema())

    writer = _AtomicPrivateJSONL(destination, benchmark.canonical_json)
    try:
        factory = _client_factory or _default_client_factory
        client = factory(
            api_key=api_key,
            timeout_seconds=float(chosen_timeout),
        )
        successful = 0
        failed = 0
        api_calls = 0
        total_cached = 0
        total_cache_write = 0

        for sequence, case in enumerate(cases, start=1):
            request = review.build_openai_review_request(
                case.candidate,
                model=chosen_model,
                prompt_version=review.AI_PROMPT_VERSION,
                system_prompt=system_prompt,
                reasoning_effort=chosen_effort,
                timeout_seconds=float(chosen_timeout),
                prompt_cache_enabled=True,
                prompt_cache_ttl="30m",
            )
            request_metadata = {
                "prompt_version": review.AI_PROMPT_VERSION,
                "prompt_sha256": prompt_sha256,
                "schema_sha256": schema_sha256,
                "requested_model": chosen_model,
                "reasoning_effort": chosen_effort,
                "max_output_tokens": _optional_nonnegative_int(
                    request.get("max_output_tokens")
                ),
            }
            started = _clock()
            response: Any = None
            api_calls += 1
            try:
                response = client.responses.parse(**request)
                parsed = _member(response, "output_parsed")
                if not isinstance(parsed, review.AIDecision):
                    raise RuntimeError("OpenAI returned no compact parsed decision")
                usage = _usage_payload(response)
                response_payload = {
                    "actual_model": _optional_string(
                        _member(response, "model")
                    ),
                    "decision": parsed.decision,
                    "confidence": parsed.confidence,
                    "quality_score": parsed.quality_score,
                    "reason": parsed.reason,
                    "risk_flags": list(parsed.risk_flags),
                }
                status = "OK"
                error_type = None
                successful += 1
            except Exception as exc:
                usage = _usage_payload(response)
                response_payload = _error_response_payload()
                status = "ERROR"
                error_type = type(exc).__name__
                failed += 1
            latency_ms = max(0, int((_clock() - started) * 1000))
            total_cached += usage["cached_input_tokens"] or 0
            total_cache_write += usage["cache_write_tokens"] or 0
            result = {
                "schema_version": benchmark.RESULT_SCHEMA_VERSION,
                "event": benchmark.RESULT_EVENT,
                "sequence": sequence,
                "case_id": case.case_id,
                "input_sha256": case.input_sha256,
                "variant": "compact",
                "provenance": "recorded_api",
                "status": status,
                "request": request_metadata,
                "response": response_payload,
                "usage": usage,
                "latency_ms": latency_ms,
                "error_type": error_type,
            }
            benchmark.validate_result_payload(
                result,
                source=str(destination),
                line_number=sequence,
            )
            writer.append(result)

        writer.commit()
    except Exception:
        writer.abort()
        raise

    return CaptureSummary(
        output_path=destination,
        cases=len(cases),
        api_calls=api_calls,
        successful_calls=successful,
        failed_calls=failed,
        cached_input_tokens=total_cached,
        cache_write_tokens=total_cache_write,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture compact OpenAI review results. Without the exact "
            "confirmation token this command performs validation only."
        )
    )
    parser.add_argument(
        "--cases",
        type=Path,
        help="deidentified case JSONL (default: benchmark fixture path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new .jsonl path; existing files are never overwritten",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help=f"must equal {LIVE_CONFIRMATION_TOKEN!r} to enable paid calls",
    )
    parser.add_argument("--model", help="override OPENAI_MODEL")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        help="override OPENAI_REASONING_EFFORT",
    )
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument(
        "--limit",
        type=int,
        help=f"capture at most this many cases (maximum {MAX_CAPTURE_CASES})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.confirm:
        # Validation-only mode deliberately does not import ai_review or OpenAI.
        try:
            benchmark = importlib.import_module("ai_review_benchmark")
            source = args.cases or benchmark.DEFAULT_CASES_PATH
            cases = _select_cases(benchmark.load_cases(source), args.limit)
        except Exception as exc:
            print(
                f"Case validation failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2
        print(
            f"VALIDATION ONLY: {len(cases)} deidentified cases; "
            "no OpenAI call made and no output created."
        )
        return 0

    try:
        summary = capture_live_reviews(
            cases_path=args.cases,
            output_path=args.output,
            confirmation_token=args.confirm,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
        )
    except (CaptureSafetyError, FileExistsError, ValueError, OSError) as exc:
        print(f"Capture refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Provider exception messages are intentionally not printed because
        # they may echo request data. Per-case failures are captured by type.
        print(f"Capture failed: {type(exc).__name__}", file=sys.stderr)
        return 2

    cache_evidence = (
        "provider cache read observed"
        if summary.provider_cache_read_observed
        else "no provider cache read observed"
    )
    print(
        f"Capture complete: calls={summary.api_calls} "
        f"successes={summary.successful_calls} errors={summary.failed_calls} "
        f"cached_input={summary.cached_input_tokens} "
        f"cache_write={summary.cache_write_tokens}; {cache_evidence}; "
        f"output={summary.output_path}"
    )
    return 0 if summary.failed_calls == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
