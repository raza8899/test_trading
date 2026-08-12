#!/usr/bin/env python3
"""Report realised and shadow performance from the bot's JSONL journals.

Only CLOSE events with explicit, finite ``net_pnl``, ``gross_pnl``, ``fees``,
and ``r_multiple`` are included.  They must also have a recognised
``ai_decision`` (``APPROVE``, ``REJECT``, ``ERROR``, or ``OFF``), except that a missing
decision is accepted when ``ai_mode`` is explicitly ``off``.  Older CLOSE
events are reported as incomplete and excluded; this module never reconstructs
or guesses P&L from entry prices, exit reasons, or other journal events.

Profit factor, expectancy, win rate, and drawdown use net P&L.  Drawdown follows
the append order of lexically sorted input paths, then line order within each
file, matching the normal ``trades_YYYYMMDD.jsonl`` journal layout.  Provenance
metadata is not required for legacy P&L to remain calculable, but reports mark
mixed or unverifiable experiments explicitly rather than implying that their
aggregate is a like-for-like comparison.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import glob
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATTERN = "trades_*.jsonl"
NUMERIC_FIELDS = ("net_pnl", "gross_pnl", "fees", "r_multiple")
AI_COHORT_DECISIONS = ("APPROVE", "REJECT")
VALID_AI_DECISIONS = (*AI_COHORT_DECISIONS, "ERROR", "OFF")


@dataclass(frozen=True)
class TradeRecord:
    """A complete CLOSE event used in performance calculations."""

    source: str
    line_number: int
    net_pnl: float
    gross_pnl: float
    fees: float
    r_multiple: float
    ai_decision: str
    execution_mode: str | None = None
    config_fingerprint: str | None = None
    ai_mode: str | None = None
    ai_response_model: str | None = None
    ai_prompt_version: str | None = None


@dataclass
class ParseDiagnostics:
    """Counts that make excluded and legacy data visible to callers."""

    total_lines: int = 0
    blank_lines: int = 0
    malformed_json_lines: int = 0
    non_object_lines: int = 0
    ignored_non_close_events: int = 0
    close_events: int = 0
    complete_close_events: int = 0
    incomplete_close_events: int = 0
    incomplete_reasons: Counter[str] = field(default_factory=Counter)
    file_errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_lines": self.total_lines,
            "blank_lines": self.blank_lines,
            "malformed_json_lines": self.malformed_json_lines,
            "non_object_lines": self.non_object_lines,
            "ignored_non_close_events": self.ignored_non_close_events,
            "close_events": self.close_events,
            "complete_close_events": self.complete_close_events,
            "incomplete_close_events": self.incomplete_close_events,
            "incomplete_reasons": dict(sorted(self.incomplete_reasons.items())),
            "file_errors": list(self.file_errors),
        }


def _finite_number(value: Any) -> float | None:
    """Return a finite JSON number as float without coercing strings or bools."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_text(
    payload: dict[str, Any],
    *names: str,
    lowercase: bool = False,
) -> str | None:
    """Read optional provenance text without making legacy records incomplete."""

    for name in names:
        value = payload.get(name)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text:
            return text.lower() if lowercase else text
    return None


def _close_event_to_trade(
    payload: dict[str, Any],
    source: Path,
    line_number: int,
) -> tuple[TradeRecord | None, list[str]]:
    issues: list[str] = []
    numbers: dict[str, float] = {}

    for name in NUMERIC_FIELDS:
        if name not in payload or payload[name] is None:
            issues.append(f"missing_field:{name}")
            continue
        value = _finite_number(payload[name])
        if value is None:
            issues.append(f"invalid_field:{name}")
            continue
        numbers[name] = value

    if "ai_decision" not in payload or payload["ai_decision"] is None:
        ai_mode = payload.get("ai_mode")
        if isinstance(ai_mode, str) and ai_mode.strip().lower() == "off":
            decision = "OFF"
        else:
            issues.append("missing_field:ai_decision")
            decision = ""
    elif not isinstance(payload["ai_decision"], str):
        issues.append("invalid_field:ai_decision")
        decision = ""
    else:
        decision = payload["ai_decision"].strip().upper()
        if decision not in VALID_AI_DECISIONS:
            issues.append("invalid_field:ai_decision")

    if issues:
        return None, issues

    return (
        TradeRecord(
            source=str(source),
            line_number=line_number,
            net_pnl=numbers["net_pnl"],
            gross_pnl=numbers["gross_pnl"],
            fees=numbers["fees"],
            r_multiple=numbers["r_multiple"],
            ai_decision=decision,
            execution_mode=_optional_text(
                payload,
                "execution_mode",
                lowercase=True,
            ),
            config_fingerprint=_optional_text(payload, "config_fingerprint"),
            ai_mode=_optional_text(payload, "ai_mode", lowercase=True),
            ai_response_model=_optional_text(
                payload,
                "ai_response_model",
                "ai_model",
            ),
            ai_prompt_version=_optional_text(payload, "ai_prompt_version"),
        ),
        [],
    )


def parse_trade_files(paths: Iterable[Path]) -> tuple[list[TradeRecord], ParseDiagnostics]:
    """Parse complete CLOSE events from already-resolved JSONL paths."""

    trades: list[TradeRecord] = []
    diagnostics = ParseDiagnostics()

    for path in sorted(paths, key=lambda item: str(item)):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    diagnostics.total_lines += 1
                    if not raw_line.strip():
                        diagnostics.blank_lines += 1
                        continue

                    try:
                        payload = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        diagnostics.malformed_json_lines += 1
                        continue

                    if not isinstance(payload, dict):
                        diagnostics.non_object_lines += 1
                        continue

                    event = payload.get("event")
                    if not isinstance(event, str) or event.strip().upper() != "CLOSE":
                        diagnostics.ignored_non_close_events += 1
                        continue

                    diagnostics.close_events += 1
                    trade, issues = _close_event_to_trade(payload, path, line_number)
                    if trade is None:
                        diagnostics.incomplete_close_events += 1
                        diagnostics.incomplete_reasons.update(issues)
                        continue

                    diagnostics.complete_close_events += 1
                    trades.append(trade)
        except OSError as exc:
            diagnostics.file_errors.append(f"{path}: {exc}")

    return trades, diagnostics


def _clean_float(value: float) -> float:
    return 0.0 if abs(value) < 1e-12 else value


def summarize_trades(records: Sequence[TradeRecord]) -> dict[str, Any]:
    """Calculate net-P&L performance metrics for records in journal order."""

    count = len(records)
    wins = sum(record.net_pnl > 0 for record in records)
    losses = sum(record.net_pnl < 0 for record in records)
    breakeven = count - wins - losses

    net_pnl = _clean_float(math.fsum(record.net_pnl for record in records))
    gross_pnl = _clean_float(math.fsum(record.gross_pnl for record in records))
    fees = _clean_float(math.fsum(record.fees for record in records))
    gross_profit = math.fsum(record.net_pnl for record in records if record.net_pnl > 0)
    gross_loss = math.fsum(record.net_pnl for record in records if record.net_pnl < 0)

    if gross_loss < 0:
        profit_factor: float | None = _clean_float(gross_profit / abs(gross_loss))
        profit_factor_status = "finite"
    elif gross_profit > 0:
        profit_factor = None
        profit_factor_status = "infinite_no_losses"
    else:
        profit_factor = None
        profit_factor_status = "undefined_no_profit_or_loss"

    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for record in records:
        cumulative = math.fsum((cumulative, record.net_pnl))
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)

    return {
        "trades": count,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate_pct": (wins / count * 100.0) if count else None,
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "expectancy": (net_pnl / count) if count else None,
        "profit_factor": profit_factor,
        "profit_factor_status": profit_factor_status,
        "max_drawdown": _clean_float(max_drawdown),
        "average_r": (
            _clean_float(math.fsum(record.r_multiple for record in records) / count)
            if count
            else None
        ),
    }


def _provenance_counts(
    records: Sequence[TradeRecord],
    field_name: str,
) -> dict[str, int]:
    counts = Counter(
        value
        for record in records
        for value in [getattr(record, field_name)]
        if value is not None
    )
    return dict(sorted(counts.items()))


def summarize_provenance(records: Sequence[TradeRecord]) -> dict[str, Any]:
    """Describe whether aggregated records are comparable experiments."""

    count = len(records)
    execution_modes = _provenance_counts(records, "execution_mode")
    config_fingerprints = _provenance_counts(records, "config_fingerprint")
    missing_execution_mode = sum(record.execution_mode is None for record in records)
    missing_config_fingerprint = sum(
        record.config_fingerprint is None for record in records
    )
    unknown_execution_modes = sorted(
        mode for mode in execution_modes if mode not in {"live", "paper"}
    )

    compatibility_issues: list[str] = []
    if len(execution_modes) > 1:
        compatibility_issues.append("mixed_execution_modes")
    if unknown_execution_modes:
        compatibility_issues.append("unrecognized_execution_modes")
    if missing_execution_mode:
        compatibility_issues.append("missing_execution_mode")
    if len(config_fingerprints) > 1:
        compatibility_issues.append("mixed_config_fingerprints")
    if missing_config_fingerprint:
        compatibility_issues.append("missing_config_fingerprint")

    if not count:
        compatibility_status = "unknown_no_records"
    elif any(issue.startswith("mixed_") for issue in compatibility_issues):
        compatibility_status = "incompatible_mixed"
    elif compatibility_issues:
        compatibility_status = "unverified"
    else:
        compatibility_status = "compatible"

    ai_records = [record for record in records if record.ai_decision != "OFF"]
    ai_response_models = _provenance_counts(ai_records, "ai_response_model")
    ai_prompt_versions = _provenance_counts(ai_records, "ai_prompt_version")
    missing_ai_response_model = sum(
        record.ai_response_model is None for record in ai_records
    )
    missing_ai_prompt_version = sum(
        record.ai_prompt_version is None for record in ai_records
    )
    ai_issues: list[str] = []
    if len(ai_response_models) > 1:
        ai_issues.append("mixed_ai_response_models")
    if missing_ai_response_model:
        ai_issues.append("missing_ai_response_model")
    if len(ai_prompt_versions) > 1:
        ai_issues.append("mixed_ai_prompt_versions")
    if missing_ai_prompt_version:
        ai_issues.append("missing_ai_prompt_version")

    if not ai_records:
        ai_status = "not_applicable"
    elif any(issue.startswith("mixed_") for issue in ai_issues):
        ai_status = "incompatible_mixed"
    elif ai_issues:
        ai_status = "unverified"
    else:
        ai_status = "compatible"

    return {
        "record_count": count,
        "compatibility_status": compatibility_status,
        "compatible": compatibility_status == "compatible",
        "compatibility_issues": compatibility_issues,
        "execution_modes": execution_modes,
        "config_fingerprints": config_fingerprints,
        "missing_execution_mode": missing_execution_mode,
        "missing_config_fingerprint": missing_config_fingerprint,
        "unrecognized_execution_modes": unknown_execution_modes,
        "ai": {
            "record_count": len(ai_records),
            "compatibility_status": ai_status,
            "compatible": ai_status in {"compatible", "not_applicable"},
            "compatibility_issues": ai_issues,
            "response_models": ai_response_models,
            "prompt_versions": ai_prompt_versions,
            "missing_response_model": missing_ai_response_model,
            "missing_prompt_version": missing_ai_prompt_version,
        },
    }


def _counted_values(values: dict[str, int]) -> str:
    return ", ".join(f"{value}={count}" for value, count in values.items())


def provenance_warnings(provenance: dict[str, Any]) -> list[str]:
    """Turn machine-readable provenance diagnostics into explicit warnings."""

    count = provenance["record_count"]
    if not count:
        return []

    warnings: list[str] = []
    execution_modes = provenance["execution_modes"]
    config_fingerprints = provenance["config_fingerprints"]
    if len(execution_modes) > 1:
        warnings.append(
            "Compatibility warning: mixed execution modes across complete CLOSE "
            f"records ({_counted_values(execution_modes)}); aggregated metrics "
            "must not be treated as one comparable experiment."
        )
    if provenance["unrecognized_execution_modes"]:
        warnings.append(
            "Provenance warning: unrecognized execution mode(s) were retained: "
            + ", ".join(provenance["unrecognized_execution_modes"])
            + ". Compatibility is unverified."
        )
    if provenance["missing_execution_mode"]:
        warnings.append(
            "Provenance warning: "
            f"{provenance['missing_execution_mode']} of {count} complete CLOSE "
            "record(s) are missing or have invalid execution_mode metadata; "
            "legacy P&L remains included, but execution-mode compatibility is "
            "unverified."
        )
    if len(config_fingerprints) > 1:
        warnings.append(
            "Compatibility warning: mixed config fingerprints across complete "
            f"CLOSE records ({_counted_values(config_fingerprints)}); aggregated "
            "metrics combine different strategy/config experiments."
        )
    if provenance["missing_config_fingerprint"]:
        warnings.append(
            "Provenance warning: "
            f"{provenance['missing_config_fingerprint']} of {count} complete CLOSE "
            "record(s) are missing or have invalid config_fingerprint metadata; "
            "legacy P&L remains included, but configuration compatibility is "
            "unverified."
        )

    ai = provenance["ai"]
    if len(ai["response_models"]) > 1:
        warnings.append(
            "AI provenance warning: mixed response models across AI-attributed "
            f"records ({_counted_values(ai['response_models'])})."
        )
    if ai["missing_response_model"]:
        warnings.append(
            "AI provenance warning: "
            f"{ai['missing_response_model']} of {ai['record_count']} AI-attributed "
            "record(s) lack response-model metadata; AI cohort attribution is "
            "unverified."
        )
    if len(ai["prompt_versions"]) > 1:
        warnings.append(
            "AI provenance warning: mixed prompt versions across AI-attributed "
            f"records ({_counted_values(ai['prompt_versions'])})."
        )
    if ai["missing_prompt_version"]:
        warnings.append(
            "AI provenance warning: "
            f"{ai['missing_prompt_version']} of {ai['record_count']} AI-attributed "
            "record(s) lack prompt-version metadata; AI cohort attribution is "
            "unverified."
        )
    return warnings


def resolve_input_paths(raw_paths: Sequence[str | os.PathLike[str]] | None) -> tuple[list[Path], list[str]]:
    """Expand files, directories, and quoted glob patterns without duplication."""

    inputs: Sequence[str | os.PathLike[str]] = raw_paths or [DEFAULT_LOG_DIR]
    candidates: list[Path] = []
    errors: list[str] = []

    for raw_path in inputs:
        value = os.path.expanduser(os.fspath(raw_path))
        path = Path(value)

        if glob.has_magic(value):
            matches = [Path(match) for match in glob.glob(value)]
            if not matches:
                errors.append(f"No files matched pattern: {value}")
            candidates.extend(match for match in matches if match.is_file())
        elif path.is_dir():
            matches = list(path.glob(LOG_PATTERN))
            if not matches:
                errors.append(f"No {LOG_PATTERN} files found in directory: {path}")
            candidates.extend(match for match in matches if match.is_file())
        elif path.is_file():
            candidates.append(path)
        else:
            errors.append(f"Input path does not exist or is not a file: {path}")

    unique: dict[str, Path] = {}
    for candidate in candidates:
        resolved = candidate.resolve()
        unique[str(resolved)] = resolved

    return [unique[key] for key in sorted(unique)], errors


def build_report(paths: Sequence[Path]) -> dict[str, Any]:
    """Build an overall report and explicit APPROVE/REJECT cohorts."""

    records, diagnostics = parse_trade_files(paths)
    provenance = summarize_provenance(records)
    approve = [record for record in records if record.ai_decision == "APPROVE"]
    reject = [record for record in records if record.ai_decision == "REJECT"]
    ai_errors = [record for record in records if record.ai_decision == "ERROR"]

    warnings: list[str] = []
    if diagnostics.incomplete_close_events:
        warnings.append(
            f"Excluded {diagnostics.incomplete_close_events} incomplete or legacy CLOSE event(s); "
            "no P&L values were inferred."
        )
    invalid_lines = diagnostics.malformed_json_lines + diagnostics.non_object_lines
    if invalid_lines:
        warnings.append(f"Ignored {invalid_lines} malformed or non-object JSONL record(s).")
    if diagnostics.file_errors:
        warnings.append("One or more files could not be read; their records are absent.")
    if not records:
        warnings.append("No complete P&L CLOSE records were found; profitability is unknown.")
    warnings.extend(provenance_warnings(provenance))

    return {
        "files": [str(path) for path in paths],
        "drawdown_order": "lexically sorted source path, then JSONL line number",
        "diagnostics": diagnostics.as_dict(),
        "provenance": provenance,
        "summary": summarize_trades(records),
        "ai_shadow_cohorts": {
            "APPROVE": summarize_trades(approve),
            "REJECT": summarize_trades(reject),
            "ERROR": summarize_trades(ai_errors),
        },
        "warnings": warnings,
    }


def _format_number(value: float | None, decimals: int = 2) -> str:
    return "N/A" if value is None else f"{value:,.{decimals}f}"


def _format_profit_factor(summary: dict[str, Any]) -> str:
    status = summary["profit_factor_status"]
    if status == "infinite_no_losses":
        return "infinite (no losing trades)"
    if status == "undefined_no_profit_or_loss":
        return "N/A (no profit or loss)"
    return _format_number(summary["profit_factor"])


def _summary_lines(title: str, summary: dict[str, Any]) -> list[str]:
    return [
        title,
        f"  Trades: {summary['trades']}",
        f"  Win rate: {_format_number(summary['win_rate_pct'])}%" if summary["win_rate_pct"] is not None else "  Win rate: N/A",
        f"  Net P&L: {_format_number(summary['net_pnl'])}",
        f"  Gross P&L: {_format_number(summary['gross_pnl'])}",
        f"  Fees: {_format_number(summary['fees'])}",
        f"  Expectancy: {_format_number(summary['expectancy'])}",
        f"  Profit factor: {_format_profit_factor(summary)}",
        f"  Max drawdown: {_format_number(summary['max_drawdown'])}",
        f"  Average R: {_format_number(summary['average_r'], decimals=3)}",
    ]


def format_text_report(report: dict[str, Any]) -> str:
    """Render a concise human-readable report."""

    diagnostics = report["diagnostics"]
    provenance = report["provenance"]
    ai_provenance = provenance["ai"]
    lines = [
        "Intraday bot performance report",
        f"Files: {len(report['files'])}",
        f"Complete CLOSE records: {diagnostics['complete_close_events']}",
        f"Incomplete/legacy CLOSE records excluded: {diagnostics['incomplete_close_events']}",
        f"Provenance compatibility: {provenance['compatibility_status']}",
        "Execution modes: "
        + (_counted_values(provenance["execution_modes"]) or "none recorded"),
        "Config fingerprints: "
        + (_counted_values(provenance["config_fingerprints"]) or "none recorded"),
        f"AI provenance compatibility: {ai_provenance['compatibility_status']}",
        "AI response models: "
        + (_counted_values(ai_provenance["response_models"]) or "none recorded"),
        "AI prompt versions: "
        + (_counted_values(ai_provenance["prompt_versions"]) or "none recorded"),
        "",
    ]
    lines.extend(_summary_lines("Overall", report["summary"]))
    lines.append("")
    lines.extend(_summary_lines("AI APPROVE cohort", report["ai_shadow_cohorts"]["APPROVE"]))
    lines.append("")
    lines.extend(_summary_lines("AI REJECT shadow cohort", report["ai_shadow_cohorts"]["REJECT"]))
    lines.append("")
    lines.extend(_summary_lines("AI ERROR/unavailable cohort", report["ai_shadow_cohorts"]["ERROR"]))

    if diagnostics["incomplete_reasons"]:
        lines.extend(("", "Excluded CLOSE reasons:"))
        for reason, count in diagnostics["incomplete_reasons"].items():
            lines.append(f"  {reason}: {count}")

    if report["warnings"]:
        lines.extend(("", "Warnings:"))
        lines.extend(f"  {warning}" for warning in report["warnings"])

    return "\n".join(lines)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report complete CLOSE-event performance from trades_*.jsonl journals. "
            "With no paths, reads the project's logs directory."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="JSONL file, directory, or quoted glob pattern (default: project logs)",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit strict JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    paths, input_errors = resolve_input_paths(args.paths)
    report = build_report(paths)
    report["input_errors"] = input_errors

    if input_errors:
        report["warnings"].extend(input_errors)

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(format_text_report(report))
        for error in input_errors:
            print(f"Input error: {error}", file=sys.stderr)

    if input_errors or report["diagnostics"]["file_errors"]:
        return 2
    if report["summary"]["trades"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
