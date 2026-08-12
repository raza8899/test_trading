from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import ai_ideas


def candidate_payload() -> dict:
    return {
        "setup": {
            "symbol": "INFY",
            "token": 123,
            "side": "LONG",
            "signal_at": "2026-08-11T10:00:00+05:30",
            "technical_score": 91.0,
            "stock_in_play_score": 88.0,
            "rvol": 2.1,
            "rsi": 61.0,
            "atr_pct": 0.012,
            "spread_bps": 3.0,
            "nifty_regime": "BULL",
        },
        "economics": {
            "entry": 100.0,
            "stop": 98.8,
            "target": 102.15,
            "qty": 100,
            "planned_risk": 140.0,
            "planned_target_profit": 220.0,
            "after_cost_payoff": 1.57,
        },
        "capacity": {"candidate_risk_budget": 200.0},
        "config_fingerprint": "abc",
    }


class AITradeIdeaTests(unittest.TestCase):
    def test_anonymization_removes_identity_time_and_composite_scores(self) -> None:
        value = ai_ideas.anonymize_candidate(candidate_payload(), "C001")

        rendered = ai_ideas.canonical_json(value)
        self.assertNotIn("INFY", rendered)
        self.assertNotIn("2026-08-11", rendered)
        self.assertNotIn("technical_score", rendered)
        self.assertNotIn("stock_in_play_score", rendered)
        self.assertEqual(value["candidate_id"], "C001")
        self.assertEqual(value["side"], "LONG")
        self.assertEqual(value["features"]["rvol"], 2.1)

    def test_load_candidates_deduplicates_idea_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades.jsonl"
            event = {
                "event": "AI_CANDIDATE",
                "idea_id": "idea-1",
                "symbol": "INFY",
                "side": "LONG",
                "candidate": candidate_payload(),
                "config_fingerprint": "abc",
            }
            path.write_text(
                json.dumps(event) + "\n" + json.dumps(event) + "\n",
                encoding="utf-8",
            )

            loaded = ai_ideas.load_candidates([path], 8)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].symbol, "INFY")

    def test_validation_rejects_invented_or_missing_candidates(self) -> None:
        envelope = ai_ideas.CandidateEnvelope(
            idea_id="idea-1",
            symbol="INFY",
            side="LONG",
            source="test",
            line_number=1,
            config_fingerprint="abc",
            anonymous_payload=ai_ideas.anonymize_candidate(
                candidate_payload(),
                "C001",
            ),
        )
        invented = ai_ideas.ShadowTradeIdeaBatch(
            schema_version="shadow-trade-idea-v1",
            market_summary="test",
            ideas=[
                ai_ideas.ShadowTradeIdea(
                    schema_version="shadow-trade-idea-v1",
                    candidate_id="C999",
                    verdict="PASS",
                    confidence_band="LOW",
                    quality_band="WEAK",
                    primary_reason="WEAK_CONFIRMATION",
                    risk_flags=["NONE"],
                    evidence_fields=["rvol"],
                    rationale="Insufficient confirmation.",
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "invented candidate"):
            ai_ideas.validate_batch(invented, [envelope])

    def test_private_jsonl_writer_uses_owner_only_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ideas.jsonl"
            ai_ideas._append_private_jsonl(path, {"status": "OK"})

            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_text())["status"], "OK")


if __name__ == "__main__":
    unittest.main()
