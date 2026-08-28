import json
import unittest
from typing import cast
from unittest.mock import AsyncMock, patch

from backend import council
from backend.agent_models import AgentResultPayload
from backend.event_store import EventStore
from backend.evidence_store import EvidenceStore


def _stage_result_with_evidence() -> AgentResultPayload:
    return cast(
        AgentResultPayload,
        {
            "agent_role": "analysis",
            "agent_name": "Analysis Agent",
            "agent_instance_id": "analysis:test/model",
            "model": "test/model",
            "response": "analysis",
            "structured_output": {},
            "evidence": [
                {
                    "id": "ev001",
                    "type": "FACT",
                    "content": "error.log shows database timeout",
                    "source": "error.log",
                    "agent_role": "analysis",
                    "confidence": 0.9,
                    "need_validation": False,
                },
                {
                    "id": "ev002",
                    "type": "FACT",
                    "content": "metrics show latency spike",
                    "source": "metrics",
                    "agent_role": "analysis",
                    "confidence": 0.8,
                    "need_validation": False,
                },
            ],
        },
    )


class EvidenceUsedByJudgeEventTest(unittest.IsolatedAsyncioTestCase):
    async def test_judge_emits_used_event_for_confirmed_evidence(self):
        event_store = EventStore()
        evidence_store = EvidenceStore(event_store=event_store)
        evidence_store.add_many(_stage_result_with_evidence()["evidence"])
        event_store.clear()
        judge_payload = {
            "verdict_summary": "Database timeout is supported by evidence.",
            "root_cause": "Database timeout",
            "confirmed_evidence": [
                {
                    "evidence_id": "ev001",
                    "type": "FACT",
                    "content": "error.log shows database timeout",
                    "source": "error.log",
                    "credibility": "HIGH",
                }
            ],
            "confidence": 0.8,
        }

        with patch(
            "backend.council.query_model",
            new=AsyncMock(return_value={"content": json.dumps(judge_payload)}),
        ):
            await council.stage2_judge_deliberation(
                "database timeout",
                [_stage_result_with_evidence()],
                evidence_store=evidence_store,
                event_store=event_store,
            )

        used_events = [
            event
            for event in event_store.get_all_events()
            if event["event_type"] == "EVIDENCE_USED_BY_JUDGE"
        ]

        self.assertEqual(len(used_events), 1)
        self.assertEqual(used_events[0]["actor"], {"type": "SYSTEM", "name": "judge_reasoning"})
        self.assertEqual(used_events[0]["evidence_id"], "ev001")
        self.assertEqual(
            used_events[0]["metadata"],
            {"judge": "Judge Agent", "reason": "supporting_decision"},
        )

    async def test_judge_emits_used_event_once_per_evidence_id(self):
        event_store = EventStore()
        evidence_store = EvidenceStore(event_store=event_store)
        evidence_store.add_many(_stage_result_with_evidence()["evidence"])
        event_store.clear()
        judge_payload = {
            "verdict_summary": "Duplicate citations still count once.",
            "root_cause": "Database timeout",
            "confirmed_evidence": [
                {
                    "evidence_id": "ev001",
                    "type": "FACT",
                    "content": "error.log shows database timeout",
                    "source": "error.log",
                },
                {
                    "evidence_id": "ev001",
                    "type": "FACT",
                    "content": "error.log shows database timeout",
                    "source": "error.log",
                },
            ],
            "confidence": 0.8,
        }

        with patch(
            "backend.council.query_model",
            new=AsyncMock(return_value={"content": json.dumps(judge_payload)}),
        ):
            await council.stage2_judge_deliberation(
                "database timeout",
                [_stage_result_with_evidence()],
                evidence_store=evidence_store,
                event_store=event_store,
            )

        used_events = [
            event
            for event in event_store.get_all_events()
            if event["event_type"] == "EVIDENCE_USED_BY_JUDGE"
        ]

        self.assertEqual(len(used_events), 1)
        self.assertEqual(used_events[0]["evidence_id"], "ev001")

    async def test_judge_without_event_store_keeps_existing_behavior(self):
        judge_payload = {
            "verdict_summary": "Database timeout is supported by evidence.",
            "root_cause": "Database timeout",
            "confirmed_evidence": [
                {
                    "evidence_id": "ev001",
                    "type": "FACT",
                    "content": "error.log shows database timeout",
                    "source": "error.log",
                }
            ],
            "confidence": 0.8,
        }

        with patch(
            "backend.council.query_model",
            new=AsyncMock(return_value={"content": json.dumps(judge_payload)}),
        ):
            result = await council.stage2_judge_deliberation(
                "database timeout",
                [_stage_result_with_evidence()],
            )

        self.assertEqual(result["agent_role"], "judge")
        self.assertEqual(
            result["structured_output"]["confirmed_evidence"][0]["evidence_id"],
            "ev001",
        )


if __name__ == "__main__":
    unittest.main()
