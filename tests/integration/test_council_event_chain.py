import json
import unittest
from unittest.mock import AsyncMock, patch

from backend import council
from backend.decision_lifecycle import create_decision
from backend.event_store import EventStore
from backend.evidence_reasoning import rank_evidences
from backend.evidence_store import EvidenceStore


def _specialist_response(
    *,
    agent_role: str,
    summary: str,
    evidence_id: str,
    content: str,
    source_name: str,
) -> dict:
    return {
        "content": json.dumps(
            {
                "agent_role": agent_role,
                "summary": summary,
                "facts": [content],
                "hypotheses": [],
                "evidence": [
                    {
                        "id": evidence_id,
                        "type": "FACT",
                        "content": content,
                        "source": {
                            "source_type": "LOG",
                            "name": source_name,
                            "location": "line 1",
                            "timestamp": "2026-08-25T10:00:01Z",
                        },
                        "confidence": 0.9,
                        "need_validation": False,
                    }
                ],
                "unknowns": [],
                "confidence": 0.8,
            }
        )
    }


def _event_indexes(events: list[dict], event_type: str) -> list[int]:
    return [
        index
        for index, event in enumerate(events)
        if event.get("event_type") == event_type
    ]


class CouncilEventChainIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_minimal_council_workflow_emits_ordered_event_chain(self):
        event_store = EventStore()
        query = """Checkout API returns 500.

```error.log
2026-08-25T10:00:01Z ERROR database timeout
```
"""
        specialist_responses = [
            _specialist_response(
                agent_role="analysis",
                summary="Database timeout is visible in error.log.",
                evidence_id="ev_db_timeout",
                content="error.log shows database timeout",
                source_name="error.log",
            ),
            _specialist_response(
                agent_role="critic",
                summary="The timeout is confirmed, but root cause still needs care.",
                evidence_id="ev_missing_pool_metrics",
                content="pool utilization metrics are not present",
                source_name="critic",
            ),
            _specialist_response(
                agent_role="investigator",
                summary="The parsed log contains a database timeout.",
                evidence_id="ev_investigator_timeout",
                content="deterministic log summary found one database timeout",
                source_name="log_input_summary",
            ),
            _specialist_response(
                agent_role="analysis",
                summary="The second analysis pass confirms the visible failure.",
                evidence_id="ev_checkout_500",
                content="checkout API returned HTTP 500 during the timeout window",
                source_name="application.log",
            ),
        ]

        with patch(
            "backend.council.query_model",
            new=AsyncMock(side_effect=specialist_responses),
        ):
            stage1_results = await council.stage1_collect_responses(query)

        self.assertTrue(
            any(result.get("evidence") for result in stage1_results),
            "Specialist agents should produce normalized evidence.",
        )

        def evidence_store_factory() -> EvidenceStore:
            return EvidenceStore(event_store=event_store)

        with patch("backend.council.EvidenceStore", side_effect=evidence_store_factory):
            evidence_store = council.build_evidence_store_from_results(stage1_results)

        evidence_items = evidence_store.get_all()
        self.assertTrue(evidence_items)
        self.assertTrue(
            any(
                event.get("event_type") == "EVIDENCE_CREATED"
                for event in event_store.get_all_events()
            )
        )

        rank_evidences(evidence_items, event_store=event_store)

        judge_payload = {
            "agent_role": "judge",
            "incident_level": "high",
            "direct_cause": "database timeout",
            "root_cause": "unknown",
            "verdict_summary": "Database timeout is confirmed by log evidence.",
            "confidence": 0.82,
            "confirmed_evidence": [
                {
                    "evidence_id": "ev_db_timeout",
                    "type": "FACT",
                    "content": "error.log shows database timeout",
                    "source": {
                        "source_type": "LOG",
                        "name": "error.log",
                        "location": "line 1",
                        "timestamp": "2026-08-25T10:00:01Z",
                    },
                    "credibility": "HIGH",
                    "reason": "The log directly reports the timeout.",
                }
            ],
            "unverified_hypothesis": [],
            "scorecard": [
                {
                    "agent_role": "analysis",
                    "agent_name": "Analysis Agent",
                    "agent_instance_id": "analysis:deepseek/deepseek-v4-flash",
                    "evidence_score": 5,
                    "reasoning_score": 4,
                    "actionability_score": 4,
                    "notes": "The result cites direct log evidence.",
                }
            ],
            "gaps": [],
            "next_actions": ["Inspect database availability during the timeout window."],
            "minority_view": "Application-layer regression is still possible.",
        }

        with patch(
            "backend.council.query_model",
            new=AsyncMock(return_value={"content": json.dumps(judge_payload)}),
        ):
            stage2_result = await council.stage2_judge_deliberation(
                query,
                stage1_results,
                evidence_store=evidence_store,
                event_store=event_store,
            )

        self.assertEqual(
            stage2_result["structured_output"]["confirmed_evidence"][0]["evidence_id"],
            "ev_db_timeout",
        )

        with patch(
            "backend.council.query_model",
            new=AsyncMock(
                return_value={
                    "content": (
                        "## Conclusion\n"
                        "Database timeout is the confirmed direct failure signal.\n\n"
                        "## Evidence\n"
                        "Judge confirmed the timeout evidence from error.log.\n"
                    )
                }
            ),
        ):
            stage3_result = await council.stage3_synthesize_final(
                query,
                stage1_results,
                stage2_result,
            )

        decision = {
            **stage3_result,
            "decision_id": "decision_event_chain_001",
        }
        create_decision(decision, stage2_result, event_store=event_store)
        create_decision(decision, stage2_result, event_store=event_store)

        events = event_store.get_all_events()
        created_indexes = _event_indexes(events, "EVIDENCE_CREATED")
        ranked_indexes = _event_indexes(events, "EVIDENCE_RANKED")
        used_indexes = _event_indexes(events, "EVIDENCE_USED_BY_JUDGE")
        decision_indexes = _event_indexes(events, "DECISION_CREATED")

        self.assertTrue(created_indexes)
        self.assertTrue(ranked_indexes)
        self.assertEqual(len(used_indexes), 1)
        self.assertEqual(len(decision_indexes), 1)
        self.assertLess(max(created_indexes), min(ranked_indexes))
        self.assertLess(max(ranked_indexes), min(used_indexes))
        self.assertLess(max(used_indexes), decision_indexes[0])

        decision_event = events[decision_indexes[0]]
        self.assertEqual(decision_event["decision_id"], "decision_event_chain_001")
        self.assertEqual(
            decision_event["metadata"]["evidence_ids"],
            ["ev_db_timeout"],
        )


if __name__ == "__main__":
    unittest.main()
