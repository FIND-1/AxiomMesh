import unittest

from backend.decision_lifecycle import create_decision
from backend.event_store import EventStore


def _decision() -> dict:
    return {
        "decision_id": "decision_001",
        "agent_role": "final_decision",
        "agent_name": "Final Decision",
        "response": "Use the database timeout finding as the decision basis.",
    }


def _judge_result() -> dict:
    return {
        "agent_role": "judge",
        "agent_name": "Judge Agent",
        "structured_output": {
            "confirmed_evidence": [
                {"evidence_id": "evidence_1", "content": "database timeout"},
                {"evidence_id": "evidence_2", "content": "latency spike"},
            ]
        },
    }


class DecisionCreatedEventTest(unittest.TestCase):
    def test_create_decision_emits_decision_created_event(self):
        event_store = EventStore()

        create_decision(_decision(), _judge_result(), event_store=event_store)

        events = event_store.get_all_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "DECISION_CREATED")
        self.assertEqual(events[0]["actor"], {"type": "SYSTEM", "name": "decision_creator"})
        self.assertEqual(events[0]["decision_id"], "decision_001")

    def test_create_decision_without_event_store_keeps_existing_behavior(self):
        decision = _decision()

        created = create_decision(decision, _judge_result())

        self.assertEqual(created, decision)

    def test_duplicate_decision_created_event_is_not_emitted(self):
        event_store = EventStore()

        create_decision(_decision(), _judge_result(), event_store=event_store)
        create_decision(_decision(), _judge_result(), event_store=event_store)

        events = [
            event
            for event in event_store.get_all_events()
            if event["event_type"] == "DECISION_CREATED"
        ]
        self.assertEqual(len(events), 1)

    def test_decision_created_metadata_contains_judge_and_evidence_ids(self):
        event_store = EventStore()

        create_decision(_decision(), _judge_result(), event_store=event_store)

        event = event_store.get_all_events()[0]
        self.assertEqual(
            event["metadata"],
            {
                "judge": "Judge Agent",
                "evidence_ids": ["evidence_1", "evidence_2"],
            },
        )


if __name__ == "__main__":
    unittest.main()
