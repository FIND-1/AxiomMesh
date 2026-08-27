import unittest

from backend.evidence_reasoning import calculate_confidence
from backend.event_store import EventStore
from backend.evidence_store import EvidenceStore


class EvidenceStoreTest(unittest.TestCase):
    def test_event_store_dependency_can_be_injected(self):
        event_store = EventStore()
        store = EvidenceStore(event_store=event_store)

        self.assertIs(store.event_store, event_store)

    def test_add_fact_evidence_can_be_saved(self):
        store = EvidenceStore()

        evidence = store.add_evidence(
            {
                "type": "FACT",
                "content": "error.log shows HikariPool connection timeout",
                "source": "logs",
                "agent_role": "analysis",
                "confidence": 0.8,
                "need_validation": False,
            }
        )

        self.assertTrue(evidence["id"])
        self.assertEqual(evidence["type"], "FACT")
        self.assertEqual(store.get_all()[0]["content"], "error.log shows HikariPool connection timeout")
        self.assertEqual(store.get_by_type("fact")[0]["id"], evidence["id"])
        self.assertEqual(store.get_by_agent("analysis")[0]["id"], evidence["id"])

    def test_duplicate_evidence_is_merged(self):
        store = EvidenceStore()

        first = store.add_evidence(
            {
                "type": "FACT",
                "content": "Redis timeout appears in application log",
                "source": "logs",
                "agent_role": "analysis",
                "confidence": 0.6,
                "need_validation": False,
            }
        )
        second = store.add_evidence(
            {
                "type": "FACT",
                "content": "redis timeout appears in application log",
                "source": "logs",
                "agent_role": "investigator",
                "confidence": 0.9,
                "need_validation": False,
            }
        )

        self.assertEqual(len(store.get_all()), 1)
        self.assertEqual(second["id"], first["id"])
        self.assertEqual(store.get_all()[0]["confidence"], 0.65)
        self.assertEqual(
            store.get_all()[0]["source_agents"],
            ["analysis", "investigator"],
        )

    def test_tool_evidence_confidence_is_higher_than_plain_hypothesis(self):
        tool_confidence = calculate_confidence(
            {
                "type": "FACT",
                "content": "Tool summarized 23 error lines",
                "source": "tool:log_input_summary",
                "agent_role": "investigator",
                "confidence": 0.8,
                "need_validation": False,
            }
        )
        hypothesis_confidence = calculate_confidence(
            {
                "type": "HYPOTHESIS",
                "content": "Database contention may be the cause",
                "source": "analysis",
                "agent_role": "analysis",
                "confidence": 0.8,
                "need_validation": False,
            }
        )

        self.assertGreater(tool_confidence, hypothesis_confidence)

    def test_need_validation_lowers_confidence(self):
        confirmed = calculate_confidence(
            {
                "type": "FACT",
                "content": "Request latency spiked at 10:00",
                "source": "metrics",
                "confidence": 0.8,
                "need_validation": False,
            }
        )
        needs_validation = calculate_confidence(
            {
                "type": "HYPOTHESIS",
                "content": "Request latency spiked at 10:00",
                "source": "metrics",
                "confidence": 0.8,
                "need_validation": True,
            }
        )

        self.assertLess(needs_validation, confirmed)


if __name__ == "__main__":
    unittest.main()
