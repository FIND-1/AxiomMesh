import unittest

from backend.event_store import EventStore
from backend.evidence_store import EvidenceStore


class EvidenceLifecycleTest(unittest.TestCase):
    def test_add_evidence_emits_created_event(self):
        event_store = EventStore()
        store = EvidenceStore(event_store=event_store)

        evidence = store.add_evidence(
            {
                "id": "evidence_created_1",
                "type": "FACT",
                "content": "application.log shows connection timeout",
                "source": "logs",
                "agent_role": "analysis",
                "confidence": 0.8,
                "need_validation": False,
            }
        )

        events = event_store.get_all_events()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "EVIDENCE_CREATED")
        self.assertEqual(events[0]["actor"], {"type": "SYSTEM", "name": "evidence_store"})
        self.assertEqual(events[0]["evidence_id"], evidence["id"])
        self.assertEqual(events[0]["metadata"]["type"], "FACT")
        self.assertEqual(events[0]["metadata"]["content"], evidence["content"])
        self.assertEqual(events[0]["metadata"]["source"], evidence["source"])

    def test_duplicate_evidence_emits_merged_without_second_created_event(self):
        event_store = EventStore()
        store = EvidenceStore(event_store=event_store)

        first = store.add_evidence(
            {
                "id": "evidence_original",
                "type": "FACT",
                "content": "Redis timeout appears in application log",
                "source": "logs",
                "agent_role": "analysis",
                "confidence": 0.6,
                "need_validation": False,
            }
        )
        store.add_evidence(
            {
                "id": "evidence_duplicate",
                "type": "FACT",
                "content": "redis timeout appears in application log",
                "source": "logs",
                "agent_role": "investigator",
                "confidence": 0.9,
                "need_validation": False,
            }
        )

        events = event_store.get_all_events()
        created_events = [
            event for event in events if event["event_type"] == "EVIDENCE_CREATED"
        ]
        merged_events = [
            event for event in events if event["event_type"] == "EVIDENCE_MERGED"
        ]

        self.assertEqual(len(created_events), 1)
        self.assertEqual(len(merged_events), 1)
        self.assertEqual(merged_events[0]["evidence_id"], first["id"])
        self.assertEqual(
            merged_events[0]["metadata"],
            {
                "source_ids": ["evidence_duplicate"],
                "target_id": "evidence_original",
            },
        )

    def test_evidence_store_without_event_store_keeps_existing_behavior(self):
        store = EvidenceStore()

        first = store.add_evidence(
            {
                "type": "FACT",
                "content": "Request latency spiked at 10:00",
                "source": "metrics",
                "agent_role": "analysis",
                "confidence": 0.8,
                "need_validation": False,
            }
        )
        second = store.add_evidence(
            {
                "type": "FACT",
                "content": "request latency spiked at 10:00",
                "source": "metrics",
                "agent_role": "investigator",
                "confidence": 0.8,
                "need_validation": False,
            }
        )

        self.assertEqual(len(store.get_all()), 1)
        self.assertEqual(second["id"], first["id"])
        self.assertEqual(
            store.get_all()[0]["source_agents"],
            ["analysis", "investigator"],
        )


if __name__ == "__main__":
    unittest.main()
