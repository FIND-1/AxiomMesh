import unittest

from backend.event_store import EventStore


class EventStoreTests(unittest.TestCase):
    def test_add_query_and_clear_events(self):
        store = EventStore()
        first = store.add_event(
            {
                "event_id": "event_1",
                "event_type": "evidence_created",
                "timestamp": "2026-08-25T10:00:00",
                "actor": {"type": "AGENT", "name": "analysis"},
                "evidence_id": "evidence_1",
                "metadata": {"status": "created"},
            }
        )
        store.add_event(
            {
                "event_id": "event_2",
                "event_type": "evidence_merged",
                "timestamp": "2026-08-25T10:01:00",
                "actor": {"type": "SYSTEM", "name": "evidence_store"},
                "evidence_id": "evidence_1",
                "metadata": {"status": "merged"},
            }
        )
        store.add_event(
            {
                "event_id": "event_3",
                "event_type": "judge_started",
                "timestamp": "2026-08-25T10:02:00",
                "actor": {"type": "AGENT", "name": "judge"},
                "evidence_id": "",
                "metadata": {"status": "running"},
            }
        )

        self.assertEqual(first["event_id"], "event_1")
        self.assertEqual(len(store.get_all_events()), 3)
        self.assertEqual(len(store.get_by_evidence_id("evidence_1")), 2)
        self.assertEqual(store.get_by_evidence_id(""), [])

        store.clear()
        self.assertEqual(store.get_all_events(), [])


if __name__ == "__main__":
    unittest.main()
