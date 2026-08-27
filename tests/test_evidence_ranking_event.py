import unittest

from backend.event_store import EventStore
from backend.evidence_reasoning import rank_evidences


class EvidenceRankingEventTest(unittest.TestCase):
    def test_rank_evidences_emits_ranked_event_for_each_evidence(self):
        event_store = EventStore()

        rank_evidences(
            [
                {
                    "id": "evidence_low",
                    "type": "HYPOTHESIS",
                    "content": "Cache pressure may contribute to latency",
                    "source_type": "AGENT",
                    "confidence": 0.4,
                    "need_validation": True,
                },
                {
                    "id": "evidence_high",
                    "type": "FACT",
                    "content": "error.log contains database timeout",
                    "source_type": "LOG",
                    "confidence": 0.9,
                    "need_validation": False,
                },
            ],
            event_store=event_store,
        )

        ranked_events = [
            event
            for event in event_store.get_all_events()
            if event["event_type"] == "EVIDENCE_RANKED"
        ]

        self.assertEqual(len(ranked_events), 2)
        self.assertEqual(
            {event["evidence_id"] for event in ranked_events},
            {"evidence_low", "evidence_high"},
        )
        self.assertTrue(
            all(event["actor"] == {"type": "SYSTEM", "name": "evidence_ranker"} for event in ranked_events)
        )

    def test_ranked_event_metadata_contains_score_and_rank(self):
        event_store = EventStore()

        result = rank_evidences(
            [
                {
                    "id": "evidence_low",
                    "type": "HYPOTHESIS",
                    "content": "Cache pressure may contribute to latency",
                    "source_type": "AGENT",
                    "confidence": 0.4,
                    "need_validation": True,
                },
                {
                    "id": "evidence_high",
                    "type": "FACT",
                    "content": "error.log contains database timeout",
                    "source_type": "LOG",
                    "confidence": 0.9,
                    "need_validation": False,
                },
            ],
            event_store=event_store,
        )

        events_by_evidence_id = {
            event["evidence_id"]: event
            for event in event_store.get_all_events()
            if event["event_type"] == "EVIDENCE_RANKED"
        }

        first_ranked = result["ranked_evidence"][0]
        first_event = events_by_evidence_id[first_ranked["id"]]

        self.assertIn("score", first_event["metadata"])
        self.assertIn("rank", first_event["metadata"])
        self.assertEqual(first_event["metadata"]["score"], first_ranked["score"])
        self.assertEqual(first_event["metadata"]["rank"], 1)

    def test_rank_evidences_without_event_store_keeps_existing_behavior(self):
        result = rank_evidences(
            [
                {
                    "id": "evidence_low",
                    "type": "HYPOTHESIS",
                    "content": "Cache pressure may contribute to latency",
                    "source_type": "AGENT",
                    "confidence": 0.4,
                    "need_validation": True,
                },
                {
                    "id": "evidence_high",
                    "type": "FACT",
                    "content": "error.log contains database timeout",
                    "source_type": "LOG",
                    "confidence": 0.9,
                    "need_validation": False,
                },
            ]
        )

        self.assertEqual(len(result["ranked_evidence"]), 2)
        self.assertEqual(result["ranked_evidence"][0]["id"], "evidence_high")
        self.assertGreaterEqual(
            result["ranked_evidence"][0]["score"],
            result["ranked_evidence"][1]["score"],
        )


if __name__ == "__main__":
    unittest.main()
