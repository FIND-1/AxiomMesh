import unittest

from backend.evidence_reasoning import EvidenceRanker, rank_evidences
from backend.evidence_store import EvidenceStore


class EvidenceReasoningTest(unittest.TestCase):
    def test_log_fact_scores_above_half(self):
        score = EvidenceRanker().calculate_evidence_score(
            {
                "type": "FACT",
                "content": "error.log shows request failures before auction close",
                "source": {
                    "source_type": "LOG",
                    "name": "error.log",
                    "location": "line 42",
                    "timestamp": "2026-08-26T01:58:22Z",
                },
                "confidence": 0.8,
                "need_validation": False,
            }
        )

        self.assertGreater(score, 0.5)

    def test_tool_fact_scores_higher_than_log_fact(self):
        ranker = EvidenceRanker()
        log_score = ranker.calculate_evidence_score(
            {
                "type": "FACT",
                "content": "error.log shows request failures",
                "source": {"source_type": "LOG", "name": "error.log"},
                "confidence": 0.8,
                "need_validation": False,
            }
        )
        tool_score = ranker.calculate_evidence_score(
            {
                "type": "FACT",
                "content": "log_input_summary detected repeated request failures",
                "source": {"source_type": "TOOL", "name": "log_input_summary"},
                "confidence": 0.8,
                "need_validation": False,
            }
        )

        self.assertGreater(tool_score, log_score)

    def test_hypothesis_needing_validation_scores_lower(self):
        ranker = EvidenceRanker()
        fact_score = ranker.calculate_evidence_score(
            {
                "type": "FACT",
                "content": "Requests failed before auction close",
                "source": {"source_type": "LOG", "name": "error.log"},
                "confidence": 0.8,
                "need_validation": False,
            }
        )
        hypothesis_score = ranker.calculate_evidence_score(
            {
                "type": "HYPOTHESIS",
                "content": "Auction service rejected bids because cache was saturated",
                "source": {"source_type": "AGENT", "name": "analysis"},
                "confidence": 0.8,
                "need_validation": True,
            }
        )

        self.assertLess(hypothesis_score, fact_score)

    def test_same_evidence_from_two_agents_scores_higher(self):
        store = EvidenceStore()
        single = store.add_evidence(
            {
                "type": "FACT",
                "content": "Bid rejected before auction close",
                "source": {"source_type": "LOG", "name": "error.log"},
                "agent_role": "analysis",
                "confidence": 0.8,
                "need_validation": False,
            }
        )
        merged = store.add_evidence(
            {
                "type": "FACT",
                "content": "bid rejected before auction close",
                "source": {"source_type": "LOG", "name": "error.log"},
                "agent_role": "critic",
                "confidence": 0.8,
                "need_validation": False,
            }
        )

        self.assertGreater(merged["score"], single["score"])
        self.assertEqual(merged["source_agents"], ["analysis", "critic"])

    def test_rank_evidences_sorts_by_score_descending(self):
        result = rank_evidences(
            [
                {
                    "id": "hypothesis_1",
                    "type": "HYPOTHESIS",
                    "content": "Cache saturation may explain the failure",
                    "source": {"source_type": "AGENT", "name": "analysis"},
                    "confidence": 0.8,
                    "need_validation": True,
                },
                {
                    "id": "fact_1",
                    "type": "FACT",
                    "content": "log_input_summary found repeated failures",
                    "source": {"source_type": "TOOL", "name": "log_input_summary"},
                    "confidence": 0.8,
                    "need_validation": False,
                },
            ]
        )

        self.assertEqual(result["ranked_evidence"][0]["id"], "fact_1")


if __name__ == "__main__":
    unittest.main()
