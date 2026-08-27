import unittest

from backend.agent_models import AgentEvent, AgentMessage, AgentResult, Evidence


class AgentModelsTest(unittest.TestCase):
    def test_agent_result_serializes_canonical_fields(self):
        evidence = Evidence(
            source="logs",
            detail="Redis timeout appears twice",
            credibility="high",
            agent_role="analysis",
        )
        message = AgentMessage(
            role="analysis",
            content="Cache timeout is a likely signal.",
            evidence=[evidence],
            confidence=0.72,
            timestamp="2026-08-25T10:00:00",
        )
        result = AgentResult(
            agent_role="analysis",
            agent_name="Analysis Agent",
            agent_instance_id="analysis:test/model",
            model="test/model",
            response="### Analysis Agent",
            structured_output={"summary": "Cache timeout", "confidence": 0.72},
            evidence=[evidence],
            messages=[message],
            confidence=0.72,
        ).to_dict()

        self.assertEqual(result["agent_role"], "analysis")
        self.assertEqual(result["evidence"][0]["source"], "logs")
        self.assertEqual(result["evidence"][0]["type"], "FACT")
        self.assertEqual(result["evidence"][0]["content"], "Redis timeout appears twice")
        self.assertFalse(result["evidence"][0]["need_validation"])
        self.assertEqual(result["messages"][0]["role"], "analysis")
        self.assertEqual(result["confidence"], 0.72)

    def test_evidence_from_mapping_supports_standard_fields(self):
        evidence = Evidence.from_mapping(
            {
                "id": "evidence_001",
                "type": "HYPOTHESIS",
                "content": "order_record may be missing a user_id index",
                "source": "analysis",
                "timestamp": "2026-08-25 10:07:22",
                "confidence": 0.42,
                "need_validation": True,
            },
            agent_role="analysis",
        ).to_dict()

        self.assertEqual(evidence["id"], "evidence_001")
        self.assertEqual(evidence["type"], "HYPOTHESIS")
        self.assertEqual(evidence["content"], "order_record may be missing a user_id index")
        self.assertEqual(evidence["detail"], evidence["content"])
        self.assertTrue(evidence["need_validation"])

    def test_agent_event_serializes_for_timeline_foundation(self):
        event = AgentEvent(
            actor={"type": "AGENT", "name": "investigator"},
            event_type="agent_status",
            metadata={
                "status": "running",
                "message": "Analyzing error.log",
                "tool": "log_input_summary",
            },
        ).to_dict()

        self.assertTrue(event["event_id"].startswith("event_"))
        self.assertEqual(event["event_type"], "agent_status")
        self.assertEqual(event["actor"]["type"], "AGENT")
        self.assertEqual(event["actor"]["name"], "investigator")
        self.assertEqual(event["evidence_id"], "")
        self.assertEqual(event["metadata"]["status"], "running")
        self.assertEqual(event["metadata"]["message"], "Analyzing error.log")
        self.assertEqual(event["type"], "agent_status")
        self.assertEqual(event["agent"], "investigator")
        self.assertEqual(event["status"], "running")


if __name__ == "__main__":
    unittest.main()
