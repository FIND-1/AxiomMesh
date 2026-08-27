import unittest
from unittest.mock import AsyncMock, patch

from backend import council


class CouncilHelpersTest(unittest.TestCase):
    def test_extract_json_object_from_fenced_block(self):
        raw = """Here is the result:

```json
{"summary": "Disk pressure", "confidence": 0.8}
```
"""
        parsed = council._extract_json_object(raw)

        self.assertEqual(parsed, {"summary": "Disk pressure", "confidence": 0.8})

    def test_normalize_specialist_payload_defaults_missing_fields(self):
        payload = council._normalize_specialist_payload(
            {"summary": "Something happened"},
            raw_text="Something happened",
            agent_role="analysis",
        )

        self.assertEqual(payload["agent_role"], "analysis")
        self.assertEqual(payload["summary"], "Something happened")
        self.assertEqual(payload["facts"], [])
        self.assertEqual(payload["findings"], [])
        self.assertEqual(payload["evidence"], [])
        self.assertEqual(payload["hypotheses"], [])
        self.assertEqual(payload["unknowns"], [])
        self.assertEqual(payload["confidence"], 0.35)

    def test_investigator_payload_does_not_promote_hypotheses(self):
        payload = council._normalize_specialist_payload(
            {
                "summary": "Collected logs",
                "facts": ["HikariPool connection is not available"],
                "hypotheses": [{"content": "Missing index caused exhaustion"}],
                "evidence": [
                    {
                        "type": "HYPOTHESIS",
                        "content": "Missing index caused exhaustion",
                        "source": "investigator",
                        "confidence": 0.4,
                    }
                ],
                "unknowns": ["Need query execution plan"],
            },
            raw_text="{}",
            agent_role="investigator",
        )

        self.assertEqual(payload["hypotheses"], [])
        self.assertIn("HikariPool connection is not available", payload["facts"])
        self.assertTrue(all(item["type"] != "HYPOTHESIS" for item in payload["evidence"]))
        self.assertEqual(payload["evidence"][0]["type"], "FACT")

    def test_judge_payload_separates_confirmed_and_unverified(self):
        payload = council._normalize_judge_payload(
            {
                "incident_level": "high",
                "direct_cause": "HikariPool exhausted",
                "root_cause": "unknown",
                "verdict_summary": "连接池耗尽已确认，根因仍需验证。",
                "confirmed_evidence": [
                    {
                        "type": "FACT",
                        "content": "HikariPool connection is not available",
                        "source": "error.log",
                        "credibility": "HIGH",
                    }
                ],
                "unverified_hypothesis": [
                    {
                        "content": "慢 SQL 占满连接池",
                        "confidence": "MEDIUM",
                        "needed_evidence": "需要 SQL 执行耗时和连接占用证据",
                    }
                ],
                "next_actions": ["采集慢查询日志"],
                "confidence": 0.7,
            },
            raw_text="{}",
        )

        self.assertEqual(payload["incident_level"], "high")
        self.assertEqual(payload["confirmed_evidence"][0]["credibility"], "HIGH")
        self.assertEqual(payload["unverified_hypothesis"][0]["confidence"], "MEDIUM")
        self.assertEqual(payload["next_actions"], ["采集慢查询日志"])

    def test_fallback_conversation_title_uses_user_intent_not_log_body(self):
        title = council._fallback_conversation_title(
            """主应用向子应用传参及常见问题

```error.log
2026-08-25 10:00:01 ERROR cache timeout
```
"""
        )

        self.assertEqual(title, "主应用向子应用传参及常见问题")


class CouncilFlowTest(unittest.IsolatedAsyncioTestCase):
    async def test_generate_conversation_title_falls_back_when_model_unavailable(self):
        with patch("backend.council.query_model", new=AsyncMock(return_value=None)):
            title = await council.generate_conversation_title(
                """订单接口 500 错误排查

[error.log]
2026-08-25 10:00:01 ERROR db timeout
"""
            )

        self.assertEqual(title, "订单接口 500 错误排查")

    async def test_stage1_collect_responses_returns_role_results(self):
        specialist_responses = [
            {"content": '{"summary":"Log spike","findings":["500 errors"],"confidence":0.7}'},
            {"content": '{"summary":"Need stronger proof","gaps":["No timeline"],"confidence":0.5}'},
            {"content": '{"summary":"Check metrics","tool_requests":["Query Prometheus"],"confidence":0.6}'},
            {"content": '{"summary":"Second analysis pass","signals":["cache timeout"],"confidence":0.65}'},
        ]

        events = []

        with patch("backend.council.query_model", new=AsyncMock(side_effect=specialist_responses)):
            results = await council.stage1_collect_responses(
                """api error logs

```error.log
2026-08-25 10:00:01 ERROR cache timeout
```
""",
                event_sink=events.append,
            )

        self.assertEqual(len(results), 4)
        self.assertEqual(results[0]["agent_role"], "analysis")
        self.assertEqual(results[1]["agent_role"], "critic")
        self.assertEqual(results[2]["agent_role"], "investigator")
        self.assertEqual(results[3]["agent_role"], "analysis")
        self.assertEqual(results[2]["tool_results"][0]["tool_name"], "log_input_summary")
        self.assertIn("messages", results[0])
        self.assertEqual(results[0]["messages"][0]["role"], "analysis")
        self.assertIn("evidence", results[0])
        self.assertIn("### Analysis Agent", results[0]["response"])
        self.assertIn("agent_instance_id", results[0])
        self.assertTrue(any(event["event_type"] == "agent_status" for event in events))
        self.assertTrue(all("event_id" in event for event in events))
        self.assertTrue(all("actor" in event for event in events))

    async def test_run_full_council_returns_structured_metadata(self):
        query_side_effect = [
            {"content": '{"summary":"Log spike","findings":["500 errors"],"confidence":0.7}'},
            {"content": '{"summary":"Need stronger proof","gaps":["No timeline"],"confidence":0.5}'},
            {"content": '{"summary":"Check metrics","tool_requests":["Query Prometheus"],"confidence":0.6}'},
            {"content": '{"summary":"Second analysis pass","signals":["cache timeout"],"confidence":0.65}'},
            {
                "content": (
                    '{"verdict_summary":"Redis latency is the leading cause.",'
                    '"root_cause":"Redis saturation",'
                    '"incident_severity":"high",'
                    '"confidence":0.78,'
                    '"evidence":[{"agent_role":"analysis","detail":"500 spikes after cache errors","credibility":"high"}],'
                    '"scorecard":[{"agent_role":"analysis","agent_name":"Analysis Agent","evidence_score":5,"reasoning_score":4,"actionability_score":4,"notes":"Strong log detail"}],'
                    '"gaps":["Need cache hit ratio"],'
                    '"recommendations":["Inspect Redis saturation"],'
                    '"minority_view":"Could still be downstream DB contention"}'
                )
            },
            {
                "content": (
                    "## Decision\nRedis saturation is the likely trigger.\n\n"
                    "## Why This Is Most Likely\nCache errors line up with the outage.\n\n"
                    "## Key Evidence\n- 500 spikes after cache errors\n\n"
                    "## Immediate Next Actions\n- Inspect Redis saturation\n\n"
                    "## Confidence\n78%"
                )
            },
        ]

        with patch("backend.council.query_model", new=AsyncMock(side_effect=query_side_effect)):
            stage1, stage2, stage3, metadata = await council.run_full_council("api error logs")

        self.assertEqual(len(stage1), 4)
        self.assertEqual(stage2["agent_role"], "judge")
        self.assertEqual(stage2["structured_output"]["root_cause"], "Redis saturation")
        self.assertEqual(stage3["agent_role"], "final_decision")
        self.assertIn("role_assignments", metadata)
        self.assertEqual(len(metadata["role_assignments"]), 4)
        self.assertIn("investigation_tools", metadata)
        self.assertIn("agent_events", metadata)
        self.assertGreater(len(metadata["agent_events"]), 0)
        self.assertTrue(all("event_type" in event for event in metadata["agent_events"]))
        self.assertIn("evidence", metadata)
        self.assertEqual(metadata["agent_messages"][0]["role"], "analysis")


if __name__ == "__main__":
    unittest.main()
