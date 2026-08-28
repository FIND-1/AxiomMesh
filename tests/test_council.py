import unittest
from unittest.mock import AsyncMock, patch

from backend import council
from backend.llm.aggregation import LLMExecutionCollector
from backend.llm.contracts import LLMUsage
from backend.llm.telemetry import LLMExecutionRecord


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
        mock_query = AsyncMock(return_value=None)
        with patch("backend.council.query_model", new=mock_query):
            title = await council.generate_conversation_title(
                """订单接口 500 错误排查

[error.log]
2026-08-25 10:00:01 ERROR db timeout
""",
                run_id="run-title",
            )

        self.assertEqual(title, "订单接口 500 错误排查")
        self.assertEqual(mock_query.await_args.kwargs["run_id"], "run-title")
        self.assertEqual(mock_query.await_args.kwargs["workflow_role"], "title")

    async def test_stage1_collect_responses_returns_role_results(self):
        specialist_responses = [
            {"content": '{"summary":"Log spike","findings":["500 errors"],"confidence":0.7}'},
            {"content": '{"summary":"Need stronger proof","gaps":["No timeline"],"confidence":0.5}'},
            {"content": '{"summary":"Check metrics","tool_requests":["Query Prometheus"],"confidence":0.6}'},
            {"content": '{"summary":"Second analysis pass","signals":["cache timeout"],"confidence":0.65}'},
        ]

        events = []

        mock_query = AsyncMock(side_effect=specialist_responses)
        with patch("backend.council.query_model", new=mock_query):
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
        run_ids = {call.kwargs["run_id"] for call in mock_query.await_args_list}
        self.assertEqual(len(run_ids), 1)
        self.assertTrue(next(iter(run_ids)))
        self.assertTrue(
            all(call.kwargs["workflow_role"] == "specialist" for call in mock_query.await_args_list)
        )

    async def test_stage2_judge_candidates_share_run_correlation(self):
        stage1_results = [
            {
                "agent_role": "analysis",
                "agent_name": "Analysis Agent",
                "model": council.COUNCIL_MODELS[0].id,
                "response": "Analysis response",
                "structured_output": {
                    "summary": "Cache errors increased",
                    "facts": ["cache timeout"],
                    "findings": [],
                    "evidence": [],
                    "hypotheses": [],
                    "unknowns": [],
                    "confidence": 0.7,
                },
                "messages": [],
                "evidence": [],
            }
        ]
        judge_response = {
            "content": (
                '{"verdict_summary":"Cache timeout confirmed.",'
                '"root_cause":"cache saturation",'
                '"confirmed_evidence":[],'
                '"unverified_hypothesis":[],'
                '"next_actions":[],'
                '"confidence":0.7}'
            )
        }
        mock_query = AsyncMock(side_effect=[None, judge_response])

        with patch("backend.council.query_model", new=mock_query):
            result = await council.stage2_judge_deliberation(
                "api error logs",
                stage1_results,
                run_id="run-judge",
            )

        self.assertEqual(result["agent_role"], "judge")
        self.assertEqual(mock_query.await_count, 2)
        self.assertEqual(
            {call.kwargs["run_id"] for call in mock_query.await_args_list},
            {"run-judge"},
        )
        self.assertTrue(
            all(call.kwargs["workflow_role"] == "judge" for call in mock_query.await_args_list)
        )
        self.assertNotEqual(
            mock_query.await_args_list[0].args[0].id,
            mock_query.await_args_list[1].args[0].id,
        )

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

        mock_query = AsyncMock(side_effect=query_side_effect)
        with patch("backend.council.query_model", new=mock_query):
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
        llm_run_ids = {call.kwargs["run_id"] for call in mock_query.await_args_list}
        self.assertEqual(len(llm_run_ids), 1)
        self.assertTrue(next(iter(llm_run_ids)))
        self.assertEqual(
            [call.kwargs["workflow_role"] for call in mock_query.await_args_list],
            ["specialist", "specialist", "specialist", "specialist", "judge", "final"],
        )

    async def test_title_and_council_share_external_execution_collector(self):
        responses = [
            {"content": "订单故障排查"},
            {"content": '{"summary":"Log spike","findings":["500 errors"],"confidence":0.7}'},
            {"content": '{"summary":"Need stronger proof","gaps":["No timeline"],"confidence":0.5}'},
            {"content": '{"summary":"Check metrics","tool_requests":["Query Prometheus"],"confidence":0.6}'},
            {"content": '{"summary":"Second analysis pass","signals":["cache timeout"],"confidence":0.65}'},
            {
                "content": (
                    '{"verdict_summary":"Cache timeout confirmed.",'
                    '"root_cause":"cache saturation",'
                    '"confirmed_evidence":[],'
                    '"unverified_hypothesis":[],'
                    '"next_actions":[],'
                    '"confidence":0.7}'
                )
            },
            {"content": "## Decision\nCache timeout confirmed."},
        ]
        collector = LLMExecutionCollector("run-api")

        async def query_side_effect(model, messages, **kwargs):
            execution_collector = kwargs["execution_collector"]
            execution_collector.add(
                LLMExecutionRecord(
                    execution_id=f"exec-{len(execution_collector.records()) + 1}",
                    run_id=kwargs["run_id"],
                    workflow_role=kwargs["workflow_role"],
                    logical_model=getattr(model, "name", None),
                    model_id=getattr(model, "id", str(model)),
                    provider=getattr(model, "provider", None),
                    provider_model_id=getattr(model, "model_id", None),
                    success=True,
                    attempt_count=1,
                    retried=False,
                    latency_ms=1.0,
                    usage=LLMUsage(total_tokens=1),
                )
            )
            return responses.pop(0)

        with patch("backend.council.query_model", new=AsyncMock(side_effect=query_side_effect)), patch(
            "backend.council.log_llm_run_summary"
        ) as mock_log:
            title = await council.generate_conversation_title(
                "订单接口 500 错误排查",
                run_id="run-api",
                execution_collector=collector,
            )
            stage1, stage2, stage3, metadata = await council.run_full_council(
                "api error logs",
                run_id="run-api",
                execution_collector=collector,
            )

        self.assertEqual(title, "订单故障排查")
        self.assertEqual(len(stage1), 4)
        self.assertEqual(stage2["agent_role"], "judge")
        self.assertEqual(stage3["agent_role"], "final_decision")
        self.assertNotIn("telemetry", metadata)
        mock_log.assert_not_called()
        summary = collector.summary()
        self.assertEqual(summary["run_id"], "run-api")
        self.assertEqual(summary["invocation_count"], 7)
        self.assertEqual(summary["confirmed_usage"]["total_tokens"]["known_sum"], 7)
        self.assertEqual(summary["by_role"]["title"]["invocation_count"], 1)
        self.assertEqual(summary["by_role"]["specialist"]["invocation_count"], 4)
        self.assertEqual(summary["by_role"]["judge"]["invocation_count"], 1)
        self.assertEqual(summary["by_role"]["final"]["invocation_count"], 1)
        self.assertEqual({record.run_id for record in collector.records()}, {"run-api"})

if __name__ == "__main__":
    unittest.main()
