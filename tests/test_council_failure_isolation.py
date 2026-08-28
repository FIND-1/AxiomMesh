import asyncio
import json
import unittest
from typing import List, cast
from unittest.mock import AsyncMock, patch

from backend import council, main
from backend.agent_models import AgentResultPayload


def _specialist_response(summary: str) -> dict:
    return {
        "content": json.dumps(
            {
                "summary": summary,
                "facts": [summary],
                "evidence": [
                    {
                        "type": "FACT",
                        "content": summary,
                        "source": "test",
                        "confidence": 0.8,
                    }
                ],
                "hypotheses": [],
                "unknowns": [],
                "confidence": 0.7,
            }
        )
    }


def _stage1_result(model_id: str | None = None) -> AgentResultPayload:
    return cast(
        AgentResultPayload,
        {
            "agent_role": "analysis",
            "agent_name": "Analysis Agent",
            "agent_instance_id": f"analysis:{model_id or council.COUNCIL_MODELS[0].id}",
            "model": model_id or council.COUNCIL_MODELS[0].id,
            "response": "analysis",
            "structured_output": {
                "summary": "Database timeout is visible.",
                "facts": ["database timeout"],
                "evidence": [
                    {
                        "id": "ev_timeout",
                        "type": "FACT",
                        "content": "database timeout",
                        "source": "error.log",
                        "confidence": 0.9,
                    }
                ],
                "hypotheses": [],
                "unknowns": [],
                "confidence": 0.8,
            },
            "evidence": [
                {
                    "id": "ev_timeout",
                    "type": "FACT",
                    "content": "database timeout",
                    "source": "error.log",
                    "confidence": 0.9,
                    "need_validation": False,
                }
            ],
            "messages": [],
            "confidence": 0.8,
            "tool_results": [],
        },
    )


def _valid_judge_response(root_cause: str = "Database timeout") -> dict:
    return {
        "content": json.dumps(
            {
                "verdict_summary": f"{root_cause} confirmed.",
                "root_cause": root_cause,
                "confirmed_evidence": [],
                "unverified_hypothesis": [],
                "next_actions": ["Inspect database availability."],
                "confidence": 0.7,
            }
        )
    }


def _stage2_result() -> AgentResultPayload:
    return cast(
        AgentResultPayload,
        {
            "agent_role": "judge",
            "agent_name": "Judge Agent",
            "agent_instance_id": f"judge:{council.CHAIRMAN_MODEL.id}",
            "model": council.CHAIRMAN_MODEL.id,
            "response": "judge",
            "structured_output": {
                "verdict_summary": "Database timeout confirmed by logs.",
                "root_cause": "Database timeout",
                "facts": ["database timeout"],
                "hypotheses": [],
                "unknowns": [],
                "evidence": [],
                "confirmed_evidence": [
                    {
                        "content": "database timeout",
                        "source": "error.log",
                        "credibility": "HIGH",
                    }
                ],
                "unverified_hypothesis": [],
                "next_actions": ["Inspect database availability."],
                "confidence": 0.7,
            },
            "evidence": [],
            "messages": [],
            "confidence": 0.7,
            "tool_results": [],
        },
    )


class Stage1FailureIsolationTest(unittest.IsolatedAsyncioTestCase):
    async def test_stage1_one_none_does_not_drop_successful_specialists(self):
        failed_model_id = council.COUNCIL_MODELS[0].id

        async def query_side_effect(model, messages, **kwargs):
            if model.id == failed_model_id:
                return None
            return _specialist_response(f"{model.name} succeeded")

        with patch("backend.council.query_model", new=AsyncMock(side_effect=query_side_effect)):
            results = await council.stage1_collect_responses("api error logs")

        self.assertEqual(len(results), len(council.COUNCIL_MODELS) - 1)
        self.assertNotIn(failed_model_id, {result["model"] for result in results})

    async def test_stage1_one_exception_does_not_cancel_other_specialists(self):
        failed_model_id = council.COUNCIL_MODELS[0].id

        async def query_side_effect(model, messages, **kwargs):
            if model.id == failed_model_id:
                raise RuntimeError("boom")
            return _specialist_response(f"{model.name} succeeded")

        with patch("backend.council.query_model", new=AsyncMock(side_effect=query_side_effect)):
            results = await council.stage1_collect_responses("api error logs")

        self.assertEqual(len(results), len(council.COUNCIL_MODELS) - 1)
        self.assertNotIn(failed_model_id, {result["model"] for result in results})

    async def test_stage1_all_none_returns_no_specialist_results(self):
        with patch("backend.council.query_model", new=AsyncMock(return_value=None)):
            results = await council.stage1_collect_responses("api error logs")

        self.assertEqual(results, [])

    async def test_specialist_cancelled_error_is_not_swallowed(self):
        incident_input = council.parse_incident_input("api error logs")
        with patch("backend.council.query_model", new=AsyncMock(side_effect=asyncio.CancelledError())):
            with self.assertRaises(asyncio.CancelledError):
                await council._run_specialist_agent(
                    council.SPECIALIST_BLUEPRINTS[0],
                    council.COUNCIL_MODELS[0],
                    incident_input,
                )

    async def test_specialist_event_sink_exception_does_not_break_specialist(self):
        def failing_event_sink(event):
            raise RuntimeError("sink failed")

        with patch("backend.council.query_model", new=AsyncMock(return_value=_specialist_response("analysis ok"))):
            result = await council._run_specialist_agent(
                council.SPECIALIST_BLUEPRINTS[0],
                council.COUNCIL_MODELS[0],
                council.parse_incident_input("api error logs"),
                event_sink=failing_event_sink,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["structured_output"]["summary"], "analysis ok")

    async def test_investigator_tool_exception_is_isolated(self):
        investigator_model_id = council.COUNCIL_MODELS[2].id

        async def query_side_effect(model, messages, **kwargs):
            return _specialist_response(f"{model.name} succeeded")

        with patch("backend.council.run_investigation_tools", side_effect=RuntimeError("tool failed")), patch(
            "backend.council.query_model",
            new=AsyncMock(side_effect=query_side_effect),
        ):
            results = await council.stage1_collect_responses("api error logs")

        self.assertEqual(len(results), len(council.COUNCIL_MODELS) - 1)
        self.assertNotIn(investigator_model_id, {result["model"] for result in results})

    async def test_run_full_all_none_returns_empty_fallback_results(self):
        with patch("backend.council.query_model", new=AsyncMock(return_value=None)):
            stage1, stage2, stage3, metadata = await council.run_full_council("api error logs")

        self.assertEqual(stage1, [])
        self.assertEqual(stage2["model"], "error")
        self.assertEqual(stage3["model"], "error")
        self.assertEqual(stage3["decision_summary"], "No decision available.")
        self.assertIn("agent_events", metadata)


class JudgeFailureIsolationTest(unittest.IsolatedAsyncioTestCase):
    async def test_judge_empty_primary_falls_back_to_next_candidate(self):
        stage1_results = [_stage1_result(council.COUNCIL_MODELS[0].id)]
        mock_query = AsyncMock(side_effect=[{"content": ""}, _valid_judge_response("DeepSeek judge")])

        with patch("backend.council.query_model", new=mock_query):
            result = await council.stage2_judge_deliberation("api error logs", stage1_results)

        self.assertEqual(mock_query.await_count, 2)
        self.assertEqual(result["model"], council.COUNCIL_MODELS[0].id)
        self.assertEqual(result["structured_output"]["root_cause"], "DeepSeek judge")

    async def test_judge_parse_failure_falls_back_to_next_candidate(self):
        stage1_results = [_stage1_result(council.COUNCIL_MODELS[0].id)]
        mock_query = AsyncMock(side_effect=[{"content": "not json"}, _valid_judge_response("Fallback judge")])

        with patch("backend.council.query_model", new=mock_query):
            result = await council.stage2_judge_deliberation("api error logs", stage1_results)

        self.assertEqual(mock_query.await_count, 2)
        self.assertEqual(result["model"], council.COUNCIL_MODELS[0].id)
        self.assertEqual(result["structured_output"]["root_cause"], "Fallback judge")

    async def test_judge_invalid_payload_falls_back_to_next_candidate(self):
        stage1_results = [_stage1_result(council.COUNCIL_MODELS[0].id)]
        mock_query = AsyncMock(side_effect=[{"content": "{}"}, _valid_judge_response("Valid judge")])

        with patch("backend.council.query_model", new=mock_query):
            result = await council.stage2_judge_deliberation("api error logs", stage1_results)

        self.assertEqual(mock_query.await_count, 2)
        self.assertEqual(result["structured_output"]["root_cause"], "Valid judge")

    async def test_judge_all_candidates_failed_uses_local_fallback(self):
        stage1_results = [_stage1_result(council.COUNCIL_MODELS[0].id)]
        mock_query = AsyncMock(side_effect=[None, {"content": "   "}])

        with patch("backend.council.query_model", new=mock_query):
            result = await council.stage2_judge_deliberation("api error logs", stage1_results)

        self.assertEqual(mock_query.await_count, 2)
        self.assertEqual(result["model"], council.CHAIRMAN_MODEL.id)
        self.assertEqual(result["agent_role"], "judge")
        self.assertIn("evidence_summary", result["structured_output"])
        self.assertIn("evidence_ranking", result["structured_output"])


class FinalFailureIsolationTest(unittest.IsolatedAsyncioTestCase):
    async def test_final_none_uses_judge_fallback(self):
        with patch("backend.council.query_model", new=AsyncMock(return_value=None)):
            result = await council.stage3_synthesize_final("api error logs", [_stage1_result()], _stage2_result())

        self.assertIn("## 结论", result["response"])
        self.assertIn("Database timeout confirmed by logs.", result["response"])
        self.assertIn("Database timeout", result["response"])
        self.assertIn("Inspect database availability.", result["response"])
        self.assertEqual(result["structured_output"]["root_cause"], "Database timeout")

    async def test_final_empty_uses_judge_fallback(self):
        with patch("backend.council.query_model", new=AsyncMock(return_value={"content": "   "})):
            result = await council.stage3_synthesize_final("api error logs", [_stage1_result()], _stage2_result())

        self.assertIn("## 结论", result["response"])
        self.assertIn("Database timeout confirmed by logs.", result["response"])
        self.assertIn("Database timeout", result["response"])
        self.assertIn("Inspect database availability.", result["response"])
        self.assertNotEqual(result["response"].strip(), "")

    async def test_final_exception_uses_judge_fallback(self):
        with patch("backend.council.query_model", new=AsyncMock(side_effect=RuntimeError("final failed"))):
            result = await council.stage3_synthesize_final("api error logs", [_stage1_result()], _stage2_result())

        self.assertIn("## 结论", result["response"])
        self.assertIn("Database timeout confirmed by logs.", result["response"])
        self.assertIn("Database timeout", result["response"])
        self.assertIn("Inspect database availability.", result["response"])
        self.assertEqual(result["decision_summary"], "Database timeout confirmed by logs.")



    async def test_final_result_construction_error_is_not_swallowed(self):
        with patch("backend.council.query_model", new=AsyncMock(return_value={"content": "## 结论\nDatabase timeout confirmed."})), patch(
            "backend.council._build_agent_result",
            side_effect=RuntimeError("local result construction bug"),
        ):
            with self.assertRaises(RuntimeError):
                await council.stage3_synthesize_final("api error logs", [_stage1_result()], _stage2_result())

class TitleFailureIsolationTest(unittest.IsolatedAsyncioTestCase):
    async def test_title_exception_uses_local_fallback(self):
        with patch("backend.council.query_model", new=AsyncMock(side_effect=RuntimeError("title failed"))):
            title = await council.generate_conversation_title("订单接口 500 错误排查")

        self.assertEqual(title, "订单接口 500 错误排查")

    async def test_title_exception_does_not_affect_council_result(self):
        async def query_side_effect(model, messages, **kwargs):
            if kwargs["workflow_role"] == "title":
                raise RuntimeError("title failed")
            if kwargs["workflow_role"] == "specialist":
                return _specialist_response(f"{model.name} succeeded")
            if kwargs["workflow_role"] == "judge":
                return _valid_judge_response()
            if kwargs["workflow_role"] == "final":
                return {"content": "## 结论\nDatabase timeout confirmed."}
            return None

        with patch("backend.council.query_model", new=AsyncMock(side_effect=query_side_effect)):
            title = await council.generate_conversation_title("订单接口 500 错误排查")
            stage1, stage2, stage3, metadata = await council.run_full_council("api error logs")

        self.assertEqual(title, "订单接口 500 错误排查")
        self.assertEqual(len(stage1), len(council.COUNCIL_MODELS))
        self.assertEqual(stage2["agent_role"], "judge")
        self.assertEqual(stage3["agent_role"], "final_decision")
        self.assertIn("agent_events", metadata)

    async def test_non_stream_title_exception_does_not_block_persistence(self):
        stage1 = [_stage1_result()]
        stage2 = _stage2_result()
        stage3 = council._build_final_fallback_result(stage2)
        metadata = {"agent_events": []}
        request = main.SendMessageRequest(content="订单接口 500 错误排查")

        with patch("backend.main.storage.get_conversation", return_value={"id": "conv-1", "messages": []}), patch(
            "backend.main.storage.add_user_message"
        ), patch("backend.main.storage.update_conversation_title") as mock_update_title, patch(
            "backend.main.storage.add_assistant_message"
        ) as mock_add_assistant, patch(
            "backend.main.run_full_council",
            new=AsyncMock(return_value=(stage1, stage2, stage3, metadata)),
        ), patch("backend.council.query_model", new=AsyncMock(side_effect=RuntimeError("title failed"))):
            response = await main.send_message("conv-1", request)

        mock_update_title.assert_called_once_with("conv-1", "订单接口 500 错误排查")
        mock_add_assistant.assert_called_once()
        self.assertEqual(response["stage3"], stage3)


class StreamPersistenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_stream_all_specialists_failed_persists_assistant_message(self):
        request = main.SendMessageRequest(content="api error logs")

        with patch("backend.main.storage.get_conversation", return_value={"id": "conv-1", "messages": []}), patch(
            "backend.main.storage.add_user_message"
        ), patch("backend.main.storage.update_conversation_title"), patch(
            "backend.main.storage.add_assistant_message"
        ) as mock_add_assistant, patch(
            "backend.main.generate_conversation_title",
            new=AsyncMock(return_value="api error logs"),
        ), patch(
            "backend.main.stage1_collect_responses",
            new=AsyncMock(return_value=[]),
        ):
            response = await main.send_message_stream("conv-1", request)
            chunks = [chunk async for chunk in response.body_iterator]

        self.assertTrue(any("stage3_complete" in chunk for chunk in chunks))
        mock_add_assistant.assert_called_once()
        _, stage1, stage2, stage3, metadata = mock_add_assistant.call_args.args
        self.assertEqual(stage1, [])
        self.assertEqual(stage2["model"], "error")
        self.assertEqual(stage3["model"], "error")
        self.assertIn("agent_events", metadata)


if __name__ == "__main__":
    unittest.main()
