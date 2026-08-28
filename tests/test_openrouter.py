import unittest
from unittest.mock import ANY, AsyncMock, patch

import httpx

from backend import openrouter
from backend.llm.aggregation import LLMExecutionCollector
from backend.llm.contracts import (
    LLMRefusalError,
    LLMResponse,
    LLMResponseError,
    LLMUnsupportedOutputError,
    LLMUsage,
)
from backend.llm.retry import RetryStats


class FlakyProvider:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def chat(self, request):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://llm.example.test/chat")
    response = httpx.Response(status_code=status_code, request=request)
    return httpx.HTTPStatusError(
        f"status {status_code}",
        request=request,
        response=response,
    )


class OpenRouterDispatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_query_model_dispatches_kimi_provider(self):
        with patch(
            "backend.openrouter._query_provider",
            new=AsyncMock(
                return_value=LLMResponse(
                    content="ok",
                    reasoning=None,
                    provider="kimi",
                    model="kimi-for-coding",
                )
            ),
        ) as mock_query, patch("backend.openrouter.log_execution_record") as mock_log:
            response = await openrouter.query_model(
                "kimi",
                [{"role": "user", "content": "hello"}],
            )

        assert response is not None
        self.assertEqual(response["content"], "ok")
        self.assertEqual(set(response.keys()), {"content", "reasoning_details"})
        assert mock_query.await_args is not None
        args = mock_query.await_args.args
        self.assertEqual(args[:5], (
            "kimi",
            openrouter.resolve_model("kimi"),
            [{"role": "user", "content": "hello"}],
            120.0,
            openrouter.KIMI_API_KEY,
        ))
        self.assertIsInstance(args[5], RetryStats)
        assert mock_log.call_args is not None
        record = mock_log.call_args.args[0]
        self.assertTrue(record.success)
        self.assertEqual(record.logical_model, "kimi")
        self.assertEqual(record.model_id, "moonshot/kimi-k2.7-code")
        self.assertEqual(record.provider, "kimi")
        self.assertEqual(record.provider_model_id, "kimi-for-coding")
        self.assertEqual(record.attempt_count, 1)
        self.assertFalse(record.retried)
        self.assertIsNone(record.run_id)
        self.assertIsNone(record.workflow_role)

    async def test_query_model_captures_success_telemetry_without_exposing_usage_to_legacy_callers(self):
        provider = FlakyProvider(
            [
                LLMResponse(
                    content="ok",
                    reasoning="done",
                    provider="openai",
                    model="gpt-5-nano",
                    request_id="req-123",
                    usage=LLMUsage(
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        cached_tokens=2,
                        reasoning_tokens=1,
                    ),
                )
            ]
        )

        with patch("backend.openrouter.OPENAI_API_KEY", "test-key"), patch(
            "backend.openrouter.get_provider",
            return_value=provider,
        ), patch("backend.openrouter.log_execution_record") as mock_log:
            response = await openrouter.query_model(
                "openai",
                [{"role": "user", "content": "hello"}],
            )

        self.assertEqual(
            response,
            {
                "content": "ok",
                "reasoning_details": "done",
            },
        )
        assert response is not None
        self.assertEqual(set(response.keys()), {"content", "reasoning_details"})
        assert mock_log.call_args is not None
        record = mock_log.call_args.args[0]
        payload = record.to_dict()
        self.assertTrue(record.success)
        self.assertEqual(record.attempt_count, 1)
        self.assertFalse(record.retried)
        self.assertIsNone(record.run_id)
        self.assertIsNone(record.workflow_role)
        self.assertEqual(record.request_id, "req-123")
        self.assertEqual(
            payload["usage"],
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cached_tokens": 2,
                "reasoning_tokens": 1,
            },
        )
        for key in ("prompt", "response", "api_key", "authorization", "messages"):
            self.assertNotIn(key, payload)

    async def test_query_model_preserves_empty_content_in_legacy_dict(self):
        provider = FlakyProvider(
            [
                LLMResponse(
                    content="",
                    reasoning=None,
                    provider="openai",
                    model="gpt-5-nano",
                )
            ]
        )

        with patch("backend.openrouter.OPENAI_API_KEY", "test-key"), patch(
            "backend.openrouter.get_provider",
            return_value=provider,
        ), patch("backend.openrouter.log_execution_record") as mock_log:
            response = await openrouter.query_model(
                "openai",
                [{"role": "user", "content": "hello"}],
            )

        self.assertEqual(response, {"content": "", "reasoning_details": None})
        mock_log.assert_called_once()
        assert mock_log.call_args is not None
        self.assertTrue(mock_log.call_args.args[0].success)

    async def test_query_model_records_correlation_fields(self):
        provider = FlakyProvider(
            [
                LLMResponse(
                    content="ok",
                    reasoning=None,
                    provider="openai",
                    model="gpt-5-nano",
                )
            ]
        )

        with patch("backend.openrouter.OPENAI_API_KEY", "test-key"), patch(
            "backend.openrouter.get_provider",
            return_value=provider,
        ), patch("backend.openrouter.log_execution_record") as mock_log:
            response = await openrouter.query_model(
                "openai",
                [{"role": "user", "content": "hello"}],
                run_id="run-1",
                workflow_role="judge",
            )

        assert response is not None
        self.assertEqual(response["content"], "ok")
        assert mock_log.call_args is not None
        record = mock_log.call_args.args[0]
        self.assertEqual(record.run_id, "run-1")
        self.assertEqual(record.workflow_role, "judge")
        self.assertTrue(record.execution_id)

    async def test_query_model_collects_execution_record_without_changing_legacy_return(self):
        provider = FlakyProvider(
            [
                LLMResponse(
                    content="ok",
                    reasoning="kept",
                    provider="openai",
                    model="gpt-5-nano",
                    usage=LLMUsage(total_tokens=7),
                )
            ]
        )
        collector = LLMExecutionCollector("run-collector")

        with patch("backend.openrouter.OPENAI_API_KEY", "test-key"), patch(
            "backend.openrouter.get_provider",
            return_value=provider,
        ), patch("backend.openrouter.log_execution_record") as mock_log:
            response = await openrouter.query_model(
                "openai",
                [{"role": "user", "content": "hello"}],
                run_id="run-collector",
                workflow_role="final",
                execution_collector=collector,
            )

        self.assertEqual(response, {"content": "ok", "reasoning_details": "kept"})
        mock_log.assert_called_once()
        self.assertEqual(len(collector.records()), 1)
        assert mock_log.call_args is not None
        self.assertIs(collector.records()[0], mock_log.call_args.args[0])
        self.assertEqual(collector.summary()["confirmed_usage"]["total_tokens"]["known_sum"], 7)

    async def test_query_model_uses_distinct_execution_ids_within_same_run(self):
        provider = FlakyProvider(
            [
                LLMResponse(
                    content="first",
                    reasoning=None,
                    provider="openai",
                    model="gpt-5-nano",
                ),
                LLMResponse(
                    content="second",
                    reasoning=None,
                    provider="openai",
                    model="gpt-5-nano",
                ),
            ]
        )

        with patch("backend.openrouter.OPENAI_API_KEY", "test-key"), patch(
            "backend.openrouter.get_provider",
            return_value=provider,
        ), patch("backend.openrouter.log_execution_record") as mock_log:
            await openrouter.query_model(
                "openai",
                [{"role": "user", "content": "first"}],
                run_id="run-shared",
                workflow_role="judge",
            )
            await openrouter.query_model(
                "openai",
                [{"role": "user", "content": "second"}],
                run_id="run-shared",
                workflow_role="judge",
            )

        records = [call.args[0] for call in mock_log.call_args_list]
        self.assertEqual(len(records), 2)
        self.assertEqual({record.run_id for record in records}, {"run-shared"})
        self.assertEqual({record.workflow_role for record in records}, {"judge"})
        self.assertNotEqual(records[0].execution_id, records[1].execution_id)

    async def test_query_model_retries_transient_failure_before_returning_legacy_dict(self):
        provider = FlakyProvider(
            [
                httpx.ReadTimeout("timed out"),
                LLMResponse(
                    content="ok after retry",
                    reasoning="retry path",
                    provider="openai",
                    model="gpt-5-nano",
                    request_id="req-retry",
                    usage=LLMUsage(total_tokens=42),
                ),
            ]
        )

        with patch("backend.openrouter.OPENAI_API_KEY", "test-key"), patch(
            "backend.openrouter.get_provider",
            return_value=provider,
        ), patch("backend.llm.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep, patch(
            "backend.openrouter.log_execution_record"
        ) as mock_log:
            response = await openrouter.query_model(
                "openai",
                [{"role": "user", "content": "hello"}],
                run_id="run-retry",
                workflow_role="judge",
            )

        self.assertEqual(
            response,
            {
                "content": "ok after retry",
                "reasoning_details": "retry path",
            },
        )
        self.assertEqual(provider.calls, 2)
        mock_sleep.assert_awaited_once()
        mock_log.assert_called_once()
        assert mock_log.call_args is not None
        record = mock_log.call_args.args[0]
        self.assertTrue(record.success)
        self.assertTrue(record.execution_id)
        self.assertEqual(record.run_id, "run-retry")
        self.assertEqual(record.workflow_role, "judge")
        self.assertEqual(record.attempt_count, 2)
        self.assertTrue(record.retried)
        self.assertGreaterEqual(record.latency_ms, 0.0)
        self.assertEqual(record.request_id, "req-retry")
        self.assertEqual(record.usage.total_tokens, 42)

    async def test_query_model_returns_none_for_valid_non_text_output(self):
        provider = FlakyProvider([LLMUnsupportedOutputError("tool only")])

        with patch("backend.openrouter.OPENAI_API_KEY", "test-key"), patch(
            "backend.openrouter.get_provider",
            return_value=provider,
        ), patch("backend.llm.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep, patch(
            "backend.openrouter.log_execution_record"
        ) as mock_log:
            response = await openrouter.query_model(
                "openai",
                [{"role": "user", "content": "hello"}],
            )

        self.assertIsNone(response)
        self.assertEqual(provider.calls, 1)
        mock_sleep.assert_not_awaited()
        assert mock_log.call_args is not None
        record = mock_log.call_args.args[0]
        self.assertFalse(record.success)
        self.assertEqual(record.attempt_count, 1)
        self.assertFalse(record.retried)
        self.assertEqual(record.error_category, "unsupported_output")

    async def test_query_model_returns_none_for_model_refusal(self):
        provider = FlakyProvider([LLMRefusalError("refused")])

        with patch("backend.openrouter.OPENAI_API_KEY", "test-key"), patch(
            "backend.openrouter.get_provider",
            return_value=provider,
        ), patch("backend.llm.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep, patch(
            "backend.openrouter.log_execution_record"
        ) as mock_log:
            response = await openrouter.query_model(
                "openai",
                [{"role": "user", "content": "hello"}],
            )

        self.assertIsNone(response)
        self.assertEqual(provider.calls, 1)
        mock_sleep.assert_not_awaited()
        assert mock_log.call_args is not None
        record = mock_log.call_args.args[0]
        self.assertFalse(record.success)
        self.assertEqual(record.attempt_count, 1)
        self.assertFalse(record.retried)
        self.assertEqual(record.error_category, "model_refusal")

    async def test_query_model_returns_none_for_response_contract_failure(self):
        provider = FlakyProvider([LLMResponseError("bad content")])

        with patch("backend.openrouter.OPENAI_API_KEY", "test-key"), patch(
            "backend.openrouter.get_provider",
            return_value=provider,
        ), patch("backend.llm.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep, patch(
            "backend.openrouter.log_execution_record"
        ) as mock_log:
            response = await openrouter.query_model(
                "openai",
                [{"role": "user", "content": "hello"}],
            )

        self.assertIsNone(response)
        self.assertEqual(provider.calls, 1)
        mock_sleep.assert_not_awaited()
        assert mock_log.call_args is not None
        record = mock_log.call_args.args[0]
        self.assertFalse(record.success)
        self.assertEqual(record.attempt_count, 1)
        self.assertFalse(record.retried)
        self.assertEqual(record.error_category, "schema_error")
        self.assertIsNone(record.http_status)

    async def test_query_model_returns_none_after_retry_exhausted(self):
        provider = FlakyProvider(
            [
                httpx.ConnectError("offline"),
                httpx.ConnectError("still offline"),
            ]
        )

        with patch("backend.openrouter.OPENAI_API_KEY", "test-key"), patch(
            "backend.openrouter.get_provider",
            return_value=provider,
        ), patch("backend.llm.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep, patch(
            "backend.openrouter.log_execution_record"
        ) as mock_log:
            response = await openrouter.query_model(
                "openai",
                [{"role": "user", "content": "hello"}],
            )

        self.assertIsNone(response)
        self.assertEqual(provider.calls, 2)
        mock_sleep.assert_awaited_once()
        assert mock_log.call_args is not None
        record = mock_log.call_args.args[0]
        self.assertFalse(record.success)
        self.assertEqual(record.attempt_count, 2)
        self.assertTrue(record.retried)
        self.assertEqual(record.error_category, "network")
        self.assertIsNone(record.http_status)
        self.assertIsNone(record.request_id)
        self.assertIsNone(record.usage)

    async def test_query_model_records_non_retryable_http_failure(self):
        provider = FlakyProvider([make_status_error(401)])

        with patch("backend.openrouter.OPENAI_API_KEY", "test-key"), patch(
            "backend.openrouter.get_provider",
            return_value=provider,
        ), patch("backend.openrouter.log_execution_record") as mock_log:
            response = await openrouter.query_model(
                "openai",
                [{"role": "user", "content": "hello"}],
            )

        self.assertIsNone(response)
        self.assertEqual(provider.calls, 1)
        assert mock_log.call_args is not None
        record = mock_log.call_args.args[0]
        self.assertFalse(record.success)
        self.assertEqual(record.attempt_count, 1)
        self.assertFalse(record.retried)
        self.assertEqual(record.error_category, "client_error")
        self.assertEqual(record.http_status, 401)

    async def test_query_model_records_missing_api_key_without_invoking_provider(self):
        with patch("backend.openrouter.OPENAI_API_KEY", None), patch(
            "backend.openrouter.get_provider",
            side_effect=AssertionError("provider should not be created"),
        ), patch("backend.openrouter.log_execution_record") as mock_log:
            response = await openrouter.query_model(
                "openai",
                [{"role": "user", "content": "hello"}],
                run_id="run-missing",
                workflow_role="judge",
            )

        self.assertIsNone(response)
        assert mock_log.call_args is not None
        record = mock_log.call_args.args[0]
        self.assertFalse(record.success)
        self.assertEqual(record.attempt_count, 0)
        self.assertFalse(record.retried)
        self.assertEqual(record.error_category, "configuration_error")
        self.assertEqual(record.run_id, "run-missing")
        self.assertEqual(record.workflow_role, "judge")
        self.assertEqual(record.provider, "openai")
        self.assertEqual(record.provider_model_id, "gpt-5-nano")


if __name__ == "__main__":
    unittest.main()
