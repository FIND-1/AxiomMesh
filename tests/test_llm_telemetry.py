import unittest
from unittest.mock import patch

import httpx

from backend.llm.contracts import (
    LLMRefusalError,
    LLMResponseError,
    LLMUnsupportedOutputError,
    LLMUsage,
)
from backend.llm.telemetry import LLMExecutionRecord, categorize_llm_error, log_execution_record


def make_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://llm.example.test/chat")
    response = httpx.Response(status_code=status_code, request=request)
    return httpx.HTTPStatusError(
        f"status {status_code}",
        request=request,
        response=response,
    )


class LLMTelemetryTest(unittest.TestCase):
    def test_execution_record_to_dict_serializes_usage_without_sensitive_fields(self):
        record = LLMExecutionRecord(
            execution_id="exec-1",
            logical_model="openai",
            model_id="openai/gpt-5-nano",
            provider="openai",
            provider_model_id="gpt-5-nano",
            success=True,
            attempt_count=2,
            retried=True,
            latency_ms=123.4,
            request_id="req-1",
            usage=LLMUsage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cached_tokens=2,
                reasoning_tokens=1,
            ),
        )

        payload = record.to_dict()

        self.assertEqual(
            payload,
            {
                "execution_id": "exec-1",
                "logical_model": "openai",
                "model_id": "openai/gpt-5-nano",
                "provider": "openai",
                "provider_model_id": "gpt-5-nano",
                "success": True,
                "attempt_count": 2,
                "retried": True,
                "latency_ms": 123.4,
                "run_id": None,
                "workflow_role": None,
                "error_category": None,
                "http_status": None,
                "request_id": "req-1",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "cached_tokens": 2,
                    "reasoning_tokens": 1,
                },
            },
        )
        for key in ("prompt", "response", "api_key", "authorization", "headers", "messages"):
            self.assertNotIn(key, payload)

    def test_categorize_llm_error_maps_current_categories(self):
        self.assertEqual(categorize_llm_error(httpx.ReadTimeout("timeout")), "timeout")
        self.assertEqual(categorize_llm_error(httpx.ConnectError("offline")), "network")
        self.assertEqual(categorize_llm_error(make_status_error(429)), "rate_limit")
        self.assertEqual(categorize_llm_error(make_status_error(503)), "server_error")
        self.assertEqual(categorize_llm_error(make_status_error(401)), "client_error")
        self.assertEqual(categorize_llm_error(ValueError("bad config")), "configuration_error")
        self.assertEqual(categorize_llm_error(TypeError("bad schema")), "schema_error")
        self.assertEqual(categorize_llm_error(LLMResponseError("bad content")), "schema_error")
        self.assertEqual(categorize_llm_error(LLMUnsupportedOutputError("tool only")), "unsupported_output")
        self.assertEqual(categorize_llm_error(LLMRefusalError("refused")), "model_refusal")

    def test_log_execution_record_uses_success_and_failure_levels(self):
        success = LLMExecutionRecord(
            execution_id="exec-success",
            logical_model="openai",
            model_id="openai/gpt-5-nano",
            provider="openai",
            provider_model_id="gpt-5-nano",
            success=True,
            attempt_count=1,
            retried=False,
            latency_ms=10.0,
        )
        failure = LLMExecutionRecord(
            execution_id="exec-failure",
            logical_model="openai",
            model_id="openai/gpt-5-nano",
            provider="openai",
            provider_model_id="gpt-5-nano",
            success=False,
            attempt_count=2,
            retried=True,
            latency_ms=20.0,
            error_category="network",
        )

        with patch("backend.llm.telemetry.logger.info") as mock_info, patch(
            "backend.llm.telemetry.logger.warning"
        ) as mock_warning:
            log_execution_record(success)
            log_execution_record(failure)

        mock_info.assert_called_once()
        mock_warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
