import unittest

from backend.llm import LLMError, LLMRequest, LLMResponse, LLMUsage


class LLMContractsTest(unittest.TestCase):
    def test_llm_request_supports_current_chat_fields(self):
        request = LLMRequest(
            model="deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.2,
            timeout=30.0,
        )

        self.assertEqual(request.model, "deepseek/deepseek-v4-flash")
        self.assertEqual(request.model_id, request.model)
        self.assertEqual(request.messages[0]["role"], "user")
        self.assertEqual(request.temperature, 0.2)
        self.assertEqual(request.timeout, 30.0)
        self.assertEqual(
            request.to_dict(),
            {
                "model": "deepseek/deepseek-v4-flash",
                "model_id": "deepseek/deepseek-v4-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.2,
                "timeout": 30.0,
            },
        )

    def test_llm_usage_keeps_unavailable_token_values_as_none(self):
        usage = LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15)

        self.assertEqual(
            usage.to_dict(),
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cached_tokens": None,
                "reasoning_tokens": None,
            },
        )

    def test_llm_response_has_provider_neutral_fields(self):
        response = LLMResponse(
            content="ok",
            reasoning="because",
            provider="deepseek",
            model="deepseek-v4-flash",
            usage=LLMUsage(input_tokens=3, output_tokens=2, total_tokens=5),
            latency_ms=123.4,
            finish_reason="stop",
            request_id="req_123",
            raw_metadata={"vendor_field": "preserved"},
        )

        self.assertEqual(
            response.to_dict(),
            {
                "content": "ok",
                "reasoning": "because",
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                    "cached_tokens": None,
                    "reasoning_tokens": None,
                },
                "latency_ms": 123.4,
                "finish_reason": "stop",
                "request_id": "req_123",
                "raw_metadata": {"vendor_field": "preserved"},
            },
        )

    def test_llm_error_preserves_provider_context(self):
        error = LLMError(
            provider="gemini",
            model="gemini-3.5-flash-lite",
            message="rate limited",
            error_type="rate_limit",
            request_id="req_error",
            raw_metadata={"status_code": 429},
        )

        self.assertEqual(
            error.to_dict(),
            {
                "provider": "gemini",
                "model": "gemini-3.5-flash-lite",
                "message": "rate limited",
                "error_type": "rate_limit",
                "request_id": "req_error",
                "raw_metadata": {"status_code": 429},
            },
        )


if __name__ == "__main__":
    unittest.main()
