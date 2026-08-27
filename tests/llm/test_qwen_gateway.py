import unittest
from unittest.mock import patch

from backend import openrouter
from backend.llm.contracts import LLMRequest, LLMResponse, LLMUsage
from backend.llm.gateway import LLMGateway
from backend.llm.providers.qwen import QwenProvider


class FakeHTTPResponse:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}
        self.raise_for_status_called = False

    def json(self):
        return self._payload

    def raise_for_status(self):
        self.raise_for_status_called = True


class FakeAsyncClient:
    response = None
    last_timeout = None
    last_post = None

    def __init__(self, *, timeout=None):
        self.timeout = timeout
        FakeAsyncClient.last_timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, *, headers=None, json=None):
        FakeAsyncClient.last_post = {
            "url": url,
            "headers": headers,
            "json": json,
        }
        return FakeAsyncClient.response


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.received_request = None

    async def chat(self, request):
        self.received_request = request
        return self.response


class QwenProviderTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeAsyncClient.response = None
        FakeAsyncClient.last_timeout = None
        FakeAsyncClient.last_post = None

    async def test_qwen_response_maps_to_llm_response(self):
        payload = {
            "id": "chatcmpl-qwen-1",
            "object": "chat.completion",
            "created": 1780000000,
            "choices": [
                {
                    "message": {
                        "content": "qwen ok",
                        "reasoning_content": "reasoning kept",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 5,
                "total_tokens": 18,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
        }
        FakeAsyncClient.response = FakeHTTPResponse(
            payload,
            headers={"x-request-id": "req-qwen-1"},
        )
        provider = QwenProvider(api_key="test-key", base_url="https://qwen.test/chat")

        with patch("backend.llm.providers.qwen.httpx.AsyncClient", FakeAsyncClient):
            response = await provider.chat(
                LLMRequest(
                    model="qwen3-235b-a22b-instruct-2507",
                    messages=[{"role": "user", "content": "hello"}],
                    timeout=9.5,
                )
            )

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "qwen ok")
        self.assertEqual(response.reasoning, "reasoning kept")
        self.assertEqual(response.provider, "qwen")
        self.assertEqual(response.model, "qwen3-235b-a22b-instruct-2507")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.request_id, "req-qwen-1")
        self.assertEqual(
            response.usage,
            LLMUsage(
                input_tokens=13,
                output_tokens=5,
                total_tokens=18,
                cached_tokens=2,
                reasoning_tokens=4,
            ),
        )
        self.assertIsNotNone(FakeAsyncClient.last_post)
        self.assertEqual(
            FakeAsyncClient.last_post["headers"]["Authorization"],
            "Bearer test-key",
        )

    async def test_qwen_usage_missing_is_not_fabricated(self):
        FakeAsyncClient.response = FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {"content": "no usage"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        provider = QwenProvider(api_key="test-key", base_url="https://qwen.test/chat")

        with patch("backend.llm.providers.qwen.httpx.AsyncClient", FakeAsyncClient):
            response = await provider.chat(
                LLMRequest(
                    model="qwen3-235b-a22b-instruct-2507",
                    messages=[{"role": "user", "content": "hello"}],
                    timeout=9.5,
                )
            )

        self.assertEqual(response.content, "no usage")
        self.assertIsNone(response.reasoning)
        self.assertEqual(response.usage, LLMUsage())
        self.assertIsNone(response.request_id)


class LLMGatewayTest(unittest.IsolatedAsyncioTestCase):
    async def test_gateway_dispatches_to_qwen_provider(self):
        expected_response = LLMResponse(
            content="ok",
            provider="qwen",
            model="qwen3-235b-a22b-instruct-2507",
        )
        provider = FakeProvider(expected_response)
        gateway = LLMGateway({"qwen": provider})
        request = LLMRequest(
            model="qwen3-235b-a22b-instruct-2507",
            messages=[{"role": "user", "content": "hello"}],
        )

        response = await gateway.chat("qwen", request)

        self.assertEqual(response, expected_response)
        self.assertEqual(provider.received_request, request)


class QwenLegacyCompatibilityTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeAsyncClient.response = None
        FakeAsyncClient.last_timeout = None
        FakeAsyncClient.last_post = None

    async def test_query_model_qwen_returns_legacy_dict(self):
        FakeAsyncClient.response = FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "legacy ok",
                            "reasoning_content": "legacy reasoning",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

        with patch("backend.openrouter.QWEN_API_KEY", "test-key"), patch(
            "backend.llm.providers.qwen.httpx.AsyncClient",
            FakeAsyncClient,
        ):
            response = await openrouter.query_model(
                "qwen",
                [{"role": "user", "content": "hello"}],
                timeout=12.0,
            )

        self.assertEqual(
            response,
            {
                "content": "legacy ok",
                "reasoning_details": "legacy reasoning",
            },
        )
        self.assertEqual(FakeAsyncClient.last_timeout, 12.0)
        self.assertEqual(
            FakeAsyncClient.last_post["json"]["model"],
            "qwen3-235b-a22b-instruct-2507",
        )
        self.assertEqual(
            FakeAsyncClient.last_post["json"]["messages"],
            [{"role": "user", "content": "hello"}],
        )


if __name__ == "__main__":
    unittest.main()
