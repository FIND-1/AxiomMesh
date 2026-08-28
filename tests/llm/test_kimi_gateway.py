import unittest
from unittest.mock import patch

from backend import openrouter
from backend.llm.contracts import LLMRequest, LLMResponse, LLMResponseError, LLMUsage
from backend.llm.gateway import LLMGateway
from backend.llm.providers.kimi import KimiProvider


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
    response: FakeHTTPResponse | None = None
    last_timeout: float | None = None
    last_post: dict | None = None

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


class KimiProviderTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeAsyncClient.response = None
        FakeAsyncClient.last_timeout = None
        FakeAsyncClient.last_post = None

    async def test_kimi_request_and_response_map_to_llm_response(self):
        payload = {
            "id": "chatcmpl-kimi-1",
            "object": "chat.completion",
            "created": 1780000000,
            "choices": [
                {
                    "message": {
                        "content": "provider ok",
                        "reasoning_content": "reasoning kept",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 9,
                "total_tokens": 22,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
        }
        FakeAsyncClient.response = FakeHTTPResponse(
            payload,
            headers={"x-request-id": "req-kimi-1"},
        )
        provider = KimiProvider(
            api_key="test-key",
            base_url="https://kimi.test/chat/completions",
        )

        with patch("backend.llm.providers.kimi.httpx.AsyncClient", FakeAsyncClient):
            response = await provider.chat(
                LLMRequest(
                    model="kimi-for-coding",
                    messages=[
                        {"role": "system", "content": "Use incident context."},
                        {"role": "user", "content": "hello"},
                    ],
                    temperature=0.3,
                    timeout=8.5,
                )
            )

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "provider ok")
        self.assertEqual(response.reasoning, "reasoning kept")
        self.assertEqual(response.provider, "kimi")
        self.assertEqual(response.model, "kimi-for-coding")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.request_id, "req-kimi-1")
        self.assertIsNotNone(response.latency_ms)
        self.assertEqual(
            response.usage,
            LLMUsage(
                input_tokens=13,
                output_tokens=9,
                total_tokens=22,
                cached_tokens=3,
                reasoning_tokens=4,
            ),
        )
        self.assertEqual(
            response.raw_metadata,
            {
                "id": "chatcmpl-kimi-1",
                "object": "chat.completion",
                "created": 1780000000,
            },
        )
        self.assertEqual(FakeAsyncClient.last_timeout, 8.5)
        self.assertEqual(
            FakeAsyncClient.last_post,
            {
                "url": "https://kimi.test/chat/completions",
                "headers": {
                    "Authorization": "Bearer test-key",
                    "Content-Type": "application/json",
                },
                "json": {
                    "model": "kimi-for-coding",
                    "messages": [
                        {"role": "system", "content": "Use incident context."},
                        {"role": "user", "content": "hello"},
                    ],
                },
            },
        )
        assert FakeAsyncClient.last_post is not None
        self.assertNotIn("temperature", FakeAsyncClient.last_post["json"])

    async def test_kimi_empty_content_is_successful(self):
        FakeAsyncClient.response = FakeHTTPResponse(
            {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        )
        provider = KimiProvider(
            api_key="test-key",
            base_url="https://kimi.test/chat/completions",
        )

        with patch("backend.llm.providers.kimi.httpx.AsyncClient", FakeAsyncClient):
            response = await provider.chat(
                LLMRequest(
                    model="kimi-for-coding",
                    messages=[{"role": "user", "content": "hello"}],
                    timeout=8.5,
                )
            )

        self.assertEqual(response.content, "")

    async def test_kimi_invalid_content_raises_response_error(self):
        provider = KimiProvider(
            api_key="test-key",
            base_url="https://kimi.test/chat/completions",
        )
        cases = {
            "none content": {"choices": [{"message": {"content": None}}]},
            "missing content": {"choices": [{"message": {}}]},
            "list content": {"choices": [{"message": {"content": ["bad"]}}]},
            "malformed choices": {"choices": {"unexpected": True}},
        }

        for name, payload in cases.items():
            with self.subTest(name=name):
                FakeAsyncClient.response = FakeHTTPResponse(payload)
                with patch("backend.llm.providers.kimi.httpx.AsyncClient", FakeAsyncClient):
                    with self.assertRaises(LLMResponseError):
                        await provider.chat(
                            LLMRequest(
                                model="kimi-for-coding",
                                messages=[{"role": "user", "content": "hello"}],
                                timeout=8.5,
                            )
                        )

    async def test_kimi_usage_missing_is_not_fabricated(self):
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
        provider = KimiProvider(
            api_key="test-key",
            base_url="https://kimi.test/chat/completions",
        )

        with patch("backend.llm.providers.kimi.httpx.AsyncClient", FakeAsyncClient):
            response = await provider.chat(
                LLMRequest(
                    model="kimi-for-coding",
                    messages=[{"role": "user", "content": "hello"}],
                    timeout=8.5,
                )
            )

        self.assertEqual(response.content, "no usage")
        self.assertEqual(response.usage, LLMUsage())
        self.assertEqual(response.finish_reason, "stop")
        self.assertIsNone(response.request_id)


class LLMGatewayKimiTest(unittest.IsolatedAsyncioTestCase):
    async def test_gateway_dispatches_to_kimi_provider(self):
        expected_response = LLMResponse(
            content="ok",
            provider="kimi",
            model="kimi-for-coding",
        )
        provider = FakeProvider(expected_response)
        gateway = LLMGateway({"kimi": provider})
        request = LLMRequest(
            model="kimi-for-coding",
            messages=[{"role": "user", "content": "hello"}],
        )

        response = await gateway.chat("kimi", request)

        self.assertEqual(response, expected_response)
        self.assertEqual(provider.received_request, request)


class KimiLegacyCompatibilityTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeAsyncClient.response = None
        FakeAsyncClient.last_timeout = None
        FakeAsyncClient.last_post = None

    async def test_query_model_kimi_returns_legacy_dict(self):
        FakeAsyncClient.response = FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "legacy kimi ok",
                            "reasoning_content": "legacy kimi reasoning",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

        with patch("backend.openrouter.KIMI_API_KEY", "test-key"), patch(
            "backend.llm.providers.kimi.httpx.AsyncClient",
            FakeAsyncClient,
        ):
            response = await openrouter.query_model(
                "kimi",
                [{"role": "user", "content": "hello"}],
                timeout=12.0,
            )

        self.assertEqual(
            response,
            {
                "content": "legacy kimi ok",
                "reasoning_details": "legacy kimi reasoning",
            },
        )
        self.assertEqual(FakeAsyncClient.last_timeout, 12.0)
        assert FakeAsyncClient.last_post is not None
        self.assertEqual(
            FakeAsyncClient.last_post["url"],
            "https://api.kimi.com/coding/v1/chat/completions",
        )
        self.assertEqual(FakeAsyncClient.last_post["json"]["model"], "kimi-for-coding")
        self.assertEqual(
            FakeAsyncClient.last_post["json"]["messages"],
            [{"role": "user", "content": "hello"}],
        )


if __name__ == "__main__":
    unittest.main()
