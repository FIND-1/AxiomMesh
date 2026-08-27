import unittest
from unittest.mock import patch

from backend import openrouter
from backend.llm.contracts import LLMRequest, LLMResponse, LLMUsage
from backend.llm.gateway import LLMGateway
from backend.llm.providers.gemini import GeminiProvider


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


class GeminiProviderTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeAsyncClient.response = None
        FakeAsyncClient.last_timeout = None
        FakeAsyncClient.last_post = None

    async def test_gemini_response_maps_to_llm_response(self):
        payload = {
            "responseId": "gemini-response-1",
            "modelVersion": "gemini-3.5-flash-lite",
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "system accepted"},
                            {"text": "user answered"},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 8,
                "totalTokenCount": 20,
                "cachedContentTokenCount": 4,
                "thoughtsTokenCount": 2,
            },
        }
        FakeAsyncClient.response = FakeHTTPResponse(
            payload,
            headers={"x-goog-request-id": "req-gemini-1"},
        )
        provider = GeminiProvider(
            api_key="test-key",
            base_url="https://gemini.test/v1beta",
        )

        with patch("backend.llm.providers.gemini.httpx.AsyncClient", FakeAsyncClient):
            response = await provider.chat(
                LLMRequest(
                    model="gemini-3.5-flash-lite",
                    messages=[
                        {"role": "system", "content": "Use incident context."},
                        {"role": "assistant", "content": "Previous answer."},
                        {"role": "user", "content": "Summarize."},
                    ],
                    temperature=0.3,
                    timeout=7.5,
                )
            )

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "system accepted\nuser answered")
        self.assertIsNone(response.reasoning)
        self.assertEqual(response.provider, "gemini")
        self.assertEqual(response.model, "gemini-3.5-flash-lite")
        self.assertEqual(response.finish_reason, "STOP")
        self.assertEqual(response.request_id, "req-gemini-1")
        self.assertIsNotNone(response.latency_ms)
        self.assertEqual(
            response.usage,
            LLMUsage(
                input_tokens=12,
                output_tokens=8,
                total_tokens=20,
                cached_tokens=4,
                reasoning_tokens=2,
            ),
        )
        self.assertEqual(
            response.raw_metadata,
            {
                "responseId": "gemini-response-1",
                "modelVersion": "gemini-3.5-flash-lite",
            },
        )
        self.assertEqual(FakeAsyncClient.last_timeout, 7.5)
        self.assertEqual(
            FakeAsyncClient.last_post,
            {
                "url": "https://gemini.test/v1beta/models/gemini-3.5-flash-lite:generateContent",
                "headers": {
                    "x-goog-api-key": "test-key",
                    "Content-Type": "application/json",
                },
                "json": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": "Use incident context."}],
                        },
                        {
                            "role": "model",
                            "parts": [{"text": "Previous answer."}],
                        },
                        {
                            "role": "user",
                            "parts": [{"text": "Summarize."}],
                        },
                    ]
                },
            },
        )
        self.assertNotIn("temperature", FakeAsyncClient.last_post["json"])

    async def test_gemini_usage_missing_is_not_fabricated(self):
        FakeAsyncClient.response = FakeHTTPResponse(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "no usage"}]},
                    }
                ]
            }
        )
        provider = GeminiProvider(
            api_key="test-key",
            base_url="https://gemini.test/v1beta",
        )

        with patch("backend.llm.providers.gemini.httpx.AsyncClient", FakeAsyncClient):
            response = await provider.chat(
                LLMRequest(
                    model="gemini-3.5-flash-lite",
                    messages=[{"role": "user", "content": "hello"}],
                    timeout=7.5,
                )
            )

        self.assertEqual(response.content, "no usage")
        self.assertEqual(response.usage, LLMUsage())
        self.assertIsNone(response.finish_reason)
        self.assertIsNone(response.request_id)


class LLMGatewayGeminiTest(unittest.IsolatedAsyncioTestCase):
    async def test_gateway_dispatches_to_gemini_provider(self):
        expected_response = LLMResponse(
            content="ok",
            provider="gemini",
            model="gemini-3.5-flash-lite",
        )
        provider = FakeProvider(expected_response)
        gateway = LLMGateway({"gemini": provider})
        request = LLMRequest(
            model="gemini-3.5-flash-lite",
            messages=[{"role": "user", "content": "hello"}],
        )

        response = await gateway.chat("gemini", request)

        self.assertEqual(response, expected_response)
        self.assertEqual(provider.received_request, request)


class GeminiLegacyCompatibilityTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeAsyncClient.response = None
        FakeAsyncClient.last_timeout = None
        FakeAsyncClient.last_post = None

    async def test_query_model_gemini_returns_legacy_dict(self):
        FakeAsyncClient.response = FakeHTTPResponse(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "legacy gemini ok"}]},
                        "finishReason": "STOP",
                    }
                ]
            }
        )

        with patch("backend.openrouter.GEMINI_API_KEY", "test-key"), patch(
            "backend.llm.providers.gemini.httpx.AsyncClient",
            FakeAsyncClient,
        ):
            response = await openrouter.query_model(
                "gemini",
                [
                    {"role": "system", "content": "Use incident context."},
                    {"role": "user", "content": "hello"},
                ],
                timeout=12.0,
            )

        self.assertEqual(
            response,
            {
                "content": "legacy gemini ok",
                "reasoning_details": None,
            },
        )
        self.assertEqual(FakeAsyncClient.last_timeout, 12.0)
        self.assertEqual(
            FakeAsyncClient.last_post["url"],
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent",
        )
        self.assertEqual(
            FakeAsyncClient.last_post["json"],
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "Use incident context."}],
                    },
                    {
                        "role": "user",
                        "parts": [{"text": "hello"}],
                    },
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
