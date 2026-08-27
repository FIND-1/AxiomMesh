import unittest
from unittest.mock import patch

from backend import openrouter
from backend.llm.contracts import LLMRequest, LLMResponse, LLMUsage
from backend.llm.gateway import LLMGateway
from backend.llm.providers.openai import OpenAIProvider


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


class OpenAIProviderTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeAsyncClient.response = None
        FakeAsyncClient.last_timeout = None
        FakeAsyncClient.last_post = None

    async def test_openai_request_and_response_map_to_llm_response(self):
        payload = {
            "id": "resp-openai-1",
            "object": "response",
            "created_at": 1780000000,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "provider ok"},
                        {"type": "text", "text": "second chunk"},
                    ],
                }
            ],
            "usage": {
                "input_tokens": 15,
                "output_tokens": 9,
                "total_tokens": 24,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens_details": {"reasoning_tokens": 3},
            },
        }
        FakeAsyncClient.response = FakeHTTPResponse(
            payload,
            headers={"x-request-id": "req-openai-1"},
        )
        provider = OpenAIProvider(
            api_key="test-key",
            base_url="https://openai.test/v1/responses",
        )

        with patch("backend.llm.providers.openai.httpx.AsyncClient", FakeAsyncClient):
            response = await provider.chat(
                LLMRequest(
                    model="gpt-5-nano",
                    messages=[
                        {"role": "system", "content": "Use incident context."},
                        {"role": "user", "content": "hello"},
                    ],
                    temperature=0.3,
                    timeout=6.5,
                )
            )

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "provider ok\nsecond chunk")
        self.assertIsNone(response.reasoning)
        self.assertEqual(response.provider, "openai")
        self.assertEqual(response.model, "gpt-5-nano")
        self.assertEqual(response.finish_reason, "completed")
        self.assertEqual(response.request_id, "req-openai-1")
        self.assertIsNotNone(response.latency_ms)
        self.assertEqual(
            response.usage,
            LLMUsage(
                input_tokens=15,
                output_tokens=9,
                total_tokens=24,
                cached_tokens=4,
                reasoning_tokens=3,
            ),
        )
        self.assertEqual(
            response.raw_metadata,
            {
                "id": "resp-openai-1",
                "object": "response",
                "created_at": 1780000000,
                "status": "completed",
            },
        )
        self.assertEqual(FakeAsyncClient.last_timeout, 6.5)
        self.assertEqual(
            FakeAsyncClient.last_post,
            {
                "url": "https://openai.test/v1/responses",
                "headers": {
                    "Authorization": "Bearer test-key",
                    "Content-Type": "application/json",
                },
                "json": {
                    "model": "gpt-5-nano",
                    "input": [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Use incident context.",
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "hello",
                                }
                            ],
                        },
                    ],
                },
            },
        )
        self.assertNotIn("temperature", FakeAsyncClient.last_post["json"])

    async def test_openai_top_level_output_text_takes_precedence(self):
        FakeAsyncClient.response = FakeHTTPResponse(
            {
                "output_text": "top-level text",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "nested text"}],
                    }
                ],
            }
        )
        provider = OpenAIProvider(
            api_key="test-key",
            base_url="https://openai.test/v1/responses",
        )

        with patch("backend.llm.providers.openai.httpx.AsyncClient", FakeAsyncClient):
            response = await provider.chat(
                LLMRequest(
                    model="gpt-5-nano",
                    messages=[{"role": "user", "content": "hello"}],
                    timeout=6.5,
                )
            )

        self.assertEqual(response.content, "top-level text")

    async def test_openai_usage_missing_is_not_fabricated(self):
        FakeAsyncClient.response = FakeHTTPResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "no usage"}],
                    }
                ]
            }
        )
        provider = OpenAIProvider(
            api_key="test-key",
            base_url="https://openai.test/v1/responses",
        )

        with patch("backend.llm.providers.openai.httpx.AsyncClient", FakeAsyncClient):
            response = await provider.chat(
                LLMRequest(
                    model="gpt-5-nano",
                    messages=[{"role": "user", "content": "hello"}],
                    timeout=6.5,
                )
            )

        self.assertEqual(response.content, "no usage")
        self.assertEqual(response.usage, LLMUsage())
        self.assertIsNone(response.request_id)

    async def test_openai_malformed_output_returns_empty_content(self):
        FakeAsyncClient.response = FakeHTTPResponse({"output": {"unexpected": True}})
        provider = OpenAIProvider(
            api_key="test-key",
            base_url="https://openai.test/v1/responses",
        )

        with patch("backend.llm.providers.openai.httpx.AsyncClient", FakeAsyncClient):
            response = await provider.chat(
                LLMRequest(
                    model="gpt-5-nano",
                    messages=[{"role": "user", "content": "hello"}],
                    timeout=6.5,
                )
            )

        self.assertEqual(response.content, "")
        self.assertEqual(response.usage, LLMUsage())


class LLMGatewayOpenAITest(unittest.IsolatedAsyncioTestCase):
    async def test_gateway_dispatches_to_openai_provider(self):
        expected_response = LLMResponse(
            content="ok",
            provider="openai",
            model="gpt-5-nano",
        )
        provider = FakeProvider(expected_response)
        gateway = LLMGateway({"openai": provider})
        request = LLMRequest(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "hello"}],
        )

        response = await gateway.chat("openai", request)

        self.assertEqual(response, expected_response)
        self.assertEqual(provider.received_request, request)


class OpenAILegacyCompatibilityTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeAsyncClient.response = None
        FakeAsyncClient.last_timeout = None
        FakeAsyncClient.last_post = None

    async def test_query_model_openai_returns_legacy_dict(self):
        FakeAsyncClient.response = FakeHTTPResponse({"output_text": "legacy openai ok"})

        with patch("backend.openrouter.OPENAI_API_KEY", "test-key"), patch(
            "backend.llm.providers.openai.httpx.AsyncClient",
            FakeAsyncClient,
        ):
            response = await openrouter.query_model(
                "openai",
                [{"role": "user", "content": "hello"}],
                timeout=12.0,
            )

        self.assertEqual(
            response,
            {
                "content": "legacy openai ok",
                "reasoning_details": None,
            },
        )
        self.assertEqual(FakeAsyncClient.last_timeout, 12.0)
        self.assertEqual(
            FakeAsyncClient.last_post["url"],
            "https://api.openai.com/v1/responses",
        )
        self.assertEqual(FakeAsyncClient.last_post["json"]["model"], "gpt-5-nano")
        self.assertEqual(
            FakeAsyncClient.last_post["json"]["input"],
            [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
