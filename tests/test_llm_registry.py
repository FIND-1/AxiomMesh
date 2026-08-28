import unittest
from dataclasses import FrozenInstanceError
from typing import Any, Dict, cast
from unittest.mock import ANY, AsyncMock, patch

from backend import openrouter
from backend.config import CHAIRMAN_MODEL, COUNCIL_MODELS, MODEL_REGISTRY, TITLE_MODEL
from backend.llm.contracts import LLMResponse
from backend.llm.retry import RetryStats
from backend.llm.registry import ModelSpec, get_model, list_models, list_models_by_name, resolve_model


class ModelRegistryTest(unittest.TestCase):
    def test_get_model_returns_known_logical_model(self):
        model = get_model("qwen")

        self.assertEqual(
            model,
            ModelSpec(
                name="qwen",
                id="qwen/qwen3-235b-a22b-instruct-2507",
                provider="qwen",
                model_id="qwen3-235b-a22b-instruct-2507",
            ),
        )

    def test_resolve_model_supports_legacy_display_id(self):
        model = resolve_model("moonshot/kimi-k2.7-code")

        self.assertEqual(model.name, "kimi")
        self.assertEqual(model.provider, "kimi")
        self.assertEqual(model.model_id, "kimi-for-coding")

    def test_resolve_model_raises_for_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "Unknown model configuration: unknown"):
            resolve_model("unknown")

    def test_registry_exports_are_read_only(self):
        with self.assertRaises(TypeError):
            cast(Dict[str, ModelSpec], list_models_by_name())["new"] = ModelSpec(
                name="new",
                id="new/model",
                provider="new",
                model_id="new-model",
            )

        self.assertIs(MODEL_REGISTRY, list_models_by_name())

    def test_model_specs_are_frozen(self):
        model = resolve_model("gemini")

        with self.assertRaises(FrozenInstanceError):
            cast(Any, model).provider = "openai"

    def test_list_models_returns_immutable_snapshot(self):
        models = list_models()

        self.assertIsInstance(models, tuple)
        self.assertEqual(
            [model.name for model in models],
            ["openai", "deepseek", "gemini", "kimi", "qwen"],
        )

    def test_config_exports_existing_participation_model_objects(self):
        self.assertEqual(
            [model.name for model in COUNCIL_MODELS],
            ["deepseek", "kimi", "gemini", "qwen"],
        )
        self.assertIs(CHAIRMAN_MODEL, resolve_model("gemini"))
        self.assertIs(TITLE_MODEL, resolve_model("gemini"))


class OpenRouterRegistryCompatibilityTest(unittest.IsolatedAsyncioTestCase):
    async def test_query_model_accepts_legacy_display_id(self):
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
        ) as mock_query, patch("backend.openrouter.log_execution_record"):
            response = await openrouter.query_model(
                "moonshot/kimi-k2.7-code",
                [{"role": "user", "content": "hello"}],
            )

        assert response is not None
        self.assertEqual(response["content"], "ok")
        assert mock_query.await_args is not None
        args = mock_query.await_args.args
        self.assertEqual(args[:5], (
            "kimi",
            resolve_model("kimi"),
            [{"role": "user", "content": "hello"}],
            120.0,
            openrouter.KIMI_API_KEY,
        ))
        self.assertIsInstance(args[5], RetryStats)


if __name__ == "__main__":
    unittest.main()
