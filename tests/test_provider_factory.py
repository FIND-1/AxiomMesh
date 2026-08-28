import unittest
from unittest.mock import patch

from backend.llm.provider_factory import get_provider
from backend.llm.providers.deepseek import DeepSeekProvider
from backend.llm.providers.gemini import GeminiProvider
from backend.llm.providers.kimi import KimiProvider
from backend.llm.providers.openai import OpenAIProvider
from backend.llm.providers.qwen import QwenProvider


class ProviderFactoryTest(unittest.TestCase):
    def test_get_provider_constructs_all_supported_providers(self):
        cases = [
            ("openai", OpenAIProvider, "openai-key", "https://openai.test/v1/responses", "OPENAI_API_KEY", "OPENAI_API_URL"),
            ("deepseek", DeepSeekProvider, "deepseek-key", "https://deepseek.test/chat", "DEEPSEEK_API_KEY", "DEEPSEEK_API_URL"),
            ("gemini", GeminiProvider, "gemini-key", "https://gemini.test/v1beta", "GEMINI_API_KEY", "GEMINI_API_URL"),
            ("qwen", QwenProvider, "qwen-key", "https://qwen.test/chat", "QWEN_API_KEY", "QWEN_API_URL"),
            ("kimi", KimiProvider, "kimi-key", "https://kimi.test/chat/completions", "KIMI_API_KEY", "KIMI_API_URL"),
        ]

        for provider_name, expected_type, api_key, base_url, api_key_attr, base_url_attr in cases:
            with self.subTest(provider=provider_name), patch(f"backend.config.{api_key_attr}", api_key), patch(
                f"backend.config.{base_url_attr}", base_url
            ):
                provider = get_provider(provider_name)

                self.assertIsInstance(provider, expected_type)
                self.assertEqual(provider.api_key, api_key)
                self.assertEqual(provider.base_url, base_url)

    def test_get_provider_raises_for_unknown_provider(self):
        with self.assertRaisesRegex(ValueError, "Unsupported provider: unknown"):
            get_provider("unknown")

    def test_get_provider_returns_fresh_instances(self):
        with patch("backend.config.QWEN_API_KEY", "qwen-key"), patch(
            "backend.config.QWEN_API_URL", "https://qwen.test/chat"
        ):
            first = get_provider("qwen")
            second = get_provider("qwen")

        self.assertIsNot(first, second)


if __name__ == "__main__":
    unittest.main()