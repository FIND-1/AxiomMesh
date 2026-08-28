"""Factory helpers for constructing provider adapter instances."""

from __future__ import annotations

from .providers.base import LLMProvider
from .providers.deepseek import DeepSeekProvider
from .providers.gemini import GeminiProvider
from .providers.kimi import KimiProvider
from .providers.openai import OpenAIProvider
from .providers.qwen import QwenProvider


def _require_api_key(api_key: str | None, provider_name: str) -> str:
    """Return the configured API key or raise a clear configuration error."""

    if not api_key:
        raise ValueError(f"Missing API key for provider: {provider_name}")
    return api_key


def get_provider(provider_name: str) -> LLMProvider:
    """Create a fresh provider adapter for the requested provider name."""

    if provider_name == "openai":
        from ..config import OPENAI_API_KEY, OPENAI_API_URL

        return OpenAIProvider(
            api_key=_require_api_key(OPENAI_API_KEY, provider_name),
            base_url=OPENAI_API_URL,
        )

    if provider_name == "deepseek":
        from ..config import DEEPSEEK_API_KEY, DEEPSEEK_API_URL

        return DeepSeekProvider(
            api_key=_require_api_key(DEEPSEEK_API_KEY, provider_name),
            base_url=DEEPSEEK_API_URL,
        )

    if provider_name == "gemini":
        from ..config import GEMINI_API_KEY, GEMINI_API_URL

        return GeminiProvider(
            api_key=_require_api_key(GEMINI_API_KEY, provider_name),
            base_url=GEMINI_API_URL,
        )

    if provider_name == "qwen":
        from ..config import QWEN_API_KEY, QWEN_API_URL

        return QwenProvider(
            api_key=_require_api_key(QWEN_API_KEY, provider_name),
            base_url=QWEN_API_URL,
        )

    if provider_name == "kimi":
        from ..config import KIMI_API_KEY, KIMI_API_URL

        return KimiProvider(
            api_key=_require_api_key(KIMI_API_KEY, provider_name),
            base_url=KIMI_API_URL,
        )

    raise ValueError(f"Unsupported provider: {provider_name}")
