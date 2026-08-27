"""Deterministic dispatch gateway for LLM provider adapters."""

from __future__ import annotations

from typing import Mapping

from .contracts import LLMRequest, LLMResponse
from .providers.base import LLMProvider


class LLMGateway:
    """Route LLM requests to a configured provider adapter."""

    def __init__(self, providers: Mapping[str, LLMProvider]) -> None:
        self._providers = dict(providers)

    async def chat(self, provider_name: str, request: LLMRequest) -> LLMResponse:
        provider = self._providers.get(provider_name)
        if provider is None:
            raise ValueError(f"Unsupported provider: {provider_name}")
        return await provider.chat(request)
