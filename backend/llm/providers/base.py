"""Provider adapter protocol for LLM chat calls."""

from __future__ import annotations

from typing import Protocol

from ..contracts import LLMRequest, LLMResponse


class LLMProvider(Protocol):
    """Minimal provider contract for chat completion adapters."""

    async def chat(self, request: LLMRequest) -> LLMResponse:
        ...
