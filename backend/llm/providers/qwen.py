"""Qwen provider adapter."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Mapping, Optional

import httpx

from ..contracts import (
    LLMRequest,
    LLMResponse,
    LLMResponseError,
    LLMUsage,
    validate_response_content,
)


def _usage_value(usage: Mapping[str, Any], key: str) -> Optional[int]:
    value = usage.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nested_usage_value(
    usage: Mapping[str, Any],
    container_key: str,
    key: str,
) -> Optional[int]:
    container = usage.get(container_key)
    if not isinstance(container, Mapping):
        return None
    return _usage_value(container, key)


def _extract_usage(data: Mapping[str, Any]) -> LLMUsage:
    usage = data.get("usage")
    if not isinstance(usage, Mapping):
        return LLMUsage()

    return LLMUsage(
        input_tokens=_usage_value(usage, "prompt_tokens"),
        output_tokens=_usage_value(usage, "completion_tokens"),
        total_tokens=_usage_value(usage, "total_tokens"),
        cached_tokens=_nested_usage_value(
            usage,
            "prompt_tokens_details",
            "cached_tokens",
        ),
        reasoning_tokens=_nested_usage_value(
            usage,
            "completion_tokens_details",
            "reasoning_tokens",
        ),
    )


def _request_id_from_headers(headers: Mapping[str, str]) -> Optional[str]:
    request_id = headers.get("x-request-id")
    if request_id:
        return request_id
    return None


class QwenProvider:
    """Adapter for Qwen's OpenAI-compatible Model Studio endpoint."""

    provider_name = "qwen"

    def __init__(self, *, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url

    async def chat(self, request: LLMRequest) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": [dict(message) for message in request.messages],
        }

        started_at = perf_counter()
        async with httpx.AsyncClient(timeout=request.timeout) as client:
            response = await client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
        latency_ms = round((perf_counter() - started_at) * 1000, 2)

        data = response.json()
        if not isinstance(data, Mapping):
            raise LLMResponseError(
                f"LLM provider response must be object, got {type(data).__name__}"
            )
        choices = data.get("choices", [])
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        message = first_choice.get("message") if isinstance(first_choice, Mapping) else None
        content = validate_response_content(
            message.get("content") if isinstance(message, Mapping) else None
        )
        reasoning = message.get("reasoning_content") if isinstance(message, Mapping) else None

        raw_metadata = {
            key: data[key]
            for key in ("id", "object", "created")
            if key in data
        }

        return LLMResponse(
            content=content,
            reasoning=reasoning,
            provider=self.provider_name,
            model=request.model,
            usage=_extract_usage(data),
            latency_ms=latency_ms,
            finish_reason=(
                first_choice.get("finish_reason")
                if isinstance(first_choice, Mapping)
                else None
            ),
            request_id=_request_id_from_headers(response.headers),
            raw_metadata=raw_metadata,
        )
