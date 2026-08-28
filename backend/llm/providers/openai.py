"""OpenAI provider adapter."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional

import httpx

from ..contracts import (
    LLMRefusalError,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
    LLMUnsupportedOutputError,
    LLMUsage,
    validate_response_content,
)


def _messages_to_openai_input(
    messages: List[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert chat-style messages into OpenAI Responses API input."""
    return [
        {
            "role": message["role"],
            "content": [
                {
                    "type": "input_text",
                    "text": message["content"],
                }
            ],
        }
        for message in messages
    ]


def _raise_if_openai_refusal(data: Mapping[str, Any]) -> None:
    output = data.get("output")
    if not isinstance(output, list):
        return

    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content_items = item.get("content")
        if not isinstance(content_items, list):
            continue
        for content in content_items:
            if not isinstance(content, Mapping):
                continue
            if content.get("type") in {"refusal", "output_refusal"}:
                refusal = validate_response_content(content.get("refusal"))
                raise LLMRefusalError(refusal)


def _extract_openai_text(data: Mapping[str, Any]) -> str:
    """Extract assistant text from an OpenAI Responses API payload."""
    _raise_if_openai_refusal(data)

    if "output_text" in data:
        return validate_response_content(data.get("output_text")).strip()

    output = data.get("output", [])
    if not isinstance(output, list):
        raise LLMResponseError(
            f"LLM response output must be list, got {type(output).__name__}"
        )

    chunks: List[str] = []
    saw_text = False
    saw_unsupported_output = False
    for item in output:
        if not isinstance(item, Mapping):
            raise LLMResponseError(
                f"LLM response output item must be object, got {type(item).__name__}"
            )

        item_type = item.get("type")
        if not isinstance(item_type, str):
            raise LLMResponseError(
                f"LLM response output item type must be str, got {type(item_type).__name__}"
            )
        if item_type != "message":
            saw_unsupported_output = True
            continue

        content_items = item.get("content", [])
        if not isinstance(content_items, list):
            raise LLMResponseError(
                f"LLM response message content must be list, got {type(content_items).__name__}"
            )

        for content in content_items:
            if not isinstance(content, Mapping):
                raise LLMResponseError(
                    f"LLM response content item must be object, got {type(content).__name__}"
                )
            content_type = content.get("type")
            if not isinstance(content_type, str):
                raise LLMResponseError(
                    f"LLM response content item type must be str, got {type(content_type).__name__}"
                )
            if content_type in {"refusal", "output_refusal"}:
                refusal = validate_response_content(content.get("refusal"))
                raise LLMRefusalError(refusal)
            if content_type in {"output_text", "text"}:
                if "text" not in content:
                    raise LLMResponseError("LLM response text content is missing text")
                text = validate_response_content(content.get("text"))
                saw_text = True
                if text:
                    chunks.append(text)
                continue
            saw_unsupported_output = True

    if not saw_text:
        if saw_unsupported_output:
            raise LLMUnsupportedOutputError(
                "LLM response did not include consumable text output"
            )
        raise LLMResponseError("LLM response content must include output text")
    return "\n".join(chunks).strip()


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
        input_tokens=_usage_value(usage, "input_tokens"),
        output_tokens=_usage_value(usage, "output_tokens"),
        total_tokens=_usage_value(usage, "total_tokens"),
        cached_tokens=_nested_usage_value(
            usage,
            "input_tokens_details",
            "cached_tokens",
        ),
        reasoning_tokens=_nested_usage_value(
            usage,
            "output_tokens_details",
            "reasoning_tokens",
        ),
    )


def _request_id_from_headers(headers: Mapping[str, str]) -> Optional[str]:
    for header_name in ("x-request-id", "openai-request-id"):
        request_id = headers.get(header_name)
        if request_id:
            return request_id
    return None


class OpenAIProvider:
    """Adapter for OpenAI's Responses API."""

    provider_name = "openai"

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
            "input": _messages_to_openai_input(request.messages),
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
        raw_metadata = {
            key: data[key]
            for key in ("id", "object", "created_at", "status")
            if key in data
        }

        return LLMResponse(
            content=_extract_openai_text(data),
            reasoning=None,
            provider=self.provider_name,
            model=request.model,
            usage=_extract_usage(data),
            latency_ms=latency_ms,
            finish_reason=data.get("finish_reason") or data.get("status"),
            request_id=_request_id_from_headers(response.headers),
            raw_metadata=raw_metadata,
        )
