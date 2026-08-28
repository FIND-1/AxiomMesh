"""Gemini provider adapter."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import httpx

from ..contracts import (
    LLMRequest,
    LLMResponse,
    LLMResponseError,
    LLMUnsupportedOutputError,
    LLMUsage,
    validate_response_content,
)

_GEMINI_NON_TEXT_PART_KEYS = {
    "functionCall",
    "functionResponse",
    "inlineData",
    "fileData",
    "executableCode",
    "codeExecutionResult",
    "toolCall",
}


def _messages_to_gemini_contents(
    messages: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert chat-style messages into Gemini contents."""
    role_map = {
        "assistant": "model",
        "model": "model",
        "system": "user",
        "user": "user",
    }
    return [
        {
            "role": role_map.get(message["role"], "user"),
            "parts": [{"text": message["content"]}],
        }
        for message in messages
    ]


def _extract_gemini_text(data: Mapping[str, Any]) -> str:
    """Extract text from a Gemini generateContent response."""
    candidates = data.get("candidates", [])
    if not isinstance(candidates, list):
        raise LLMResponseError(
            f"LLM response candidates must be list, got {type(candidates).__name__}"
        )

    chunks: List[str] = []
    saw_text = False
    saw_unsupported_output = False

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise LLMResponseError(
                f"LLM response candidate must be object, got {type(candidate).__name__}"
            )
        content = candidate.get("content")
        if not isinstance(content, Mapping):
            raise LLMResponseError(
                f"LLM response candidate content must be object, got {type(content).__name__}"
            )
        parts = content.get("parts")
        if not isinstance(parts, list):
            raise LLMResponseError(
                f"LLM response content parts must be list, got {type(parts).__name__}"
            )
        for part in parts:
            if not isinstance(part, Mapping):
                raise LLMResponseError(
                    f"LLM response part must be object, got {type(part).__name__}"
                )
            if "text" not in part:
                if any(key in part for key in _GEMINI_NON_TEXT_PART_KEYS):
                    saw_unsupported_output = True
                    continue
                raise LLMResponseError("LLM response part is missing text")
            text = validate_response_content(part.get("text"))
            if part.get("thought") is True:
                saw_unsupported_output = True
                continue
            saw_text = True
            if text:
                chunks.append(text)

    if not saw_text:
        if saw_unsupported_output:
            raise LLMUnsupportedOutputError(
                "LLM response did not include consumable text output"
            )
        raise LLMResponseError("LLM response content must include a text part")
    return "\n".join(chunks).strip()


def _usage_value(usage: Mapping[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = usage.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _extract_usage(data: Mapping[str, Any]) -> LLMUsage:
    usage = data.get("usageMetadata")
    if not isinstance(usage, Mapping):
        usage = data.get("usage_metadata")
    if not isinstance(usage, Mapping):
        return LLMUsage()

    return LLMUsage(
        input_tokens=_usage_value(usage, "promptTokenCount", "prompt_token_count"),
        output_tokens=_usage_value(
            usage,
            "candidatesTokenCount",
            "candidates_token_count",
        ),
        total_tokens=_usage_value(usage, "totalTokenCount", "total_token_count"),
        cached_tokens=_usage_value(
            usage,
            "cachedContentTokenCount",
            "cached_content_token_count",
        ),
        reasoning_tokens=_usage_value(
            usage,
            "thoughtsTokenCount",
            "thoughts_token_count",
        ),
    )


def _request_id_from_headers(headers: Mapping[str, str]) -> Optional[str]:
    for header_name in ("x-request-id", "x-goog-request-id"):
        request_id = headers.get(header_name)
        if request_id:
            return request_id
    return None


class GeminiProvider:
    """Adapter for Gemini's generateContent API."""

    provider_name = "gemini"

    def __init__(self, *, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url

    async def chat(self, request: LLMRequest) -> LLMResponse:
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "contents": _messages_to_gemini_contents(request.messages),
        }
        url = f"{self.base_url}/models/{request.model}:generateContent"

        started_at = perf_counter()
        async with httpx.AsyncClient(timeout=request.timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        latency_ms = round((perf_counter() - started_at) * 1000, 2)

        data = response.json()
        if not isinstance(data, Mapping):
            raise LLMResponseError(
                f"LLM provider response must be object, got {type(data).__name__}"
            )
        candidates = data.get("candidates", [])
        first_candidate = candidates[0] if isinstance(candidates, list) and candidates else {}

        raw_metadata = {
            key: data[key]
            for key in ("responseId", "modelVersion")
            if key in data
        }

        return LLMResponse(
            content=_extract_gemini_text(data),
            reasoning=None,
            provider=self.provider_name,
            model=request.model,
            usage=_extract_usage(data),
            latency_ms=latency_ms,
            finish_reason=(
                first_candidate.get("finishReason")
                if isinstance(first_candidate, Mapping)
                else None
            ),
            request_id=_request_id_from_headers(response.headers),
            raw_metadata=raw_metadata,
        )
