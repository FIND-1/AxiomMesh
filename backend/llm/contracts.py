"""Stable data contracts for LLM gateway and provider adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class LLMUsage:
    """Token usage reported by a provider when available."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Optional[int]]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass(frozen=True)
class LLMRequest:
    """Provider-neutral chat request."""

    model: str
    messages: Sequence[Mapping[str, Any]]
    temperature: Optional[float] = None
    timeout: Optional[float] = None

    @property
    def model_id(self) -> str:
        return self.model

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "model_id": self.model_id,
            "messages": [dict(message) for message in self.messages],
            "temperature": self.temperature,
            "timeout": self.timeout,
        }


@dataclass(frozen=True)
class LLMResponse:
    """Provider-neutral chat response."""

    content: str
    provider: str
    model: str
    reasoning: Optional[str] = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_ms: Optional[float] = None
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
            "request_id": self.request_id,
            "raw_metadata": dict(self.raw_metadata),
        }


@dataclass(frozen=True)
class LLMError:
    """Provider-neutral error payload for failed model calls."""

    provider: str
    model: str
    message: str
    error_type: Optional[str] = None
    request_id: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "message": self.message,
            "error_type": self.error_type,
            "request_id": self.request_id,
            "raw_metadata": dict(self.raw_metadata),
        }
