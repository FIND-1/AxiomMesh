"""Static model registry for logical model routing."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ModelSpec:
    """Static model identity and routing metadata."""

    name: str
    id: str
    provider: str
    model_id: str


_MODEL_SPECS_BY_NAME = MappingProxyType(
    {
        "openai": ModelSpec(
            name="openai",
            id="openai/gpt-5-nano",
            provider="openai",
            model_id="gpt-5-nano",
        ),
        "deepseek": ModelSpec(
            name="deepseek",
            id="deepseek/deepseek-v4-flash",
            provider="deepseek",
            model_id="deepseek-v4-flash",
        ),
        "gemini": ModelSpec(
            name="gemini",
            id="google/gemini-3.5-flash-lite",
            provider="gemini",
            model_id="gemini-3.5-flash-lite",
        ),
        "kimi": ModelSpec(
            name="kimi",
            id="moonshot/kimi-k2.7-code",
            provider="kimi",
            model_id="kimi-for-coding",
        ),
        "qwen": ModelSpec(
            name="qwen",
            id="qwen/qwen3-235b-a22b-instruct-2507",
            provider="qwen",
            model_id="qwen3-235b-a22b-instruct-2507",
        ),
    }
)

_MODEL_SPECS_BY_ID = MappingProxyType(
    {spec.id: spec for spec in _MODEL_SPECS_BY_NAME.values()}
)


def get_model(name: str) -> ModelSpec | None:
    """Return a model spec by logical name."""

    return _MODEL_SPECS_BY_NAME.get(name)


def resolve_model(model: ModelSpec | str) -> ModelSpec:
    """Resolve a model spec from a logical name, legacy id, or spec."""

    if isinstance(model, ModelSpec):
        return model

    resolved = _MODEL_SPECS_BY_NAME.get(model) or _MODEL_SPECS_BY_ID.get(model)
    if resolved is None:
        raise ValueError(f"Unknown model configuration: {model}")
    return resolved


def list_models() -> tuple[ModelSpec, ...]:
    """Return all known model specs as an immutable snapshot."""

    return tuple(_MODEL_SPECS_BY_NAME.values())


def list_models_by_name() -> Mapping[str, ModelSpec]:
    """Return a read-only view keyed by logical model name."""

    return _MODEL_SPECS_BY_NAME
