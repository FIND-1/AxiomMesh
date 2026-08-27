"""LLM API client for making OpenAI and DeepSeek requests."""

from typing import Any, Dict, List, Optional, Sequence

import httpx

from .config import (
    CHAIRMAN_MODEL,
    COUNCIL_MODELS,
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_URL,
    GEMINI_API_KEY,
    GEMINI_API_URL,
    KIMI_API_KEY,
    KIMI_API_URL,
    MODEL_REGISTRY,
    OPENAI_API_KEY,
    OPENAI_API_URL,
    QWEN_API_KEY,
    QWEN_API_URL,
    TITLE_MODEL,
    ModelConfig,
)
from .llm.contracts import LLMRequest, LLMResponse
from .llm.gateway import LLMGateway
from .llm.providers.deepseek import DeepSeekProvider
from .llm.providers.gemini import GeminiProvider
from .llm.providers.kimi import KimiProvider
from .llm.providers.qwen import QwenProvider



def resolve_model(model: ModelConfig | str) -> ModelConfig:
    """Resolve a model config from either a config object or a registry id."""
    if isinstance(model, ModelConfig):
        return model

    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]

    known_models = [*COUNCIL_MODELS, CHAIRMAN_MODEL, TITLE_MODEL]
    for config in known_models:
        if config.id == model:
            return config

    raise ValueError(f"Unknown model configuration: {model}")


def _messages_to_openai_input(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
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


def _extract_openai_text(data: Dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI Responses API payload."""
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = data.get("output", [])
    chunks: List[str] = []

    for item in output:
        if item.get("type") != "message":
            continue

        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                chunks.append(content.get("text", ""))

    return "\n".join(chunk for chunk in chunks if chunk).strip()


async def _query_openai(
    model: ModelConfig,
    messages: List[Dict[str, str]],
    timeout: float,
) -> Optional[Dict[str, Any]]:
    """Query the OpenAI Responses API."""
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY is not configured.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model.api_model,
        "input": _messages_to_openai_input(messages),
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()

    data = response.json()
    return {
        "content": _extract_openai_text(data),
        "reasoning_details": None,
    }


async def _query_deepseek(
    model: ModelConfig,
    messages: List[Dict[str, str]],
    timeout: float,
) -> Optional[Dict[str, Any]]:
    """Query the DeepSeek Chat Completions API."""
    if not DEEPSEEK_API_KEY:
        print("DEEPSEEK_API_KEY is not configured.")
        return None

    request = LLMRequest(
        model=model.api_model,
        messages=messages,
        timeout=timeout,
    )
    gateway = LLMGateway(
        {
            "deepseek": DeepSeekProvider(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_API_URL,
            )
        }
    )
    response = await gateway.chat("deepseek", request)
    return _llm_response_to_legacy_dict(response)


def _llm_response_to_legacy_dict(response: LLMResponse) -> Dict[str, Any]:
    return {
        "content": response.content,
        "reasoning_details": response.reasoning,
    }


async def _query_gemini(
    model: ModelConfig,
    messages: List[Dict[str, str]],
    timeout: float,
) -> Optional[Dict[str, Any]]:
    """Query the Gemini generateContent API."""
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY is not configured.")
        return None

    request = LLMRequest(
        model=model.api_model,
        messages=messages,
        timeout=timeout,
    )
    gateway = LLMGateway(
        {
            "gemini": GeminiProvider(
                api_key=GEMINI_API_KEY,
                base_url=GEMINI_API_URL,
            )
        }
    )
    response = await gateway.chat("gemini", request)
    return _llm_response_to_legacy_dict(response)


async def _query_qwen(
    model: ModelConfig,
    messages: List[Dict[str, str]],
    timeout: float,
) -> Optional[Dict[str, Any]]:
    """Query Qwen via Alibaba Model Studio's OpenAI-compatible endpoint."""
    if not QWEN_API_KEY:
        print("QWEN_API_KEY / DASHSCOPE_API_KEY is not configured.")
        return None

    request = LLMRequest(
        model=model.api_model,
        messages=messages,
        timeout=timeout,
    )
    gateway = LLMGateway(
        {
            "qwen": QwenProvider(
                api_key=QWEN_API_KEY,
                base_url=QWEN_API_URL,
            )
        }
    )
    response = await gateway.chat("qwen", request)
    return _llm_response_to_legacy_dict(response)


async def _query_kimi(
    model: ModelConfig,
    messages: List[Dict[str, str]],
    timeout: float,
) -> Optional[Dict[str, Any]]:
    """Query Kimi via the official OpenAI-compatible coding endpoint."""
    if not KIMI_API_KEY:
        print("KIMI_API_KEY / MOONSHOT_API_KEY is not configured.")
        return None

    request = LLMRequest(
        model=model.api_model,
        messages=messages,
        timeout=timeout,
    )
    gateway = LLMGateway(
        {
            "kimi": KimiProvider(
                api_key=KIMI_API_KEY,
                base_url=KIMI_API_URL,
            )
        }
    )
    response = await gateway.chat("kimi", request)
    return _llm_response_to_legacy_dict(response)


async def query_model(
    model: ModelConfig | str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via its configured provider.

    Args:
        model: Model configuration or registry id
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    try:
        config = resolve_model(model)
        if config.provider == "openai":
            return await _query_openai(config, messages, timeout)
        if config.provider == "deepseek":
            return await _query_deepseek(config, messages, timeout)
        if config.provider == "gemini":
            return await _query_gemini(config, messages, timeout)
        if config.provider == "kimi":
            return await _query_kimi(config, messages, timeout)
        if config.provider == "qwen":
            return await _query_qwen(config, messages, timeout)
        raise ValueError(f"Unsupported provider: {config.provider}")
    except Exception as e:
        model_id = model.id if isinstance(model, ModelConfig) else model
        print(f"Error querying model {model_id}: {e}")
        return None


async def query_models_parallel(
    models: Sequence[ModelConfig | str],
    messages: List[Dict[str, str]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of model configurations
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {
        resolve_model(model).id: response
        for model, response in zip(models, responses)
    }
