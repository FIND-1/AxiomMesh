"""LLM API client for making OpenAI and DeepSeek requests."""

from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from .config import (
    DEEPSEEK_API_KEY,
    GEMINI_API_KEY,
    KIMI_API_KEY,
    OPENAI_API_KEY,
    QWEN_API_KEY,
    ModelConfig,
)
from .llm.contracts import LLMRequest, LLMResponse, LLMUsage
from .llm.aggregation import LLMExecutionCollector
from .llm.gateway import LLMGateway
from .llm.provider_factory import get_provider
from .llm.registry import resolve_model
from .llm.retry import RetryStats, retry_async
from .llm.telemetry import (
    LLMExecutionRecord,
    categorize_llm_error,
    http_status_from_exception,
    log_execution_record,
)


async def _query_provider(
    provider_name: str,
    model: ModelConfig,
    messages: List[Dict[str, str]],
    timeout: float,
    api_key: Optional[str],
    retry_stats: RetryStats,
) -> Optional[LLMResponse]:
    """Query a single provider through the shared gateway contract."""
    if not api_key:
        return None

    request = LLMRequest(
        model=model.model_id,
        messages=messages,
        timeout=timeout,
    )
    provider = get_provider(provider_name)
    gateway = LLMGateway({provider_name: provider})
    response = await retry_async(
        lambda: gateway.chat(provider_name, request),
        stats=retry_stats,
    )
    return response


def _llm_response_to_legacy_dict(response: LLMResponse) -> Dict[str, Any]:
    return {
        "content": response.content,
        "reasoning_details": response.reasoning,
    }


async def query_model(
    model: ModelConfig | str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
    run_id: Optional[str] = None,
    workflow_role: Optional[str] = None,
    execution_collector: Optional[LLMExecutionCollector] = None,
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via its configured provider.

    Args:
        model: Model specification, logical model name, or legacy model id
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        run_id: Optional Council execution correlation id
        workflow_role: Optional workflow role for this invocation
        execution_collector: Optional per-run collector for execution records

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    execution_id = str(uuid4())
    started_at = perf_counter()
    retry_stats = RetryStats()
    config: Optional[ModelConfig] = None

    try:
        config = resolve_model(model)
        if config.provider == "openai":
            response = await _query_provider(
                "openai",
                config,
                messages,
                timeout,
                OPENAI_API_KEY,
                retry_stats,
            )
        elif config.provider == "deepseek":
            response = await _query_provider(
                "deepseek",
                config,
                messages,
                timeout,
                DEEPSEEK_API_KEY,
                retry_stats,
            )
        elif config.provider == "gemini":
            response = await _query_provider(
                "gemini",
                config,
                messages,
                timeout,
                GEMINI_API_KEY,
                retry_stats,
            )
        elif config.provider == "kimi":
            response = await _query_provider(
                "kimi",
                config,
                messages,
                timeout,
                KIMI_API_KEY,
                retry_stats,
            )
        elif config.provider == "qwen":
            response = await _query_provider(
                "qwen",
                config,
                messages,
                timeout,
                QWEN_API_KEY,
                retry_stats,
            )
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

        if response is None:
            _record_execution(
                _build_execution_record(
                    execution_id=execution_id,
                    run_id=run_id,
                    workflow_role=workflow_role,
                    model=model,
                    config=config,
                    success=False,
                    attempt_count=0,
                    retried=False,
                    latency_ms=_elapsed_ms(started_at),
                    error_category="configuration_error",
                ),
                execution_collector,
            )
            return None

        _record_execution(
            _build_execution_record(
                execution_id=execution_id,
                run_id=run_id,
                workflow_role=workflow_role,
                model=model,
                config=config,
                success=True,
                attempt_count=retry_stats.attempt_count or 1,
                retried=retry_stats.retried,
                latency_ms=_elapsed_ms(started_at),
                request_id=response.request_id,
                usage=response.usage,
            ),
            execution_collector,
        )
        return _llm_response_to_legacy_dict(response)
    except Exception as e:
        _record_execution(
            _build_execution_record(
                execution_id=execution_id,
                run_id=run_id,
                workflow_role=workflow_role,
                model=model,
                config=config,
                success=False,
                attempt_count=retry_stats.attempt_count,
                retried=retry_stats.retried,
                latency_ms=_elapsed_ms(started_at),
                error_category=categorize_llm_error(e),
                http_status=retry_stats.last_http_status or http_status_from_exception(e),
            ),
            execution_collector,
        )
        return None


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


def _record_execution(
    record: LLMExecutionRecord,
    execution_collector: Optional[LLMExecutionCollector],
) -> None:
    log_execution_record(record)
    if execution_collector is not None:
        execution_collector.add(record)


def _build_execution_record(
    *,
    execution_id: str,
    run_id: Optional[str],
    workflow_role: Optional[str],
    model: ModelConfig | str,
    config: Optional[ModelConfig],
    success: bool,
    attempt_count: int,
    retried: bool,
    latency_ms: float,
    error_category: Optional[str] = None,
    http_status: Optional[int] = None,
    request_id: Optional[str] = None,
    usage: Optional[LLMUsage] = None,
) -> LLMExecutionRecord:
    resolved = config or (model if isinstance(model, ModelConfig) else None)
    fallback_model_id = model.id if isinstance(model, ModelConfig) else str(model)
    fallback_logical_model = model.name if isinstance(model, ModelConfig) else None

    return LLMExecutionRecord(
        execution_id=execution_id,
        run_id=run_id,
        workflow_role=workflow_role,
        logical_model=resolved.name if resolved is not None else fallback_logical_model,
        model_id=resolved.id if resolved is not None else fallback_model_id,
        provider=resolved.provider if resolved is not None else None,
        provider_model_id=resolved.model_id if resolved is not None else None,
        success=success,
        attempt_count=attempt_count,
        retried=retried,
        latency_ms=latency_ms,
        error_category=error_category,
        http_status=http_status,
        request_id=request_id,
        usage=usage,
    )


async def query_models_parallel(
    models: Sequence[ModelConfig | str],
    messages: List[Dict[str, str]],
    run_id: Optional[str] = None,
    workflow_role: Optional[str] = None,
    execution_collector: Optional[LLMExecutionCollector] = None,
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
    tasks = [
        query_model(
            model,
            messages,
            run_id=run_id,
            workflow_role=workflow_role,
            execution_collector=execution_collector,
        )
        for model in models
    ]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {
        resolve_model(model).id: response
        for model, response in zip(models, responses)
    }
