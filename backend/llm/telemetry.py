"""Minimal structured execution records for LLM invocations."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from typing import Any, Dict, Optional

import httpx

from .contracts import LLMUsage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMExecutionRecord:
    """Structured summary of one logical LLM invocation."""

    execution_id: str
    logical_model: Optional[str]
    model_id: str
    provider: Optional[str]
    provider_model_id: Optional[str]
    success: bool
    attempt_count: int
    retried: bool
    latency_ms: float
    run_id: Optional[str] = None
    workflow_role: Optional[str] = None
    error_category: Optional[str] = None
    http_status: Optional[int] = None
    request_id: Optional[str] = None
    usage: Optional[LLMUsage] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def http_status_from_exception(exc: Exception) -> Optional[int]:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None


def categorize_llm_error(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.NetworkError):
        return "network"
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code == 429:
            return "rate_limit"
        if 500 <= status_code <= 599:
            return "server_error"
        if 400 <= status_code <= 499:
            return "client_error"
        return "unknown"
    if isinstance(exc, JSONDecodeError):
        return "parse_error"
    if isinstance(exc, (TypeError, AttributeError)):
        return "schema_error"
    if isinstance(exc, ValueError):
        return "configuration_error"
    return "unknown"


def log_execution_record(record: LLMExecutionRecord) -> None:
    payload = record.to_dict()
    if record.success:
        logger.info("llm_execution %s", payload)
        return
    logger.warning("llm_execution %s", payload)
