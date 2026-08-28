"""Minimal async retry helpers for transient LLM transport failures."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, TypeVar

import httpx

T = TypeVar("T")

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class RetryStats:
    """Side-channel retry metadata for one logical invocation."""

    attempt_count: int = 0
    retried: bool = False
    last_error: Optional[Exception] = None
    last_http_status: Optional[int] = None


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.NetworkError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    return False


def _retry_delay_seconds(attempt: int) -> float:
    base_delay = 0.05 * attempt
    return base_delay + random.uniform(0.0, 0.05)


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 2,
    stats: Optional[RetryStats] = None,
) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        if stats is not None:
            stats.attempt_count = attempt
        try:
            return await operation()
        except Exception as exc:
            if stats is not None:
                stats.last_error = exc
                stats.last_http_status = (
                    exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None
                )
            if attempt >= max_attempts or not _is_retryable_error(exc):
                raise
            if stats is not None:
                stats.retried = True
            await asyncio.sleep(_retry_delay_seconds(attempt))

    raise RuntimeError("retry_async reached an unexpected state")
