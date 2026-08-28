"""Run-scoped aggregation helpers for LLM execution telemetry."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from .contracts import LLMUsage
from .telemetry import LLMExecutionRecord

logger = logging.getLogger(__name__)

USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
)


class LLMExecutionCollector:
    """Local collector for one Council run."""

    def __init__(self, run_id: Optional[str] = None) -> None:
        self.run_id = run_id
        self._records: List[LLMExecutionRecord] = []

    def add(self, record: LLMExecutionRecord) -> None:
        self._records.append(record)

    def records(self) -> List[LLMExecutionRecord]:
        return list(self._records)

    def summary(self) -> Dict[str, Any]:
        return aggregate_llm_executions(self._records, run_id=self.run_id)


def aggregate_llm_executions(
    records: Iterable[LLMExecutionRecord],
    *,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    records_list = list(records)
    summary = _summarize_records(records_list)
    summary["run_id"] = run_id or _infer_run_id(records_list)
    summary["by_role"] = _summarize_by(records_list, lambda record: record.workflow_role)
    summary["by_logical_model"] = _summarize_by(records_list, lambda record: record.logical_model)
    summary["by_provider"] = _summarize_by(records_list, lambda record: record.provider)
    return summary


def log_llm_run_summary(summary: Dict[str, Any]) -> None:
    logger.info("llm_run_summary %s", summary)


def _infer_run_id(records: List[LLMExecutionRecord]) -> Optional[str]:
    run_ids = {record.run_id for record in records if record.run_id is not None}
    if len(run_ids) == 1:
        return next(iter(run_ids))
    return None


def _summarize_by(
    records: List[LLMExecutionRecord],
    key_fn,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[LLMExecutionRecord]] = {}
    for record in records:
        key = key_fn(record) or "unknown"
        grouped.setdefault(key, []).append(record)
    return {key: _summarize_records(group_records) for key, group_records in grouped.items()}


def _summarize_records(records: List[LLMExecutionRecord]) -> Dict[str, Any]:
    invocation_count = len(records)
    attempt_count = sum(record.attempt_count for record in records)
    retried_invocations = sum(1 for record in records if record.retried)
    retry_attempts = sum(max(record.attempt_count - 1, 0) for record in records)
    success_count = sum(1 for record in records if record.success)
    failure_count = invocation_count - success_count
    latency_values = [record.latency_ms for record in records]

    return {
        "invocation_count": invocation_count,
        "attempt_count": attempt_count,
        "retried_invocations": retried_invocations,
        "retry_attempts": retry_attempts,
        "success_count": success_count,
        "failure_count": failure_count,
        "confirmed_usage": _summarize_usage(records),
        "latency": {
            "sum_invocation_latency_ms": round(sum(latency_values), 2),
            "avg_invocation_latency_ms": round(sum(latency_values) / invocation_count, 2)
            if invocation_count
            else 0.0,
            "max_invocation_latency_ms": max(latency_values) if latency_values else 0.0,
        },
    }


def _summarize_usage(records: List[LLMExecutionRecord]) -> Dict[str, Dict[str, Any]]:
    summary = {
        field: {"known_sum": 0, "has_unknown": False, "is_complete": True}
        for field in USAGE_FIELDS
    }
    for record in records:
        _add_usage(summary, record.usage)
    return summary


def _add_usage(summary: Dict[str, Dict[str, Any]], usage: Optional[LLMUsage]) -> None:
    if usage is None:
        for field in USAGE_FIELDS:
            _mark_unknown(summary[field])
        return

    for field in USAGE_FIELDS:
        value = getattr(usage, field)
        if value is None:
            _mark_unknown(summary[field])
        else:
            summary[field]["known_sum"] += value


def _mark_unknown(field_summary: Dict[str, Any]) -> None:
    field_summary["has_unknown"] = True
    field_summary["is_complete"] = False
