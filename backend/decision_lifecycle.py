"""Decision lifecycle event helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .agent_models import utc_timestamp
from .event_store import EventStore


def _decision_id(decision: Mapping[str, Any]) -> str:
    return str(
        decision.get("decision_id")
        or decision.get("id")
        or decision.get("agent_instance_id")
        or ""
    ).strip()


def _judge_name(judge_result: Mapping[str, Any]) -> str:
    return str(judge_result.get("agent_name") or "Judge Agent").strip() or "Judge Agent"


def _confirmed_evidence_ids(judge_result: Mapping[str, Any]) -> List[str]:
    structured_output = judge_result.get("structured_output")
    if not isinstance(structured_output, Mapping):
        structured_output = {}

    confirmed_evidence = structured_output.get("confirmed_evidence")
    if not isinstance(confirmed_evidence, list):
        confirmed_evidence = judge_result.get("confirmed_evidence")
    if not isinstance(confirmed_evidence, list):
        return []

    evidence_ids: List[str] = []
    seen: set[str] = set()
    for evidence in confirmed_evidence:
        if not isinstance(evidence, Mapping):
            continue
        evidence_id = str(evidence.get("evidence_id") or evidence.get("id") or "").strip()
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        evidence_ids.append(evidence_id)
    return evidence_ids


def _decision_created_event_exists(event_store: EventStore, decision_id: str) -> bool:
    return any(
        event.get("event_type") == "DECISION_CREATED"
        and str(event.get("decision_id") or "").strip() == decision_id
        for event in event_store.get_all_events()
    )


def _emit_decision_created_event(
    decision: Mapping[str, Any],
    judge_result: Mapping[str, Any],
    event_store: Optional[EventStore],
) -> None:
    if event_store is None:
        return

    decision_id = _decision_id(decision)
    if not decision_id or _decision_created_event_exists(event_store, decision_id):
        return

    event_store.add_event(
        {
            "event_type": "DECISION_CREATED",
            "timestamp": utc_timestamp(),
            "actor": {"type": "SYSTEM", "name": "decision_creator"},
            "decision_id": decision_id,
            "metadata": {
                "judge": _judge_name(judge_result),
                "evidence_ids": _confirmed_evidence_ids(judge_result),
            },
        }
    )


def create_decision(
    decision: Mapping[str, Any],
    judge_result: Mapping[str, Any],
    event_store: Optional[EventStore] = None,
) -> Dict[str, Any]:
    """Return the existing decision payload and emit its lifecycle event."""
    created_decision = dict(decision)
    _emit_decision_created_event(created_decision, judge_result, event_store)
    return created_decision
