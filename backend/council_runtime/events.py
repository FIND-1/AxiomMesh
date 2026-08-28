"""Event helpers for Council runtime."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..agent_models import AgentEvent
from ..event_store import EventStore


logger = logging.getLogger("backend.council")


AgentEventSink = Callable[[Dict[str, Any]], Awaitable[None] | None]


async def _emit_agent_event(
    event_sink: Optional[AgentEventSink],
    *,
    agent: str,
    status: str,
    message: str,
    event_type: str = "agent_status",
    actor_type: str = "AGENT",
    evidence_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload_metadata = dict(metadata or {})
    payload_metadata.setdefault("status", status)
    payload_metadata.setdefault("message", message)
    event = AgentEvent(
        actor={"type": actor_type, "name": agent},
        event_type=event_type,
        evidence_id=str(evidence_id or "").strip(),
        metadata=payload_metadata,
    ).to_dict()
    if event_sink is not None:
        try:
            result = event_sink(event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.warning("Agent event sink failed for %s", event_type, exc_info=True)
    return event

def _emit_judge_evidence_usage_events(
    event_store: Optional[EventStore],
    confirmed_evidence: List[Dict[str, Any]],
    *,
    judge_name: str,
) -> None:
    if event_store is None:
        return

    seen_evidence_ids: set[str] = set()
    for evidence in confirmed_evidence:
        evidence_id = str(evidence.get("evidence_id") or "").strip()
        if not evidence_id or evidence_id in seen_evidence_ids:
            continue

        seen_evidence_ids.add(evidence_id)
        event = AgentEvent(
            actor={"type": "SYSTEM", "name": "judge_reasoning"},
            event_type="EVIDENCE_USED_BY_JUDGE",
            evidence_id=evidence_id,
            metadata={
                "judge": judge_name,
                "reason": "supporting_decision",
            },
        ).to_dict()
        event_store.add_event(event)
