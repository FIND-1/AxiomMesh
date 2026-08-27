"""Evidence confidence and ranking helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .agent_models import AgentEvent
from .event_store import EventStore


def _as_mapping(evidence: Any) -> Mapping[str, Any]:
    if isinstance(evidence, Mapping):
        return evidence
    if is_dataclass(evidence):
        return asdict(evidence)
    return {}


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _source_is_tool(evidence: Mapping[str, Any]) -> bool:
    source_value = evidence.get("source")
    source = str(source_value or "").strip().lower()
    metadata = evidence.get("metadata") if isinstance(evidence.get("metadata"), Mapping) else {}
    if isinstance(source_value, Mapping):
        source = str(source_value.get("name") or "").strip().lower()
        source_type = str(
            source_value.get("source_type")
            or source_value.get("type")
            or metadata.get("source_type")
            or evidence.get("source_type")
            or ""
        ).lower()
    else:
        source_type = str(metadata.get("source_type") or evidence.get("source_type") or "").lower()

    return (
        source in {"tool", "tools", "deterministic_tool", "investigation_tool"}
        or source.startswith("tool:")
        or source_type == "tool"
        or bool(metadata.get("tool_name"))
    )


def _has_multiple_agents(evidence: Mapping[str, Any]) -> bool:
    source_agents = evidence.get("source_agents")
    if not isinstance(source_agents, list):
        metadata = evidence.get("metadata") if isinstance(evidence.get("metadata"), Mapping) else {}
        source_agents = metadata.get("source_agents")

    if not isinstance(source_agents, list):
        return False

    normalized = {str(agent).strip() for agent in source_agents if str(agent).strip()}
    return len(normalized) > 1


def _source_type(evidence: Mapping[str, Any]) -> str:
    source_value = evidence.get("source")
    metadata = evidence.get("metadata") if isinstance(evidence.get("metadata"), Mapping) else {}
    if isinstance(source_value, Mapping):
        source_type = source_value.get("source_type") or source_value.get("type")
    else:
        source_type = evidence.get("source_type") or metadata.get("source_type")
    return str(source_type or "UNKNOWN").strip().upper()


def calculate_confidence(evidence: Any) -> float:
    """Calculate bounded confidence without fully trusting model-provided values."""
    item = _as_mapping(evidence)
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    model_confidence = metadata.get("model_confidence", item.get("confidence"))
    confidence = _coerce_confidence(model_confidence) * 0.5

    if _source_is_tool(item):
        confidence += 0.3
    if _has_multiple_agents(item):
        confidence += 0.2
    if bool(item.get("need_validation")):
        confidence -= 0.3

    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return round(confidence, 2)


class EvidenceRanker:
    """Rank evidence using source trace, support count, and validation signals."""

    def __init__(self, event_store: Optional[EventStore] = None) -> None:
        self.event_store = event_store

    def _emit_event(
        self,
        event_type: str,
        evidence_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.event_store is None:
            return

        event = AgentEvent(
            actor={"type": "SYSTEM", "name": "evidence_ranker"},
            event_type=event_type,
            evidence_id=evidence_id,
            metadata=metadata or {},
        ).to_dict()
        self.event_store.add_event(event)

    def calculate_evidence_score(self, evidence: Any) -> float:
        item = _as_mapping(evidence)
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        model_confidence = metadata.get("model_confidence", item.get("confidence"))
        score = _coerce_confidence(model_confidence) * 0.5

        source_type = _source_type(item)
        if source_type == "TOOL":
            score += 0.3
        elif source_type == "LOG":
            score += 0.2

        if _has_multiple_agents(item):
            score += 0.2
        if bool(item.get("need_validation")):
            score -= 0.3
        if str(item.get("type") or "").strip().upper() == "HYPOTHESIS":
            score -= 0.15

        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return round(score, 2)


def rank_evidences(
    evidences: Sequence[Mapping[str, Any]],
    event_store: Optional[EventStore] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return evidence ranked by descending evidence score."""
    ranker = EvidenceRanker(event_store=event_store)
    ranked = []
    for evidence in evidences:
        item = dict(evidence)
        ranked.append(
            {
                "id": item.get("id", ""),
                "type": item.get("type", "UNKNOWN"),
                "content": item.get("content") or item.get("detail", ""),
                "source": item.get("source", {}),
                "source_agents": item.get("source_agents", []),
                "score": ranker.calculate_evidence_score(item),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        ranker._emit_event(
            "EVIDENCE_RANKED",
            str(item.get("id") or ""),
            metadata={
                "score": item["score"],
                "rank": rank,
            },
        )
    return {"ranked_evidence": ranked}
