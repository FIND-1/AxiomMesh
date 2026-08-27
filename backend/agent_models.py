"""Shared structured models for Agent Council communication."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, TypedDict

from typing_extensions import NotRequired


EVIDENCE_TYPES = {"FACT", "HYPOTHESIS", "CORRELATION", "RECOMMENDATION", "UNKNOWN"}
SOURCE_TYPES = {"LOG", "TOOL", "AGENT", "UNKNOWN"}
EVENT_ACTOR_TYPES = {"AGENT", "TOOL", "SYSTEM"}


def utc_timestamp() -> str:
    """Return a compact UTC timestamp for agent communication records."""
    return datetime.utcnow().isoformat()


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def _coerce_evidence_type(value: Any) -> str:
    evidence_type = str(value or "FACT").strip().upper()
    return evidence_type if evidence_type in EVIDENCE_TYPES else "UNKNOWN"


def _coerce_confidence(value: Any, *, default: float = 0.6) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return round(number, 2)


def _confidence_from_credibility(value: Any) -> float:
    credibility = str(value or "").strip().lower()
    if credibility == "high":
        return 0.9
    if credibility == "low":
        return 0.3
    return 0.6


class SourceTrace(dict):
    """Dictionary source trace with lenient equality for legacy string checks."""

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return other in {
                str(self.get("name") or ""),
                str(self.get("source_label") or ""),
            }
        return super().__eq__(other)


def _coerce_source_type(value: Any) -> str:
    source_type = str(value or "UNKNOWN").strip().upper()
    return source_type if source_type in SOURCE_TYPES else "UNKNOWN"


def _coerce_actor_type(value: Any) -> str:
    actor_type = str(value or "SYSTEM").strip().upper()
    return actor_type if actor_type in EVENT_ACTOR_TYPES else "SYSTEM"


def _infer_source_type(source_name: str, metadata: Mapping[str, Any]) -> str:
    source_type = _coerce_source_type(
        metadata.get("source_type") or metadata.get("type")
    )
    if source_type != "UNKNOWN":
        return source_type

    name = source_name.strip().lower()
    if not name:
        return "UNKNOWN"
    if name.startswith("tool:") or name in {"tool", "tools", "deterministic_tool", "investigation_tool"}:
        return "TOOL"
    if name in {"logs", "log"} or name.endswith(".log"):
        return "LOG"
    if name in {"analysis", "critic", "investigator", "judge", "final_decision"}:
        return "AGENT"
    return "UNKNOWN"


class AgentMessagePayload(TypedDict):
    role: str
    content: str
    evidence: List[Dict[str, Any]]
    confidence: float
    timestamp: str


class AgentResultPayload(TypedDict):
    agent_role: str
    agent_name: str
    agent_instance_id: str
    model: str
    response: str
    structured_output: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    messages: List[AgentMessagePayload]
    confidence: float
    tool_results: List[Dict[str, Any]]
    decision_summary: NotRequired[str]


def _normalize_source_trace(
    source: Any,
    *,
    agent_role: Optional[str] = None,
    timestamp: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> SourceTrace:
    """Normalize legacy and structured evidence source payloads."""
    metadata = metadata or {}
    if isinstance(source, Mapping):
        source_name = str(source.get("name") or source.get("source_name") or "").strip()
        source_type = _coerce_source_type(source.get("source_type") or source.get("type"))
        location = str(source.get("location") or "").strip()
        source_timestamp = str(source.get("timestamp") or timestamp or "").strip()
    else:
        source_name = str(source or "").strip()
        source_type = "UNKNOWN"
        location = str(metadata.get("location") or "").strip()
        source_timestamp = str(timestamp or metadata.get("timestamp") or "").strip()

    if source_type == "UNKNOWN":
        source_type = _infer_source_type(source_name, metadata)

    return SourceTrace(
        {
            "source_type": source_type,
            "name": source_name,
            "location": location,
            "timestamp": source_timestamp,
        }
    )


def _evidence_id(*, evidence_type: str, source: str, content: str) -> str:
    digest = hashlib.sha1(
        f"{evidence_type}|{source}|{content}".encode("utf-8")
    ).hexdigest()[:12]
    return f"evidence_{digest}"


@dataclass(frozen=True)
class Evidence:
    """Evidence item used by agents, tools, and judge decisions."""

    source: Any = field(default_factory=lambda: {"source_type": "UNKNOWN", "name": "", "location": "", "timestamp": ""})
    detail: str = ""
    credibility: str = "medium"
    agent_role: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    type: str = "FACT"
    content: Optional[str] = None
    timestamp: Optional[str] = None
    confidence: float = 0.6
    need_validation: bool = False

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        agent_role: Optional[str] = None,
    ) -> "Evidence":
        """Create evidence from a model/tool payload while preserving extras."""
        detail = str(value.get("content") or value.get("detail", "")).strip()
        credibility = str(value.get("credibility", "medium")).strip() or "medium"
        role = str(value.get("agent_role") or agent_role or "").strip() or None
        timestamp = str(value.get("timestamp") or "").strip() or None
        metadata = {
            key: item
            for key, item in value.items()
            if key
            not in {
                "id",
                "type",
                "content",
                "source",
                "timestamp",
                "confidence",
                "need_validation",
                "detail",
                "credibility",
                "agent_role",
            }
        }
        source_trace = _normalize_source_trace(
            value.get("source"),
            agent_role=role,
            timestamp=timestamp,
            metadata=metadata,
        )
        evidence_type = _coerce_evidence_type(value.get("type"))
        confidence = _coerce_confidence(
            value.get("confidence"),
            default=_confidence_from_credibility(credibility),
        )
        need_validation = bool(
            value.get("need_validation", evidence_type != "FACT" or confidence < 0.8)
        )
        evidence_id = str(
            value.get("id")
            or _evidence_id(
                evidence_type=evidence_type,
                source=source_trace.get("name") or source_trace.get("source_type") or "UNKNOWN",
                content=detail,
            )
        )
        return cls(
            source=source_trace,
            detail=detail,
            credibility=credibility,
            agent_role=role,
            metadata=metadata,
            id=evidence_id,
            type=evidence_type,
            content=detail,
            timestamp=timestamp,
            confidence=confidence,
            need_validation=need_validation,
        )

    def to_dict(self) -> Dict[str, Any]:
        content = (self.content or self.detail or "").strip()
        evidence_type = _coerce_evidence_type(self.type)
        source_trace = _normalize_source_trace(
            self.source,
            agent_role=self.agent_role,
            timestamp=self.timestamp,
            metadata=self.metadata,
        )
        evidence_id = self.id or _evidence_id(
            evidence_type=evidence_type,
            source=source_trace.get("name") or source_trace.get("source_type") or "UNKNOWN",
            content=content,
        )
        return {
            "id": evidence_id,
            "type": evidence_type,
            "content": content,
            "source": source_trace,
            "source_type": source_trace["source_type"],
            "source_name": source_trace["name"],
            "source_location": source_trace["location"],
            "source_timestamp": source_trace["timestamp"],
            "timestamp": self.timestamp,
            "confidence": _coerce_confidence(self.confidence),
            "need_validation": self.need_validation,
            # Backward-compatible aliases for the existing frontend/tests.
            "detail": self.detail,
            "credibility": self.credibility,
            "agent_role": self.agent_role,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AgentEvent:
    """Timeline event emitted while the council workflow is running."""

    actor: Dict[str, str]
    event_id: str = field(default_factory=_event_id)
    event_type: str = "agent_status"
    timestamp: str = field(default_factory=utc_timestamp)
    evidence_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        actor_name = str(self.actor.get("name") or "").strip()
        actor = {
            "type": _coerce_actor_type(self.actor.get("type")),
            "name": actor_name,
        }
        status = str(self.metadata.get("status") or "").strip()
        message = str(self.metadata.get("message") or "").strip()
        return {
            "event_id": self.event_id,
            "event_type": str(self.event_type or "").strip() or "agent_status",
            "timestamp": self.timestamp,
            "actor": actor,
            "evidence_id": str(self.evidence_id or "").strip(),
            "metadata": self.metadata,
            # Backward-compatible aliases for existing streaming UI/tests.
            "type": str(self.event_type or "").strip() or "agent_status",
            "agent": actor_name,
            "status": status,
            "message": message,
        }


@dataclass(frozen=True)
class AgentMessage:
    """A structured message exchanged between agent workflow stages."""

    role: str
    content: str
    evidence: List[Evidence] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> AgentMessagePayload:
        return {
            "role": self.role,
            "content": self.content,
            "evidence": [item.to_dict() for item in self.evidence],
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class AgentResult:
    """Canonical result envelope returned by an agent stage."""

    agent_role: str
    agent_name: str
    model: str
    response: str
    structured_output: Dict[str, Any]
    agent_instance_id: str
    evidence: List[Evidence] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)
    confidence: float = 0.0
    tool_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> AgentResultPayload:
        return {
            "agent_role": self.agent_role,
            "agent_name": self.agent_name,
            "agent_instance_id": self.agent_instance_id,
            "model": self.model,
            "response": self.response,
            "structured_output": self.structured_output,
            "evidence": [item.to_dict() for item in self.evidence],
            "messages": [item.to_dict() for item in self.messages],
            "confidence": self.confidence,
            "tool_results": self.tool_results,
        }
