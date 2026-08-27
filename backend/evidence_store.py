"""In-memory evidence registry for a single council run."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .agent_models import AgentEvent, Evidence
from .event_store import EventStore
from .evidence_reasoning import EvidenceRanker, calculate_confidence


_PUNCTUATION_PATTERN = re.compile(r"[\s\W_]+", re.UNICODE)
_WORD_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _as_dict(evidence: Any) -> Dict[str, Any]:
    if isinstance(evidence, Evidence):
        return evidence.to_dict()
    if isinstance(evidence, Mapping):
        return dict(evidence)
    if is_dataclass(evidence):
        return asdict(evidence)
    return {}


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return _PUNCTUATION_PATTERN.sub("", text)


def _tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    words = set(_WORD_PATTERN.findall(text))
    if words:
        return words
    return set(_normalize_text(text))


def _is_similar_text(left: str, right: str) -> bool:
    normalized_left = _normalize_text(left)
    normalized_right = _normalize_text(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True

    shorter, longer = sorted([normalized_left, normalized_right], key=len)
    if len(shorter) >= 20 and shorter in longer and len(shorter) / len(longer) >= 0.8:
        return True

    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return False

    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return bool(union) and overlap / union >= 0.85


def _agent_from_evidence(item: Mapping[str, Any], fallback: Optional[str] = None) -> Optional[str]:
    agent = str(item.get("agent_role") or fallback or "").strip()
    return agent or None


def _source_agents(item: Mapping[str, Any], fallback: Optional[str] = None) -> List[str]:
    raw_agents = item.get("source_agents")
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    if not isinstance(raw_agents, list):
        raw_agents = metadata.get("source_agents")

    agents = [
        str(agent).strip()
        for agent in (raw_agents if isinstance(raw_agents, list) else [])
        if str(agent).strip()
    ]
    primary_agent = _agent_from_evidence(item, fallback)
    if primary_agent:
        agents.append(primary_agent)

    deduped: List[str] = []
    seen: set[str] = set()
    for agent in agents:
        if agent in seen:
            continue
        seen.add(agent)
        deduped.append(agent)
    return deduped


def _sources(item: Mapping[str, Any]) -> List[str]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    raw_sources = item.get("sources")
    if not isinstance(raw_sources, list):
        raw_sources = metadata.get("sources")

    sources = [
        str(source).strip()
        for source in (raw_sources if isinstance(raw_sources, list) else [])
        if str(source).strip()
    ]
    source_value = item.get("source")
    if isinstance(source_value, Mapping):
        source = str(source_value.get("name") or source_value.get("source_type") or "").strip()
    else:
        source = str(source_value or "").strip()
    if source:
        sources.append(source)

    deduped: List[str] = []
    seen: set[str] = set()
    for source in sources:
        if source in seen:
            continue
        seen.add(source)
        deduped.append(source)
    return deduped


class EvidenceStore:
    """Collect, deduplicate, and query evidence for one council execution."""

    def __init__(self, event_store: Optional[EventStore] = None) -> None:
        self._items: List[Dict[str, Any]] = []
        self._ranker = EvidenceRanker()
        self.event_store = event_store

    def _emit_event(
        self,
        event_type: str,
        evidence_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        actor: Optional[Dict[str, str]] = None,
    ) -> None:
        if self.event_store is None:
            return

        event = AgentEvent(
            actor=actor or {"type": "SYSTEM", "name": "evidence_store"},
            event_type=event_type,
            evidence_id=evidence_id or "",
            metadata=metadata or {},
        ).to_dict()
        self.event_store.add_event(event)

    def add_evidence(self, evidence: Any) -> Dict[str, Any]:
        """Register evidence, generating ids and merging similar evidence."""
        item = _as_dict(evidence)
        if not item:
            raise ValueError("evidence must be a mapping or Evidence instance")

        agent_role = _agent_from_evidence(item)
        normalized = Evidence.from_mapping(item, agent_role=agent_role).to_dict()
        source_agents = _source_agents(item, agent_role)
        sources = _sources(item)
        metadata = dict(normalized.get("metadata") or {})
        metadata["model_confidence"] = normalized.get("confidence", 0.0)
        metadata["source_agents"] = source_agents
        metadata["sources"] = sources
        normalized["source_agents"] = source_agents
        normalized["sources"] = sources
        normalized["metadata"] = metadata
        normalized["confidence"] = calculate_confidence(normalized)
        normalized["score"] = self._ranker.calculate_evidence_score(normalized)

        existing = self._find_duplicate(normalized)
        if existing is None:
            self._items.append(normalized)
            self._emit_event(
                "EVIDENCE_CREATED",
                evidence_id=normalized.get("id"),
                metadata={
                    "type": normalized.get("type"),
                    "content": normalized.get("content"),
                    "source": normalized.get("source"),
                },
            )
            return dict(normalized)

        self._merge(existing, normalized)
        return dict(existing)

    def add_many(self, evidence_items: Iterable[Any]) -> List[Dict[str, Any]]:
        return [self.add_evidence(item) for item in evidence_items]

    def get_all(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._items]

    def get_by_type(self, evidence_type: str) -> List[Dict[str, Any]]:
        normalized_type = str(evidence_type or "").strip().upper()
        return [dict(item) for item in self._items if item.get("type") == normalized_type]

    def get_by_agent(self, agent_role: str) -> List[Dict[str, Any]]:
        role = str(agent_role or "").strip()
        if not role:
            return []
        return [
            dict(item)
            for item in self._items
            if item.get("agent_role") == role or role in item.get("source_agents", [])
        ]

    def _find_duplicate(self, candidate: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        for item in self._items:
            if item.get("type") != candidate.get("type"):
                continue
            if _is_similar_text(
                str(item.get("content") or item.get("detail") or ""),
                str(candidate.get("content") or candidate.get("detail") or ""),
            ):
                return item
        return None

    def _merge(self, existing: Dict[str, Any], candidate: Mapping[str, Any]) -> None:
        existing_agents = _source_agents(existing)
        candidate_agents = _source_agents(candidate)
        source_agents = [*existing_agents]
        for agent in candidate_agents:
            if agent not in source_agents:
                source_agents.append(agent)

        existing_sources = _sources(existing)
        candidate_sources = _sources(candidate)
        sources = [*existing_sources]
        for source in candidate_sources:
            if source not in sources:
                sources.append(source)

        candidate_confidence = calculate_confidence({**candidate, "source_agents": source_agents})
        existing_confidence = calculate_confidence({**existing, "source_agents": source_agents})
        candidate_score = self._ranker.calculate_evidence_score({**candidate, "source_agents": source_agents})
        existing_score = self._ranker.calculate_evidence_score({**existing, "source_agents": source_agents})
        if candidate_confidence > existing_confidence:
            existing.update(
                {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"id", "metadata", "source_agents", "sources", "score"}
                }
            )

        metadata = dict(existing.get("metadata") or {})
        metadata["source_agents"] = source_agents
        metadata["sources"] = sources
        existing["source_agents"] = source_agents
        existing["sources"] = sources
        existing["metadata"] = metadata
        existing["confidence"] = max(existing_confidence, candidate_confidence)
        existing["score"] = max(existing_score, candidate_score)
        self._emit_event(
            "EVIDENCE_MERGED",
            evidence_id=existing.get("id"),
            metadata={
                "source_ids": [candidate.get("id")],
                "target_id": existing.get("id"),
            },
        )
