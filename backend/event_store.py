"""In-memory event registry for a single council run."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping


class EventStore:
    """Collect and query lifecycle events during one council execution."""

    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []

    def add_event(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        """Register one event and return a detached copy."""
        item = dict(event)
        self._events.append(item)
        return dict(item)

    def get_all_events(self) -> List[Dict[str, Any]]:
        """Return all stored events as detached copies."""
        return [dict(item) for item in self._events]

    def get_by_evidence_id(self, evidence_id: str) -> List[Dict[str, Any]]:
        """Return all events associated with the given evidence id."""
        target = str(evidence_id or "").strip()
        if not target:
            return []
        return [
            dict(item)
            for item in self._events
            if str(item.get("evidence_id") or "").strip() == target
        ]

    def clear(self) -> None:
        """Remove all stored events."""
        self._events.clear()
