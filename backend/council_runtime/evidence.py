"""Evidence transformation helpers for Council runtime."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

from ..agent_models import Evidence


def _evidence_models_from_payload(
    payload: Dict[str, Any],
    *,
    agent_role: str,
) -> List[Evidence]:
    evidence_models: List[Evidence] = []
    for item in payload.get("evidence", []) or []:
        if not isinstance(item, dict) or not str(item.get("content") or item.get("detail") or "").strip():
            continue
        evidence_models.append(Evidence.from_mapping(item, agent_role=agent_role))
    return evidence_models

_EVIDENCE_ROLE_ORDER = {
    "analysis": 0,
    "investigator": 1,
    "critic": 2,
}

def _tool_results_to_evidence(
    stage_result: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    evidence_items: List[Dict[str, Any]] = []
    agent_role = str(stage_result.get("agent_role") or "").strip() or None

    for result in stage_result.get("tool_results", []) or []:
        if not isinstance(result, dict):
            continue
        tool_name = str(result.get("tool_name") or "unknown_tool").strip() or "unknown_tool"
        status = str(result.get("status") or "unknown").strip() or "unknown"
        summary = str(result.get("summary") or "").strip()
        findings = result.get("findings", [])
        content_parts = [summary] if summary else []
        if findings:
            content_parts.append(json.dumps(findings, ensure_ascii=False, sort_keys=True))
        content = " ".join(content_parts).strip() or f"{tool_name} returned status {status}."
        source = result.get("source") or {
            "source_type": "TOOL",
            "name": tool_name,
            "location": "",
            "timestamp": "",
        }
        evidence_items.append(
            {
                "type": "FACT" if status == "completed" else "UNKNOWN",
                "content": content,
                "source": source,
                "confidence": 0.9 if status == "completed" else 0.2,
                "need_validation": status != "completed",
                "agent_role": agent_role,
                "metadata": {
                    "source_type": "TOOL",
                    "tool_name": tool_name,
                    "tool_status": status,
                    "findings": findings,
                },
            }
        )

    return evidence_items
