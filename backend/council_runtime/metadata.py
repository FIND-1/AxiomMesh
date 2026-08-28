"""Council metadata construction helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..agent_models import AgentResultPayload
from ..config import CHAIRMAN_MODEL
from ..evidence_reasoning import rank_evidences
from ..evidence_store import EvidenceStore
from ..incident_input import IncidentInput, summarize_incident_input
from ..investigation_tools import investigation_tool_catalog
from .normalization import _summarize_evidence_items


def build_council_metadata(
    stage1_results: List[AgentResultPayload],
    stage2_result: AgentResultPayload,
    incident_input: Optional[IncidentInput] = None,
    agent_events: Optional[List[Dict[str, Any]]] = None,
    evidence_store: Optional[EvidenceStore] = None,
) -> Dict[str, Any]:
    evidence_items = evidence_store.get_all() if evidence_store else [
        evidence
        for stage_result in [*stage1_results, stage2_result]
        for evidence in stage_result.get("evidence", [])
    ]
    return {
        "workflow": [
            "council_manager_dispatch",
            "input_parsing",
            "specialist_analysis",
            "investigation_tooling",
            "judge_deliberation",
            "final_decision",
        ],
        "incident_input": summarize_incident_input(incident_input) if incident_input else {},
        "investigation_tools": {
            "available": investigation_tool_catalog(),
            "runs": [
                result
                for stage1_result in stage1_results
                for result in stage1_result.get("tool_results", [])
            ],
        },
        "agent_messages": [
            message
            for stage_result in [*stage1_results, stage2_result]
            for message in stage_result.get("messages", [])
        ],
        "agent_events": agent_events or [],
        "evidence": evidence_items,
        "evidence_summary": _summarize_evidence_items(evidence_items),
        "evidence_ranking": rank_evidences(evidence_items),
        "role_assignments": [
            {
                "agent_role": result["agent_role"],
                "agent_name": result["agent_name"],
                "agent_instance_id": result.get("agent_instance_id", result["agent_role"]),
                "model": result["model"],
            }
            for result in stage1_results
        ],
        "judge": {
            "agent_name": stage2_result.get("agent_name", "Judge Agent"),
            "model": stage2_result.get("model", CHAIRMAN_MODEL.id),
        },
    }
