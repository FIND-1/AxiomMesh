"""Judge prompt and formatting helpers."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..agent_models import AgentResultPayload
from .schemas import JUDGE_JSON_SCHEMA


def _build_judge_prompt(
    user_query: str,
    stage1_results: List[AgentResultPayload],
    evidence_items: Optional[List[Dict[str, Any]]] = None,
    evidence_ranking: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> str:
    specialist_packets = []
    for result in stage1_results:
        specialist_packets.append(
            f"Agent Role: {result['agent_role']}\n"
            f"Agent Name: {result['agent_name']}\n"
            f"Agent Instance: {result.get('agent_instance_id', result['agent_role'])}\n"
            f"Model: {result['model']}\n"
            f"Summary: {result.get('structured_output', {}).get('summary', '')}\n"
            f"Confidence: {result.get('confidence', 0.0)}"
        )

    joined_packets = "\n\n".join(specialist_packets)
    evidence_packet = json.dumps(evidence_items or [], ensure_ascii=False, indent=2)
    ranking_packet = json.dumps(evidence_ranking or {"ranked_evidence": []}, ensure_ascii=False, indent=2)
    return f"""You are the Judge Agent in an Agent Council for AI incident response.
Your job is to evaluate evidence, separate confirmed facts from unverified hypotheses, and make the most defensible decision.

User incident input:
{user_query}

Specialist roster:
{joined_packets}

Unified EvidenceStore evidence:
{evidence_packet}

Evidence Ranking Result:
{ranking_packet}

Rules:
- Return valid JSON only. Do not wrap it in Markdown.
- Use Simplified Chinese for every human-readable string value. Keep JSON field names unchanged.
- Score every specialist in the scorecard.
- Evidence, reasoning, and actionability scores must be integers from 1 to 5.
- Confidence must be a number between 0.0 and 1.0.
- Use Unified EvidenceStore evidence as the source of truth for confirmed_evidence and unverified_hypothesis.
- Use Evidence Ranking Result to prioritize stronger evidence, especially when deciding confirmed_evidence.
- Treat FACT evidence from logs or deterministic tools as stronger than agent hypotheses.
- Do not promote a HYPOTHESIS to root_cause unless confirmed_evidence supports it.
- Put uncertain theories in unverified_hypothesis and state what evidence is needed.
- Prefer validation-oriented next_actions when root cause confidence is not high.
- Include evidence_summary counts based on the Unified EvidenceStore evidence.

Return JSON with exactly this shape:
{JUDGE_JSON_SCHEMA}
"""

def _format_judge_markdown(structured_output: Dict[str, Any]) -> str:
    lines = [structured_output["verdict_summary"]]

    if structured_output.get("direct_cause"):
        lines.append(f"\n**直接原因：** {structured_output['direct_cause']}")

    if structured_output.get("root_cause"):
        lines.append(f"\n**根因判断：** {structured_output['root_cause']}")

    confirmed_evidence = structured_output.get("confirmed_evidence") or []
    if confirmed_evidence:
        lines.append("\n**已确认事实**")
        lines.extend(
            f"- [{item.get('credibility', 'MEDIUM')}] {item.get('content', '')} ({item.get('source', '')})"
            for item in confirmed_evidence
        )

    unverified_hypothesis = structured_output.get("unverified_hypothesis") or []
    if unverified_hypothesis:
        lines.append("\n**未验证假设**")
        lines.extend(
            f"- [{item.get('confidence', 'MEDIUM')}] {item.get('content', '')}: {item.get('needed_evidence', '')}".rstrip(": ")
            for item in unverified_hypothesis
        )

    gaps = structured_output.get("gaps") or []
    if gaps:
        lines.append("\n**剩余风险**")
        lines.extend(f"- {item}" for item in gaps)

    recommendations = structured_output.get("next_actions") or structured_output.get("recommendations") or []
    if recommendations:
        lines.append("\n**建议行动**")
        lines.extend(f"- {item}" for item in recommendations)

    if structured_output.get("minority_view"):
        lines.append(f"\n**少数观点：** {structured_output['minority_view']}")

    lines.append(f"\n**置信度：** {int(structured_output['confidence'] * 100)}%")
    return "\n".join(lines)
