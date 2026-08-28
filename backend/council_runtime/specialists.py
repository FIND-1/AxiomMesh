"""Specialist prompt and formatting helpers."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..incident_input import IncidentInput, format_incident_context
from ..investigation_tools import format_tool_results, investigation_tool_catalog
from .schemas import RoleBlueprint, SPECIALIST_JSON_SCHEMAS


def _build_specialist_prompt(
    blueprint: RoleBlueprint,
    incident_input: IncidentInput,
    tool_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    emphasis = "\n".join(f"- {item}" for item in blueprint.emphasis)
    role_rules = {
        "analysis": (
            "- Treat facts as observations only.\n"
            "- Put suspected causes in hypotheses with need_validation=true.\n"
            "- Do not write that a hypothesis is the root cause."
        ),
        "investigator": (
            "- You are an Evidence Collector, not a resolver.\n"
            "- Extract timeline, facts, tool_results, and unknowns only.\n"
            "- Never output direct fixes, final root cause, or remediation plans."
        ),
        "critic": (
            "- You are a Fault Reviewer.\n"
            "- Challenge only weak or under-supported hypotheses.\n"
            "- For every challenge, name the missing evidence or a credible alternative."
        ),
    }[blueprint.role_id]
    investigation_context = ""
    if blueprint.role_id == "investigator":
        investigation_context = f"""

Available deterministic investigation tools:
{json.dumps(investigation_tool_catalog(), ensure_ascii=False, indent=2)}

Tool execution results:
{format_tool_results(tool_results or [])}

Use completed tool results as evidence. If you need tools that are not available yet, list them in "tool_requests".
"""

    return f"""You are the {blueprint.agent_name} inside an Agent Council for AI incident response.

Mission:
{blueprint.mission}

Operating rules:
{emphasis}
{role_rules}
- Return valid JSON only. Do not wrap it in Markdown.
- Use Simplified Chinese for every human-readable string value. Keep JSON field names unchanged.
- Use the unified output fields: agent_role, facts, hypotheses, evidence, unknowns, confidence.
- Separate FACT, HYPOTHESIS, CORRELATION, RECOMMENDATION, and UNKNOWN evidence types.
- Keep all evidence grounded in the provided user input or deterministic tool results. If evidence is missing, say so explicitly in "unknowns".
- Confidence values must be numbers between 0.0 and 1.0.

User incident input:
{incident_input.user_query}

Normalized incident context:
{format_incident_context(incident_input)}
{investigation_context}

Return JSON with exactly this shape:
{SPECIALIST_JSON_SCHEMAS[blueprint.role_id]}
"""

def _format_specialist_markdown(
    blueprint: RoleBlueprint,
    structured_output: Dict[str, Any],
) -> str:
    lines = [f"### {blueprint.agent_name}", structured_output["summary"]]

    signals = structured_output.get("signals") or []
    if signals:
        lines.append("\n**异常信号**")
        lines.extend(f"- {item}" for item in signals)

    timeline = structured_output.get("timeline") or []
    if timeline:
        lines.append("\n**时间线**")
        for item in timeline:
            if isinstance(item, dict):
                timestamp = item.get("timestamp") or "unknown time"
                lines.append(f"- {timestamp}: {item.get('event', '')} ({item.get('source', '')})")
            else:
                lines.append(f"- {item}")

    patterns = structured_output.get("patterns") or []
    if patterns:
        lines.append("\n**模式**")
        lines.extend(f"- {item}" for item in patterns)

    if structured_output.get("reliability_assessment"):
        lines.append(f"\n**可靠性评估：** {structured_output['reliability_assessment']}")

    evidence = structured_output.get("evidence") or []
    if evidence:
        lines.append("\n**证据**")
        lines.extend(
            f"- [{item.get('type', 'FACT')}/{item.get('credibility', 'medium')}] "
            f"{item.get('content') or item.get('detail')} ({item.get('source')})"
            for item in evidence
        )

    missing_evidence = structured_output.get("missing_evidence") or []
    if missing_evidence:
        lines.append("\n**缺失证据**")
        lines.extend(f"- {item}" for item in missing_evidence)

    alternative_causes = structured_output.get("alternative_causes") or []
    if alternative_causes:
        lines.append("\n**替代原因**")
        lines.extend(f"- {item}" for item in alternative_causes)

    hypotheses = structured_output.get("hypotheses") or []
    if hypotheses:
        lines.append("\n**故障假设**")
        lines.extend(
            f"- {item['content']} ({int(item['confidence'] * 100)}%): {item['rationale']}".rstrip(": ")
            for item in hypotheses
        )

    tool_requests = structured_output.get("tool_requests") or []
    if tool_requests:
        lines.append("\n**工具请求**")
        lines.extend(f"- {item}" for item in tool_requests)

    tool_results = structured_output.get("tool_results") or []
    if tool_results:
        lines.append("\n**工具结果**")
        for result in tool_results:
            lines.append(
                f"- {result.get('tool_name', 'unknown')} [{result.get('status', 'unknown')}]: "
                f"{result.get('summary', '')}"
            )

    next_actions = structured_output.get("next_actions") or []
    if next_actions:
        lines.append("\n**下一步行动**")
        lines.extend(f"- {item}" for item in next_actions)

    gaps = structured_output.get("gaps") or []
    if gaps:
        lines.append("\n**信息缺口**")
        lines.extend(f"- {item}" for item in gaps)

    lines.append(f"\n**置信度：** {int(structured_output['confidence'] * 100)}%")
    return "\n".join(lines)
