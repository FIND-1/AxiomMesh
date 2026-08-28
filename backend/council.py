"""Role-based 3-stage Agent Council orchestration."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple
from uuid import uuid4

from .agent_models import AgentEvent, AgentMessage, AgentResult, AgentResultPayload, Evidence
from .config import CHAIRMAN_MODEL, COUNCIL_MODELS, TITLE_MODEL, ModelConfig
from .event_store import EventStore
from .evidence_reasoning import rank_evidences
from .evidence_store import EvidenceStore
from .incident_input import (
    IncidentInput,
    format_incident_context,
    parse_incident_input,
    summarize_incident_input,
)
from .investigation_tools import (
    InvestigationToolContext,
    format_tool_results,
    investigation_tool_catalog,
    run_investigation_tools,
)
from .llm.aggregation import LLMExecutionCollector, log_llm_run_summary
from .openrouter import query_model


@dataclass(frozen=True)
class RoleBlueprint:
    """Configuration for a specialist council role."""

    role_id: str
    agent_name: str
    mission: str
    emphasis: List[str]


AgentEventSink = Callable[[Dict[str, Any]], Awaitable[None] | None]


SPECIALIST_BLUEPRINTS = [
    RoleBlueprint(
        role_id="analysis",
        agent_name="Analysis Agent",
        mission=(
            "Identify abnormal patterns in incident descriptions and logs, propose "
            "initial hypotheses, and explicitly mark what still needs validation."
        ),
        emphasis=[
            "Extract concrete error signals, stack traces, time correlations, and failure signatures.",
            "Separate facts from hypotheses; do not claim a final root cause.",
            "Mark every unverified cause as a hypothesis that needs validation.",
        ],
    ),
    RoleBlueprint(
        role_id="critic",
        agent_name="Critic Agent",
        mission=(
            "Review and challenge hypotheses, identify weak assumptions, and surface "
            "credible alternative explanations without arguing for its own sake."
        ),
        emphasis=[
            "Do not restate the likely answer unless you are explicitly challenging it.",
            "Focus on missing evidence, shaky logic, blind spots, and counter-hypotheses.",
            "Explain what would falsify or strengthen each competing theory.",
        ],
    ),
    RoleBlueprint(
        role_id="investigator",
        agent_name="Investigation Agent",
        mission=(
            "Collect observable facts, organize evidence, build a timeline, run "
            "deterministic tools, and identify information gaps."
        ),
        emphasis=[
            "Do not output final root cause conclusions.",
            "Do not output repair recommendations.",
            "Report tool results and unknowns; only state what the input or tools directly support.",
        ],
    ),
]

SPECIALIST_JSON_SCHEMAS = {
    "analysis": """{
  "agent_role": "analysis",
  "summary": "short incident pattern analysis summary",
  "facts": ["facts directly visible in the provided input"],
  "patterns": ["abnormal pattern or correlation that is visible but not final proof"],
  "hypotheses": [
    {
      "content": "possible cause, not a final root cause",
      "cause": "same possible cause",
      "rationale": "why this cause is plausible",
      "confidence": 0.0,
      "need_validation": true
    }
  ],
  "evidence": [
    {
      "id": "evidence_001",
      "type": "FACT | HYPOTHESIS | CORRELATION | RECOMMENDATION | UNKNOWN",
      "content": "specific evidence or hypothesis content",
      "source": {
        "source_type": "LOG | TOOL | AGENT | UNKNOWN",
        "name": "source name such as error.log, log_input_summary, or agent role",
        "location": "line number, log section, tool finding path, or empty string",
        "timestamp": "timestamp if available, otherwise empty string"
      },
      "timestamp": "timestamp if available, otherwise null",
      "confidence": 0.0,
      "need_validation": false
    }
  ],
  "unknowns": ["missing information required to validate hypotheses"],
  "confidence": 0.0
}""",
    "investigator": """{
  "agent_role": "investigator",
  "summary": "short evidence collection summary",
  "timeline": [
    {
      "timestamp": "timestamp if available",
      "event": "observable event only",
      "source": "log or tool source"
    }
  ],
  "facts": ["facts directly extracted from logs, user input, or tools"],
  "hypotheses": [],
  "evidence": [
    {
      "id": "evidence_001",
      "type": "FACT | CORRELATION | UNKNOWN",
      "content": "observable evidence only",
      "source": {
        "source_type": "LOG | TOOL | AGENT | UNKNOWN",
        "name": "source name such as error.log, log_input_summary, or agent role",
        "location": "line number, log section, tool finding path, or empty string",
        "timestamp": "timestamp if available, otherwise empty string"
      },
      "timestamp": "timestamp if available, otherwise null",
      "confidence": 0.0,
      "need_validation": false
    }
  ],
  "tool_results": [],
  "unknowns": ["what is still unknown"],
  "tool_requests": ["additional tool call needed"],
  "confidence": 0.0
}""",
    "critic": """{
  "agent_role": "critic",
  "summary": "short critique summary",
  "facts": ["facts the critique relies on"],
  "hypotheses": [],
  "challenged_hypothesis": "specific hypothesis being challenged",
  "risks": ["risk in the current reasoning"],
  "alternative_hypotheses": [
    {
      "content": "credible alternative explanation",
      "rationale": "why it may explain the same facts",
      "confidence": 0.0,
      "need_validation": true
    }
  ],
  "missing_evidence": ["missing evidence 1"],
  "evidence": [
    {
      "id": "evidence_001",
      "type": "FACT | HYPOTHESIS | UNKNOWN",
      "content": "evidence, alternative hypothesis, or missing-evidence statement",
      "source": {
        "source_type": "LOG | TOOL | AGENT | UNKNOWN",
        "name": "source name such as error.log, log_input_summary, or agent role",
        "location": "line number, log section, tool finding path, or empty string",
        "timestamp": "timestamp if available, otherwise empty string"
      },
      "timestamp": "timestamp if available, otherwise null",
      "confidence": 0.0,
      "need_validation": true
    }
  ],
  "unknowns": ["unknowns that prevent a reliable conclusion"],
  "confidence": 0.0
}""",
}

JUDGE_JSON_SCHEMA = """{
  "agent_role": "judge",
  "incident_level": "critical | high | medium | low | unknown",
  "direct_cause": "direct observable failure mechanism, or unknown",
  "root_cause": "most likely root cause only if evidence supports it, otherwise unknown",
  "verdict_summary": "overall decision summary",
  "confidence": 0.0,
  "confirmed_evidence": [
    {
      "evidence_id": "evidence_001",
      "type": "FACT | CORRELATION",
      "content": "evidence content",
      "source": {
        "source_type": "LOG | TOOL | AGENT | UNKNOWN",
        "name": "source name",
        "location": "line number, log section, tool finding path, or empty string",
        "timestamp": "timestamp if available, otherwise empty string"
      },
      "credibility": "HIGH | MEDIUM | LOW",
      "reason": "why this can be trusted"
    }
  ],
  "unverified_hypothesis": [
    {
      "content": "hypothesis that still needs validation",
      "confidence": "HIGH | MEDIUM | LOW",
      "needed_evidence": "what would validate or falsify it"
    }
  ],
  "evidence_summary": {
    "total": 0,
    "fact": 0,
    "hypothesis": 0,
    "correlation": 0,
    "recommendation": 0,
    "unknown": 0
  },
  "scorecard": [
    {
      "agent_role": "analysis | critic | investigator",
      "agent_name": "display name",
      "agent_instance_id": "agent role and model instance id",
      "evidence_score": 1,
      "reasoning_score": 1,
      "actionability_score": 1,
      "notes": "why these scores were assigned"
    }
  ],
  "gaps": ["remaining uncertainty or unknowns"],
  "next_actions": ["next validation or mitigation action"],
  "minority_view": "best competing explanation if the leading view is wrong"
}"""


def _specialist_assignments() -> List[Tuple[RoleBlueprint, ModelConfig]]:
    """Assign every configured model to a specialist role instance."""
    assignments: List[Tuple[RoleBlueprint, ModelConfig]] = []
    models = COUNCIL_MODELS or [CHAIRMAN_MODEL]

    for index, model in enumerate(models):
        blueprint = SPECIALIST_BLUEPRINTS[index % len(SPECIALIST_BLUEPRINTS)]
        assignments.append((blueprint, model))

    return assignments


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


def _build_final_decision_prompt(
    user_query: str,
    stage1_results: List[AgentResultPayload],
    stage2_result: AgentResultPayload,
) -> str:
    specialist_summaries = "\n".join(
        f"- {result['agent_name']}: {result['structured_output'].get('summary', '')}"
        for result in stage1_results
    )
    judge_packet = json.dumps(stage2_result.get("structured_output", {}), ensure_ascii=False, indent=2)

    return f"""You are writing the final decision memo for an Agent Council focused on AI incident response.

Original incident input:
{user_query}

Specialist summaries:
{specialist_summaries}

Judge assessment:
{judge_packet}

Write a concise final decision in Simplified Chinese Markdown with these sections:
## 结论
## 判断依据
## 关键证据
## 立即行动
## 置信度

Keep the answer practical, decisive, and grounded in the evidence that the council already discussed. Do not mix English into the answer unless it is a log excerpt, field name, command, code identifier, or model name.
"""


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of the first valid JSON object from model output."""
    candidates: List[str] = []
    stripped = text.strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    fence_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fence_matches)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return parsed

    return None


def _ensure_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce_confidence(value: Any, *, default: float = 0.35) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return round(number, 2)


def _coerce_score(value: Any, *, default: int = 3) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default

    if number < 1:
        return 1
    if number > 5:
        return 5
    return number


def _credibility_from_confidence(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence < 0.5:
        return "low"
    return "medium"


def _normalize_evidence_type(value: Any, *, default: str = "FACT") -> str:
    evidence_type = str(value or default).strip().upper()
    if evidence_type in {"FACT", "HYPOTHESIS", "CORRELATION", "RECOMMENDATION", "UNKNOWN"}:
        return evidence_type
    return "UNKNOWN"


def _normalize_fact_items(value: Any) -> List[str]:
    facts: List[str] = []
    if not isinstance(value, list):
        return _ensure_string_list(value)

    for item in value:
        if isinstance(item, dict):
            content = str(item.get("content") or item.get("detail") or item.get("fact") or "").strip()
            if content:
                facts.append(content)
        elif str(item).strip():
            facts.append(str(item).strip())

    return facts


def _normalize_timeline(value: Any) -> List[Dict[str, str]]:
    timeline: List[Dict[str, str]] = []
    if not isinstance(value, list):
        return timeline

    for item in value:
        if isinstance(item, dict):
            event = str(item.get("event") or item.get("content") or item.get("detail") or "").strip()
            if not event:
                continue
            timeline.append(
                {
                    "timestamp": str(item.get("timestamp") or "").strip(),
                    "event": event,
                    "source": str(item.get("source") or "user_input").strip() or "user_input",
                }
            )
        elif str(item).strip():
            timeline.append(
                {
                    "timestamp": "",
                    "event": str(item).strip(),
                    "source": "user_input",
                }
            )

    return timeline


def _normalize_evidence_items(value: Any) -> List[Dict[str, Any]]:
    evidence_items: List[Dict[str, Any]] = []
    if not isinstance(value, list):
        return evidence_items

    for item in value:
        if isinstance(item, dict):
            content = str(
                item.get("content")
                or item.get("detail")
                or item.get("cause")
                or item.get("event")
                or ""
            ).strip()
            if not content:
                continue
            source = item.get("source") if "source" in item else None
            evidence_type = _normalize_evidence_type(item.get("type"))
            confidence = _coerce_confidence(
                item.get("confidence"),
                default={
                    "high": 0.9,
                    "medium": 0.6,
                    "low": 0.3,
                }.get(str(item.get("credibility", "")).lower(), 0.6),
            )
            evidence = Evidence.from_mapping(
                {
                    **item,
                    "id": item.get("id") or item.get("evidence_id"),
                    "type": evidence_type,
                    "content": content,
                    "source": source,
                    "confidence": confidence,
                    "need_validation": item.get(
                        "need_validation",
                        evidence_type != "FACT" or confidence < 0.8,
                    ),
                    "credibility": str(item.get("credibility") or _credibility_from_confidence(confidence)),
                }
            ).to_dict()
            evidence_items.append(evidence)
        elif str(item).strip():
            evidence_items.append(Evidence(content=str(item).strip()).to_dict())

    return evidence_items


def _normalize_hypotheses(value: Any) -> List[Dict[str, Any]]:
    hypotheses: List[Dict[str, Any]] = []
    if not isinstance(value, list):
        return hypotheses

    for item in value:
        if isinstance(item, dict):
            content = str(item.get("content") or item.get("cause") or "").strip()
            if not content:
                continue
            confidence = _coerce_confidence(item.get("confidence"), default=0.35)
            hypotheses.append(
                {
                    "content": content,
                    "cause": content,
                    "rationale": str(item.get("rationale", "")).strip(),
                    "confidence": confidence,
                    "need_validation": bool(item.get("need_validation", True)),
                }
            )
        elif str(item).strip():
            content = str(item).strip()
            hypotheses.append(
                {
                    "content": content,
                    "cause": content,
                    "rationale": "",
                    "confidence": 0.35,
                    "need_validation": True,
                }
            )

    return hypotheses


def _evidence_from_strings(
    values: List[str],
    *,
    evidence_type: str,
    source: str,
    confidence: float,
    need_validation: bool,
) -> List[Dict[str, Any]]:
    return [
        Evidence.from_mapping(
            {
                "type": evidence_type,
                "content": value,
                "source": source,
                "confidence": confidence,
                "need_validation": need_validation,
            }
        ).to_dict()
        for value in values
    ]


def _dedupe_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in evidence:
        key = str(item.get("id") or item.get("content") or item.get("detail"))
        if key in seen:
            continue
        deduped.append(item)
        seen.add(key)
    return deduped


def _summarize_evidence_items(evidence_items: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "total": len(evidence_items),
        "fact": 0,
        "hypothesis": 0,
        "correlation": 0,
        "recommendation": 0,
        "unknown": 0,
    }
    for item in evidence_items:
        key = str(item.get("type") or "UNKNOWN").strip().lower()
        if key not in summary:
            summary[key] = 0
        summary[key] += 1
    return summary


def _normalize_evidence_summary(
    value: Any,
    evidence_items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    fallback = _summarize_evidence_items(evidence_items or [])
    if not isinstance(value, dict):
        return fallback

    summary = {**fallback}
    for key in ["total", "fact", "hypothesis", "correlation", "recommendation", "unknown"]:
        try:
            summary[key] = int(value.get(key, summary[key]))
        except (TypeError, ValueError):
            continue
        if summary[key] < 0:
            summary[key] = 0
    return summary


def _normalize_specialist_payload(
    raw_payload: Optional[Dict[str, Any]],
    raw_text: str,
    *,
    agent_role: str = "analysis",
) -> Dict[str, Any]:
    payload = raw_payload or {}
    summary = str(payload.get("summary", "")).strip() or "No structured summary was produced."
    facts = _normalize_fact_items(payload.get("facts"))
    signals = _ensure_string_list(payload.get("signals") or payload.get("findings"))
    timeline = _normalize_timeline(payload.get("timeline"))
    patterns = _ensure_string_list(payload.get("patterns"))
    unknowns = _ensure_string_list(payload.get("unknowns") or payload.get("gaps") or payload.get("missing_evidence"))
    missing_evidence = _ensure_string_list(payload.get("missing_evidence"))
    risks = _ensure_string_list(payload.get("risks") or payload.get("concerns"))
    alternative_hypotheses = _normalize_hypotheses(
        payload.get("alternative_hypotheses") or payload.get("alternative_causes")
    )
    hypotheses = _normalize_hypotheses(payload.get("hypotheses"))
    findings = _ensure_string_list(
        payload.get("findings")
        or payload.get("signals")
        or payload.get("patterns")
        or payload.get("concerns")
    )
    if not facts:
        facts = signals if agent_role in {"analysis", "investigator"} else []

    if agent_role == "investigator":
        hypotheses = []

    evidence = _normalize_evidence_items(payload.get("evidence"))
    if agent_role == "investigator":
        evidence = [
            item
            for item in evidence
            if item.get("type") in {"FACT", "CORRELATION", "UNKNOWN"}
        ]
    evidence.extend(
        _evidence_from_strings(
            facts,
            evidence_type="FACT",
            source=agent_role,
            confidence=0.85,
            need_validation=False,
        )
    )
    evidence.extend(
        _evidence_from_strings(
            patterns,
            evidence_type="CORRELATION",
            source=agent_role,
            confidence=0.6,
            need_validation=True,
        )
    )
    evidence.extend(
        _evidence_from_strings(
            [item["content"] for item in hypotheses + alternative_hypotheses],
            evidence_type="HYPOTHESIS",
            source=agent_role,
            confidence=0.35,
            need_validation=True,
        )
    )
    evidence.extend(
        _evidence_from_strings(
            unknowns,
            evidence_type="UNKNOWN",
            source=agent_role,
            confidence=0.0,
            need_validation=True,
        )
    )

    return {
        "agent_role": str(payload.get("agent_role") or agent_role),
        "summary": summary,
        "facts": facts,
        "signals": signals,
        "timeline": timeline,
        "patterns": patterns,
        "unknowns": unknowns,
        "reliability_assessment": str(payload.get("reliability_assessment", "")).strip(),
        "missing_evidence": missing_evidence,
        "challenged_hypothesis": str(payload.get("challenged_hypothesis", "")).strip(),
        "risks": risks,
        "alternative_hypotheses": alternative_hypotheses,
        "alternative_causes": [item["content"] for item in alternative_hypotheses],
        "concerns": risks,
        "findings": findings,
        "evidence": _dedupe_evidence(evidence),
        "hypotheses": hypotheses,
        "gaps": unknowns,
        "next_actions": _ensure_string_list(payload.get("next_actions") or payload.get("recommendations")),
        "tool_requests": _ensure_string_list(payload.get("tool_requests")),
        "confidence": _coerce_confidence(payload.get("confidence")),
        "raw_text": raw_text.strip(),
    }


def _normalize_scorecard(value: Any) -> List[Dict[str, Any]]:
    scorecard: List[Dict[str, Any]] = []
    if not isinstance(value, list):
        return scorecard

    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("agent_role", "")).strip()
        if not role:
            continue
        scorecard.append(
            {
                "agent_role": role,
                "agent_name": str(item.get("agent_name", role)).strip() or role,
                "agent_instance_id": str(item.get("agent_instance_id", "")).strip() or role,
                "evidence_score": _coerce_score(item.get("evidence_score")),
                "reasoning_score": _coerce_score(item.get("reasoning_score")),
                "actionability_score": _coerce_score(item.get("actionability_score")),
                "notes": str(item.get("notes", "")).strip(),
            }
        )

    return scorecard


def _normalize_judge_payload(raw_payload: Optional[Dict[str, Any]], raw_text: str) -> Dict[str, Any]:
    payload = raw_payload or {}
    root_cause = str(payload.get("root_cause") or payload.get("winning_hypothesis") or "Unclear").strip() or "Unclear"
    evidence = _normalize_evidence_items(
        payload.get("evidence")
        or payload.get("supporting_evidence")
        or payload.get("confirmed_evidence")
    )
    confirmed_evidence = _normalize_confirmed_evidence(payload.get("confirmed_evidence") or evidence)
    unverified_hypothesis = _normalize_unverified_hypotheses(
        payload.get("unverified_hypothesis") or payload.get("hypotheses")
    )
    recommendations = _ensure_string_list(
        payload.get("next_actions")
        or payload.get("recommendations")
        or payload.get("recommended_actions")
    )
    incident_level = str(
        payload.get("incident_level")
        or payload.get("incident_severity")
        or "unknown"
    ).strip() or "unknown"
    return {
        "agent_role": str(payload.get("agent_role") or "judge"),
        "verdict_summary": str(payload.get("verdict_summary", "")).strip() or "Judge did not provide a structured verdict.",
        "direct_cause": str(payload.get("direct_cause") or "").strip() or "unknown",
        "root_cause": root_cause,
        "winning_hypothesis": root_cause,
        "incident_level": incident_level,
        "incident_severity": incident_level,
        "confidence": _coerce_confidence(payload.get("confidence")),
        "evidence": evidence,
        "supporting_evidence": evidence,
        "confirmed_evidence": confirmed_evidence,
        "unverified_hypothesis": unverified_hypothesis,
        "evidence_summary": _normalize_evidence_summary(payload.get("evidence_summary"), evidence),
        "scorecard": _normalize_scorecard(payload.get("scorecard")),
        "gaps": _ensure_string_list(payload.get("gaps")),
        "unknowns": _ensure_string_list(payload.get("unknowns") or payload.get("gaps")),
        "facts": [item["content"] for item in confirmed_evidence],
        "hypotheses": unverified_hypothesis,
        "recommendations": recommendations,
        "recommended_actions": recommendations,
        "next_actions": recommendations,
        "minority_view": str(payload.get("minority_view", "")).strip(),
        "raw_text": raw_text.strip(),
    }


def _normalize_confirmed_evidence(value: Any) -> List[Dict[str, Any]]:
    items = _normalize_evidence_items(value)
    confirmed: List[Dict[str, Any]] = []
    for item in items:
        if item.get("type") not in {"FACT", "CORRELATION"}:
            continue
        credibility = str(item.get("credibility") or "").upper()
        confirmed.append(
            {
                "evidence_id": item.get("id"),
                "type": item.get("type", "FACT"),
                "content": item.get("content") or item.get("detail", ""),
                "source": item.get("source", "user_input"),
                "credibility": credibility or "MEDIUM",
                "reason": item.get("reason") or item.get("metadata", {}).get("reason", ""),
            }
        )
    return confirmed


def _normalize_unverified_hypotheses(value: Any) -> List[Dict[str, Any]]:
    hypotheses: List[Dict[str, Any]] = []
    if not isinstance(value, list):
        return hypotheses

    for item in value:
        if isinstance(item, dict):
            content = str(item.get("content") or item.get("cause") or "").strip()
            if not content:
                continue
            confidence = item.get("confidence")
            if isinstance(confidence, (int, float)):
                confidence_label = _credibility_from_confidence(_coerce_confidence(confidence)).upper()
            else:
                confidence_label = str(confidence or "MEDIUM").strip().upper()
            hypotheses.append(
                {
                    "content": content,
                    "confidence": confidence_label or "MEDIUM",
                    "needed_evidence": str(
                        item.get("needed_evidence")
                        or item.get("validation")
                        or item.get("rationale")
                        or ""
                    ).strip(),
                }
            )
        elif str(item).strip():
            hypotheses.append(
                {
                    "content": str(item).strip(),
                    "confidence": "MEDIUM",
                    "needed_evidence": "",
                }
            )

    return hypotheses


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
        result = event_sink(event)
        if inspect.isawaitable(result):
            await result
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


def _build_agent_result(
    *,
    agent_role: str,
    agent_name: str,
    model: str,
    response: str,
    structured_output: Dict[str, Any],
    agent_instance_id: Optional[str] = None,
    tool_results: Optional[List[Dict[str, Any]]] = None,
) -> AgentResultPayload:
    evidence = _evidence_models_from_payload(structured_output, agent_role=agent_role)
    confidence = _coerce_confidence(structured_output.get("confidence"), default=0.0)
    message = AgentMessage(
        role=agent_role,
        content=structured_output.get("summary")
        or structured_output.get("verdict_summary")
        or structured_output.get("root_cause")
        or response,
        evidence=evidence,
        confidence=confidence,
    )
    return AgentResult(
        agent_role=agent_role,
        agent_name=agent_name,
        agent_instance_id=agent_instance_id or f"{agent_role}:{model}",
        model=model,
        response=response,
        structured_output=structured_output,
        evidence=evidence,
        messages=[message],
        confidence=confidence,
        tool_results=tool_results or [],
    ).to_dict()


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


def build_evidence_store_from_results(stage1_results: List[AgentResultPayload]) -> EvidenceStore:
    """Build the run-scoped EvidenceStore after AgentResult normalization."""
    evidence_store = EvidenceStore()
    ordered_results = sorted(
        stage1_results,
        key=lambda result: _EVIDENCE_ROLE_ORDER.get(str(result.get("agent_role")), 99),
    )

    for result in ordered_results:
        evidence_store.add_many(result.get("evidence", []) or [])
        evidence_store.add_many(_tool_results_to_evidence(result))

    return evidence_store


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


async def _run_specialist_agent(
    blueprint: RoleBlueprint,
    model: ModelConfig,
    incident_input: IncidentInput,
    event_sink: Optional[AgentEventSink] = None,
    run_id: Optional[str] = None,
    execution_collector: Optional[LLMExecutionCollector] = None,
) -> Optional[AgentResultPayload]:
    await _emit_agent_event(
        event_sink,
        agent=blueprint.role_id,
        status="running",
        message=f"{blueprint.agent_name} started on {model.id}",
        metadata={"model": model.id},
    )
    tool_results: List[Dict[str, Any]] = []
    if blueprint.role_id == "investigator":
        await _emit_agent_event(
            event_sink,
            agent=blueprint.role_id,
            status="running",
            message="Running deterministic investigation tools",
            metadata={"tools": [tool["name"] for tool in investigation_tool_catalog()]},
        )
        tool_results = run_investigation_tools(InvestigationToolContext(incident_input))
        await _emit_agent_event(
            event_sink,
            agent=blueprint.role_id,
            status="completed",
            message="Investigation tools completed",
            metadata={"tool_count": len(tool_results)},
        )

    prompt = _build_specialist_prompt(blueprint, incident_input, tool_results)
    response = await query_model(
        model,
        [{"role": "user", "content": prompt}],
        run_id=run_id,
        workflow_role="specialist",
        execution_collector=execution_collector,
    )
    if response is None:
        await _emit_agent_event(
            event_sink,
            agent=blueprint.role_id,
            status="failed",
            message=f"{blueprint.agent_name} failed on {model.id}",
            metadata={"model": model.id},
        )
        return None

    raw_text = response.get("content", "")
    payload = _normalize_specialist_payload(
        _extract_json_object(raw_text),
        raw_text,
        agent_role=blueprint.role_id,
    )
    payload["tool_results"] = tool_results
    result = _build_agent_result(
        agent_role=blueprint.role_id,
        agent_name=blueprint.agent_name,
        model=model.id,
        response=_format_specialist_markdown(blueprint, payload),
        structured_output=payload,
        agent_instance_id=f"{blueprint.role_id}:{model.id}",
        tool_results=tool_results,
    )
    await _emit_agent_event(
        event_sink,
        agent=blueprint.role_id,
        status="completed",
        message=f"{blueprint.agent_name} produced structured output",
        metadata={
            "model": model.id,
            "facts": len(payload.get("facts", [])),
            "hypotheses": len(payload.get("hypotheses", [])),
            "evidence": len(payload.get("evidence", [])),
            "unknowns": len(payload.get("unknowns", [])),
        },
    )
    return result


def _judge_fallback_models(stage1_results: List[AgentResultPayload]) -> List[ModelConfig]:
    stage1_model_ids = {result["model"] for result in stage1_results}
    candidates: List[ModelConfig] = []

    for model in [CHAIRMAN_MODEL, *COUNCIL_MODELS]:
        if model.id in stage1_model_ids or model.id == CHAIRMAN_MODEL.id:
            candidates.append(model)

    deduped: List[ModelConfig] = []
    seen: set[str] = set()
    for model in candidates:
        if model.id not in seen:
            deduped.append(model)
            seen.add(model.id)

    return deduped


async def stage1_collect_responses(
    user_query: str,
    incident_input: Optional[IncidentInput] = None,
    event_sink: Optional[AgentEventSink] = None,
    run_id: Optional[str] = None,
    execution_collector: Optional[LLMExecutionCollector] = None,
) -> List[AgentResultPayload]:
    """
    Stage 1: Run specialist agent roles in parallel.

    Args:
        user_query: The user's question or incident input

    Returns:
        List of structured specialist results
    """
    parsed_input = incident_input or parse_incident_input(user_query)
    stage_run_id = run_id or str(uuid4())
    assignments = _specialist_assignments()
    tasks = [
        _run_specialist_agent(
            blueprint,
            model,
            parsed_input,
            event_sink,
            run_id=stage_run_id,
            execution_collector=execution_collector,
        )
        for blueprint, model in assignments
    ]
    responses = await asyncio.gather(*tasks)
    return [response for response in responses if response is not None]


async def stage2_judge_deliberation(
    user_query: str,
    stage1_results: List[AgentResultPayload],
    event_sink: Optional[AgentEventSink] = None,
    evidence_store: Optional[EvidenceStore] = None,
    event_store: Optional[EventStore] = None,
    run_id: Optional[str] = None,
    execution_collector: Optional[LLMExecutionCollector] = None,
) -> AgentResultPayload:
    """
    Stage 2: Judge evaluates specialist outputs and produces a scorecard.

    Args:
        user_query: The original user query
        stage1_results: Structured outputs from the specialist agents

    Returns:
        Structured judge assessment
    """
    await _emit_agent_event(
        event_sink,
        agent="judge",
        status="running",
        message="Judge Agent is evaluating evidence quality",
        metadata={"specialist_count": len(stage1_results)},
    )
    evidence_store = evidence_store or build_evidence_store_from_results(stage1_results)
    judge_event_store = event_store or evidence_store.event_store
    evidence_items = evidence_store.get_all()
    evidence_summary = _summarize_evidence_items(evidence_items)
    evidence_ranking = rank_evidences(evidence_items)
    prompt = _build_judge_prompt(user_query, stage1_results, evidence_items, evidence_ranking)
    response: Optional[Dict[str, Any]] = None
    judge_model = CHAIRMAN_MODEL
    stage_run_id = run_id or str(uuid4())

    for candidate in _judge_fallback_models(stage1_results):
        response = await query_model(
            candidate,
            [{"role": "user", "content": prompt}],
            run_id=stage_run_id,
            workflow_role="judge",
            execution_collector=execution_collector,
        )
        if response is not None:
            judge_model = candidate
            break

    if response is None:
        structured_output = _normalize_judge_payload(
            None,
            "Judge model was unavailable; returning a fallback verdict.",
        )
        structured_output["evidence_summary"] = evidence_summary
        structured_output["evidence_ranking"] = evidence_ranking
        fallback_result = _build_agent_result(
            agent_role="judge",
            agent_name="Judge Agent",
            model=CHAIRMAN_MODEL.id,
            response=_format_judge_markdown(structured_output),
            structured_output=structured_output,
        )
        await _emit_agent_event(
            event_sink,
            agent="judge",
            status="failed",
            message="Judge model was unavailable; fallback verdict returned",
            metadata={"model": CHAIRMAN_MODEL.id},
        )
        return fallback_result

    raw_text = response.get("content", "")
    raw_payload = _extract_json_object(raw_text)
    structured_output = _normalize_judge_payload(raw_payload, raw_text)
    if not raw_payload or not isinstance(raw_payload.get("evidence_summary"), dict):
        structured_output["evidence_summary"] = evidence_summary
    else:
        structured_output["evidence_summary"] = _normalize_evidence_summary(
            structured_output.get("evidence_summary"),
            evidence_items,
        )
    structured_output["evidence_ranking"] = evidence_ranking
    _emit_judge_evidence_usage_events(
        judge_event_store,
        structured_output.get("confirmed_evidence", []),
        judge_name="Judge Agent",
    )
    result = _build_agent_result(
        agent_role="judge",
        agent_name="Judge Agent",
        model=judge_model.id,
        response=_format_judge_markdown(structured_output),
        structured_output=structured_output,
    )
    await _emit_agent_event(
        event_sink,
        agent="judge",
        status="completed",
        message="Judge Agent completed evidence evaluation",
        metadata={
            "model": judge_model.id,
            "confirmed_evidence": len(structured_output.get("confirmed_evidence", [])),
            "unverified_hypothesis": len(structured_output.get("unverified_hypothesis", [])),
            "evidence_summary": structured_output.get("evidence_summary", {}),
            "evidence_ranking_count": len(evidence_ranking.get("ranked_evidence", [])),
        },
    )
    return result


def _fallback_final_response(stage2_result: AgentResultPayload) -> str:
    verdict = stage2_result.get("structured_output", {})
    actions = verdict.get("next_actions") or verdict.get("recommendations") or ["Collect more evidence before acting."]
    evidence = verdict.get("confirmed_evidence") or verdict.get("evidence") or []

    lines = [
        "## 结论",
        verdict.get("verdict_summary", "A structured final decision was not available."),
        "",
        "## 判断依据",
        verdict.get("root_cause", "The council could not confirm a leading hypothesis."),
        "",
        "## 关键证据",
    ]
    if evidence:
        lines.extend(f"- {item.get('content') or item.get('detail', '')}" for item in evidence)
    else:
        lines.append("- Judge Agent 指出当前可用的直接证据仍然有限。")

    lines.extend(["", "## 立即行动"])
    lines.extend(f"- {item}" for item in actions)
    lines.extend(["", "## 置信度", f"{int(verdict.get('confidence', 0.35) * 100)}%"])
    return "\n".join(lines)


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[AgentResultPayload],
    stage2_result: AgentResultPayload,
    event_sink: Optional[AgentEventSink] = None,
    run_id: Optional[str] = None,
    execution_collector: Optional[LLMExecutionCollector] = None,
) -> AgentResultPayload:
    """
    Stage 3: Produce the final decision memo for the council.

    Args:
        user_query: The original user query
        stage1_results: Specialist results from Stage 1
        stage2_result: Judge assessment from Stage 2

    Returns:
        Dict with the final decision response
    """
    await _emit_agent_event(
        event_sink,
        agent="final_decision",
        status="running",
        message="Final Decision is synthesizing the judge assessment",
        metadata={"model": CHAIRMAN_MODEL.id},
    )
    prompt = _build_final_decision_prompt(user_query, stage1_results, stage2_result)
    response = await query_model(
        CHAIRMAN_MODEL,
        [{"role": "user", "content": prompt}],
        run_id=run_id or str(uuid4()),
        workflow_role="final",
        execution_collector=execution_collector,
    )

    if response is None:
        final_response = _fallback_final_response(stage2_result)
        structured_output = {
            "summary": stage2_result["structured_output"].get("verdict_summary", ""),
            "decision_summary": stage2_result["structured_output"].get("verdict_summary", ""),
            "facts": stage2_result["structured_output"].get("facts", []),
            "hypotheses": stage2_result["structured_output"].get("hypotheses", []),
            "unknowns": stage2_result["structured_output"].get("unknowns", []),
            "root_cause": stage2_result["structured_output"].get("root_cause", ""),
            "confidence": stage2_result["structured_output"].get("confidence", 0.35),
            "evidence": stage2_result["structured_output"].get("evidence", []),
            "confirmed_evidence": stage2_result["structured_output"].get("confirmed_evidence", []),
            "unverified_hypothesis": stage2_result["structured_output"].get("unverified_hypothesis", []),
            "next_actions": stage2_result["structured_output"].get("next_actions", []),
        }
        result = _build_agent_result(
            agent_role="final_decision",
            agent_name="Final Decision",
            model=CHAIRMAN_MODEL.id,
            response=final_response,
            structured_output=structured_output,
        )
        result["decision_summary"] = structured_output["decision_summary"]
        await _emit_agent_event(
            event_sink,
            agent="final_decision",
            status="failed",
            message="Final model unavailable; fallback decision returned",
            metadata={"model": CHAIRMAN_MODEL.id},
        )
        return result

    structured_output = {
        "summary": stage2_result["structured_output"].get("verdict_summary", ""),
        "decision_summary": stage2_result["structured_output"].get("verdict_summary", ""),
        "facts": stage2_result["structured_output"].get("facts", []),
        "hypotheses": stage2_result["structured_output"].get("hypotheses", []),
        "unknowns": stage2_result["structured_output"].get("unknowns", []),
        "root_cause": stage2_result["structured_output"].get("root_cause", ""),
        "confidence": stage2_result["structured_output"].get("confidence", 0.35),
        "evidence": stage2_result["structured_output"].get("evidence", []),
        "confirmed_evidence": stage2_result["structured_output"].get("confirmed_evidence", []),
        "unverified_hypothesis": stage2_result["structured_output"].get("unverified_hypothesis", []),
        "next_actions": stage2_result["structured_output"].get("next_actions", []),
        "raw_text": response.get("content", ""),
    }
    result = _build_agent_result(
        agent_role="final_decision",
        agent_name="Final Decision",
        model=CHAIRMAN_MODEL.id,
        response=response.get("content", ""),
        structured_output=structured_output,
    )
    result["decision_summary"] = structured_output["decision_summary"]
    await _emit_agent_event(
        event_sink,
        agent="final_decision",
        status="completed",
        message="Final Decision completed",
        metadata={"model": CHAIRMAN_MODEL.id},
    )
    return result


async def generate_conversation_title(
    user_query: str,
    run_id: Optional[str] = None,
    execution_collector: Optional[LLMExecutionCollector] = None,
) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    fallback_title = _fallback_conversation_title(user_query)
    title_prompt = f"""Generate a very short Simplified Chinese title (3-5 words maximum) that summarizes the following incident or question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]
    response = await query_model(
        TITLE_MODEL,
        messages,
        timeout=30.0,
        run_id=run_id,
        workflow_role="title",
        execution_collector=execution_collector,
    )

    if response is None:
        return fallback_title

    title = response.get("content", "New Conversation").strip().strip("\"'")
    if not title or title.lower() == "new conversation":
        return fallback_title

    if len(title) > 50:
        title = title[:47] + "..."
    return title


def _fallback_conversation_title(user_query: str) -> str:
    """Generate a local title when the title model is unavailable."""
    without_fences = re.sub(r"```.*?```", " ", user_query, flags=re.DOTALL)
    lines = [line.strip() for line in without_fences.splitlines()]

    for line in lines:
        if not line:
            continue
        if re.match(r"^(?:#{1,6}\s*)?(?:\[)?(?:error|application|system)\.log(?:\])?:?\s*$", line, re.IGNORECASE):
            continue

        candidate = re.sub(r"^[#>\-\s]+", "", line)
        candidate = re.sub(r"\s+", " ", candidate).strip(" ，。,.!！?？:：;；\"'")
        if len(candidate) < 2:
            continue
        if len(candidate) > 24:
            candidate = candidate[:21].rstrip() + "..."
        return candidate

    return "新会话"




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


async def run_full_council(
    user_query: str,
    structured_logs: Optional[Dict[str, str]] = None,
    run_id: Optional[str] = None,
    execution_collector: Optional[LLMExecutionCollector] = None,
) -> Tuple[List[AgentResultPayload], AgentResultPayload, AgentResultPayload, Dict[str, Any]]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's incident prompt

    Returns:
        Tuple of (stage1_results, stage2_result, stage3_result, metadata)
    """
    council_run_id = run_id or str(uuid4())
    owns_collector = execution_collector is None
    collector = execution_collector or LLMExecutionCollector(council_run_id)
    incident_input = parse_incident_input(user_query, structured_logs)
    agent_events: List[Dict[str, Any]] = []

    def collect_event(event: Dict[str, Any]) -> None:
        agent_events.append(event)

    stage1_results = await stage1_collect_responses(
        user_query,
        incident_input,
        event_sink=collect_event,
        run_id=council_run_id,
        execution_collector=collector,
    )

    if not stage1_results:
        empty_judge_output = _normalize_judge_payload(
            None,
            "All specialist agents failed to respond. No judge assessment is available.",
        )
        empty_stage2_result = _build_agent_result(
            agent_role="judge",
            agent_name="Judge Agent",
            model="error",
            response="All specialist agents failed to respond. No judge assessment is available.",
            structured_output=empty_judge_output,
        )
        empty_stage3_result = _build_agent_result(
            agent_role="final_decision",
            agent_name="Final Decision",
            model="error",
            response="All specialist agents failed to respond. Please try again.",
            structured_output={
                "summary": "All specialist agents failed to respond.",
                "decision_summary": "No decision available.",
                "facts": [],
                "hypotheses": [],
                "unknowns": ["No specialist outputs were available."],
                "root_cause": "unknown",
                "confidence": 0.0,
                "evidence": [],
                "confirmed_evidence": [],
                "unverified_hypothesis": [],
                "next_actions": ["Retry after checking model availability and API configuration."],
            },
        )
        empty_stage3_result["decision_summary"] = "No decision available."
        if owns_collector:
            log_llm_run_summary(collector.summary())
        return [], empty_stage2_result, empty_stage3_result, {"agent_events": agent_events}

    evidence_store = build_evidence_store_from_results(stage1_results)
    stage2_result = await stage2_judge_deliberation(
        user_query,
        stage1_results,
        event_sink=collect_event,
        evidence_store=evidence_store,
        run_id=council_run_id,
        execution_collector=collector,
    )
    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_result,
        event_sink=collect_event,
        run_id=council_run_id,
        execution_collector=collector,
    )
    metadata = build_council_metadata(
        stage1_results,
        stage2_result,
        incident_input,
        agent_events,
        evidence_store,
    )
    if owns_collector:
        log_llm_run_summary(collector.summary())
    return stage1_results, stage2_result, stage3_result, metadata
