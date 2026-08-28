"""Normalization helpers for Council model payloads."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..agent_models import Evidence


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

def _is_valid_judge_payload(raw_payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(raw_payload, dict):
        return False

    text_fields = {
        "verdict_summary",
        "direct_cause",
        "root_cause",
        "winning_hypothesis",
        "minority_view",
    }
    list_fields = {
        "evidence",
        "supporting_evidence",
        "confirmed_evidence",
        "unverified_hypothesis",
        "hypotheses",
        "scorecard",
        "next_actions",
        "recommendations",
        "recommended_actions",
        "gaps",
        "unknowns",
    }

    for field in text_fields:
        value = raw_payload.get(field)
        if isinstance(value, str) and value.strip():
            return True
    for field in list_fields:
        value = raw_payload.get(field)
        if isinstance(value, list) and value:
            return True
    return False

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
