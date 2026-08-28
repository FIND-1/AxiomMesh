"""Schemas and role definitions for Council runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RoleBlueprint:
    """Configuration for a specialist council role."""

    role_id: str
    agent_name: str
    mission: str
    emphasis: List[str]

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
