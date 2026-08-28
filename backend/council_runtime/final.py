"""Final decision prompt and fallback helpers."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..agent_models import AgentResultPayload


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

def _final_structured_output_from_judge(stage2_result: AgentResultPayload) -> Dict[str, Any]:
    verdict = stage2_result.get("structured_output", {})
    return {
        "summary": verdict.get("verdict_summary", ""),
        "decision_summary": verdict.get("verdict_summary", ""),
        "facts": verdict.get("facts", []),
        "hypotheses": verdict.get("hypotheses", []),
        "unknowns": verdict.get("unknowns", []),
        "root_cause": verdict.get("root_cause", ""),
        "confidence": verdict.get("confidence", 0.35),
        "evidence": verdict.get("evidence", []),
        "confirmed_evidence": verdict.get("confirmed_evidence", []),
        "unverified_hypothesis": verdict.get("unverified_hypothesis", []),
        "next_actions": verdict.get("next_actions", []),
    }
