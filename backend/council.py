"""Role-based 3-stage Agent Council orchestration."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
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
from .council_runtime.events import (
    AgentEventSink,
    _emit_agent_event,
    _emit_judge_evidence_usage_events,
)
from .council_runtime.evidence import (
    _EVIDENCE_ROLE_ORDER,
    _evidence_models_from_payload,
    _tool_results_to_evidence,
)
from .council_runtime.final import (
    _build_final_decision_prompt,
    _fallback_final_response,
    _final_structured_output_from_judge,
)
from .council_runtime.judge import _build_judge_prompt, _format_judge_markdown
from .council_runtime.metadata import build_council_metadata
from .council_runtime.normalization import (
    _coerce_confidence,
    _coerce_score,
    _credibility_from_confidence,
    _dedupe_evidence,
    _ensure_string_list,
    _evidence_from_strings,
    _extract_json_object,
    _is_valid_judge_payload,
    _normalize_confirmed_evidence,
    _normalize_evidence_items,
    _normalize_evidence_summary,
    _normalize_evidence_type,
    _normalize_fact_items,
    _normalize_hypotheses,
    _normalize_judge_payload,
    _normalize_scorecard,
    _normalize_specialist_payload,
    _normalize_timeline,
    _normalize_unverified_hypotheses,
    _summarize_evidence_items,
)
from .council_runtime.schemas import (
    JUDGE_JSON_SCHEMA,
    SPECIALIST_BLUEPRINTS,
    SPECIALIST_JSON_SCHEMAS,
    RoleBlueprint,
)
from .council_runtime.specialists import (
    _build_specialist_prompt,
    _format_specialist_markdown,
)
from .council_runtime.titles import _fallback_conversation_title


logger = logging.getLogger(__name__)




def _specialist_assignments() -> List[Tuple[RoleBlueprint, ModelConfig]]:
    """Assign every configured model to a specialist role instance."""
    assignments: List[Tuple[RoleBlueprint, ModelConfig]] = []
    models = COUNCIL_MODELS or [CHAIRMAN_MODEL]

    for index, model in enumerate(models):
        blueprint = SPECIALIST_BLUEPRINTS[index % len(SPECIALIST_BLUEPRINTS)]
        assignments.append((blueprint, model))

    return assignments






















































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






async def _run_specialist_agent(
    blueprint: RoleBlueprint,
    model: ModelConfig,
    incident_input: IncidentInput,
    event_sink: Optional[AgentEventSink] = None,
    run_id: Optional[str] = None,
    execution_collector: Optional[LLMExecutionCollector] = None,
) -> Optional[AgentResultPayload]:
    try:
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
    except Exception as exc:
        logger.warning(
            "Specialist agent failed: role=%s model=%s",
            blueprint.role_id,
            model.id,
            exc_info=True,
        )
        await _emit_agent_event(
            event_sink,
            agent=blueprint.role_id,
            status="failed",
            message=f"{blueprint.agent_name} failed on {model.id}",
            metadata={"model": model.id, "error": str(exc)},
        )
        return None

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


def _build_fallback_judge_result(
    evidence_summary: Dict[str, int],
    evidence_ranking: Dict[str, List[Dict[str, Any]]],
) -> AgentResultPayload:
    structured_output = _normalize_judge_payload(
        None,
        "Judge model was unavailable; returning a fallback verdict.",
    )
    structured_output["evidence_summary"] = evidence_summary
    structured_output["evidence_ranking"] = evidence_ranking
    return _build_agent_result(
        agent_role="judge",
        agent_name="Judge Agent",
        model=CHAIRMAN_MODEL.id,
        response=_format_judge_markdown(structured_output),
        structured_output=structured_output,
    )


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
    raw_text = ""
    raw_payload: Optional[Dict[str, Any]] = None
    judge_model = CHAIRMAN_MODEL
    stage_run_id = run_id or str(uuid4())

    for candidate in _judge_fallback_models(stage1_results):
        try:
            response = await query_model(
                candidate,
                [{"role": "user", "content": prompt}],
                run_id=stage_run_id,
                workflow_role="judge",
                execution_collector=execution_collector,
            )
        except Exception:
            logger.warning("Judge candidate failed: model=%s", candidate.id, exc_info=True)
            continue

        if response is None:
            continue

        candidate_text = response.get("content", "")
        if not isinstance(candidate_text, str) or not candidate_text.strip():
            logger.warning("Judge candidate returned empty content: model=%s", candidate.id)
            continue

        candidate_payload = _extract_json_object(candidate_text)
        if not _is_valid_judge_payload(candidate_payload):
            logger.warning("Judge candidate returned invalid payload: model=%s", candidate.id)
            continue

        raw_text = candidate_text
        raw_payload = candidate_payload
        judge_model = candidate
        break

    if raw_payload is None:
        fallback_result = _build_fallback_judge_result(evidence_summary, evidence_ranking)
        await _emit_agent_event(
            event_sink,
            agent="judge",
            status="failed",
            message="Judge model was unavailable; fallback verdict returned",
            metadata={"model": CHAIRMAN_MODEL.id},
        )
        return fallback_result

    structured_output = _normalize_judge_payload(raw_payload, raw_text)
    if not isinstance(raw_payload.get("evidence_summary"), dict):
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





def _build_final_fallback_result(stage2_result: AgentResultPayload) -> AgentResultPayload:
    structured_output = _final_structured_output_from_judge(stage2_result)
    result = _build_agent_result(
        agent_role="final_decision",
        agent_name="Final Decision",
        model=CHAIRMAN_MODEL.id,
        response=_fallback_final_response(stage2_result),
        structured_output=structured_output,
    )
    result["decision_summary"] = structured_output["decision_summary"]
    return result


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
    try:
        response = await query_model(
            CHAIRMAN_MODEL,
            [{"role": "user", "content": prompt}],
            run_id=run_id or str(uuid4()),
            workflow_role="final",
            execution_collector=execution_collector,
        )
        final_response = "" if response is None else response.get("content", "")
    except Exception:
        logger.warning("Final model failed; returning judge-based fallback", exc_info=True)
        result = _build_final_fallback_result(stage2_result)
        await _emit_agent_event(
            event_sink,
            agent="final_decision",
            status="failed",
            message="Final model failed; fallback decision returned",
            metadata={"model": CHAIRMAN_MODEL.id},
        )
        return result

    if response is None:
        result = _build_final_fallback_result(stage2_result)
        await _emit_agent_event(
            event_sink,
            agent="final_decision",
            status="failed",
            message="Final model unavailable; fallback decision returned",
            metadata={"model": CHAIRMAN_MODEL.id},
        )
        return result

    if not isinstance(final_response, str) or not final_response.strip():
        result = _build_final_fallback_result(stage2_result)
        await _emit_agent_event(
            event_sink,
            agent="final_decision",
            status="failed",
            message="Final model returned empty content; fallback decision returned",
            metadata={"model": CHAIRMAN_MODEL.id},
        )
        return result

    structured_output = _final_structured_output_from_judge(stage2_result)
    structured_output["raw_text"] = final_response
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
    try:
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
    except Exception:
        logger.warning("Title generation failed; using local fallback", exc_info=True)
        return fallback_title







def build_empty_council_results(
    agent_events: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[AgentResultPayload], AgentResultPayload, AgentResultPayload, Dict[str, Any]]:
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
    return [], empty_stage2_result, empty_stage3_result, {"agent_events": agent_events or []}

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
        empty_results = build_empty_council_results(agent_events)
        if owns_collector:
            log_llm_run_summary(collector.summary())
        return empty_results

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
