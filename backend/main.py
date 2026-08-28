"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import asyncio
import time

from . import storage
from .council import (
    build_council_metadata,
    build_empty_council_results,
    build_evidence_store_from_results,
    generate_conversation_title,
    run_full_council,
    stage1_collect_responses,
    stage2_judge_deliberation,
    stage3_synthesize_final,
)
from .incident_input import parse_incident_input
from .llm.aggregation import LLMExecutionCollector, log_llm_run_summary

app = FastAPI(title="Agent Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str
    logs: Dict[str, str] | None = None


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Agent Council API"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    reusable_conversation = storage.get_reusable_empty_conversation()
    if reusable_conversation is not None:
        return reusable_conversation

    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0
    council_run_id = str(uuid.uuid4())
    execution_collector = LLMExecutionCollector(council_run_id)

    try:
        # Add user message
        storage.add_user_message(conversation_id, request.content)

        # If this is the first message, generate a title
        if is_first_message:
            title = await generate_conversation_title(
                request.content,
                run_id=council_run_id,
                execution_collector=execution_collector,
            )
            storage.update_conversation_title(conversation_id, title)

        # Run the 3-stage council process
        stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
            request.content,
            request.logs,
            run_id=council_run_id,
            execution_collector=execution_collector,
        )

        # Add assistant message with all stages
        storage.add_assistant_message(
            conversation_id,
            stage1_results,
            stage2_results,
            stage3_result,
            metadata,
        )

        # Return the complete response with metadata
        return {
            "stage1": stage1_results,
            "stage2": stage2_results,
            "stage3": stage3_result,
            "metadata": metadata
        }
    finally:
        log_llm_run_summary(execution_collector.summary())


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0
    council_run_id = str(uuid.uuid4())
    execution_collector = LLMExecutionCollector(council_run_id)

    async def event_generator():
        try:
            incident_input = parse_incident_input(request.content, request.logs)
            agent_events: List[Dict[str, Any]] = []
            agent_event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

            async def queue_agent_event(event: Dict[str, Any]) -> None:
                agent_events.append(event)
                await agent_event_queue.put(event)

            async def forward_agent_events_while(task: asyncio.Task):
                last_heartbeat = time.monotonic()
                while not task.done():
                    try:
                        event = await asyncio.wait_for(agent_event_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        if time.monotonic() - last_heartbeat >= 15:
                            last_heartbeat = time.monotonic()
                            yield ": heartbeat\n\n"
                        continue
                    yield f"data: {json.dumps(event)}\n\n"

                while not agent_event_queue.empty():
                    event = agent_event_queue.get_nowait()
                    yield f"data: {json.dumps(event)}\n\n"

            # Add user message
            storage.add_user_message(conversation_id, request.content)

            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(
                    generate_conversation_title(
                        request.content,
                        run_id=council_run_id,
                        execution_collector=execution_collector,
                    )
                )

            # Stage 1: Collect responses
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_task = asyncio.create_task(
                stage1_collect_responses(
                    request.content,
                    incident_input,
                    event_sink=queue_agent_event,
                    run_id=council_run_id,
                    execution_collector=execution_collector,
                )
            )
            async for event in forward_agent_events_while(stage1_task):
                yield event
            stage1_results = await stage1_task
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            if not stage1_results:
                stage1_results, stage2_results, stage3_result, metadata = build_empty_council_results(agent_events)
                yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"
                if title_task:
                    title = await title_task
                    storage.update_conversation_title(conversation_id, title)
                    yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"
                storage.add_assistant_message(
                    conversation_id,
                    stage1_results,
                    stage2_results,
                    stage3_result,
                    metadata,
                )
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                return

            # Stage 2: Judge deliberation
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            evidence_store = build_evidence_store_from_results(stage1_results)
            stage2_task = asyncio.create_task(
                stage2_judge_deliberation(
                    request.content,
                    stage1_results,
                    event_sink=queue_agent_event,
                    evidence_store=evidence_store,
                    run_id=council_run_id,
                    execution_collector=execution_collector,
                )
            )
            async for event in forward_agent_events_while(stage2_task):
                yield event
            stage2_results = await stage2_task
            metadata = build_council_metadata(
                stage1_results,
                stage2_results,
                incident_input,
                agent_events,
                evidence_store,
            )
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': metadata})}\n\n"

            # Stage 3: Synthesize final answer
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_task = asyncio.create_task(
                stage3_synthesize_final(
                    request.content,
                    stage1_results,
                    stage2_results,
                    event_sink=queue_agent_event,
                    run_id=council_run_id,
                    execution_collector=execution_collector,
                )
            )
            async for event in forward_agent_events_while(stage3_task):
                yield event
            stage3_result = await stage3_task
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"
            metadata = build_council_metadata(
                stage1_results,
                stage2_results,
                incident_input,
                agent_events,
                evidence_store,
            )

            # Wait for title generation if it was started
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save complete assistant message
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result,
                metadata,
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            log_llm_run_summary(execution_collector.summary())

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
