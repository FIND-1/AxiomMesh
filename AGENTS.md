# AGENTS.md

This file gives Codex-oriented guidance for working in `AxiomMesh`.

## Communication

- Default to Chinese when replying to the user unless they ask for another language.
- Be concise, but include concrete file paths, commands, and validation results when making changes.

## Project Overview

LLM Council is a local Agent Council web app for incident analysis. It lets several configured LLMs act as specialist agents, collect and normalize evidence, ask a Judge Agent to evaluate the evidence, and produce a final decision through a chairman model.

Core flow:

1. Stage 1: run specialist agents in parallel.
2. Stage 2: build an evidence store, rank evidence, and run Judge deliberation.
3. Stage 3: synthesize a final decision memo from the specialist and judge outputs.

## Working Style

- Read the existing implementation before changing it.
- Prefer small, local edits over refactors.
- Reuse existing modules, utilities, and UI patterns.
- Do not remove existing business behavior unless the user explicitly asks for it.
- Ignore unrelated dirty worktree changes unless they are necessary to understand your task.

## Architecture

### Backend (`backend/`)

- `config.py`
  - Defines `ModelConfig`, `MODEL_REGISTRY`, `COUNCIL_MODELS`, `CHAIRMAN_MODEL`, and `TITLE_MODEL`.
  - Reads provider keys from `.env`: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `KIMI_API_KEY`, and `QWEN_API_KEY`.
  - Supports compatibility aliases: `OPENROUTER_API_KEY` for OpenAI, `MOONSHOT_API_KEY` for Kimi, and `DASHSCOPE_API_KEY` for Qwen.
  - Backend port is `8001`.

- `openrouter.py`
  - Provider client layer despite the legacy filename.
  - `query_model()`: single async model call routed by provider.
  - `query_models_parallel()`: parallel model calls with graceful degradation.
  - Failed model calls should not crash the whole request if others succeed.

- `agent_models.py`
  - Defines structured payloads and dataclasses for `Evidence`, `AgentMessage`, `AgentEvent`, and `AgentResult`.
  - Normalizes evidence source traces, confidence values, ids, and backward-compatible aliases.

- `incident_input.py`
  - Parses plain user input plus optional structured logs.
  - Detects `error.log`, `application.log`, and `system.log` from fenced code blocks or section headers.

- `investigation_tools.py`
  - Hosts deterministic tools for the Investigation Agent.
  - Current tool: `log_input_summary`, which summarizes parsed logs, severity counts, timestamps, and repeated error signatures.

- `evidence_store.py`
  - Per-run in-memory evidence registry.
  - Deduplicates similar evidence, tracks source agents/sources, calculates confidence, scores evidence, and emits evidence lifecycle events when an `EventStore` is attached.

- `evidence_reasoning.py`
  - Evidence confidence and ranking helpers.
  - Tool/log-backed evidence and multi-agent agreement increase score; validation needs and hypotheses reduce score.

- `event_store.py`
  - Per-run in-memory lifecycle event registry.
  - Supports event lookup by evidence id.

- `decision_lifecycle.py`
  - Emits `DECISION_CREATED` events for decision payloads.

- `council.py`
  - `stage1_collect_responses()`: runs role-based specialist agents.
  - `stage2_judge_deliberation()`: evaluates evidence and produces a Judge scorecard/verdict.
  - `stage3_synthesize_final()`: asks the chairman model for the final decision memo.
  - `run_full_council()`: complete non-streaming orchestration.
  - `build_council_metadata()`: returns workflow, incident summary, tool runs, agent messages/events, evidence, rankings, role assignments, and judge metadata.

- `storage.py`
  - Stores JSON conversations under `data/conversations/`.
  - Persisted assistant messages include `stage1`, `stage2`, `stage3`, and `metadata`.
  - Reuses the newest empty conversation to avoid duplicate empty threads.

- `main.py`
  - FastAPI entrypoint.
  - CORS is enabled for local frontend origins.
  - `POST /api/conversations/{id}/message` returns full stage data plus metadata.
  - `POST /api/conversations/{id}/message/stream` streams Server-Sent Events for stage progress and agent status.

### Frontend (`frontend/src/`)

- `App.jsx`
  - Owns conversation state, selection, streaming message updates, visibility refresh, and transient loading state.

- `api.js`
  - Fetch client for conversation APIs.
  - Parses Server-Sent Events from the streaming message endpoint.

- `components/ChatInterface.jsx`
  - Multiline textarea.
  - `Enter` sends, `Shift+Enter` inserts newline.
  - Shows recent agent status events while streaming.

- `components/Stage1.jsx`
  - Displays specialist agent responses in tabs.

- `components/Stage2.jsx`
  - Displays the Judge Agent assessment.
  - Shows scorecard, supporting evidence, remaining risks, and recommended actions when present.

- `components/Stage3.jsx`
  - Shows the final decision memo from the chairman model.

- Styling
  - Light theme.
  - Primary color is `#4a90e2`.
  - Markdown content should stay wrapped in `.markdown-content` for readable spacing.

## Important Invariants

- Preserve role separation: Analysis can hypothesize, Critic challenges weak reasoning, Investigator collects observable evidence and should not promote hypotheses or recommend fixes.
- Preserve evidence typing: `FACT`, `HYPOTHESIS`, `CORRELATION`, `RECOMMENDATION`, and `UNKNOWN`.
- Do not promote a hypothesis to `root_cause` unless confirmed evidence supports it.
- Keep Judge output structured enough for scorecards, evidence summaries, rankings, and final synthesis.
- Use relative backend imports such as `from .config import ...`.
- Run backend from the project root, not from inside `backend/`.
- Keep backend/frontend port expectations aligned:
  - backend: `8001`
  - frontend: `5173`

## Common Gotchas

- Import errors usually come from running the backend from the wrong directory, using non-relative imports, or using Python-version-specific typing imports without `typing_extensions`.
- CORS issues usually mean frontend origin and `main.py` configuration no longer match.
- Provider failures should degrade gracefully; a single model failure should not crash the whole council run.
- Prompt JSON parsing is best-effort; if you touch prompts, verify the normalization helpers still tolerate real model output.
- Streaming UI expects specific event types from `main.py`; update `frontend/src/App.jsx` and `frontend/README.md` if events change.

## Commit Message Rules

- Use Conventional Commits format: `<type>: <简体中文描述>`.
- Prefer these types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `style`, `build`, `ci`, `perf`.
- Write the subject in concise Simplified Chinese unless a proper noun, command, API name, or package name must stay in English.
- Keep the subject focused on the user-visible or technical change, for example `feat: 添加 Agent Council 应用实现` or `docs: 更新项目架构说明`.
- Do not use plain English sentence-style commit titles such as `Add Agent Council app implementation`.
- Before committing, confirm secrets, local runtime data, virtual environments, dependency folders, and build outputs are not staged.

## Dev Commands

Install dependencies:

```bash
uv sync
cd frontend
npm install
```

Run locally:

```bash
uv run python -m backend.main
cd frontend
npm run dev
```

Useful checks:

```bash
uv run python -m unittest discover tests
uv run python -c "from backend.main import app; print('app import ok')"
cd frontend
npm run build
npm run lint
```

## Change Checklist

Before finishing a task:

1. Confirm how the relevant stage currently works before editing.
2. Keep changes scoped to the requested behavior.
3. Run an appropriate check when code changes are made:
   - backend changes: targeted Python run or existing tests if available
   - frontend changes: `npm run build` or another relevant project check
   - docs-only changes: inspect rendered-sensitive Markdown structure and mention that no runtime check was needed
4. Mention what you changed, what you verified, and anything you did not verify.

## When Updating This File

- Keep it practical and execution-oriented.
- Prefer durable implementation facts over aspirational process language.
- Update this file when architecture, commands, ports, storage shape, event contracts, provider configuration, or stage behavior changes.
