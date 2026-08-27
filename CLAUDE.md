# CLAUDE.md - Technical Notes for LLM Council

This file contains technical details, architectural decisions, and important implementation notes for future development sessions.

## Project Overview

LLM Council is now a role-based Agent Council for incident analysis. Multiple configured LLMs run as specialist agents, their evidence is normalized into a shared evidence model, a Judge Agent evaluates the evidence, and a chairman model produces the final decision memo.

The current implementation is not the original anonymous peer-ranking flow. Stage 2 is Judge deliberation over structured evidence.

## Runtime Flow

```text
User incident/question/logs
    |
    v
Incident input parser
    |
    v
Stage 1: Specialist agents in parallel
    - Analysis Agent
    - Critic Agent
    - Investigation Agent
      - deterministic tools such as log_input_summary
    |
    v
EvidenceStore
    - normalize evidence
    - deduplicate similar evidence
    - track source agents and sources
    - calculate confidence and score
    |
    v
Stage 2: Judge Agent
    - confirmed evidence
    - unverified hypotheses
    - scorecard
    - gaps and next actions
    |
    v
Stage 3: Final Decision
    - concise Simplified Chinese Markdown decision memo
```

The streaming endpoint emits Server-Sent Events throughout this flow so the frontend can show progress before the full response is saved.

## Backend Structure (`backend/`)

### `config.py`

- Defines `ModelConfig`, `MODEL_REGISTRY`, `COUNCIL_MODELS`, `CHAIRMAN_MODEL`, and `TITLE_MODEL`.
- Reads direct provider keys:
  - `OPENAI_API_KEY`
  - `DEEPSEEK_API_KEY`
  - `GEMINI_API_KEY`
  - `KIMI_API_KEY`
  - `QWEN_API_KEY`
- Supports aliases:
  - `OPENROUTER_API_KEY` as a backward-compatible fallback for OpenAI.
  - `MOONSHOT_API_KEY` as a fallback for Kimi.
  - `DASHSCOPE_API_KEY` as a fallback for Qwen.
- Backend runs on port `8001`.

### `openrouter.py`

The filename is legacy. The module is now the provider client layer.

- `resolve_model()`: resolves a registry key, model id, or `ModelConfig`.
- `query_model()`: routes a request to OpenAI, DeepSeek, Gemini, Kimi, or Qwen.
- `query_models_parallel()`: runs multiple model requests with graceful degradation.

Provider failures return `None` and should not crash the whole council flow unless every required model fails.

### `agent_models.py`

Shared structured models:

- `Evidence`
- `AgentEvent`
- `AgentMessage`
- `AgentResult`
- typed payload dictionaries

This module normalizes confidence, source traces, ids, evidence types, and backward-compatible aliases used by the frontend/tests.

### `incident_input.py`

Parses user input into `IncidentInput`.

Supported log names:

- `error.log`
- `application.log`
- `system.log`

Logs can be provided as fenced blocks, section headers, or structured request payloads.

### `investigation_tools.py`

Contains deterministic tools available to the Investigation Agent.

Current tool:

- `log_input_summary`: summarizes parsed log inputs, severity counts, timestamps, and repeated failure signatures.

### `evidence_store.py`

Per-run in-memory evidence registry.

Responsibilities:

- normalize evidence
- deduplicate similar evidence
- merge source agents and sources
- calculate bounded confidence
- score evidence
- emit `EVIDENCE_CREATED` and `EVIDENCE_MERGED` lifecycle events when an `EventStore` is attached

### `evidence_reasoning.py`

Evidence scoring and ranking helpers.

Important scoring tendencies:

- tool evidence is strongest
- log evidence is strong
- evidence supported by multiple agents gets a boost
- `need_validation` lowers confidence
- `HYPOTHESIS` evidence is penalized for final ranking

### `event_store.py`

Small in-memory lifecycle event registry for one council run.

### `decision_lifecycle.py`

Creates decision lifecycle events, currently `DECISION_CREATED`.

### `council.py`

Core orchestration.

- `stage1_collect_responses()`: runs role-based specialists in parallel.
- `stage2_judge_deliberation()`: builds/ranks evidence and asks the Judge Agent for a structured assessment.
- `stage3_synthesize_final()`: asks the chairman model for the final Markdown decision.
- `run_full_council()`: complete non-streaming orchestration.
- `build_council_metadata()`: builds workflow, incident, tool, message, event, evidence, ranking, role, and judge metadata.

### `storage.py`

JSON-based conversation storage in `data/conversations/`.

Assistant messages persist:

- `stage1`
- `stage2`
- `stage3`
- `metadata`

The storage layer also reuses the newest empty conversation so repeated "new conversation" actions do not create many empty JSON files.

### `main.py`

FastAPI app with CORS for local frontend origins.

Endpoints:

- `GET /`
- `GET /api/conversations`
- `POST /api/conversations`
- `GET /api/conversations/{conversation_id}`
- `POST /api/conversations/{conversation_id}/message`
- `POST /api/conversations/{conversation_id}/message/stream`

Streaming event types used by the frontend:

- `stage1_start`
- `agent_status`
- `stage1_complete`
- `stage2_start`
- `stage2_complete`
- `stage3_start`
- `stage3_complete`
- `title_complete`
- `complete`
- `error`

## Frontend Structure (`frontend/src/`)

### `App.jsx`

Owns:

- conversation list and current conversation state
- new conversation creation/reuse
- optimistic user/assistant messages
- streaming updates from the SSE endpoint
- title refresh and visibility refresh

### `api.js`

Fetch client and SSE parser. The backend base URL is currently:

```js
const API_BASE = 'http://localhost:8001';
```

### `components/ChatInterface.jsx`

- Renders user and assistant messages.
- Displays recent agent status events while streaming.
- Uses `Enter` to send and `Shift+Enter` for newline.
- Currently shows the input form only for empty conversations.

### `components/Stage1.jsx`

Tabbed specialist output display.

### `components/Stage2.jsx`

Judge Agent display, including:

- raw Judge Markdown response
- scorecard
- supporting evidence
- remaining risks
- recommended actions

### `components/Stage3.jsx`

Final decision memo display.

## Important Invariants

- Stage 1 specialists have different responsibilities:
  - Analysis Agent may propose hypotheses but must mark validation needs.
  - Critic Agent challenges assumptions and highlights missing evidence.
  - Investigation Agent collects observable facts and tool results; it should not output final root causes or fixes.
- Preserve evidence type separation: `FACT`, `HYPOTHESIS`, `CORRELATION`, `RECOMMENDATION`, `UNKNOWN`.
- Judge should not promote a hypothesis to `root_cause` unless supported by confirmed evidence.
- Keep provider failures graceful.
- Keep backend imports relative, for example `from .config import ...`.
- Run backend from the project root.
- Keep backend port `8001` and frontend port `5173` aligned across docs, CORS, and `frontend/src/api.js`.
- Keep Markdown output wrapped in `.markdown-content` in React components.

## Common Gotchas

1. **Python typing compatibility**
   - Project supports Python 3.10.
   - Use `typing_extensions` for typing features that are not available in Python 3.10's standard `typing`.

2. **Module import errors**
   - Run `uv run python -m backend.main` or `uv run uvicorn backend.main:app --reload --port 8001` from the project root.

3. **Provider naming**
   - `backend/openrouter.py` is a legacy filename. Do not assume all calls go through OpenRouter.

4. **Prompt JSON parsing**
   - Models may return fenced JSON, plain JSON, or extra prose.
   - Keep normalization helpers tolerant when modifying prompts.

5. **Streaming contract**
   - If backend event names change, update `frontend/src/App.jsx`, `frontend/src/api.js` if needed, and `frontend/README.md`.

6. **Storage shape**
   - Assistant messages persist `metadata`; do not assume metadata is only transient UI state.

## Commit Message Rules

Use Conventional Commits with a concise Simplified Chinese subject:

```text
<type>: <简体中文描述>
```

Preferred types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `style`, `build`, `ci`, `perf`.

Examples:

```text
feat: 添加 Agent Council 应用实现
docs: 更新项目架构说明
chore: 整理本地运行数据忽略规则
```

Avoid plain English sentence-style titles such as `Add Agent Council app implementation`. Before committing, verify `.env`, virtual environments, local runtime data, dependency folders, and build outputs are not staged.

## Development Commands

Install:

```bash
uv sync
cd frontend
npm install
```

Run:

```bash
uv run python -m backend.main
cd frontend
npm run dev
```

Backend checks:

```bash
uv run python -m unittest discover tests
uv run python -c "from backend.main import app; print('app import ok')"
```

Frontend checks:

```bash
cd frontend
npm run build
npm run lint
```

## Future Enhancement Ideas

- UI model/provider configuration.
- Multi-turn incident follow-up input after the first council run.
- Export conversations to Markdown/PDF.
- More deterministic investigation tools.
- Evidence graph/timeline visualization.
- Provider health diagnostics in the UI.
