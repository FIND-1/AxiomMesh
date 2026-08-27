# LLM Council Frontend

React + Vite frontend for the local Agent Council app.

## Responsibilities

- Lists and opens stored conversations.
- Creates or reuses an empty conversation.
- Sends the first user message to the backend streaming endpoint.
- Renders progressive council stages:
  - Stage 1: specialist agent analysis tabs.
  - Stage 2: Judge Agent assessment, scorecard, risks, and actions.
  - Stage 3: final decision memo.
- Displays recent `agent_status` events while a stream is running.

## Development

Install dependencies from `frontend/`:

```bash
npm install
```

Run the dev server:

```bash
npm run dev
```

Build:

```bash
npm run build
```

Lint:

```bash
npm run lint
```

## Backend Contract

The API client in `src/api.js` expects the backend at:

```text
http://localhost:8001
```

The main UI path uses:

```http
POST /api/conversations/{conversation_id}/message/stream
```

Expected stream event types:

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

Conversation messages persisted by the backend contain user messages plus assistant messages shaped around `stage1`, `stage2`, `stage3`, and `metadata`.

## Component Map

```text
src/App.jsx                       Conversation state, SSE orchestration, refresh behavior
src/api.js                        Fetch client and SSE parser
src/components/Sidebar.jsx        Conversation list and new conversation button
src/components/ChatInterface.jsx  Message rendering and input form
src/components/Stage1.jsx         Specialist result tabs
src/components/Stage2.jsx         Judge result display
src/components/Stage3.jsx         Final decision display
```

## Notes

- `Enter` sends, `Shift+Enter` inserts a newline.
- Markdown output should stay wrapped in `.markdown-content`.
- Keep the frontend port at Vite's default `5173` unless backend CORS and docs are updated together.
