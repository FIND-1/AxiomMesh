# AxiomMesh

AxiomMesh 是一个面向 **AI SRE / Incident Response** 的多模型 Agent Council。

它不会让多个模型简单“投票”，而是让不同模型分别承担分析、质疑、调查和裁决职责，再基于统一 Evidence 生成最终结论。

## 工作流程

```text id="6iiq7j"
Incident / Logs
      ↓
Stage 1 Specialists
Analysis / Critic / Investigation
      ↓
EvidenceStore
Merge / Rank / Validate
      ↓
Stage 2 Judge
Confirm Evidence / Compare Hypotheses
      ↓
Stage 3 Final Decision
Conclusion / Evidence / Actions
```

当前默认 Council：

```text id="5tv5j5"
DeepSeek
Kimi
Gemini
Qwen
```

同时已支持 OpenAI Provider。

## Features

* 多模型 Specialist 并发分析
* Analysis / Critic / Investigation 角色分工
* Evidence 创建、合并、排序与追踪
* Judge 多候选 fallback
* Final Decision 降级生成
* 单个模型失败不会拖垮整个 Council
* DeepSeek / Gemini / Kimi / Qwen / OpenAI Provider
* 统一 LLM Contract / Retry / Telemetry
* FastAPI + SSE 流式输出
* React + Vite 前端
* 本地 Conversation 持久化
* Evidence → Judge → Decision 事件链

## Architecture

```text id="0vclys"
backend/
├── council.py
├── council_runtime/
│   ├── specialists.py
│   ├── judge.py
│   ├── final.py
│   ├── evidence.py
│   ├── events.py
│   └── normalization.py
│
├── llm/
│   ├── contracts.py
│   ├── gateway.py
│   ├── registry.py
│   ├── retry.py
│   ├── telemetry.py
│   └── providers/
│
├── evidence_store.py
├── evidence_reasoning.py
├── event_store.py
└── main.py
```

`council.py` 负责 Stage orchestration，`council_runtime` 保存各阶段的纯运行时逻辑；Provider 调用统一通过 `backend.llm` 管理。

## Reliability

AxiomMesh 支持局部故障降级：

```text id="cwqjrv"
Specialist fails
→ other Specialists continue

Judge fails
→ try another candidate

All Judges fail
→ local Judge fallback

Final model fails
→ build Final from Judge result

Title fails
→ local title fallback
```

目标是让单个 Provider 或单个 Stage 的失败尽量只降低结果质量，而不是终止整个请求。

## Getting Started

### Backend

```bash id="h5t9xk"
git clone https://github.com/FIND-1/AxiomMesh.git
cd AxiomMesh

uv sync
```

创建 `.env`：

```env id="ugje9i"
DEEPSEEK_API_KEY=
GEMINI_API_KEY=
KIMI_API_KEY=
QWEN_API_KEY=
OPENAI_API_KEY=
```

启动：

```bash id="f9pfo0"
uv run python -m backend.main
```

默认：

```text id="xqjrqw"
http://localhost:8001
```

### Frontend

```bash id="wdgcv0"
cd frontend
npm install
npm run dev
```

默认：

```text id="vkhm56"
http://localhost:5173
```

## Tests

```bash id="pmsjgk"
uv run python -m unittest discover -s tests
```

当前基线：

```text id="y3vrz8"
122 tests passed
commit: 559ea0e
```

测试覆盖：

* LLM Contract / Provider / Retry
* Council Stage orchestration
* Judge / Final fallback
* Failure isolation
* Evidence lifecycle
* Event Chain
* Import compatibility

## Project Direction

AxiomMesh 当前主要用于 AI SRE / Incident Response，后续希望继续探索：

```text id="sbyb5c"
Multi-Agent
+
Structured Evidence
+
Multi-Model Reliability
+
Auditable Decision Making
```

## License

Apache License 2.0.
