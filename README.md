# Agent Orchestration Platform

A platform that turns a single natural-language **goal** into coordinated multi-agent work: a planner
decomposes the goal into a dependency DAG, four specialist agents execute the steps with continuous
concurrency, a monitor tracks every step live, and a synthesizer combines the outputs into one
answer with provenance.

```
                  ┌─────────────────── LangGraph outer loop ───────────────────┐
                  │                                                             │
Goal ─▶  plan ─▶ execute ─▶ evaluate ──(structural failure: re-plan?)── yes ────┘
       (LLM)      │       (LLM decider)            │
                  ▼                                no
              dispatch                            ▼
                  │                          synthesize ─▶ API
   ┌─────────┬────┴────┬──────────┐         (LLM: combine    (status / result /
   ▼         ▼         ▼          ▼            + provenance)   trace, any time)
 research  analysis   code      writing
 web +     compute,  generate,  polished     Monitor: status, trace, tokens, cost
 cites     compare,  explain,   prose +      (readable while the run is still live)
           patterns  debug      format

      the four specialist agents — the planner may only route to an (agent, action)
      pair in the registry; every step runs as one of these four.
```

- **Outer loop = LangGraph** — a fixed state machine (`plan → execute → evaluate → synthesize`) with
  one LLM-routed conditional edge (bounded re-plan) and a `MemorySaver` checkpointer.
- **Inner `execute` = plain `asyncio`** — the runtime step-DAG runs with *continuous* concurrency (a
  fast step's successor starts before a slow sibling finishes), capped by a semaphore that doubles as
  the active provider's free-tier rate limiter. A structural failure preemptively cancels in-flight
  steps and
  routes straight to the re-plan decider.

The only LLM calls are the **planner**, the **re-plan decider**, and the **synthesizer**; step
execution is ordinary observable code. See [DECISIONS.md](DECISIONS.md) for the design rationale and
[AI_USAGE.md](AI_USAGE.md) for how the AI was driven. For the cross-cutting concerns, see
[app/ARCHITECTURE.md](app/ARCHITECTURE.md) (the full graph diagram — outer loop, sub-agents, middleware),
[app/SECURITY.md](app/SECURITY.md) (untrusted input, secrets, schema validation, prompt-injection) and
[app/OBSERVABILITY.md](app/OBSERVABILITY.md) (execution trace, progress tracking, failure diagnosis).

## Architecture

```
app/
  main.py                 # entry point: starts the FastAPI server (uvicorn)
  cli.py                  # run one goal locally for debugging
  config.yaml             # runtime parameters (active provider, per-agent model tier, bounds)
  llm_config.yml          # per-provider model ids (big/small tiers) + pricing (Groq, Gemini)
  src/
    api/                  # FastAPI app + request/response models (6 endpoints)
    engine/               # orchestration layer
      graph.py            # LangGraph outer loop (build + run_task)
      nodes.py            # plan/execute/evaluate/synthesize nodes + routers
      planner.py          # goal -> validated plan (one reasoning-first structured call)
      validation.py       # schema -> Kahn cycle check -> referential checks -> parallel_groups
      scheduler.py        # plain-async inner DAG executor (continuous concurrency)
      dispatch.py         # routes a step to the right agent, injects trimmed upstream context
      monitor.py          # observability: trace, status, totals, failure classification
      evaluation.py       # LLM re-plan decider + bounded merge protocol
      synthesizer.py      # results -> final answer with provenance
      registry.py         # the post-LLM allowlist + capability catalog
      runs.py             # in-process registry of live monitors
    schemas/              # plan, run-state, config (pydantic)
    sub_agents/           # the four specialist agents (each a self-contained package)
      research/           # grounded web research — Tavily search + cite sources
      analysis/           # quantitative analysis; tools.py = deterministic compute tool
      code/               # generate/explain/debug; nodes.py + routing.py (two-tier validation)
      writing/            # polished prose with output-format control
    general_utils/        # shared agent contract, model init, retry; tokens.py = cost estimation
tests/                    # offline tests (fake agents/models) + run_all_tests.py
Dockerfile, docker-compose.yml
```

## Run

The whole platform runs as one container via Docker Compose — no local Python needed.

> **API keys matter — set them in `app/.env` before running:**
> - **LLM provider key (required)** — the active `provider:` in `app/config.yaml` decides which key
>   every LLM call (planner, the four agents, synthesizer) needs: `groq` (the default) uses
>   **`GROQ_API_KEY`**, `gemini` uses **`GOOGLE_API_KEY`**. Groq is the default because its free tier
>   is far larger than Gemini's (~14,400 req/day vs Gemini's 20/day), so multi-step runs complete
>   without hitting a daily cap. Without the active provider's key the server starts but any
>   `POST /tasks` fails.
> - **`TAVILY_API_KEY` (strongly recommended)** — the Research Agent's live web search. Without it
>   the research step has no grounded sources, so research-driven goals degrade badly. Optional only
>   if you never run a research step.

### Quickstart on any machine (Docker, clean clone)

Run every command from the **repository root** (the folder that contains `docker-compose.yml`):

```bash
# 1. clone
git clone https://github.com/YishaiDev/Agent-Orchestration-Platform-Assignment
cd Agent-Orchestration-Platform-Assignment

# 2. create your secrets file from the template
cp app/.env.example app/.env          # Windows: copy app\.env.example app\.env

# 3. edit app/.env -> set GROQ_API_KEY (default provider; or GOOGLE_API_KEY if provider: gemini);
#    TAVILY_API_KEY (strongly recommended, research)

# 4. build and run — Compose does everything: builds the image, installs deps, starts the server
docker compose up --build             # legacy Docker: docker-compose up --build
```

The server is then live on `http://localhost:8000` (interactive docs at `/docs`). Compose reads
`app/.env` and injects the keys into the container's environment; the file is never baked into the
image (the Dockerfile copies only tracked source — `main.py`, `cli.py`, `config.yaml`,
`llm_config.yml`, `src/`, `evals/` — never `.env` or the local `.venv`).

> Both `docker compose` (v2, bundled with Docker) and `docker-compose` (v1) work. Run from the repo
> root, not from `app/`. If `app/.env` is missing, Compose stops with an error — do step 2 first.

### Verify the API works

With the server running, from a second terminal. The first two calls need **no** API key:

```bash
curl http://localhost:8000/health
# -> {"status":"ok"}

curl http://localhost:8000/agents
# -> [{"name":"research",...}, {"name":"analysis",...}, {"name":"code",...}, {"name":"writing",...}]
```

Submit a real multi-step task (this **uses your active provider key**, `GROQ_API_KEY` by default),
then poll and fetch the result:

```bash
# submit -> {"task_id": "<id>", "status": "planning"}
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"goal": "Explain what a topological sort is, then write a 3-sentence beginner summary"}'

# poll until "status":"completed" (watch progress + trace grow)
curl http://localhost:8000/tasks/<id>

# synthesized answer with provenance + token/cost totals (409 until ready)
curl http://localhost:8000/tasks/<id>/result

# optional: cooperative cancel -> {"status":"cancelled","completed_steps":[...]}
curl -X POST http://localhost:8000/tasks/<id>/cancel
```

`constraints` and `output_format` are optional. The body accepts the assignment's full **Task
Format** — `constraints` may be a JSON object (normalized to planner text) or a plain string:

```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"goal": "Write a comparison blog post about Python vs JavaScript for beginners",
       "constraints": {"max_words": 1500, "tone": "friendly", "include_code_examples": true},
       "output_format": "markdown"}'
```

Tasks complete in a steady stream rather than instantly: concurrency is capped at `3` to respect the
active provider's free-tier rate limits (generous on Groq; tight on Gemini).

### Run locally without Docker (uv)

```bash
cd app
cp .env.example .env                   # Windows: copy .env.example .env   (then set GROQ_API_KEY)
uv sync
uv run python main.py                  # serves on http://localhost:8000
```

### Run one goal from the CLI (no server)

```bash
cd app                                 # needs app/.env with GROQ_API_KEY (the default provider's key)
uv run python cli.py "Compare Postgres and MySQL for analytics and write a short brief"
```

## Test

All tests run **offline** with fake agents and fake models (no provider quota, no network). Each test
file is independently runnable; `run_all_tests.py` runs the whole suite via pytest.

```powershell
cd app
uv run python ../tests/run_all_tests.py        # full suite
uv run pytest ../tests/engine -q                # just the engine + API tests
uv run python ../tests/engine/test_api.py       # one file, standalone
```

The six spec-named scenarios are covered explicitly: task **submission**, **plan generation**,
**sequential execution** (dependency chain, outputs passed downstream), **parallel execution**,
**error handling**, and **result synthesis** — plus continuous-concurrency timing, preemptive-cancel
timing, bounded re-plan correctness, cooperative cancel, and a prompt-injection probe.

## Agents and capabilities

The planner may only emit steps whose `(agent, action)` pair is in the registry
(`app/src/engine/registry.py`); anything else is rejected before execution. `GET /agents` returns
this catalog live.

| Agent | Capabilities (actions) | What it does |
| --- | --- | --- |
| `research` | `research` | Grounded web research: searches, summarizes, and cites sources. |
| `analysis` | `analyze`, `compare`, `identify_patterns` | Quantitative analysis and comparison over data. |
| `code` | `generate`, `explain`, `debug` | Generates/explains/debugs code (no execution). |
| `writing` | `write` | Synthesizes polished prose from source material with format control. |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/tasks` | Submit a goal; returns a `task_id` and starts the run in the background. |
| `GET` | `/tasks/{id}` | Live status, progress, totals, and the per-step execution trace. |
| `GET` | `/tasks/{id}/result` | The synthesized final result with provenance (`409` until ready). |
| `GET` | `/tasks/{id}/plan` | The validated execution plan: steps, dependencies, and parallel groups (`409` until planned). |
| `GET` | `/tasks/{id}/stream` | Server-Sent Events stream of live progress until the task reaches a terminal state. |
| `POST` | `/tasks/{id}/cancel` | Cooperative cancel; returns the steps completed so far. |
| `GET` | `/agents` | The registered agents, their capabilities, and status. |
| `GET` | `/health` | Liveness probe. |

## Example: task → plan → result

Submit a goal:

```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"goal": "Research electric vehicle adoption trends and write a one-page brief"}'
# -> {"task_id": "a1b2...", "status": "planning"}
```

The planner decomposes it into a DAG (research → analyze → write):

```json
{
  "task_id": "a1b2...",
  "reasoning": "Gather sources first, analyze the trend data, then write the brief from both.",
  "steps": [
    {"id": "s1", "agent": "research", "action": "research",
     "input": {"subtopic": "global EV adoption trends and drivers"}, "dependencies": []},
    {"id": "s2", "agent": "analysis", "action": "analyze",
     "input": {"instruction": "identify the strongest growth signals"}, "dependencies": ["s1"]},
    {"id": "s3", "agent": "writing", "action": "write",
     "input": {"instruction": "one-page brief", "output_format": "markdown"},
     "dependencies": ["s1", "s2"]}
  ],
  "parallel_groups": [["s1"], ["s2"], ["s3"]]
}
```

Fetch the synthesized result once the run completes. The shape matches the assignment's **Final
Result Format** — a nested `result` (`content`/`format`/`word_count`), the full `execution_trace`, and
run totals — plus additive fields (`provenance`, `confidence`, `failed_steps`, `skipped_steps`,
`total_cost_usd`):

```json
{
  "task_id": "a1b2...",
  "status": "completed",
  "result": { "content": "# EV adoption brief...", "format": "markdown", "word_count": 612 },
  "execution_trace": [
    { "step_id": "s1", "agent": "research", "action": "research", "status": "completed",
      "tokens_used": 5570, "execution_time_ms": 6541,
      "started_at": "2026-06-18T09:46:38Z", "completed_at": "2026-06-18T09:46:45Z",
      "input": { "subtopic": "..." }, "output": { "content": "...", "sources": ["..."], "confidence": 0.9 } }
  ],
  "total_tokens": 14595,
  "total_time_ms": 20247,
  "confidence": 0.95,
  "provenance": [ { "step_id": "s1", "agent": "research", "action": "research", "status": "completed",
                    "confidence": 0.9, "sources": ["..."] } ],
  "failed_steps": [],
  "skipped_steps": [],
  "total_cost_usd": 0.001987
}
```

### Reading the answer as formatted markdown

`result.content` is already markdown, but the `/result` endpoint (and the CLI) return it inside a JSON
**string**, so newlines show as `\n` and any typographic characters (e.g. `–`, `‑`) show as `\uXXXX`.
That is correct JSON — to *render* the answer, extract and decode that one field rather than copying
the raw blob:

```bash
# from the live API
curl -s http://localhost:8000/tasks/a1b2.../result | jq -r '.result.content' > answer.md

# from a saved CLI result file
jq -r '.result.content' result.json > answer.md
```

`jq -r` ("raw") unescapes the string — `\n` becomes real line breaks and `\uXXXX` becomes the actual
character — so `answer.md` renders cleanly. Equivalents without `jq`:

```bash
# Python
python -c "import json,sys; print(json.load(open('result.json'))['result']['content'])" > answer.md
```
```powershell
# PowerShell
(Get-Content result.json -Raw | ConvertFrom-Json).result.content | Out-File answer.md -Encoding utf8
```

> If you ever embed the full raw JSON inside a markdown code block, wrap it in a **four-backtick**
> fence (` ````json `) — the answer can itself contain triple-backtick ` ```python ` blocks, and a
> three-backtick outer fence would be closed early by them.

## Track progress

`GET /tasks/{id}` is readable **while the task is still running** — the API reads the live monitor, so
status moves `pending → planning → executing → completed`, `progress.completed_steps` climbs, and the
`execution_trace` grows one entry per finished step (`agent`, `action`, `input`, `output`, `status`,
`tokens_used`, `execution_time_ms`, `started_at`, `completed_at`):

```bash
curl http://localhost:8000/tasks/a1b2...
# -> { "status": "executing",
#      "progress": {"total_steps": 3, "completed_steps": 1, "current_step": "s2"},
#      "total_tokens": 420, "total_cost_usd": 0.001,
#      "execution_trace": [ {"step_id": "s1", "agent": "research", "action": "research",
#                            "status": "completed", "tokens_used": 412, "execution_time_ms": 6500, ...} ] }
```

## Configuration

Runtime parameters live in `app/config.yaml` (the active `provider:`, per-agent model **tier**
(`big`/`small`) + temperature, and bounds: `max_replans`, `concurrency`, `max_steps`,
`step_timeout_seconds`, `planner_max_attempts`, `context_char_budget`). Model ids and pricing for each
provider live in `app/llm_config.yml` — switch the whole platform between **Groq** and **Gemini** by
changing the single `provider:` line. Secrets are read from `app/.env` (never hardcoded). Concurrency
defaults to `3` to stay within free-tier rate limits (Groq's is generous; Gemini's allows only ~20
requests/day).

**Sequential vs parallel:** `orchestrator.bounds.concurrency: 1` runs the DAG fully sequentially (one
step at a time); higher values run independent steps in parallel. Dependency order holds at any value.

## Requirements Coverage

What's applied against the assignment's Technical Requirements:

| Requirement | Status | Notes |
| --- | --- | --- |
| Python 3.11+ | ✅ | `app/pyproject.toml` |
| Async web framework | ✅ | FastAPI |
| ≥4 agent types | ✅ | research, writing, analysis, code |
| Task planning / decomposition | ✅ | `engine/planner.py` → `engine/validation.py` |
| Dependency-based execution | ✅ | `engine/scheduler.py` (deps gate each launch) |
| Result synthesis | ✅ | `engine/synthesizer.py`, conflict-resolving + provenance |
| Containerized (docker-compose) | ✅ | `Dockerfile`, `docker-compose.yml` |
| ≥6 meaningful tests | ✅ | 125 passing: submission, plan generation, sequential & parallel execution, error handling, synthesis |
| Sequential execution | ✅ | `concurrency: 1` (same scheduler) |
| Parallel step execution | ✅ | continuous-concurrency scheduler |
| Progress tracking | ✅ | `GET /tasks/{id}` (live) |
| Execution traces | ✅ | per-step agent/action/io/duration/tokens |
| Token usage tracking | ✅ | per step + run totals |
| Dynamic re-planning on failure | ✅ | structural-failure preempt → re-plan decider |
| Agent capability matching | ✅ | `(agent, action)` registry allowlist |
| Cost estimation before execution | ✅ | `general_utils/cost.py::estimate_cost` → `est_cost_usd` |
| Streaming intermediate results | ✅ | SSE via `GET /tasks/{id}/stream` |
