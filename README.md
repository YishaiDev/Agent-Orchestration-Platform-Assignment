# Agent Orchestration Platform

A platform that turns a single natural-language **goal** into coordinated multi-agent work: a planner
decomposes the goal into a dependency DAG, four specialist agents execute the steps with continuous
concurrency, a monitor tracks every step live, and a synthesizer combines the outputs into one
answer with provenance.

```
                  ┌─────────────────── LangGraph outer loop ───────────────────┐
                  │                                                             │
Goal ─▶  plan ─▶ execute ─▶ evaluate ──(structural failure: re-plan?)── yes ────┘
       (LLM)   (run DAG)    (LLM decider)            │
                  │                                  no
                  ▼                                  ▼
              Monitor                           synthesize ─▶ API
        (status, trace,                        (LLM: combine +    (status / result /
         tokens, cost)                          provenance)        trace, any time)
```

- **Outer loop = LangGraph** — a fixed state machine (`plan → execute → evaluate → synthesize`) with
  one LLM-routed conditional edge (bounded re-plan) and a `MemorySaver` checkpointer.
- **Inner `execute` = plain `asyncio`** — the runtime step-DAG runs with *continuous* concurrency (a
  fast step's successor starts before a slow sibling finishes), capped by a semaphore that doubles as
  the Gemini free-tier rate limiter. A structural failure preemptively cancels in-flight steps and
  routes straight to the re-plan decider.

The only LLM calls are the **planner**, the **re-plan decider**, and the **synthesizer**; step
execution is ordinary observable code. See [DECISIONS.md](DECISIONS.md) for the design rationale and
[AI_USAGE.md](AI_USAGE.md) for how the AI was driven.

## Architecture

```
app/
  main.py                 # entry point: starts the FastAPI server (uvicorn)
  cli.py                  # run one goal locally for debugging
  config.yaml             # runtime parameters (models, bounds)
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
    sub_agents/           # the four specialist agents (research, analysis, code, writing)
    general_utils/        # shared agent contract, model init, retry, cost capture
tests/                    # offline tests (fake agents/models) + run_all_tests.py
Dockerfile, docker-compose.yml
```

## Run

The whole platform runs as one container via Docker Compose — no local Python needed. It requires a
Google Gemini API key (and, for the research agent's web search, an optional Tavily key).

### Quickstart on any machine (Docker, clean clone)

Run every command from the **repository root** (the folder that contains `docker-compose.yml`):

```bash
# 1. clone
git clone https://github.com/YishaiDev/Agent-Orchestration-Platform-Assignment
cd Agent-Orchestration-Platform-Assignment

# 2. create your secrets file from the template
cp app/.env.example app/.env          # Windows: copy app\.env.example app\.env

# 3. edit app/.env -> set GOOGLE_API_KEY (required); TAVILY_API_KEY is optional (research)

# 4. build and run — Compose does everything: builds the image, installs deps, starts the server
docker compose up --build             # legacy Docker: docker-compose up --build
```

The server is then live on `http://localhost:8000` (interactive docs at `/docs`). Compose reads
`app/.env` and injects the keys into the container's environment; the file is never baked into the
image (the Dockerfile copies only tracked source — `main.py`, `cli.py`, `config.yaml`, `src/`,
`evals/` — never `.env` or the local `.venv`).

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

Submit a real multi-step task (this **uses your GOOGLE_API_KEY**), then poll and fetch the result:

```bash
# submit -> {"task_id": "<id>", "status": "pending"}
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"goal": "Explain what a topological sort is, then write a 3-sentence beginner summary"}'

# poll until "status":"completed" (watch progress + trace grow)
curl http://localhost:8000/tasks/<id>

# synthesized answer with provenance + token/cost totals (409 until ready)
curl http://localhost:8000/tasks/<id>/result

# optional: cooperative cancel -> {"status":"cancelling","completed_steps":[...]}
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
Gemini free tier's 5 requests/minute.

### Run locally without Docker (uv)

```bash
cd app
cp .env.example .env                   # Windows: copy .env.example .env   (then set GOOGLE_API_KEY)
uv sync
uv run python main.py                  # serves on http://localhost:8000
```

### Run one goal from the CLI (no server)

```bash
cd app                                 # needs app/.env with GOOGLE_API_KEY
uv run python cli.py "Compare Postgres and MySQL for analytics and write a short brief"
```

## Test

All tests run **offline** with fake agents and fake models (no Gemini quota, no network). Each test
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
| `POST` | `/tasks/{id}/cancel` | Cooperative cancel; returns the steps completed so far. |
| `GET` | `/agents` | The registered agents, their capabilities, and status. |
| `GET` | `/health` | Liveness probe. |

## Example: task → plan → result

Submit a goal:

```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"goal": "Research electric vehicle adoption trends and write a one-page brief"}'
# -> {"task_id": "a1b2...", "status": "pending"}
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

Fetch the synthesized result once the run completes:

```bash
curl http://localhost:8000/tasks/a1b2.../result
# -> { "status": "completed", "content": "...", "confidence": 0.86,
#      "provenance": [...], "failed_steps": [], "skipped_steps": [],
#      "total_tokens": 1234, "total_cost_usd": 0.004, "total_time_ms": 8200 }
```

## Track progress

`GET /tasks/{id}` is readable **while the task is still running** — the API reads the live monitor, so
status moves `pending → planning → executing → completed`, `progress.completed_steps` climbs, and the
`trace` grows one entry per finished step (agent, action, input, output, status, duration, tokens):

```bash
curl http://localhost:8000/tasks/a1b2...
# -> { "status": "executing",
#      "progress": {"total_steps": 3, "completed_steps": 1, "current_step": "s2"},
#      "total_tokens": 420, "total_cost_usd": 0.001,
#      "trace": [ {"step_id": "s1", "agent": "research", "status": "completed", ...} ] }
```

## Configuration

Runtime parameters live in `app/config.yaml` (models, temperatures, and bounds: `max_replans`,
`concurrency`, `max_steps`, `step_timeout_seconds`, `planner_max_attempts`, `context_char_budget`).
Secrets are read from `app/.env` (never hardcoded). Concurrency defaults to `3` to stay within the
Gemini free tier's 5 requests/minute.
