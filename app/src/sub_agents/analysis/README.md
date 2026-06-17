# Analysis Agent

Autonomous sub-agent for **analyzing data**, **comparing options**, and **identifying patterns** over
upstream structured data. Built as a LangChain `create_agent` (compiles to a LangGraph) ReAct loop:
the model *thinks*, *computes* exact figures, inspects the result, and reasons again.

## Why this shape
LLMs are unreliable at arithmetic, so precise math is offloaded to a deterministic tool rather than
done "in the model's head". The agent has two tools:

- **`think`** — private reasoning scratchpad (interleaved thinking; not user-visible).
- **`compute`** — deterministic statistics over the dataset **or** raw arithmetic.
  Evaluation is **regex-guarded arithmetic only** (digits/operators/parens), never a code
  interpreter — a deliberate security choice (no RCE). Modes:
  - *expression*: e.g. `(2.25 / (2 * 0.25)) * (1 / 7)`
  - *metrics*: `count`, `count_where`, `sum`, `average`, `min`, `max`, `distinct`, `group_by` over
    dot-notation fields, plus a `formula` combining results via `m[0], m[1], …`.

## Flow
```
run_analysis_agent(instruction, action, data, sources, step_id)
  → build_analysis_agent(model, summarizer, cfg, price)        # create_agent → LangGraph
  → model ⇄ {think, compute}                                   # bounded ReAct loop
       middleware: ModelCallLimit + compaction(before) + token/cost(after)
  → AnalysisSummary{content, findings[], confidence}           # final structured output
  → AgentResult                                                # uniform platform output
```

## Output
`AgentResult` with:
- `output = { content, findings: list[str], confidence: 0-1, sources: list[str] }`
- `status`: `completed` | `completed_degraded` (confidence < threshold) | `failed`
- `tokens_used`, `execution_time_ms`, `est_cost_usd`, `actual_cost_usd`

The run is wrapped in `try/except` and never raises; failures return `status="failed"` with an error.

## Files
- `agent.py` — `build_analysis_agent`, async `run_analysis_agent`, summarization, result assembly.
- `schemas.py` — `AnalysisContext` (runtime state, mutated in-loop), `AnalysisSummary`, `Action`,
  `CAPABILITIES`.
- `prompts.py` — per-action system prompt and `<instruction>`/`<data>` fencing (prompt-injection
  defense).

Shared building blocks live in `app/src/general_utils/`: `tools.py` (`think`, `compute`),
`middleware.py` (compaction + token/cost capture), `agent_base.py` (`AgentResult`).

## Config (`app/config.yaml` → `analysis_agent`)
```yaml
analysis_agent:
  model: { id: gemini-2.5-flash, temperature: 0.2 }
  summarizer_model: { id: gemini-2.5-flash-lite, temperature: 0.0 }
  recursion_limit: 10            # caps the ReAct loop
  max_compute_calls: 6           # caps compute tool calls
  confidence_threshold: 0.5      # below → completed_degraded
  summarization: { trigger_messages: 16, keep_recent: 6 }
```

## Usage
```python
from app.src.sub_agents.analysis.agent import run_analysis_agent

result = await run_analysis_agent(
    "What percentage of tickets were resolved?",
    action="analyze",
    data=[{"status": "resolved"}, {"status": "open"}, {"status": "resolved"}],
    step_id="step-1",
)
```

## Tests & eval
- Unit (offline, mocked Gemini): `python tests/sub_agents/test_analysis_agent.py`
- Eval (live Gemini, LLM-judge): `uv run python evals/judges/analysis_judge.py --all`
  (or `--id A-01`); reports land in `evals/reports/`.
