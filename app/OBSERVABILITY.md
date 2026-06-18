# Observability

Execution-trace completeness, real-time progress accuracy, and failure diagnosis. The per-run
observability hub is `engine/monitor.py` (`RunMonitor`), created per task and shared with the scheduler.

## Execution trace (agent, action, input, output, duration, tokens per step)

`TraceEntry` (`engine/monitor.py`) carries exactly the required fields plus timestamps:
`step_id, agent, action, status, input, output, execution_time_ms, tokens_used, started_at,
completed_at`. It is built for **every finished step** in `record_result` → `_entry`, capturing the
step's `input` and the agent's full `output` (a research step shows its findings, a code step its
code). The same list is served live at `GET /tasks/{id}` and embedded in the final `/result` envelope.

## Real-time progress tracking

The monitor lives **off** the checkpointed graph state — in `engine/runs.py::RunRegistry`, keyed by
`task_id` — so the API reads live state while the graph is still running, not only at the end.
`progress()` returns `{total_steps, completed_steps, current_step}`, each sourced to stay accurate:

- `total_steps` from the **live plan** (grows correctly after a re-plan),
- `completed_steps` counts only `COMPLETED` status (failed/skipped never inflate it),
- `current_step` is set in `start_step` the instant a step launches.

Updates are synchronous store mutations under single-threaded asyncio (no lock, no races), and every
change stamps `updated_at`. An SSE endpoint `GET /tasks/{id}/stream` pushes a snapshot whenever
progress changes.

## Diagnosing a failed step

Three complementary layers:

1. **Trace** — agents never raise; a failure becomes `AgentResult(status="failed",
   output={"error": ...})`, so the failed step is recorded (not lost) with its agent/action, the
   `input` it received, and the error — exactly which step failed and why.
2. **Logs** — `_RunLogAdapter` prefixes every line with `[task=<id>]`, giving an ordered causal
   timeline per run: plan attached → step start → completed (tokens/ms) → **failed (error)** →
   skipped/cancelled → structural failure → re-plan → final result. (This is how the live Groq
   rate-limit was diagnosed: `step s3 failed: 429 rate_limit_exceeded` → `structural failure at s3 ->
   re-plan requested` → `cancelled 1 step(s): s5`.)
3. **Result + classifier** — `classify_failure` records `failed_step_id`/`failure_error`, and
   `skip_cascade`/`transitive_dependents` compute the blast radius (which downstream steps were lost).
   The final envelope surfaces `failed_steps`, `skipped_steps`, and a calibrated `confidence`.

## Next step — connect an observability platform

This is structured stdlib logging plus an in-memory trace in a single process — not distributed
tracing or a metrics backend. Connecting a dedicated LLM-observability app such as **Langfuse** (or
OpenTelemetry export) would be a good addition: per-step spans, token/cost dashboards, latency
percentiles, and trace search across runs, without the monitor having to own all of that itself.
