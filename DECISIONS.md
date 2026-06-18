# Design Decisions

> **✅ Implemented** = code exists at the cited path. All four specialist agents and the full
> orchestration layer are built, unit-tested, and eval'd. Paths are under `app/src/` unless noted.

## Cross-cutting decisions

**Switchable LLM provider (Groq default / Gemini).** The platform runs on either provider, selected by
one line — `provider:` in `config.yaml`. Model ids + pricing for both live in `llm_config.yml`, keyed
by a `big`/`small` tier; each agent role references a tier, so switching provider never touches agent
code (`general_utils/llm.py::build_chat_model` picks the provider + key). **Driven by a real failure:**
Gemini's free tier (~20 req/day) 429'd multi-step runs, so Groq — far larger free tier — became the
default while Gemini stays a first-class option.

**Code Agent — verification without execution (two-tier gate).** The spec's Code Agent only generates /
explains / debugs (all text); nothing requires *running* code, and executing LLM-generated code from an
untrusted goal would be the project's largest attack surface. So the agent has **no sandbox, no
execution** — a bounded generate→judge→refine reflection graph producing a typed `CodeOutput`, gated per
language (`sub_agents/code/validation.py`):

- **Tier 1 — deterministic parser (ground truth):** Python via `ast.parse`; JavaScript via `tree-sitter`
  (checks `root_node.has_error` **and** an `is_missing` walk). A parse failure feeds a bounded correction
  loop (`max_syntax_retries`, default 4); after the budget it returns best-effort code with `parses:false`
  rather than failing. tree-sitter is error-recovering, so it's a deliberately coarser gate than `ast` —
  documented honestly.
- **Tier 2 — LLM critic (fallback only):** for languages with no parser (Ruby, Go …), a cheaper,
  independent reviewer (temp 0) returns `revise|return`, bounded by `max_review_retries`. Parser-backed
  languages never invoke it — ground-truth-first, no added cost.

The gate lands on `output["parses"]`, which the eval judge reads from the output (never recomputes, so
the two can't drift). (✅ `sub_agents/code/validation.py`, `agent.py`; tests in `test_code_agent.py`;
eval 5/5, including the JS Tier-1 case.)

---

## 1. Task Decomposition Strategy

**Approach chosen:** A dedicated planner emits a strict-JSON `ExecutionPlan` (the spec's shape: `steps[]`
with `id/agent/action/input/dependencies` + `parallel_groups`), **validated as a DAG before any
execution** — never run raw. One structured planning call + deterministic validation beats a monolithic
autonomous agent because the plan becomes an *inspectable, testable artifact*, not hidden
chain-of-thought. The planner is pinned on three axes: *when* — **static** (full plan upfront), not
per-step dynamic; *what* — a **dependency DAG**, not a flat list; *how* — **one structured call +
validation**, not a `create_agent` loop mutating the plan live. An autonomous orchestrator was rejected
on the rubric: inherently sequential (no continuous concurrency), non-deterministic (hurts Tests), opaque
(hurts Observability), and a single point of failure. The project keeps **autonomy at the leaves**
(Research *is* a `create_agent` loop) and **determinism at the backbone** (planner → scheduler →
synthesizer). (✅ `engine/planner.py` emits a reasoning-first `PlannerDraft`; `engine/validation.py`
finalizes it with derived `parallel_groups`.)

**Planner prompt:** the embedded output JSON schema; a few-shot goal→plan example; the **agent registry**
(each agent's name + allowed actions + capabilities) so only routable steps can be emitted; and
pass-through of `constraints`/`output_format` into step inputs. The untrusted `goal` is **fenced as data**
(`sub_agents/_prompt_utils.py::fence`, applied in `engine/prompts.py`).

**Validation:** Pydantic parse → topological **cycle check (Kahn)** → referential checks (agent exists,
action valid for that agent, every dependency id exists). An invalid plan triggers **one bounded re-ask**,
then fails cleanly — the same principle as the agents: LLM output is parsed into a typed schema before it
is trusted (`agent_base.py::invoke_structured`). (✅ `engine/validation.py::validate_and_finalize`.)

---

## 2. Dependency Management

**Approach chosen:** The **dependency edges are the source of truth**; the LLM's `parallel_groups` is a
hint/cross-check, not the executor's authority. Execution order derives from **in-degree readiness**
computed from `dependencies`, so the system is correct even if `parallel_groups` is wrong.
(✅ `engine/scheduler.py`.)

**Data passing:** Each finished step's `AgentResult.output` is injected into its dependents' input. The
output contract is uniform across agents (`agent_base.py::AgentResult`: `step_id, agent, status, output,
tokens_used, execution_time_ms` + additive `est_cost_usd`/`actual_cost_usd`), so the orchestrator consumes
any agent's result unchanged. Upstream outputs are **summarized/trimmed** to `context_char_budget` before
injection (not raw-concatenated) to protect the token budget (`engine/dispatch.py`).

**Cycle detection:** Kahn's algorithm at validation time; a plan that can't be fully topologically ordered
is rejected **before** execution — no partial run on a cyclic plan
(`engine/validation.py::derive_parallel_groups`).

*Design journey — cost estimate became a measured prompt, not a flat constant.* The pre-run `est_cost_usd`
(distinct from the measured post-run `actual_cost_usd`) first multiplied a hardcoded per-agent
`_AVG_INPUT_TOKENS` by call count, so two prompts differing 10× in size got the same estimate. Asked *"is
there a better way?"*, a web sweep and the `agent-building` cost reference both said: count the real
tokens. **Option A — call the provider tokenizer** for an exact count: rejected — it's a *synchronous
network call*, and the agents are async, so it would block the event loop (killing §3 concurrency) and
crash the offline test doubles, adding a network failure mode to a mere estimate. **Option B —
deterministic character-ratio over the real assembled prompt** (chosen): the estimate now tracks
instruction + context + data size, with no blocking call and full determinism; exact accounting stays the
measured `actual_cost`. (✅ `general_utils/tokens.py::count_prompt_tokens`; tests in `test_tokens.py`.)

---

## 3. Parallel Execution

**Approach chosen:** A **continuous ready-set scheduler** on `asyncio` (`scheduler.py::execute_plan`): all
ready steps launch as tasks, the driver awaits `asyncio.wait(FIRST_COMPLETED)`, and **the instant any step
finishes** its newly-ready successors launch — without waiting for siblings. This is genuinely concurrent
and non-wave (the spec's "not sequential async"): a fast step's successor starts before a slow sibling
finishes, which a `gather`/super-step model would block. Real overlap because agents are async at the call
boundary (Research awaits Tavily + LLM I/O). (✅ tests in `test_scheduler.py`.)

**Concurrency limit:** an `asyncio.Semaphore` sized from `orchestrator.concurrency` (default 3) caps
simultaneous LLM/search calls — doubling as the active provider's free-tier rate-limit throttle.

**Error handling:** per-step policy (§4). A failed step marks its transitive *dependents* `skipped` but
**independent branches keep running** — no task-wide abort. Safe because **agents never raise**: each
catches all exceptions and returns a `status="failed"` `AgentResult`, so the scheduler always gets a
result to route on. The one exception that propagates by design is `asyncio.CancelledError` (not caught by
`except Exception`), so a cancelled step dies immediately and is recorded `cancelled`.
(✅ classification in `engine/monitor.py`.)

---

## 4. Failure Recovery

**Approach chosen:** a five-rung escalation ladder, cheapest first — (1) **retry** the step (transient
errors, bounded backoff, at the agent layer); (2) **skip + continue** (a failed step marks only its
dependents `skipped`; independent branches survive); (3) **partial result** (a non-critical branch is lost
— the synthesizer reports the omission); (4) **partial/failed task** (a *load-bearing* step fails —
classified `structural`); (5) **bounded re-plan** (`structural` + re-planning still allowed; the merge
freezes completed steps, namespaces new ids under `r{n}_`, re-validates the DAG). Rungs 1–4 keep the engine
deterministic; only rung 5 re-invokes the LLM, capped by `max_replans` (default 1). Final status follows
the spec set: `completed` (with partial result + failed/skipped lists) whenever *any* step completed,
`failed` only when nothing did — no non-spec "partial" status (`engine/nodes.py::_final_status`).

**Retry logic:** **exponential backoff with jitter** at the model-call boundary, bounded attempts —
`@retry(stop=stop_after_attempt(6), wait=wait_exponential_jitter(initial=4, max=70))` in
`agent_base.py::invoke_structured`, plus provider-level `max_retries=6`. **Driven by a real failure:** the
live eval hit Gemini `429 RESOURCE_EXHAUSTED` (free tier = 5 req/min **and** 20/day per model); the backoff
+ a single-worker eval + a `--model` quota-fallback override are the direct response.

**Partial results:** `AgentResult.status` already encodes degraded completion — the Writing Agent returns
`completed_degraded` with `unresolved_issues` at its reflection cap. The synthesizer composes from whatever
succeeded, **tags provenance** (which agent produced what, with confidence + sources), lists skipped/failed
steps, and is prompted to **reconcile conflicts** (prefer higher confidence / more sources, note the
disagreement) rather than concatenate. (✅ `engine/synthesizer.py`, tests in `test_synthesizer.py`.)

The remaining mechanisms all follow one principle — **a free deterministic check decides whether to spend
an LLM call; the LLM makes only the judgment a parser can't:**

- **Criticality classification → re-plan decider.** When a step fails, the Monitor classifies it
  (`monitor.py::classify_failure`): **structural** if it loses *crucial* work (the failed step is
  non-`optional`, or its skip-cascade strips a non-`optional` dependent), else **skippable** (loss confined
  to `optional` steps — no LLM call). Only `structural` consults the LLM **re-plan decider**
  (`evaluation.py::decide_replan`), one bounded call given the whole plan + completed outputs + the failed
  step, returning `continue` or `replan`. *Design journey:* the classifier first said `skippable` whenever
  *any* non-`optional` step survived — a live run where a load-bearing research failure gutted 5 of 6 steps
  but one branch survived wrongly suppressed re-plan. **Option A — a numeric loss-threshold (≥X% removed):**
  rejected as semantically blind (treats every step as equal weight). **Option B — per-step criticality**
  (chosen): no magic number, and it reuses the `optional` flag the schema already carries. (✅ tests in
  `test_monitor.py`, `test_evaluation.py`.)
- **Synthesis never crashes the run.** Both LLM calls in the synthesize→judge tail can fail terminally
  (`400 tool_use_failed`, `429`, network); previously such a failure crashed the run and discarded every
  completed step. `synthesize_node` now extends the agents' "never raise" guarantee: on any exception it
  assembles a **deterministic best-effort answer** from completed outputs (`synthesizer.py::fallback_synthesis`
  — Writing prose first, then every code block verbatim) flagged `synth_failed`; `judge_node` short-circuits
  to `accept` as `completed_degraded`, guarded the same way. A model swap was rejected as *the* fix: routing
  synthesis to the tool-reliable tier only lowers one failure mode's probability and can't cover
  429/quota/network; the fallback covers the whole class for zero tokens, with model choice left as an
  independent dial. (✅ tests in `test_graph.py`; confirmed live.)
- **Research re-asks a malformed tool call.** The provider intermittently rejects a tool call `400
  tool_use_failed` (Llama emits its own `<function=…>` dialect; GPT-OSS leaks a harmony channel token into
  the tool name). It's sampling-dependent, so backoff can't help (a 400 isn't transient). The agent wraps
  just the loop in a **bounded re-ask** with a **bumped temperature** and fresh context
  (`research/agent.py::_invoke_with_tool_retry`) — the perturbation is the point (research runs at temp 0,
  so a plain retry re-samples the identical bad call). After the budget it degrades to a clean `failed`
  result. (✅ tests in `test_research_agent.py`.)
- **Synthesis quality judge — the last unguarded stage gets the same two-tier gate.** Synthesis was the
  only outer-loop stage shipping unchecked LLM output. The fix mirrors the planner: free deterministic
  checks first (`synthesis_judge.py::check_synthesis` — empty content, `output_format` non-compliance,
  attribution to non-existent step ids), one structured LLM judge second, as **two flat graph nodes**
  (`synthesize`→`judge`) with corrective loops as edges. The judge returns **3 actions**: `accept`,
  `resynthesize` (cheap retry over the same outputs, bounded by `max_resynth`=2), or `replan` (the outputs
  lack the info; reuses the shared `max_replans` budget). Reported confidence is **calibrated** down to the
  completion ratio at accept time. On budget exhaustion it takes `accept` as `completed_degraded` rather
  than looping. *Design journey (5 forks):* (1) I caught the gap with one question — *"where is synthesis's
  validation layer, and should it retrigger the graph?"*; (2) deterministic-only rejected — a parser can't
  catch a *fabricated* statistic, so the judge is mandatory (checks stay as a free first tier); (3) reusing
  the `evaluate` node rejected — it's purpose-built for step-failure classification and moving it would
  un-guard execution → a separate judge; (4) 4 actions → 3 (`rerun` folded into `replan` — a degenerate
  replan); (5) nested subgraph rejected for flat nodes — `replan` must reach the sibling `execute` node, and
  an in-node loop hides from the trace + checkpointer. (✅ `engine/synthesis_judge.py`, `nodes.py`; tests in
  `test_synthesis_judge.py` + `test_graph.py`; eval 5/5.)

**Monitoring (Observability).** The Monitor is the platform's context bus and required monitoring surface:
per-step **execution trace** (agent, action, in/out, status, duration, tokens, timestamps), live **status +
progress**, **agent outputs**, **token usage**. It lives **off** the checkpointed LangGraph state
(`engine/runs.py::RunRegistry`, keyed by `task_id`) so the API can read live status mid-run and the
checkpointed state stays small + msgpack-serializable. Trace appends are synchronous store mutations (safe
under single-threaded asyncio). (✅ `engine/monitor.py`, `runs.py`.)

**Caveat — cancellation is cooperative.** `POST /tasks/{id}/cancel` sets a flag the scheduler checks between
launches, so it stops new work and reports completed steps, but doesn't interrupt a step already blocked
inside a provider call — a run can briefly report `executing` right after a `cancelled` ack. Preemptive kill
is deferred (§5).

---

## 5. One Thing I Would Do Differently With More Time

The orchestration layer is built (§§1–4 ✅), so the remaining gaps are about depth, validation, and fit —
not coverage. In rough priority:

1. **More validation.** I validated against the assignment's literal Task Format / API / Final Result
   examples and caught three real defects (see the note below and AI_USAGE.md) — but I'd want more: a wider
   sweep of malformed and adversarial goals, constraint edge cases (conflicting/oversized constraints), and
   concurrent multi-task load against the *live* providers, not just the offline doubles. I did a pass; it
   deserves a deeper one.
2. **More thinking about the actual use cases — and which sub-agents follow from them.** The four agents
   (research / analysis / code / writing) satisfy the spec's example, but the *right* roster depends on what
   the platform is genuinely for. With more time I'd profile the target workloads first and add agents to
   match — e.g. a **data/SQL** agent, a **fact-checking / verification** agent, a **summarization** agent, or
   a **tool/API-calling** agent — rather than assume the spec's four are the final set. The registry makes
   adding one cheap (a new self-contained package + one registry entry), so the limiting factor is *deciding*
   the roster, not wiring it.
3. **More on tests — the area I'd extend most.** Today the suite is fully offline against fakes (deterministic
   and fast, but blind to live behavior). I'd add: (a) an **end-to-end live test** behind a quota-gated CI
   flag against the real APIs; (b) **property/fuzz tests** over the planner→validation path (random DAGs,
   injected cycles, dangling deps); (c) **concurrency/race tests** asserting genuine overlap and correct
   skip-cascades under many simultaneous tasks; and (d) **regression fixtures for each spec shape** so
   conformance can't silently drift again. Tests are where this system most needs hardening before it carries
   real traffic.
4. **Durability.** Move task/plan state from the in-memory `RunRegistry` to a **persistent store** (and swap
   `MemorySaver` for a durable checkpointer) so `GET /tasks/{id}` survives a restart.
5. **Deeper dynamic re-planning** — currently one bounded structural-failure re-ask; richer would be a
   multi-attempt patch-planner that re-scopes only the failed sub-DAG. And **true streaming of intermediate
   content** (the SSE endpoint today emits progress, not partial content) — the one honest nice-to-have skip.

> Closed during a spec-validation pass: wiring `output_format` through the `POST /tasks` body — the request
> model rejected the spec's JSON-object `constraints` (422) and never surfaced `output_format`.
> `api/models.py::TaskRequest` now accepts `constraints` as object **or** string and threads `output_format`
> end-to-end. ✅
>
> Component-level plans and the full requirements-compliance matrix are kept as internal notes outside the
> submission tree.
