# Design Decisions

> Status legend: **✅ Implemented** (code exists, pointed at by file) · **🟡 Designed** (decided
> and specified, not yet coded). The platform is built component-by-component; all four specialized
> agents **and** the orchestration layer are complete and tested offline.
> File paths are under `app/src/` unless noted.

**Build status at time of writing**

| Layer | State | Where |
| --- | --- | --- |
| Writing Agent (LangGraph reflection loop) | ✅ built + unit-tested + eval'd | `sub_agents/writing/` |
| Research Agent (`create_agent` autonomous loop) | ✅ built + unit-tested + eval'd | `sub_agents/research/` |
| Analysis Agent (single structured call) | ✅ built + unit-tested | `sub_agents/analysis/` |
| Code Agent (two-tier gate: tree-sitter parse + LLM-critic fallback) | ✅ built + unit-tested + eval'd | `sub_agents/code/` |
| Shared agent contract / retry / token + cost capture | ✅ built | `general_utils/agent_base.py`, `general_utils/` |
| Planner + DAG validation | ✅ built + unit-tested | `engine/planner.py`, `engine/validation.py` |
| Scheduler (continuous-concurrency inner executor) + dispatch | ✅ built + unit-tested | `engine/scheduler.py`, `engine/dispatch.py` |
| Monitor (trace, status, totals, failure classification) | ✅ built + unit-tested | `engine/monitor.py` |
| Re-plan decider + bounded merge | ✅ built + unit-tested | `engine/evaluation.py` |
| Synthesizer (provenance) | ✅ built + unit-tested | `engine/synthesizer.py` |
| Synthesis quality judge (deterministic checks + LLM judge, 3 actions) | ✅ built + unit-tested + eval'd | `engine/synthesis_judge.py` |
| LangGraph outer loop (plan→execute→evaluate→synthesize→judge) | ✅ built + unit-tested | `engine/graph.py`, `engine/nodes.py` |
| API (6 endpoints) + entry points | ✅ built + unit-tested | `api/`, `main.py`, `cli.py` |

**Code Agent — verification without execution (two-tier gate).** The spec's Code Agent does
generate / explain / debug, all text-producing tasks; **nothing in the spec requires running code**,
and executing LLM-generated code from an untrusted goal string would be the single largest attack
surface in the project. So the agent ships with **no sandbox, no tool loop, no execution** — it
mirrors the Writing agent shape (one structured call → typed `CodeOutput`). Correctness is gated by
a **two-tier quality gate** (`sub_agents/code/validation.py`, `agent.py`), the tier chosen per
language by `has_validator(language)`:

- **Tier 1 — deterministic parser (ground truth).** Python via stdlib `ast.parse` (exact);
  JavaScript via `tree-sitter` + `tree-sitter-javascript` (a parse fails on `root_node.has_error`
  **or any `is_missing` node** — the missing-node walk catches inserted MISSING nodes `has_error`
  alone skips). A real parse error is unambiguous and feeds a bounded correction loop (one refine
  call per failure, capped by `max_syntax_retries`, default 4). tree-sitter is error-recovering, so
  it is a **deliberately coarser gate** than `ast` (catches gross breakage, not every subtle error)
  — documented honestly. **Graceful give-up:** after the retry budget the agent returns the
  best-effort code, **not** a failure — status stays `completed` and the output records the
  unverified state (`parses: false` + `validation_error`).
- **Tier 2 — LLM critic, fallback only.** For languages with **no** registered parser (Ruby, Go,
  …) — the only signal where a parser can't reach. A **cheaper, independent** reviewer
  (`review_model_id`, default `gemini-2.5-flash`, temp 0) returns
  `CodeVerdict{verdict: revise|return, issues}`; `revise` recalls the generator with concrete
  issues, bounded by `max_review_retries`. Independence (different model, temp 0) avoids the
  self-approval bias of a model grading its own output, at a fraction of the generator's cost.
  Parser-backed languages **never** invoke the critic — ground-truth-first, no added cost.

The gate result lands on a single key, `output["parses"]` (`true`/`false`/`null` when no parser),
which the eval judge reads **from the agent output** rather than recomputing, so the two never
drift. **(✅ built + unit-tested, `tests/sub_agents/test_code_agent.py` — 19 cases covering both
tiers, alias normalization, graceful give-up, parser-vs-critic routing, fencing, and
generator-vs-reviewer cost accounting; eval'd via `evals/judges/code_judge.py` +
`evals/datasets/code_agent.yaml` — generate Python/JavaScript, explain, debug, and an
injection-safety probe, judged against the deterministic parse signal. The latest live run scored
**5/5 PASS** — including the JavaScript case, where `parses=yes` confirms the Tier-1 tree-sitter gate
now actually fires on JS (a prior run showed `n/a` before the grammar was wired in).)**

---

## 1. Task Decomposition Strategy

**Approach chosen:** A dedicated **Gemini planner** emits a strict-JSON `ExecutionPlan` (the spec's
plan shape: `steps[]` with `id/agent/action/input/dependencies`, plus `parallel_groups`). It is
**validated as a DAG before any execution** — never executed raw. A single planning LLM call over a
custom scheduler is preferred to one monolithic autonomous agent because the plan is then an
*inspectable, testable artifact* (satisfies the spec's "View execution trace" / "plan generation"
test) rather than hidden chain-of-thought. **(✅ built — `engine/planner.py` emits a reasoning-first
`PlannerDraft`; `engine/validation.py` finalizes it into an `ExecutionPlan` with derived
`parallel_groups`.)**

**Planner pinned on three design axes.** Researching how to define a planner surfaced three
independent choices, and this project pins each: *when* it plans — **static** (a full plan upfront),
not dynamic per-step; *what* it emits — a **dependency DAG**, not a flat ordered list; *how* it is
controlled — **one structured call + deterministic validation**, not a `create_agent` reasoning loop
that holds and mutates the plan live. An autonomous LLM orchestrator was considered and rejected on
the rubric: it is *inherently sequential* (a ReAct loop cannot deliver the continuous concurrency
the Performance dimension grades), non-deterministic (hurts Tests), opaque (hurts Observability),
and a single point of failure — if that one orchestrator LLM rate-limits mid-run the whole task
dies. The project instead keeps **autonomy at the leaves** (the Research agent *is* a `create_agent`
loop, where the next move is genuinely unknowable upfront) and **determinism at the backbone**
(planner → scheduler → synthesizer), injecting LLM judgment only at one bounded decision point
(re-plan, §4). The result is live-validated, monitored, updatable execution that still keeps
concurrency, observability, and testability. **(✅ built — the LangGraph outer loop in
`engine/graph.py` wraps the plain-async inner executor in `engine/scheduler.py`.)**

**Planner prompt — key elements:** the embedded output JSON schema; a few-shot example mapping a
goal → `steps`/`dependencies`/`parallel_groups`; the **agent registry** (each agent's name +
allowed `action`s + capabilities) so the planner can only emit routable steps; and pass-through of
the task `constraints`/`output_format` into the relevant step inputs. The untrusted `goal` is
**fenced as data**, reusing the same defense already applied in the Writing/Research prompts
(`sub_agents/_prompt_utils.py::fence`, applied in `engine/prompts.py`). **(✅ built.)**

**Validation:** Pydantic parse of the model output → topological **cycle check (Kahn)** →
referential checks (every `step.agent` exists in the registry; every `action` is valid for that
agent; every dependency id exists). An invalid plan triggers **one bounded re-ask**, then fails
cleanly. This mirrors the principle already implemented in the agents: **LLM output is parsed into
a typed schema before it is trusted** (`general_utils/agent_base.py::invoke_structured` uses
`with_structured_output(..., include_raw=True)` so every node returns a validated Pydantic object,
✅). **(Plan-level validator: ✅ built — `engine/validation.py::validate_and_finalize`; one bounded
re-ask in `engine/planner.py` feeds the validation errors back to the model.)**

---

## 2. Dependency Management

**Approach chosen:** The **dependency edges are the source of truth**; the LLM's `parallel_groups`
is treated as a *hint / cross-check*, not as the executor's authority. Execution order is derived
from **in-degree readiness** computed from `dependencies`, so the system is correct even if the
planner's `parallel_groups` are wrong. **(✅ built — `engine/scheduler.py` launches steps off live
dependency readiness, not `parallel_groups`.)**

**Data passing:** Each finished step's `AgentResult.output` is injected into its dependents' input.
The agent output contract is **already implemented and uniform** across agents
(`general_utils/agent_base.py::AgentResult`: `step_id, agent, status, output, tokens_used, execution_time_ms`, plus
additive `est_cost_usd`/`actual_cost_usd`), so the orchestrator can consume any agent's result
unchanged (✅ contract). To protect the token budget, upstream outputs are **summarized/trimmed
before injection** rather than raw-concatenated, trimmed to `context_char_budget`
(`engine/dispatch.py` builds each dependent's upstream context). **(Wiring into the
scheduler: ✅ built.)**

**Cycle detection:** Kahn's algorithm at plan-validation time; a plan whose steps cannot be fully
topologically ordered is rejected **before** execution (no partial run on a cyclic plan).
**(✅ built — `engine/validation.py::derive_parallel_groups` raises on any cycle.)**

**Pre-execution cost estimate — measured prompt, not a flat constant.** The additive `est_cost_usd`
is a *pre-run* forecast (distinct from `actual_cost_usd`, which the token-cost middleware measures
from `usage_metadata` after the fact). It was first computed from a hardcoded per-agent
`_AVG_INPUT_TOKENS` constant × expected calls, so two requests whose prompts differed 10× in size
got an identical estimate. It now counts the **real assembled prompt** via
`general_utils/tokens.py::count_prompt_tokens` (a character-ratio heuristic, `chars_per_token` in
`app/config.yaml`), so the estimate scales with instruction + upstream context + data-preview size;
only the genuinely-unknowable parts (output length, loop-turn count) stay as config knobs
(`avg_output_tokens` per agent). **(✅ built — `general_utils/tokens.py`, wired into
`sub_agents/{analysis,research,code}/agent.py`; unit-tested `tests/general_utils/test_tokens.py` —
**5/5 PASS** covering empty prompts, real-length counting, monotonic growth with size, and
`chars_per_token` sensitivity.)**

**Design journey (how we got here):** flat per-agent constant → questioned directly (*"is there a
better way to estimate the cost?"*); a web sweep **and** the `agent-building` cost-optimization
reference both said the same thing — *count the real tokens, don't assume averages*. **Option A —
call the provider tokenizer** (`get_num_tokens_from_messages`) for an exact count: rejected on
Performance + Tests — Gemini's counter is a **synchronous network call**, and the agents are
`async`, so invoking it on the hot path would block the event loop (killing the continuous
concurrency §3 grades) and would fail on the injected offline test doubles, adding a network failure
mode to what is only an estimate. **Option B — deterministic character-ratio over the real prompt**
(chosen): captures the actual win (the estimate now tracks prompt size) with no blocking call, no new
failure mode, and full determinism. The `agent-building` reference's own stance sealed it —
*pre-execution estimates are inherently rough; the precise signal is the measured `actual_cost`* — so
paying a per-run network round-trip for exactness on the *estimate* is the wrong trade.

---

## 3. Parallel Execution

**Approach chosen:** A **continuous ready-set scheduler** on `asyncio`
(`engine/scheduler.py::execute_plan`): all currently-ready steps are launched as tasks, the driver
awaits `asyncio.wait(..., FIRST_COMPLETED)`, and **the instant any step finishes** its newly-ready
successors are launched — *without* waiting for its siblings. This is **genuinely concurrent and
non-wave** (the spec calls out "not sequential async"): a fast step's successor starts before a slow
sibling finishes, which a wave-synchronous `gather`/super-step model would block. It is real overlap
because the agents are async at the call boundary — Research is `async def run_research_agent(...)`
and awaits real I/O (Tavily + Gemini), so the waits overlap rather than serialize (✅ agent is async).
**(Scheduler: ✅ built + unit-tested, `tests/engine/test_scheduler.py`.)**

**Concurrency limit:** an `asyncio.Semaphore` sized from `orchestrator.concurrency` in
`app/config.yaml` (default 3) caps simultaneous LLM/search calls — this doubles as the **rate-limit
throttle** the Gemini free tier forces (the live eval in `evals/` had to drop to one worker; see
Failure Recovery). **(✅ built.)**

**Error handling (parallel step fails):** per-step policy (section 4). A failed step marks its
transitive *dependents* `skipped` but **independent branches keep running** — no task-wide abort. This
is safe because **agents never raise**: every agent catches all exceptions and returns a
`status="failed"` `AgentResult` with an `error` field (✅), so the scheduler always gets a result
object to route on, never an exception mid-flight. The one exception that *does* propagate by design
is `asyncio.CancelledError` (a `BaseException`, not caught by the agents' `except Exception`), so a
preemptively cancelled step dies immediately and is recorded `cancelled`. **(✅ built — failure
classification in `engine/monitor.py`, cancellation in `engine/scheduler.py`.)**

---

## 4. Failure Recovery

**Approach chosen — a five-rung escalation ladder, cheapest first:** (1) **retry** the step
(transient errors — bounded backoff, already at the agent layer); (2) **skip + continue** (a failed
step marks only its dependents `skipped`; independent branches survive); (3) **partial result** (a
non-critical branch is lost — the synthesizer reports the omission honestly); (4) **partial or failed
task** (a *load-bearing* step fails and its skip-cascade kills **every** non-`optional` step — this is
classified `structural`; criticality is computed from a per-step `optional` flag plus the dependency
graph, not an undefined "sink"); (5) **bounded re-plan** (the failure is `structural` and re-planning
is still allowed; the merge protocol freezes completed steps, namespaces new ids under an `r{n}_`
prefix, and re-validates the merged DAG). Rungs 1–4 keep the planner static and the engine
deterministic; only rung 5 re-invokes the LLM, and it is capped by `max_replans` (default 1). Final
**task status follows the spec set**: a run keeps `status="completed"` with a partial result + the
failed/skipped lists whenever *any* step completed, and only reports `failed` when nothing completed
at all — there is no non-spec "partial" status (`engine/nodes.py::_final_status`). **(✅ built +
unit-tested across `tests/engine/test_scheduler.py`, `test_monitor.py`, `test_evaluation.py`,
`test_graph.py`.)**

**Deterministic classification decides *whether* to consult the LLM; the LLM decides *whether to
re-plan*.** The same cost-aware principle as the agents — a free deterministic check first, an LLM
call only where it changes a decision — but at the orchestrator the deterministic pre-filter is a
**structural classification, not a confidence threshold**. When a step fails, the Monitor classifies
it (`engine/monitor.py::classify_failure`): it is **skippable** when independent non-`optional` work
survives its skip-cascade (mark the branch skipped, let healthy branches keep running, no LLM call),
and **structural** when the cascade removes *every* remaining non-`optional` step. Only a `structural`
failure consults the LLM **re-plan decider** (`engine/evaluation.py::decide_replan`), one bounded call
given **the whole current plan, which steps finished and their outputs, and the failed step + its
error**, that returns `continue` (the goal is still reachable from what completed) or `replan` (revised
steps for the unfinished part). This keeps the planner static and the hot path deterministic, calls the
LLM at exactly one bounded decision point, and never exhausts the Gemini free tier (5 rpm / 20-per-day)
the way a judge-after-every-step would. Offline LLM-as-judge stays in `evals/` for regression, separate
from this online control path. **(✅ built + unit-tested, `tests/engine/test_monitor.py`,
`test_evaluation.py`.)**

**Monitoring (Observability dimension).** The Monitor is the platform's context bus and the spec's
required monitoring surface: a per-step **execution trace** (agent, action, in/out, status,
duration, tokens, timestamps), live **task status + progress** (`completed/total`, current step),
**agent outputs**, and **token usage**. Trace appends and counter bumps are synchronous store
mutations (safe under single-threaded asyncio — no lock needed). It is the single source feeding the
API status/result/trace endpoints and the structural classification above; it lives **off** the
checkpointed LangGraph state (in `engine/runs.py::RunRegistry`, keyed by `task_id`) so the API can
read live status while the graph is still running, and so the checkpointed state stays small and
msgpack-serializable. **(✅ built — `engine/monitor.py`, `engine/runs.py`.)**

**Retry logic:** **exponential backoff with jitter** at the model-call boundary, bounded attempts.
Implemented in `general_utils/agent_base.py::invoke_structured` via
`@retry(stop=stop_after_attempt(6), wait=wait_exponential_jitter(initial=4, max=70), reraise=True)`,
plus provider-level `max_retries=6` in `general_utils/llm.py::build_chat_model` (✅). This was
**driven by a real failure**: the live eval hit Gemini `429 RESOURCE_EXHAUSTED` (free tier =
5 req/min **and** 20 req/day per model); the backoff/jitter and a single-worker eval are the direct
response (the eval also gained a `--model` override to fall back to a model with separate quota).
The Research eval (`evals/judges/research_judge.py` + `evals/datasets/research_agent.yaml`, 5 LLM-judged
examples spanning factual lookup, comparison, recent-developments, how-to, and an **unanswerable /
anti-hallucination** probe) ran end-to-end and scored **4/5 PASS**; the lone failure was the daily-quota
casualty above (the agent returned a clean `status="failed"` `AgentResult`, never crashed the batch),
not a grounding regression — directly exercising this section's "agents never raise" guarantee.

**Partial results:** the spec's `AgentResult.status` already encodes degraded completion. The
Writing Agent returns **`completed_degraded`** with `unresolved_issues` when it hits the reflection
cap with open judge issues (`sub_agents/writing/agent.py::assemble_result`, ✅). The synthesizer
composes from whatever succeeded, **tags provenance per the spec** (which agent produced
what, with per-step confidence and sources), and lists skipped/failed steps; final task status
reflects partial vs. full completion. The synthesizer prompt is also told to **reconcile conflicts**
(prefer higher `confidence` / more sources and note the disagreement) rather than concatenate.
**(Synthesizer: ✅ built — `engine/synthesizer.py`, unit-tested `tests/engine/test_synthesizer.py`.)**

**Synthesis quality judge — the last unguarded stage gets the same two-tier gate as planning.**
Synthesis was the only outer-loop stage whose output reaches the user with no quality gate: one LLM
call, and whatever parsed became the `FinalResult`. The fix mirrors the planner's discipline — **free
deterministic checks first, one structured LLM judge second** — modelled as **two thin graph nodes**
(`synthesize` → `judge`) with the corrective loops as graph edges, so the loop is traced,
checkpointed, and unit-testable rather than hidden in an in-node `for`-loop. The judge returns one of
**three actions**: `accept` (ship), `resynthesize` (cheap retry with scoped feedback over the *same*
outputs — for an unsupported claim, incoherence, or a format violation), or `replan` (the outputs
genuinely lack the information to answer — author replacement steps for the gap). `resynthesize` is a
`judge → synthesize` edge bounded by `max_resynth` (default 2); `replan` reuses `merge_replan` and the
**shared `max_replans` budget** via a `judge → execute` edge, so the two corrective paths can never
loop unbounded. The deterministic pre-checks (`engine/synthesis_judge.py::check_synthesis`) are free
and catch mechanical faults the LLM shouldn't be paid to find — empty content despite completed steps,
`output_format` non-compliance (JSON/bullet), and attribution to step ids that don't exist — and they
are fed to the judge as evidence. Reported **confidence is calibrated** down to the completion ratio at
accept time (`calibrated_confidence`), so a partial run can't claim full confidence. When either budget
is exhausted the judge takes the `accept` path with status **`completed_degraded`** rather than looping
— graceful degrade, never a hang. The deterministic checks decide nothing alone; as with the re-plan
decider, the LLM is the only thing that adjudicates faithfulness/coverage, and it fires exactly once per
synthesis pass. **(✅ built — `engine/synthesis_judge.py`, `engine/nodes.py::{synthesize_node,judge_node,
route_after_judge}`; unit-tested `tests/engine/test_synthesis_judge.py` + loop/budget tests in
`tests/engine/test_graph.py`; eval'd via `evals/judges/synthesis_judge_eval.py` +
`evals/datasets/synthesis_judge.yaml` — a **label-based** eval (the component under test is itself an
LLM judge, so each scenario carries a known-correct expected verdict and the harness compares the live
verdict to that label; no second judge). The latest run scored **5/5 PASS** across grounded-accept,
hallucinated-claim resynthesize, JSON-format resynthesize, coverage-gap replan, and a
conflicting-sources accept — the judge named the two fabricated figures verbatim in its feedback and,
on the gap case, authored two replacement `code` steps.)**

**Design journey (how we got here).** The final shape above was not the first proposal — it came out
of five forks I pushed the AI through, each tied to a rubric dimension:

1. **Spotting the gap at all.** The AI's outer-loop design guarded planning (DAG validation),
   execution (agent retries), and step-failure (bounded re-plan), but left `synthesize` shipping
   whatever the model returned. I caught it with one question — *"where is the validation layer for
   synthesis, and should it be able to retrigger the graph?"* The gap was the same blind spot as the
   original static planner: a stage with no recovery path.
2. **Deterministic-only vs. an LLM judge.** First instinct was to lean on cheap deterministic checks
   alone. Rejected: mechanical checks can catch an empty answer or bad JSON, but they cannot tell that
   a fluent paragraph contains a *fabricated* statistic. Hallucination detection needs a model that
   reads the claims against the source outputs → the judge is mandatory, **but** the deterministic
   checks stay as a free first tier so the LLM is never paid to find faults a parser can catch
   (Performance / cost).
3. **A new judge vs. moving the existing `evaluate` node.** The AI's reflex was to reuse `evaluate` by
   moving it to after synthesis. Rejected (Architecture): `evaluate` is purpose-built for step-failure
   classification — it takes a `failed_id`/`error` and decides re-plan — so it can't double as a
   faithfulness judge, and relocating it would *un-guard* execution. Decision: a separate
   post-synthesis judge; the pre-synthesis `evaluate` guard stays so broken executions never pay for a
   synthesis call.
4. **Four actions vs. three.** The draft had `accept | resynthesize | replan | rerun`. Rejected the
   fourth: "re-run one step with a fix" is just a degenerate replan where the judge authors the
   replacement step, so `rerun` folded into `replan` (KISS — fewer routes, one budget to reason about).
5. **Where the loop lives — fat in-node loop vs. nested subgraph vs. flat nodes.** The AI proposed a
   nested synthesis subgraph. Rejected (Observability / testability): a `replan` has to reach the
   *sibling* `execute` node, which is clumsy across a subgraph boundary, and an in-node `for`-loop
   hides the loop from the trace and the checkpointer. Decision: **two flat graph nodes**
   (`synthesize` → `judge`) with the corrective loops as ordinary edges, so every hop is traced,
   checkpointed, and unit-testable — and `judge → execute` is a normal edge to a sibling. The cheap
   remedy (`resynthesize`) is a tight `judge → synthesize` edge; the expensive one (`replan`) reuses
   the shared `max_replans` budget so neither can loop unbounded.

---

## 5. One Thing I Would Do Differently With More Time

The orchestration layer is now built (sections 1–4 are ✅), so the remaining gaps are about
**depth and durability**, not coverage. With more time I would (in order): (1) **deepen dynamic
re-planning** — it is currently bounded to a single structural-failure re-ask; richer would be a
multi-attempt patch-planner that re-scopes only the failed sub-DAG instead of re-planning the
remainder; (2) move task/plan state from the in-memory `RunRegistry` to a **persistent store** (and
swap `MemorySaver` for a durable checkpointer) so `GET /tasks/{id}` survives a restart; (3) **stream
intermediate results** over the trace (SSE/WebSocket) instead of only returning them at the end —
this is the one honest spec skip (nice-to-have); (4) add an **end-to-end live test** against the real
Gemini/Tavily APIs behind a quota-gated CI flag, complementing the fully-offline suite that ships
today.

> Closed during a spec-validation pass: item (5) of an earlier draft — wiring `output_format` through
> the `POST /tasks` body so the synthesis judge's `_check_format` is no longer dormant — is now done.
> While validating against the assignment's literal **Task Format**, I found the request model rejected
> the spec's JSON-object `constraints` (422) and never surfaced `output_format`; `api/models.py::TaskRequest`
> now accepts `constraints` as object **or** string (normalized to fenced planner text) and threads
> `output_format` end-to-end. ✅

> Component-level plans and the full requirements-compliance matrix are kept as internal working
> notes outside the submission tree.
