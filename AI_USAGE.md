# AI Tool Usage

## Tools I Used

- **Claude Code (Opus 4.8)** — primary pair-programming agent for design, implementation, test
  harnesses, refactoring, and keeping `DECISIONS.md` in sync.
- **Switchable LLM provider** — a `big`/`small`/`tools` tier per role, resolved from `llm_config.yml` for
  whichever `provider:` is active. **Groq** (Llama-3.3-70b big / 3.1-8b small / GPT-OSS-20b tools,
  `GROQ_API_KEY`) is the default — its free tier is far larger than Gemini's, so multi-step runs don't hit
  a daily cap; the `tools` tier is a function-calling-reliable model the Research Agent binds its search
  tool to (Llama on Groq emits malformed tool calls). **Google Gemini** (`GOOGLE_API_KEY`) stays fully
  supported. Both via LangChain `init_chat_model`.
- **Tavily** — the grounded web-search backend for the Research Agent.
- **`agent-building` skill** — the architecture reference (complexity ladder; pattern catalog:
  Plan-and-Execute, evaluator-optimizer, scatter-gather) used to ground the orchestration design.
- **Web search** — because an LLM's training knowledge goes stale, I searched for current information
  instead of trusting the model: comparing external positions on online vs offline LLM-as-judge and on
  reflection/evaluator-optimizer cost before committing to an evaluation strategy.
- **Custom documentation tooling I built (skill + hook).** Strong SDD documentation is a
  differentiator, so rather than reconstruct it at the end I built tooling to capture it as I work: a
  project skill, `decision-docs-from-conversation` (`.claude/skills/`), that turns a design
  conversation into template-conformant `DECISIONS.md` / `AI_USAGE.md` entries, and a **`Stop` hook**
  (`.claude/hooks/update_docs.py`) that triggers it after every session — a detached, recursion-guarded
  `claude -p` worker that folds new decisions into both files in place, so important knowledge always
  lands in the docs.

## What Helped Most

- **Offline test harness for the Research Agent.** The agent is an async `create_agent` loop over a live
  search tool, so naive tests would hit the network and quota. AI built deterministic doubles instead — a
  scripted `BaseChatModel` cycling canned tool-call/finish turns, a `with_structured_output` override, and a
  counting fake searcher — so all behavioral asserts (search cap, recursion cap, grounded sources,
  compaction, concurrency, output shape) run fully offline and fast.
- **LLM-as-judge eval harness.** AI scaffolded the dataset + judge mirroring the Writing-Agent eval — runs
  the real agent, scores grounding/confidence/no-hallucination, writes a markdown report. Ran end-to-end at
  4/5 PASS and surfaced a real operational constraint (below).
- **Grounding the orchestration design in references, not vibes.** The `agent-building` complexity ladder
  (`with_structured_output` when the schema is known upfront → the planner is a structured call, *not* a
  `create_agent` loop) plus a web sweep turned a vague "add a judge" idea into a specific tiered design:
  deterministic checks on 100% of steps, an LLM judge only at the bounded re-plan decision.
- **Two-tier Code-Agent validation, researched before coding.** I had AI first search how others validate
  LLM-written code (the Aider/Plandex reflection pattern) rather than invent one, then extend the
  Python-only `ast` gate. It scaffolded the tree-sitter JS validator, one generic correction loop reused by
  both tiers (DRY), and an LLM-critic fallback for parser-less languages — with offline tests for both
  paths, Ruff/mypy clean.
- **Mechanically applying an agreed design across seven files without drift.** Once the synthesis-judge
  design was locked, AI carried it through schema → judge → prompts → node split → graph edges → config →
  tests as one coherent change, every function under the 20-line rule, reusing
  `merge_replan`/`invoke_structured` instead of cloning — full offline suite green. The repetitive,
  error-prone wiring is where it earns its keep once *I* own the design.

## What I Had to Fix

- **AI invented model IDs *and their prices*.** The first plan named `gemini-3.5-flash` /
  `gemini-3.1-flash-lite` with **fabricated** per-token pricing — confident and wrong. I refused to trust
  AI cost numbers and made the **pricing table the single gate**: every model id must carry a verified price
  row or cost accounting fails fast. That later caught a **half-applied migration** (a cost test + two config
  defaults still naming a retired model). A blanket `git add -A` had nearly folded it in mid-flight; I now
  stage explicitly and re-confirm the suite before tagging.
- **Misread of the 429 error.** AI first treated the eval's `RESOURCE_EXHAUSTED` as transient and added
  backoff. The retries were a good resilience fix, but the root cause was a **per-model-per-day** free-tier
  cap (20/day) — no in-run retry clears it. The real fix was a single-worker eval + a `--model` override to a
  model with a separate daily bucket.
- **AI's first planner had no failure adaptivity.** Its opening pick was a purely static plan-once planner. I
  pushed back twice — first *"what happens on an unexpected failure?"*, then on the opposite extreme (an
  autonomous orchestrator that adapts live). Neither was right; I made AI cost the autonomous option against
  the rubric (loses concurrency, observability, testability; adds a single point of failure), and we
  converged on a **hybrid**: deterministic backbone + autonomy at the leaves + one bounded re-plan rung.
- **"Judge every step" was my idea, and AI went along with it.** Before committing I had it research the
  trade-off; the evidence (free-tier caps + non-determinism injected into tests) showed per-step judging is
  both a budget and a concurrency killer. Corrected to a **selective** judge, fired only on a deterministic
  suspect-signal, off the hot path.
- **AI left synthesis with no quality gate — the same blind spot as the planner.** Its outer loop guarded
  planning, execution, and step-failure, but `synthesize` shipped whatever the model returned. I caught it
  with *"where is the validation layer for synthesis — and should it retrigger the graph?"* AI's first reflex
  was to **move the existing `evaluate` node** after synthesis; I rejected that (it's built for step-failure
  classification and would un-guard execution). We converged on a **separate post-synthesis judge** with **3
  actions not 4** (rerun folded into replan), deterministic checks first, and I made AI keep it as **flat
  graph nodes** (the nested subgraph it proposed can't cleanly reach the sibling `execute` node).
- **AI's first code-validation plan over-spent and rubber-stamped itself.** Two wrong defaults: an
  **8-regeneration** syntax give-up cap (cost-overkill — cut to 4), and running the LLM critic on the **same
  generator model for every language**, including parser-backed ones (redundant cost + self-approval bias). I
  made it **fallback-only** (parser-backed languages never invoke the critic) on a **cheaper, independent
  reviewer at temp 0**, and collapsed two near-duplicate correction loops into one generic loop.
- **A flat per-agent cost constant masqueraded as a real estimate.** Each agent's `est_cost_usd` multiplied a
  hardcoded `_AVG_INPUT_TOKENS` by call count, so a one-line goal and a goal carrying 6 KB of context produced
  the same number. I asked *"is there a better way?"* and had AI check the web + the cost-optimization
  reference. Its first proposal — Gemini's `get_num_tokens_from_messages` for an exact count — failed once
  wired against the constraints: a **synchronous network call** on async agents would block concurrency and
  crash the offline doubles, all for precision on a mere estimate. We landed on a deterministic
  character-ratio over the **real assembled prompt**, leaving exact accounting to the measured `actual_cost`.
- **AI reached for repo-root CI; the spec forbade it.** To prove the image built, AI added a
  `.github/workflows/` pipeline — but the assignment fixes the submission tree and I'd been told `.github/`
  may not live at repo root. Relocating it failed (Actions only discovers workflows at `.github/workflows/`),
  so I dropped CI and leaned on `docker compose up --build` being self-contained. Removing the ignore files
  then exposed a real hazard: the Dockerfile's blanket `COPY app/ ./` would bake my real `.env` and the
  host's **Windows** `.venv` over the image's Linux venv. I had AI rewrite it to copy **only tracked source**
  — a clean, secret-free image from one change.
- **AI's first re-plan trigger was a magic number; I redirected it to criticality.** A live run showed a
  load-bearing research failure gutting 5 of 6 steps yet not re-planning, because the classifier said
  "skippable" the moment any branch survived. AI's first fix was a **numeric loss-threshold (≥50%)**. I didn't
  like it: *"maybe ask the planner to tag each step crucial / not crucial."* That pushed us to per-step
  **criticality** — and the schema **already** had the tag (`optional`), so no new field. I had AI revert the
  half-built threshold before implementing it.
- **AI proposed swapping the synthesis model; I made it defend the fallback as the real fix.** When a
  `tool_use_failed` call crashed a run *at synthesis* (losing 7 steps), the reflex was "route synthesis to the
  tool-reliable model." I interrogated that (*"why not just replace it with the gpt?"*, *"how will a
  deterministic process structure the output well enough?"*): a model swap only lowers one failure mode's
  probability and can't cover 429/quota/network, and the reliable tier's 8K-TPM window is too small for the
  heaviest call. So we built a **deterministic fallback** from completed steps. Verified live.
- **The synthesizer silently dropped code from the final answer.** I noticed a code-gen run came back with
  prose but no code — *"i don't see a code example in the response."* Root cause, not a prompt tweak:
  `_render_outputs` read only the output's `content` field and ignored the separate `code` field, so the model
  never saw the code. I had AI emit the code in a fenced block verbatim, with a regression test asserting it
  reaches the synthesis prompt.

A green suite hid three **spec-conformance** defects until I ran the app against the assignment's literal
Task Format / API / Final Result examples:

1. **Request body rejected (422).** The documented body — `constraints` as a JSON object + `output_format` —
   failed validation (the model accepted only free-text constraints, never surfaced `output_format`). Fix:
   `api/models.py::TaskRequest` accepts object **or** string and exposes `output_format`.
2. **Local run crashed on import.** `uv run python main.py` died with `ModuleNotFoundError: No module named
   'app'` — only the container's `PYTHONPATH` made it work. Fix: `main.py` mirrors the tests' path-insertion.
3. **Result envelope was flat, not nested.** `/tasks/{id}/result` returned a flat body while the spec shows a
   nested `result` (`content`/`format`/`word_count`) + top-level `execution_trace`. Fix: restructured
   `FinalResult` to the spec shape (provenance retained as additive fields), stamped per-step
   `started_at`/`completed_at`.

These are the gaps a passing suite can't see — the tests drove fakes against their *own* shapes; they never
POST the spec's JSON or assert its literal envelope. I then added tests that lock those shapes.

## What AI Struggled With

- **Model-intrinsic malformed tool calls it can lower but not eliminate.** Two signatures on Groq: Llama
  emits its own dialect (`<function=Name>{json}`, args in the name slot), and GPT-OSS leaks a harmony channel
  token into the tool name (`web_search<|channel|>commentary`) — both `400 tool_use_failed`. It's
  sampling-dependent and format-rooted, so no prompt removes it and retry-backoff (for transient errors)
  doesn't apply to a 400. AI could mitigate (tool-reliable tier, temperature-perturbed re-ask, synthesis
  fallback) but couldn't make a model stop mis-formatting calls. The honest posture: defense-in-depth, not a
  cure.
- **Per-organization (not per-key) Groq quota it can't see.** When a run exhausted the daily budget I asked
  *"if I give a fresh API token, will it work?"* — it won't: Groq's free-tier limits are enforced **per
  organization**, so a fresh key on the same account shares the drained pool. AI could route around it but the
  ceiling is an account fact invisible from the key.
- **External quota/billing limits it can't see.** The Gemini per-model-per-day cap meant the eval couldn't
  fully complete in one sitting (the 5th example, the anti-hallucination probe, was a quota casualty, not a
  behavioral failure). AI could diagnose and route around it but couldn't make the daily limit go away.
- **tree-sitter's error model is subtler than AI first assumed.** Its initial JS validator checked only
  `root_node.has_error`, which silently passes some malformed inputs (tree-sitter is error-recovering and
  inserts MISSING nodes `has_error` doesn't always flag). I had it verify the API empirically and add an
  `is_missing` walk, and document the honest limit — a coarser parse than `ast`, catching gross breakage, not
  every subtle error.
- **Dependency state it can't see across environments.** AI declared `tree-sitter` in `pyproject.toml` but the
  runner kept failing on import: a stale root `.venv` was resolving ahead of the real `app/` env. AI couldn't
  tell from the declared deps that the *active* interpreter lacked the package — it took `uv sync` in the right
  dir and pointing the suite at the `app/` env. An environment fact, not a code one.
