# AI Tool Usage

## Tools I Used

- **Claude Code (Opus 4.8)** — primary pair-programming agent for design, implementation,
  test-harness authoring, refactoring, and keeping `DECISIONS.md` in sync.
- **Google Gemini** (`gemini-2.5-flash` + `gemini-2.5-flash-lite` via LangChain
  `init_chat_model`, `model_provider="google_genai"`) — the LLM the platform itself runs on,
  and the judge model in the eval harness.
- **Tavily** — the grounded web-search backend for the Research Agent.
- **Claude Code `agent-building` skill** — the architecture reference (complexity ladder, pattern
  catalog: Plan-and-Execute, evaluator-optimizer, scatter-gather) I used to ground the
  orchestration-layer design rather than improvise it.
- **Web search** — to compare external positions on online vs offline LLM-as-judge and on
  reflection / evaluator-optimizer cost before committing to an evaluation strategy.

## What Helped Most

- **Offline test harness for the Research Agent.** The agent is an async LangChain
  `create_agent` loop over a live search tool, so naive tests would hit the network and the
  Gemini quota. AI built deterministic test doubles instead — a scripted `BaseChatModel` that
  cycles canned tool-call/finish turns, a `with_structured_output` override returning a fixed
  `ResearchSummary`, and a counting fake searcher — letting all nine behavioral asserts (search
  cap, recursion cap, grounded sources, compaction-between-rounds, concurrency, output shape)
  run fully offline and fast.
- **LLM-as-judge eval harness.** AI scaffolded the dataset (`evals/datasets/research_agent.yaml`)
  and judge (`evals/judges/research_judge.py`) mirroring the existing Writing-Agent eval — runs
  the real agent, has Gemini score grounding/confidence/no-hallucination, and writes a markdown
  report. It ran end-to-end at **4/5 PASS** and surfaced a real operational constraint (below).
- **Grounding the orchestration design in references, not vibes.** For the planner + failure layer,
  the `agent-building` skill's complexity ladder (`with_structured_output` when the output schema is
  known upfront → so the planner is a structured call, *not* a `create_agent` loop) plus a web sweep
  of current practice turned a vague "add a judge" idea into a specific, defensible **tiered** design:
  deterministic checks on 100% of steps, an LLM judge only at the bounded re-plan decision.
- **Two-tier Code-Agent validation, researched before coding.** I had the AI first search how others
  validate LLM-written code (the Aider/Plandex reflection pattern) rather than invent one, then extend
  the Python-only `ast` gate to other languages. It scaffolded the `tree-sitter` JavaScript validator,
  a single generic correction loop reused by **both** tiers (DRY), and an LLM-critic fallback for
  parser-less languages — then wrote 19 offline tests covering both paths, all green and Ruff/mypy
  clean. Same principle as the orchestrator: a free deterministic check first, the LLM only where a
  parser can't reach.
- **Mechanically applying an agreed design across seven files without drift.** Once the synthesis-judge
  design was locked (below), AI carried it through schema → judge module → prompts → node split → graph
  edges → config → tests as one coherent change, kept every function under the 20-line house rule,
  reused `merge_replan`/`invoke_structured`/the `repair_message` shape instead of cloning them, and ran
  the full suite green (107 tests) plus Ruff clean. The repetitive, error-prone wiring is exactly where
  it earns its keep once *I* own the design.

## What I Had to Fix

- **Speculative model IDs in the original plan.** The first research-agent plan named
  `gemini-3.5-flash` / `gemini-3.1-flash-lite` with invented pricing. Those IDs don't exist for
  this project — I corrected the config to the real `gemini-2.5-flash` / `gemini-2.5-flash-lite`
  and their actual prices before anything ran. The same phantom `gemini-3.5-flash` came back to
  bite at submission time: a blanket `git add -A` nearly folded an unfinished `gemini-3.5-flash`
  migration (the bumped config plus its matched pricing test) into the tagged commit. It was
  catchable only because the **pricing table** ships exactly two IDs (`gemini-2.5-flash` /
  `-flash-lite`) — the bumped models had no price, so cost accounting would have crashed on the
  first live task. I reverted the whole trio (`config.yaml` + `config.py` + test) to the validated
  baseline before tagging `v1`, and re-confirmed all 107 offline tests green.
- **Initial misread of the 429 error.** AI first treated the eval's `RESOURCE_EXHAUSTED` as a
  transient rate-limit and added `max_retries`/backoff. The retries were a good resilience fix,
  but the root cause was a **per-model per-day** free-tier cap (20/day for `gemini-2.5-flash`) —
  no amount of in-run retry clears it. The real fix was a single-worker eval + a `--model`
  override to a model with a separate daily bucket.
- **AI's first planner design had no failure adaptivity.** Its opening recommendation was a purely
  static plan-once planner. I pushed back twice — first on *"what happens on an unexpected
  failure?"*, then on the opposite extreme (*an autonomous `create_agent` orchestrator that holds the
  plan and adapts live has a real advantage*). Neither first answer was right. I made the AI cost the
  autonomous option against the actual rubric (it loses concurrency, observability, and testability,
  and adds a single point of failure), and we converged on a **hybrid**: deterministic backbone +
  autonomy only at the leaves + one **bounded re-plan** rung.
- **"Judge every step" was my idea, and the AI initially went along with it.** Before committing I
  had it research the trade-off instead of accepting it. The evidence (free-tier 5 rpm / 20-per-day
  caps + the non-determinism a per-step LLM call injects into the test suite) showed per-step judging
  would be both a budget and a concurrency killer. Corrected to a **selective** judge, fired only on
  a deterministic suspect-signal and off the parallel hot path.
- **AI left the synthesis stage with no quality gate — the same blind spot as the planner.** Its
  outer-loop design guarded planning (DAG validation), execution (agent retries), and step-failure
  (bounded re-plan), but `synthesize` shipped whatever the model returned straight to the user: no
  grounding check, no format check, no confidence calibration, no recovery. I caught it by asking
  *"where is the validation layer for synthesis — and should it be able to retrigger the graph?"* The
  AI's first reflex was to **move the existing `evaluate` node to after synthesis**; I rejected that
  (it's purpose-built for step-failure classification and takes a `failed_id`/`error`, so it can't
  double as a faithfulness judge, and moving it would un-guard execution). We converged through several
  rounds on the design that shipped: a **separate post-synthesis judge** with **3 actions, not 4**
  (rerun folded into replan), deterministic checks first, the cheap remedy (`resynthesize`) as a tight
  edge and the expensive one (`replan`) reusing the shared budget — and I made the AI keep it as **flat
  graph nodes**, not the nested subgraph it proposed, because `replan` has to reach the sibling
  `execute` node and that's clumsy across a subgraph boundary.
- **AI's first code-validation plan over-spent and rubber-stamped itself.** Two defaults were wrong:
  it set the syntax give-up cap to **8 regenerations** (cost-overkill — if the model can't fix syntax
  in 3–4 tries it rarely fixes it later; I cut it to 4), and it ran the LLM critic on the **same
  generator model for every language**, including parser-backed ones. That is both redundant cost and
  a self-approval bias (a model grading its own output). I made it **fallback-only** (parser-backed
  languages never invoke the critic) and moved it to a **cheaper, independent reviewer model at
  temperature 0**. I also collapsed two near-duplicate correction loops it had written into one
  generic loop parameterized by signal + refine functions.
- **AI reached for repo-root CI; the spec forbade it.** To prove the Docker image built, the AI
  added a `.github/workflows/` GitHub Actions pipeline at the repo root. But the assignment fixes
  the submission tree (`README` / `AI_USAGE` / `DECISIONS` / `docker-compose.yml` / `Dockerfile` /
  `app/` / `tests/`) and I had been told not to add `.github/` — *"`.github/` stays at repo root is
  not acceptable."* Relocating it under `app/infrastructure/` was a dead end (Actions only discovers
  workflows at `.github/workflows/`), so I dropped CI entirely and leaned on `docker compose up
  --build` being self-contained. Removing the ignore files then exposed a real hazard: the
  `Dockerfile`'s blanket `COPY app/ ./` would bake my real `app/.env` and the host's **Windows**
  `.venv` over the image's Linux venv. I had the AI rewrite it to copy **only tracked source**
  (`src/`, `main.py`, `cli.py`, `config.yaml`, `evals/`) — a clean tree *and* a secret-free,
  unbroken image from one change, with `.git/info/exclude` preserving the gitignore protection
  invisibly.

## What AI Struggled With

- **External quota/billing limits it can't see.** Because the Gemini free tier caps requests
  per-model-per-day, the eval couldn't fully complete in one sitting (the 5th example,
  the anti-hallucination probe, was a quota casualty rather than a behavioral failure). AI could
  diagnose and route around it (backoff, worker throttling, model fallback) but couldn't make the
  underlying daily limit go away — that ceiling is an account constraint, not a code one.
- **tree-sitter's error model is subtler than AI first assumed.** Its initial JS validator checked
  only `root_node.has_error`, which silently passes some malformed inputs because tree-sitter is
  error-recovering and inserts **MISSING** nodes that `has_error` doesn't always flag. I had it verify
  the API empirically and add a missing-node walk (`is_missing`) on top of `has_error`. It also kept
  describing tree-sitter as an exact gate like `ast`; I made it document the honest limit — a coarser,
  permissive parse that catches gross breakage, not every subtle error.
- **Dependency state it can't see across environments.** AI declared `tree-sitter` in `pyproject.toml`
  but the test runner kept failing on import: the repo had a second, stale `.venv` at the root that
  `uv run` was resolving to, while the real project env lives under `app/`. AI couldn't tell from the
  declared deps that the *active* interpreter lacked the package — it took running `uv sync` in the
  correct project dir and pointing the suite at the `app/` env to fix. An environment fact, not a code
  one.
