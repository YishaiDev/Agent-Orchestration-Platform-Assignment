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

These are the working practices — not the tools — that moved this assignment forward.

- **Plan mode as the default gate.** Every non-trivial change went through plan mode first: I read the
  proposed approach, confirmed it was the path I actually wanted, and only then let it execute. Immediately
  after, I had it summarize what it had done, so I could verify the result matched the intent before moving
  on — design, confirm, execute, then check.
- **Prompting against sycophancy when I needed real advice.** When I wanted a genuine judgment rather than
  agreement, I said so in the prompt — to challenge me and push back, not just satisfy me — so the answer was
  an actual recommendation instead of an echo of what I'd proposed.
- **A self-critique judge per task.** For anything that had to be exactly right, I had AI build a judge that
  criticized its own output against everything I had specified, confirming the work covered the full
  requirement rather than assuming it did.
- **Tests as the documented, executable form of the logic.** Because the code implements real logic, I
  treated the test suite as how that logic is both written down and proven — the flow that guarantees the
  load-bearing paths keep working as the system changes.
- **Evals after every feature.** Once a feature was finished I had AI create and run an eval over it and
  produce a report, so I could measure its quality and cost instead of guessing.
- **Automatic prompt engineering driven by the evals.** AI is strong at prompt engineering, so I closed the
  loop: it analyzed the eval results and revised the prompts and approach accordingly. I leaned on this most
  for the planning prompts, early in the assignment.
- **The orchestrator agent as a manager on complex tasks.** For complex work I treated the Claude Code agent
  as an orchestrator — a manager that spun up multiple subagents per task and coordinated them. Delegating
  the heavy reading and execution to subagents kept the main context clean and focused.
- **A 25%-context auto-compact hook.** I set a hook that auto-compacts at the 25%-context threshold, both to
  cut token cost and to keep each conversation tight around the context that still mattered.
- **One session per topic.** I scoped a separate session to each topic so its context stayed exact; finishing
  a piece of work in its own session let me return to it later without dragging stale context along.

## What I Had to Fix

- **Plan mode is a long multi-planning loop, not a one-shot.** Getting the right context into the model
  before it implemented anything took real time and constant correction — I refined the plan, caught what it
  had wrong, and refined again. That iteration is the point, not overhead: paying it up front is what kept the
  implementation on the path I actually intended.
- **Wiring the three small-model tiers and trusting their output was the real challenge.** This app wasn't
  designed around LLM usage to begin with, so I couldn't be certain the three-tier setup was intended — but it
  behaved as though it was, so I kept it and weighed each tier carefully. To trust the outputs rather than
  assume them, I leaned on an LLM-as-judge to confirm they were genuinely accurate, not merely plausible.
- **I re-read and validated every regenerated batch of code myself.** After AI regenerated code I went over
  it, understood the load-bearing parts, validated them, fixed what was off, and questioned its choices. This
  is how I closed two gaps the model can't close alone: between what it produced and what the requirement
  actually was, and between what I intended and what it understood. Reviewing the code every time is what
  surfaced the things it had silently assumed without ever raising them.

## What AI Struggled With

- ** Validating itself from a certion point- a humen must review the code.