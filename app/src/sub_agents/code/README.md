# Code Agent

The 4th specialized agent. Does three text-producing tasks over a `language` param:
**generate**, **explain**, **debug**. No execution — it mirrors the Writing agent shape
(one structured LLM call → typed `CodeOutput` → uniform `AgentResult`).

## Why no execution

Running untrusted LLM-generated code from a goal string would be the largest attack surface in
the project and would hurt the graded Security dimension. The agent has **no sandbox, no tool loop,
no shell/filesystem/network** — zero blast radius. Untrusted input is fenced as data so injected
text can't redirect it. Both gate tiers below only **parse or read** the code — never run it.

## Quality gate — two tiers

Correctness is gated without an LLM-judge rubber stamp. The tier is chosen per language by
`has_validator(language)`:

### Tier 1 — deterministic parser (ground truth)

Used for any language with a registered parser. A real parse error is unambiguous and feeds a
bounded correction loop: on an error the agent fires a refine call with the error fed back, then
re-parses, up to `max_syntax_retries` (default 4).

- **Python** — stdlib `ast.parse`: exact, catches every syntax error.
- **JavaScript** — `tree-sitter` + `tree-sitter-javascript`: a parse fails when
  `root_node.has_error` is true **or any node `is_missing`** (the missing-node walk catches inserted
  MISSING nodes that `has_error` alone can skip). tree-sitter is **error-recovering, so this is a
  coarser gate than `ast`** — it reliably catches gross breakage (unbalanced braces/parens, dangling
  tokens), not every subtle error.

**Graceful give-up:** if the code still won't parse after the retry budget, the agent returns the
**best-effort code, not a failure** — status stays `completed` and the output records the unverified
state (`parses: false` + `validation_error`). Downstream consumers decide what to do with it.

### Tier 2 — LLM critic (fallback only)

Used **only for languages with no registered parser** (e.g. Ruby, Go-without-grammar) — the only
signal available where a parser can't reach. A cheaper, **independent** reviewer model
(`review_model_id`, default `gemini-2.5-flash`, temperature 0) reads the code and returns a
`CodeVerdict{verdict: revise|return, issues}`. `revise` recalls the generator with the concrete
issues; `return` accepts. Bounded by `max_review_retries` (default 1). Independence (different model,
temp 0) avoids the self-approval bias of a model grading its own output. Parser-backed languages
**never** invoke the critic, preserving ground-truth-first with no added cost.

## Output keys

`output` always carries `{content, code, language}` plus the gate result:

| Key | Meaning |
|-----|---------|
| `parses` | `true` / `false` after a Tier-1 parse; `null` when no parser exists (Tier-2 or trusted) |
| `validation_error` | present only when `parses` is `false` — the parser's message |

`parses` is the **single source of truth**: the eval judge (`code_judge.py`) reads it from the
output instead of recomputing, so the agent and the eval can never drift.

## Files

| File | Responsibility |
|------|----------------|
| `agent.py` | `run_code_agent(...)` entrypoint + one generic `_correct` loop; never raises |
| `schemas.py` | `CodeInput` / `CodeOutput`, `CodeVerdict`, `Action`, tolerant `coerce_action` |
| `prompts.py` | action-dispatched, injection-safe message builders (generate/refine/critic) |
| `validation.py` | language-alias normalization, per-language parser registry, `has_validator` |

## Usage

```python
from app.src.sub_agents.code.agent import run_code_agent

result = await run_code_agent(
    "Write a function that returns the nth Fibonacci number iteratively.",
    action="generate",     # generate | explain | debug (off-vocab -> generate)
    language="python",     # steers the prompt and picks the gate tier
    step_id="step_3",
)
# result.output -> {"content": ..., "code": ..., "language": ..., "parses": True}
```

## Multi-language support

`language` is first-class: it steers the prompt, is echoed in the output, and selects the gate.
Short names from the planner are normalized first (`js`/`node`/`jsx` → `javascript`,
`py`/`python3` → `python`) via `_normalize_lang`, so the validators actually fire. Parsers live in
`validation.py::_VALIDATORS` — currently **Python** (`ast`) and **JavaScript** (tree-sitter). Any
other language falls through to the Tier-2 critic.

### Adding a parser for another language

Touch only the registry — the agent code never changes. tree-sitter scales to many languages with
one mechanism: `uv add tree-sitter-typescript` (or `-go`, `-rust`, …), then register a validator
that flags `root_node.has_error` / missing nodes, mirroring `_js_syntax_error`:

```python
_VALIDATORS["typescript"] = _ts_syntax_error  # same has_error + missing-node walk
```

Registering a language moves it from the Tier-2 critic to the deterministic Tier-1 gate
automatically.

## Tests & eval

- Unit: `tests/sub_agents/test_code_agent.py` (offline, mocked Gemini) — runnable standalone or via
  `tests/run_all_tests.py`. Covers both tiers, alias normalization, graceful give-up, the
  parser-vs-critic routing, fencing, and generator-vs-reviewer cost accounting.
- Eval: `evals/judges/code_judge.py` + `evals/datasets/code_agent.yaml` (live LLM judge that reads
  the `parses` signal from the agent output).
