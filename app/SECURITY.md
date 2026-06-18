# Security

How the platform handles untrusted input, secrets, LLM-output validation, and prompt-injection.

## Untrusted input (goal, constraints)

Everything enters through one Pydantic gate, `api/models.py::TaskRequest`: typed, bounded fields
(`goal` non-empty, `max_replans ≥ 0`, `deadline_seconds > 0`), so malformed bodies are rejected `422`
before any engine code runs. `constraints` (string **or** JSON object) is normalized to text, never
`eval`/`exec`'d. Goal and constraints are treated as **data, not instructions** — they reach a prompt
only inside a fence (see below). No user input touches a shell, and the Code agent parses/judges
generated code rather than executing it.

## API keys & secrets

All keys are `SecretStr` on `AppConfig` (`schemas/config.py`), loaded from `app/.env` (gitignored;
`.env.example` is the committed template) — so logs, reprs, and exceptions render `**********`. The raw
value is unwrapped at exactly one site (`general_utils/llm.py::_resolve_provider_key`,
`.get_secret_value()`) and handed straight to `init_chat_model`. It never reaches the execution trace,
the `/result` envelope, or any log line.

## Validating LLM output before execution

Two layers, and a plan is **never trusted just because it parsed**:

1. **Type** — every model call is forced into a Pydantic schema via `with_structured_output`
   (`general_utils/agent_base.py::invoke_structured`), with bounded retry on a parse failure.
2. **Structure** — `engine/validation.py` runs deterministic, no-LLM checks before a single agent
   runs: unique step ids, every `agent/action` in the registry **allowlist**, dependencies resolve,
   no self-deps, and **Kahn's algorithm rejects any cycle**. A bad plan raises `PlanValidationError`
   and never executes; the allowlist means the model cannot invent an agent, action, or tool.

## Prompt-injection

The system prompt is the only authority; all untrusted text is wrapped as data by
`fence(label, body)` (`sub_agents/_prompt_utils.py`), and the system prompts instruct the model to
treat fenced content as data. This is applied across the planner, the four sub-agents, **and** the
synthesizer/judge — which fence each upstream agent output before re-feeding it to an LLM, since an
agent output can itself carry injected text from a fetched page. (The Writing agent's `instruction` is
the one intentional non-fence: it is the legitimate directive.)

## What's missing — a dedicated guardrail layer

The above is data-boundary discipline plus the hard deterministic allowlist/DAG gate. There is no
separate input/output guardrail stage — an injection/jailbreak classifier on the goal, or a
moderation/PII pass on agent and final outputs. A thin guardrail layer wrapping these boundaries would
have been a good addition and is the natural next security step beyond the assignment's scope.
