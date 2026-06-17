"""Dispatch: route one step to its agent and inject upstream outputs as grounding.

Each agent has a different entrypoint and input shape, so dispatch adapts the generic
:class:`ExecutionStep` to the concrete call and folds completed upstream outputs into the field that
agent consumes (analysis ``data``, code/writing context, research subtopic). Upstream text is
trimmed to a deterministic character budget before injection — quota-free context management that
keeps each first call bounded while the agents' own compaction handles overflow during their loop.

Agents never raise (they return ``failed`` results), so dispatch only guards against unexpected
mapping errors with ``except Exception`` — which, being narrower than ``BaseException``, lets a
scheduler ``CancelledError`` propagate cleanly for preemptive re-plans.
"""

from __future__ import annotations

import asyncio
import json

from app.src.general_utils.agent_base import AgentResult
from app.src.schemas.config import get_config
from app.src.schemas.plan import ExecutionStep
from app.src.sub_agents.analysis import run_analysis_agent
from app.src.sub_agents.code import run_code_agent
from app.src.sub_agents.research.agent import run_research_agent
from app.src.sub_agents.writing import WritingInput, run_writing_agent


def _completed_upstream(
    step: ExecutionStep, results: dict[str, AgentResult]
) -> list[AgentResult]:
    """Return upstream dependency results that completed successfully."""
    return [
        results[dep]
        for dep in step.dependencies
        if dep in results and results[dep].status == "completed"
    ]


def _summarize(result: AgentResult) -> str:
    """Render one upstream output as a labeled text block."""
    output = result.output
    text = output.get("content") or output.get("summary") or json.dumps(output, default=str)
    return f"[{result.step_id}/{result.agent}] {text}"


def _build_context(upstream: list[AgentResult]) -> str:
    """Concatenate and trim upstream outputs to the configured character budget."""
    budget = get_config().orchestrator.context_char_budget
    joined = "\n\n".join(_summarize(result) for result in upstream)
    return joined[:budget]


def _text(step: ExecutionStep, key: str, fallback: str = "") -> str:
    """Read a string input field from the step, falling back when absent."""
    value = step.input.get(key)
    return str(value) if value not in (None, "") else fallback


def _analysis_data(step: ExecutionStep, upstream: list[AgentResult]) -> object:
    """Resolve the analysis ``data`` argument from explicit input or upstream outputs."""
    if "data" in step.input:
        return step.input["data"]
    return [result.output for result in upstream] or None


def _collect_sources(upstream: list[AgentResult]) -> list[str] | None:
    """Gather any ``sources`` lists carried by upstream outputs."""
    sources = [src for result in upstream for src in (result.output.get("sources") or [])]
    return sources or None


async def _route(
    step: ExecutionStep, upstream: list[AgentResult], context: str, session_id: str
) -> AgentResult:
    """Invoke the agent named by the step with adapted, context-grounded inputs."""
    if step.agent == "research":
        subtopic = _text(step, "subtopic", fallback=_text(step, "instruction", context))
        return await run_research_agent(subtopic, step_id=step.id, session_id=session_id)
    if step.agent == "analysis":
        return await run_analysis_agent(
            instruction=_text(step, "instruction"),
            action=step.action,  # type: ignore[arg-type]
            data=_analysis_data(step, upstream),
            sources=_collect_sources(upstream),
            step_id=step.id,
            session_id=session_id,
        )
    if step.agent == "code":
        return await run_code_agent(
            task_input=_text(step, "task_input", fallback=_text(step, "instruction")),
            action=step.action,
            step_id=step.id,
            language=step.input.get("language"),  # type: ignore[arg-type]
            upstream_context=context,
        )
    inp = WritingInput(
        instruction=_text(step, "instruction"),
        source_material=context,
        constraints=step.input.get("constraints", {}),  # type: ignore[arg-type]
        output_format=str(step.input.get("output_format", "markdown")),
    )
    return await asyncio.to_thread(run_writing_agent, inp, step.id)


def _failed(step: ExecutionStep, error: str) -> AgentResult:
    """Build a ``failed`` result for an unexpected dispatch-level error."""
    return AgentResult(
        step_id=step.id,
        agent=step.agent,
        status="failed",
        output={"error": error},
        tokens_used=0,
        execution_time_ms=0,
    )


async def dispatch(
    step: ExecutionStep, results: dict[str, AgentResult], session_id: str = "local"
) -> AgentResult:
    """Execute one step against its agent, grounding it in upstream outputs.

    Args:
        step: The step to run.
        results: Completed results so far, keyed by step id (for dependency injection).
        session_id: Session identifier carried into the agent call.

    Returns:
        The agent's :class:`AgentResult`; a ``failed`` result on unexpected mapping errors.
    """
    upstream = _completed_upstream(step, results)
    context = _build_context(upstream)
    try:
        return await _route(step, upstream, context, session_id)
    except Exception as exc:  # noqa: BLE001 - agents never raise; this guards mapping errors only
        return _failed(step, str(exc))
