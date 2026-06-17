"""Re-plan evaluation: the LLM decider plus the deterministic merge of its replacement steps.

The decider is one structured call given the whole current plan, the frozen completed outputs, and
the failed step + error; it returns ``continue`` (the remaining work still reaches the goal) or
``replan`` with replacement steps for the unfinished part. The merge is pure code: completed steps
stay frozen, new steps are namespaced under a reserved ``r{n}_`` prefix (intra-batch dependencies
rewritten, collisions rejected), and the merged plan is re-validated as a DAG before it runs.
"""

from __future__ import annotations

import asyncio
import json

from langchain_core.language_models.chat_models import BaseChatModel

from app.src.engine import prompts
from app.src.engine.monitor import skip_cascade
from app.src.engine.registry import capabilities_catalog
from app.src.engine.validation import PlanValidationError, revalidate
from app.src.general_utils.agent_base import AgentResult, invoke_structured
from app.src.general_utils.llm import build_chat_model
from app.src.schemas.config import AppConfig, get_config
from app.src.schemas.plan import (
    ExecutionPlan,
    ExecutionStep,
    ReplanDecision,
    StepStatus,
)

_SUMMARY_CHARS = 800


def _default_decider_model(cfg: AppConfig) -> BaseChatModel:
    """Build the configured re-plan decider model."""
    orchestrator = cfg.orchestrator
    return build_chat_model(
        orchestrator.decider_model_id,
        orchestrator.decider_temperature,
        cfg.google_api_key.get_secret_value(),
    )


def _completed_summary(plan: ExecutionPlan, results: dict[str, AgentResult]) -> str:
    """Render completed steps and trimmed outputs for the decider prompt."""
    lines = []
    for step in plan.steps:
        result = results.get(step.id)
        if result is None or result.status != "completed":
            continue
        text = result.output.get("content") or json.dumps(result.output, default=str)
        lines.append(f"{step.id} ({step.agent}/{step.action}): {str(text)[:_SUMMARY_CHARS]}")
    return "\n".join(lines)


def _failure_text(plan: ExecutionPlan, failed_id: str, error: str) -> str:
    """Render the failed step and the steps lost with it for the decider prompt."""
    lost = sorted(skip_cascade(plan, failed_id))
    step = plan.step_by_id(failed_id)
    agent = f"{step.agent}/{step.action}" if step else "unknown"
    return f"failed step {failed_id} ({agent}); error: {error}; downstream lost: {lost or 'none'}"


async def decide_replan(
    plan: ExecutionPlan,
    goal: str,
    results: dict[str, AgentResult],
    failed_id: str,
    error: str,
    model: BaseChatModel | None = None,
) -> tuple[ReplanDecision, int]:
    """Ask the decider whether to continue or re-plan the unfinished work.

    Args:
        plan: The current plan.
        goal: The untrusted task goal (fenced as data).
        results: Completed results so far, keyed by step id.
        failed_id: The id of the step that triggered the decision.
        error: The failure error text.
        model: Optional injected decider model (defaults to the configured Gemini model).

    Returns:
        A tuple of (ReplanDecision, tokens used).
    """
    model = model or _default_decider_model(get_config())
    messages = prompts.decider_messages(
        goal,
        _completed_summary(plan, results),
        _failure_text(plan, failed_id, error),
        capabilities_catalog(),
    )
    decision, tokens = await asyncio.to_thread(invoke_structured, model, ReplanDecision, messages)
    return decision, tokens


def _renamed(step: ExecutionStep, rename: dict[str, str]) -> ExecutionStep:
    """Namespace a new step's id and rewrite intra-batch dependency references."""
    new_deps = [rename.get(dep, dep) for dep in step.dependencies]
    return step.model_copy(update={"id": rename[step.id], "dependencies": new_deps})


def merge_replan(
    plan: ExecutionPlan,
    decision: ReplanDecision,
    step_status: dict[str, StepStatus],
    round_no: int,
) -> ExecutionPlan:
    """Merge the decider's replacement steps with the frozen completed steps.

    Args:
        plan: The plan being revised.
        decision: The decider verdict carrying ``new_steps``.
        step_status: Live status per step id (completed steps are kept).
        round_no: 1-based re-plan round, used for the reserved id prefix.

    Returns:
        A re-validated merged ExecutionPlan with recomputed parallel groups.

    Raises:
        PlanValidationError: On id collisions or any structural/reference/cycle failure.
    """
    completed = [s for s in plan.steps if step_status.get(s.id) == StepStatus.COMPLETED]
    existing_ids = {s.id for s in completed}
    prefix = f"r{round_no}_"
    rename = {s.id: f"{prefix}{s.id}" for s in decision.new_steps}
    collisions = [nid for nid in rename.values() if nid in existing_ids]
    if collisions:
        raise PlanValidationError([f"re-plan id collision: {nid}" for nid in collisions])
    namespaced = [_renamed(step, rename) for step in decision.new_steps]
    merged = ExecutionPlan(
        reasoning=f"{plan.reasoning}\n\n[replan {round_no}] {decision.reasoning}",
        task_id=plan.task_id,
        steps=completed + namespaced,
    )
    return revalidate(merged)
