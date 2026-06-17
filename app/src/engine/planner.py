"""Planner: turn an untrusted goal into a validated, executable plan.

A single reasoning-first structured call (strong model, native thinking) produces a
:class:`PlannerDraft`; deterministic validation then lifts it into an :class:`ExecutionPlan` or
feeds the errors back for one bounded repair attempt. There is no LLM judge here — a runnable plan
is guaranteed structurally, and a poor plan is corrected reactively by the bounded re-plan.
"""

from __future__ import annotations

import asyncio

from langchain_core.language_models.chat_models import BaseChatModel

from app.src.engine import prompts
from app.src.engine.registry import capabilities_catalog
from app.src.engine.validation import PlanValidationError, validate_and_finalize
from app.src.general_utils.agent_base import invoke_structured
from app.src.general_utils.llm import build_chat_model
from app.src.schemas.config import AppConfig, get_config
from app.src.schemas.plan import ExecutionPlan, PlannerDraft


def _default_planner_model(cfg: AppConfig) -> BaseChatModel:
    """Build the configured planner model from the app config.

    Args:
        cfg: The loaded application config.

    Returns:
        An initialized chat model for planning.
    """
    orchestrator = cfg.orchestrator
    return build_chat_model(
        orchestrator.planner_model_id,
        orchestrator.planner_temperature,
        cfg.google_api_key.get_secret_value(),
    )


async def build_plan(
    goal: str, constraints: str, task_id: str, model: BaseChatModel | None = None
) -> tuple[ExecutionPlan, int]:
    """Plan a goal into a validated DAG, with one bounded repair attempt on validation failure.

    Args:
        goal: The untrusted task goal (fenced as data in the prompt).
        constraints: Optional untrusted constraints text.
        task_id: Engine-assigned task id stamped onto the plan.
        model: Optional injected planner model (defaults to the configured Gemini model).

    Returns:
        A tuple of (validated ExecutionPlan, total planner tokens used).

    Raises:
        PlanValidationError: When every bounded attempt produces an invalid plan.
    """
    cfg = get_config()
    model = model or _default_planner_model(cfg)
    messages = prompts.initial_messages(goal, constraints, capabilities_catalog())
    total_tokens = 0
    errors: list[str] = []
    for _ in range(cfg.orchestrator.planner_max_attempts):
        draft, tokens = await asyncio.to_thread(invoke_structured, model, PlannerDraft, messages)
        total_tokens += tokens
        try:
            return validate_and_finalize(draft, task_id), total_tokens
        except PlanValidationError as exc:
            errors = exc.errors
            messages = messages + [prompts.repair_message(errors)]
    raise PlanValidationError(errors)
