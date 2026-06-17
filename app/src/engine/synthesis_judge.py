"""Synthesis quality gate: deterministic checks plus one LLM-as-judge over the final draft.

Synthesis is the only outer-loop stage whose output reaches the user, so it gets the same two-tier
guard as planning: free deterministic checks first (empty content, output-format compliance,
attribution sanity, confidence calibration), then a single structured LLM judge that adjudicates
faithfulness and goal coverage and returns one of three actions — ``accept`` (ship),
``resynthesize`` (cheap retry with feedback, same data) or ``replan`` (fresh work for a gap). The
judge never touches accounting; it only decides the action and authors any replacement steps.
"""

from __future__ import annotations

import asyncio
import json

from langchain_core.language_models.chat_models import BaseChatModel

from app.src.engine import prompts
from app.src.engine.registry import capabilities_catalog
from app.src.engine.synthesizer import Synthesis, _render_outputs
from app.src.general_utils.agent_base import AgentResult, invoke_structured
from app.src.general_utils.llm import build_chat_model
from app.src.schemas.config import AppConfig, get_config
from app.src.schemas.plan import ExecutionPlan, SynthesisVerdict

_BULLET_PREFIXES = ("-", "*", "•")


def _default_judge_model(cfg: AppConfig) -> BaseChatModel:
    """Build the configured synthesis-judge model."""
    orchestrator = cfg.orchestrator
    return build_chat_model(
        orchestrator.judge_model_id,
        orchestrator.judge_temperature,
        cfg.google_api_key.get_secret_value(),
    )


def calibrated_confidence(confidence: float, completed: int, total: int) -> float:
    """Cap reported confidence by the fraction of steps that actually completed.

    Args:
        confidence: The model's self-reported confidence.
        completed: Number of steps that completed successfully.
        total: Total number of steps in the plan.

    Returns:
        ``confidence`` clamped to the completion ratio (0.0 when nothing completed).
    """
    ratio = completed / total if total else 0.0
    return min(confidence, ratio)


def _check_format(content: str, output_format: str) -> list[str]:
    """Check the draft against a lightweight, well-known output-format hint."""
    fmt = output_format.strip().lower()
    if "json" in fmt:
        try:
            json.loads(content)
        except (ValueError, TypeError):
            return ["output_format requested JSON but content is not valid JSON"]
    elif any(word in fmt for word in ("bullet", "list")):
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not any(line.startswith(_BULLET_PREFIXES) for line in lines):
            return ["output_format requested a bullet list but content has no bullet lines"]
    return []


def _check_attribution(content: str, valid_ids: set[str]) -> list[str]:
    """Flag bracketed step citations (e.g. ``[s1]``) that reference no real step."""
    tokens = content.split()
    cited = {tok.strip("[]") for tok in tokens if tok.startswith("[") and "]" in tok}
    unknown = sorted(c for c in cited if c and c not in valid_ids and c.replace("_", "").isalnum())
    return [f"attribution cites unknown step '{cid}'" for cid in unknown]


def check_synthesis(
    synthesis: Synthesis,
    plan: ExecutionPlan,
    completed: int,
    output_format: str | None,
) -> list[str]:
    """Run the free, no-LLM checks over a synthesis draft.

    Args:
        synthesis: The draft (content + confidence) to inspect.
        plan: The executed plan (for valid step ids).
        completed: Number of steps that completed successfully.
        output_format: Optional requested output format hint.

    Returns:
        A list of problem strings (empty when the draft passes every check).
    """
    errors: list[str] = []
    if completed > 0 and not synthesis.content.strip():
        errors.append("content is empty despite completed steps")
    if output_format:
        errors.extend(_check_format(synthesis.content, output_format))
    errors.extend(_check_attribution(synthesis.content, plan.step_ids()))
    return errors


async def judge_synthesis(
    goal: str,
    plan: ExecutionPlan,
    results: dict[str, AgentResult],
    synthesis: Synthesis,
    det_errors: list[str],
    model: BaseChatModel | None = None,
) -> tuple[SynthesisVerdict, int]:
    """Adjudicate a synthesis draft for faithfulness and goal coverage.

    Args:
        goal: The untrusted task goal (fenced as data).
        plan: The executed plan.
        results: Completed results keyed by step id.
        synthesis: The draft to judge.
        det_errors: Deterministic findings to surface to the judge.
        model: Optional injected judge model (defaults to the configured Gemini model).

    Returns:
        A tuple of (SynthesisVerdict, tokens used).
    """
    model = model or _default_judge_model(get_config())
    messages = prompts.judge_messages(
        goal, _render_outputs(plan, results), synthesis.content, det_errors, capabilities_catalog()
    )
    verdict, tokens = await asyncio.to_thread(
        invoke_structured, model, SynthesisVerdict, messages
    )
    return verdict, tokens
