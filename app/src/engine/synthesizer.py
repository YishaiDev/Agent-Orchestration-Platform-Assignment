"""Synthesizer: combine completed step outputs into one coherent, attributed final answer.

The LLM call is told to reconcile conflicts (prefer higher confidence / more sources) and note
disagreements rather than concatenate. Provenance and run totals are assembled deterministically by
the engine from the monitor — the model produces only the prose and an overall confidence — so the
final result's attribution and accounting never depend on the model getting them right.
"""

from __future__ import annotations

import asyncio
import json

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from app.src.engine import prompts
from app.src.engine.monitor import RunMonitor
from app.src.general_utils.agent_base import AgentResult, invoke_structured
from app.src.general_utils.llm import build_chat_model
from app.src.schemas.config import AppConfig, get_config
from app.src.schemas.plan import ExecutionPlan, ExecutionStep, StepStatus
from app.src.schemas.run_state import FinalResult, ProvenanceEntry

_OUTPUT_CHARS = 1200
_LOST_STATES = {StepStatus.SKIPPED, StepStatus.CANCELLED}
_FALLBACK_PREAMBLE = "[Auto-assembled from completed steps; the synthesis step was unavailable.]"
_FALLBACK_CONFIDENCE = 0.3


class Synthesis(BaseModel):
    """The synthesizer's structured output: the final prose plus an overall confidence."""

    content: str = Field(description="The coherent, attributed answer to the goal.")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in the answer.")


def _default_synth_model(cfg: AppConfig) -> BaseChatModel:
    """Build the configured synthesizer model."""
    orchestrator = cfg.orchestrator
    return build_chat_model(
        orchestrator.synthesizer_model_id,
        orchestrator.synthesizer_temperature,
    )


def _render_block(step: ExecutionStep, result: AgentResult) -> str:
    """Render one completed step as a confidence-tagged block, preserving any code verbatim."""
    confidence = result.output.get("confidence")
    text = result.output.get("content") or json.dumps(result.output, default=str)
    block = (
        f"[{step.id} | {result.agent}/{step.action} | confidence={confidence}] "
        f"{str(text)[:_OUTPUT_CHARS]}"
    )
    code = result.output.get("code")
    if not code:
        return block
    language = result.output.get("language") or ""
    return f"{block}\n```{language}\n{code}\n```"


def _render_outputs(plan: ExecutionPlan, results: dict[str, AgentResult]) -> str:
    """Render completed, confidence-tagged outputs for the synthesis prompt."""
    blocks = [
        _render_block(step, result)
        for step in plan.steps
        if (result := results.get(step.id)) is not None and result.status == "completed"
    ]
    return "\n\n".join(blocks)


def _writing_content(plan: ExecutionPlan, results: dict[str, AgentResult]) -> str:
    """Return the writing agent's content if a writing step completed, else an empty string."""
    for step in plan.steps:
        result = results.get(step.id)
        if result is not None and result.status == "completed" and result.agent == "writing":
            return str(result.output.get("content") or "")
    return ""


def _all_code_blocks(plan: ExecutionPlan, results: dict[str, AgentResult]) -> str:
    """Render every completed code-bearing step as a fenced block, verbatim."""
    blocks = []
    for step in plan.steps:
        result = results.get(step.id)
        if result is None or result.status != "completed" or not result.output.get("code"):
            continue
        language = result.output.get("language") or ""
        blocks.append(f"```{language}\n{result.output['code']}\n```")
    return "\n\n".join(blocks)


def fallback_synthesis(plan: ExecutionPlan, results: dict[str, AgentResult]) -> Synthesis:
    """Deterministic best-effort answer used when the synthesizer LLM call fails terminally.

    Leads with the writing agent's already-coherent prose when present (appending code blocks the
    prose omits), and otherwise stitches all completed outputs (which already carry code fences).
    Confidence is held low and calibrated down later, so a fallback never claims full confidence.

    Args:
        plan: The executed plan.
        results: Completed results keyed by step id.

    Returns:
        A low-confidence Synthesis assembled without any model call.
    """
    lead = _writing_content(plan, results)
    parts = [_FALLBACK_PREAMBLE, lead, _all_code_blocks(plan, results)] if lead else [
        _FALLBACK_PREAMBLE,
        _render_outputs(plan, results),
    ]
    return Synthesis(content="\n\n".join(p for p in parts if p), confidence=_FALLBACK_CONFIDENCE)


async def synthesize(
    goal: str,
    plan: ExecutionPlan,
    results: dict[str, AgentResult],
    model: BaseChatModel | None = None,
    feedback: str | None = None,
) -> tuple[Synthesis, int]:
    """Synthesize the final answer from completed step outputs.

    Args:
        goal: The untrusted task goal (fenced as data).
        plan: The executed plan.
        results: Completed results keyed by step id.
        model: Optional injected synthesizer model (defaults to the configured Gemini model).
        feedback: Optional judge feedback that turns this into a corrective re-synthesis pass.

    Returns:
        A tuple of (Synthesis, tokens used).
    """
    model = model or _default_synth_model(get_config())
    messages = prompts.synth_messages(goal, _render_outputs(plan, results))
    if feedback:
        messages = [*messages, prompts.synth_repair_message(feedback)]
    synthesis, tokens = await asyncio.to_thread(invoke_structured, model, Synthesis, messages)
    return synthesis, tokens


def _provenance(monitor: RunMonitor) -> list[ProvenanceEntry]:
    """Build deterministic per-step provenance from the monitor's plan and results."""
    if monitor.plan is None:
        return []
    entries = []
    for step in monitor.plan.steps:
        result = monitor.results.get(step.id)
        if result is None:
            continue
        entries.append(
            ProvenanceEntry(
                step_id=step.id,
                agent=result.agent,
                action=step.action,
                status=result.status,
                confidence=result.output.get("confidence"),
                sources=result.output.get("sources") or [],
            )
        )
    return entries


def _steps_in(monitor: RunMonitor, states: set[StepStatus]) -> list[str]:
    """Return step ids whose live status is in ``states``."""
    return [sid for sid, status in monitor.step_status.items() if status in states]


def build_final_result(monitor: RunMonitor, synthesis: Synthesis, status: str) -> FinalResult:
    """Assemble the final result with provenance and run totals.

    Args:
        monitor: The run monitor holding plan, results, status, and totals.
        synthesis: The synthesizer's prose and confidence.
        status: The task-level status string to stamp on the result.

    Returns:
        A populated FinalResult.
    """
    elapsed_ms = int((monitor.updated_at - monitor.created_at).total_seconds() * 1000)
    return FinalResult(
        task_id=monitor.task_id,
        status=status,
        content=synthesis.content,
        confidence=synthesis.confidence,
        provenance=_provenance(monitor),
        failed_steps=_steps_in(monitor, {StepStatus.FAILED}),
        skipped_steps=_steps_in(monitor, _LOST_STATES),
        total_tokens=monitor.total_tokens,
        total_cost_usd=round(monitor.total_cost_usd, 6),
        total_time_ms=elapsed_ms,
    )
