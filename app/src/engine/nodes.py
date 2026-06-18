"""LangGraph outer-loop nodes: plan -> execute -> evaluate -> synthesize -> judge, plus routers.

The nodes are thin: they fetch the shared monitor from the registry, call one engine service, and
return a minimal state delta. Two LLM-routed edges exist: ``evaluate`` -> ``execute`` (re-plan) on a
structural step-failure, and ``judge`` -> {``synthesize`` (re-synthesize), ``execute`` (replan),
END (accept)} on the synthesis quality verdict. Both share the ``max_replans`` budget. Engine
dependencies (models, step runner, registry) are injected through the run config so the whole loop
is testable offline with fakes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from app.src.engine.evaluation import decide_replan, merge_replan
from app.src.engine.monitor import RunMonitor
from app.src.engine.planner import build_plan
from app.src.engine.runs import RunRegistry
from app.src.engine.scheduler import StepRunner, execute_plan
from app.src.engine.synthesis_judge import calibrated_confidence, check_synthesis, judge_synthesis
from app.src.engine.synthesizer import (
    Synthesis,
    build_final_result,
    fallback_synthesis,
    synthesize,
)
from app.src.engine.validation import PlanValidationError
from app.src.schemas.plan import ExecutionPlan, ReplanDecision, SynthesisVerdict, TaskState
from app.src.schemas.run_state import RunState

logger = logging.getLogger("app.engine.nodes")


@dataclass
class EngineDeps:
    """Injected engine dependencies threaded through the run config."""

    registry: RunRegistry
    runner: StepRunner | None = None
    planner_model: BaseChatModel | None = None
    decider_model: BaseChatModel | None = None
    synth_model: BaseChatModel | None = None
    judge_model: BaseChatModel | None = None
    concurrency: int | None = None


def _deps(config: RunnableConfig) -> EngineDeps:
    """Extract the injected engine dependencies from the run config."""
    return cast(EngineDeps, config["configurable"]["deps"])


def _monitor(config: RunnableConfig, state: RunState) -> RunMonitor:
    """Fetch the shared monitor for the current task."""
    monitor = _deps(config).registry.get(state["task_id"])
    if monitor is None:
        raise RuntimeError(f"no monitor registered for task {state['task_id']}")
    return monitor


def _require_plan(monitor: RunMonitor) -> ExecutionPlan:
    """Return the monitor's active plan, or fail loudly if a node ran before planning."""
    if monitor.plan is None:
        raise RuntimeError(f"task {monitor.task_id} has no plan attached")
    return monitor.plan


async def plan_node(state: RunState, config: RunnableConfig) -> dict[str, object]:
    """Plan the goal into a validated DAG; route to synthesize on planning failure."""
    deps = _deps(config)
    monitor = _monitor(config, state)
    monitor.set_state(TaskState.PLANNING)
    try:
        plan, tokens = await build_plan(
            state["goal"], state["constraints"], state["task_id"], model=deps.planner_model
        )
    except PlanValidationError as exc:
        return {"error": str(exc)}
    monitor.attach_plan(plan)
    monitor.total_tokens += tokens
    return {}


def route_after_plan(state: RunState) -> str:
    """Route to execution when planning succeeded, else straight to synthesis (failed run)."""
    return "synthesize" if state.get("error") else "execute"


async def execute_node(state: RunState, config: RunnableConfig) -> dict[str, object]:
    """Run the current plan's DAG via the inner scheduler (continuous concurrency)."""
    deps = _deps(config)
    monitor = _monitor(config, state)
    monitor.set_state(TaskState.EXECUTING)
    await execute_plan(
        _require_plan(monitor),
        monitor,
        runner=deps.runner,
        concurrency=deps.concurrency,
        session_id=state.get("session_id", "local"),
    )
    return {}


async def _decide_and_merge(
    state: RunState, deps: EngineDeps, monitor: RunMonitor
) -> dict[str, object]:
    """Consult the LLM decider and merge replacement steps on a re-plan verdict."""
    plan = _require_plan(monitor)
    decision, tokens = await decide_replan(
        plan,
        state["goal"],
        monitor.results,
        monitor.failed_step_id or "",
        monitor.failure_error or "",
        model=deps.decider_model,
    )
    monitor.total_tokens += tokens
    round_no = state.get("replans", 0) + 1
    if decision.decision == "continue":
        monitor.clear_replan()
        return {"decision": "continue"}
    try:
        merged = merge_replan(plan, decision, monitor.step_status, round_no)
    except PlanValidationError:
        monitor.clear_replan()
        return {"decision": "continue"}
    monitor.clear_replan()
    monitor.attach_plan(merged)
    return {"decision": "replan", "replans": round_no}


async def evaluate_node(state: RunState, config: RunnableConfig) -> dict[str, object]:
    """On a structural failure, decide continue vs bounded re-plan; otherwise pass through."""
    deps = _deps(config)
    monitor = _monitor(config, state)
    if not monitor.replan_requested:
        return {"decision": "continue"}
    if state.get("replans", 0) >= state.get("max_replans", 1):
        monitor.clear_replan()
        return {"decision": "continue"}
    return await _decide_and_merge(state, deps, monitor)


def route_after_evaluate(state: RunState) -> str:
    """Loop back to execute on a re-plan decision, else proceed to synthesis."""
    return "execute" if state.get("decision") == "replan" else "synthesize"


def _final_status(monitor: RunMonitor) -> str:
    """Derive the task-level status from the monitor's terminal state."""
    if monitor.cancelled:
        return "cancelled"
    if monitor.completed_count() == 0:
        return "failed"
    return "completed"


def _draft(monitor: RunMonitor) -> Synthesis:
    """Reconstruct the latest synthesis draft from the monitor (empty when absent)."""
    return Synthesis(**(monitor.draft or {"content": "", "confidence": 0.0}))


async def synthesize_node(state: RunState, config: RunnableConfig) -> dict[str, object]:
    """Produce a synthesis draft (feedback-aware on re-synthesis passes); the judge node decides."""
    deps = _deps(config)
    monitor = _monitor(config, state)
    if monitor.plan is None or monitor.completed_count() == 0:
        monitor.set_draft(Synthesis(content="", confidence=0.0).model_dump())
        return {}
    try:
        synthesis, tokens = await synthesize(
            state["goal"],
            monitor.plan,
            monitor.results,
            model=deps.synth_model,
            feedback=state.get("synth_feedback") or None,
        )
    except Exception:
        logger.exception("synthesis call failed; falling back to deterministic assembly")
        monitor.set_draft(fallback_synthesis(monitor.plan, monitor.results).model_dump())
        return {"synth_failed": True}
    monitor.total_tokens += tokens
    monitor.set_draft(synthesis.model_dump())
    return {}


def _accept(
    monitor: RunMonitor, synthesis: Synthesis, degraded: bool
) -> dict[str, object]:
    """Calibrate confidence, build the final result, and stamp terminal state."""
    base = _final_status(monitor)
    status = "completed_degraded" if degraded and base == "completed" else base
    final_synth = synthesis
    if base == "completed" and monitor.plan is not None:
        calibrated = calibrated_confidence(
            synthesis.confidence, monitor.completed_count(), len(monitor.plan.steps)
        )
        final_synth = Synthesis(content=synthesis.content, confidence=calibrated)
    final = build_final_result(monitor, final_synth, status)
    monitor.set_state(TaskState.COMPLETED if status == "completed_degraded" else TaskState(status))
    monitor.set_final_result(final.model_dump())
    return {"decision": "accept", "final_result": final.model_dump(),
            "final_output": final_synth.content}


def _stage_replan(
    state: RunState, monitor: RunMonitor, verdict: SynthesisVerdict
) -> dict[str, object]:
    """Merge the judge's replacement steps and route back to execution, or accept on failure."""
    round_no = state.get("replans", 0) + 1
    decision = ReplanDecision(
        reasoning=verdict.feedback or verdict.reasoning,
        decision="replan",
        new_steps=verdict.new_steps,
    )
    try:
        merged = merge_replan(_require_plan(monitor), decision, monitor.step_status, round_no)
    except PlanValidationError:
        return _accept(monitor, _draft(monitor), degraded=True)
    monitor.attach_plan(merged)
    return {"decision": "replan", "replans": round_no, "resynth_rounds": 0, "synth_feedback": ""}


async def judge_node(state: RunState, config: RunnableConfig) -> dict[str, object]:
    """Adjudicate the draft: accept, re-synthesize, or replan (each bounded by its budget)."""
    deps = _deps(config)
    monitor = _monitor(config, state)
    synthesis = _draft(monitor)
    if monitor.plan is None or monitor.completed_count() == 0:
        return _accept(monitor, synthesis, degraded=False)
    if state.get("synth_failed"):
        return _accept(monitor, synthesis, degraded=True)
    det_errors = check_synthesis(
        synthesis, monitor.plan, monitor.completed_count(), state.get("output_format") or None
    )
    try:
        verdict, tokens = await judge_synthesis(
            state["goal"], monitor.plan, monitor.results, synthesis, det_errors,
            model=deps.judge_model,
        )
    except Exception:
        logger.exception("judge call failed; accepting current draft as degraded")
        return _accept(monitor, synthesis, degraded=True)
    monitor.total_tokens += tokens
    if verdict.verdict == "resynthesize" and state.get("resynth_rounds", 0) < state["max_resynth"]:
        return {"decision": "resynthesize", "synth_feedback": verdict.feedback,
                "resynth_rounds": state.get("resynth_rounds", 0) + 1}
    if verdict.verdict == "replan" and state.get("replans", 0) < state["max_replans"]:
        return _stage_replan(state, monitor, verdict)
    return _accept(monitor, synthesis, degraded=bool(det_errors) or verdict.verdict != "accept")


def route_after_judge(state: RunState) -> str:
    """Route the judge's verdict: re-synthesize, replan via execute, or finish."""
    decision = state.get("decision")
    if decision == "resynthesize":
        return "synthesize"
    if decision == "replan":
        return "execute"
    return "end"
