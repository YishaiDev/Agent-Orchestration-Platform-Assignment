"""Inner async scheduler: continuous-concurrency execution of the step DAG.

This is the plain-``asyncio`` core the LangGraph ``execute`` node wraps. Unlike a wave-synchronous
super-step model, a step launches the instant all its dependencies complete — so a fast step's
successor starts while a slow sibling is still running. A semaphore bounds real concurrency (and
doubles as the Gemini rate limiter). The scheduler mutates the shared :class:`RunMonitor` in place;
it never returns results directly.

Failure handling is split for clarity: a *skippable* failure is realised lazily (a step whose
dependency was lost is marked skipped at readiness time, so independent branches keep running),
while a *structural* failure raises the monitor's re-plan event and the scheduler **preempts** —
cancelling every in-flight task so control returns to the re-plan decider without draining the DAG.
Cancellation uses ``Task.cancel`` (``CancelledError`` is a ``BaseException``), which the agents'
``except Exception`` cannot swallow.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from app.src.engine import dispatch as dispatch_module
from app.src.engine.monitor import RunMonitor, classify_failure
from app.src.general_utils.agent_base import AgentResult
from app.src.schemas.config import get_config
from app.src.schemas.plan import ExecutionPlan, ExecutionStep, StepStatus

StepRunner = Callable[[ExecutionStep, dict[str, AgentResult], str], Awaitable[AgentResult]]

_LOST = {StepStatus.FAILED, StepStatus.SKIPPED, StepStatus.CANCELLED}


def _classify_pending(
    plan: ExecutionPlan, monitor: RunMonitor, pending: set[str]
) -> tuple[list[str], set[str]]:
    """Split pending steps into those ready to run and those to skip (a dependency was lost)."""
    ready: list[str] = []
    skipped: set[str] = set()
    for step_id in pending:
        step = plan.step_by_id(step_id)
        statuses = [monitor.step_status.get(dep) for dep in step.dependencies] if step else []
        if any(status in _LOST for status in statuses):
            skipped.add(step_id)
        elif all(status == StepStatus.COMPLETED for status in statuses):
            ready.append(step_id)
    return ready, skipped


async def _run_one(
    step: ExecutionStep,
    monitor: RunMonitor,
    runner: StepRunner,
    sem: asyncio.Semaphore,
    session: str,
) -> AgentResult | None:
    """Run a single step under the concurrency semaphore, recording its result."""
    async with sem:
        if monitor.should_stop_launching():
            monitor.mark_cancelled({step.id})
            return None
        monitor.start_step(step)
        result = await runner(step, monitor.results, session)
        monitor.record_result(step, result)
        return result


def _launch_ready(
    plan: ExecutionPlan,
    monitor: RunMonitor,
    pending: set[str],
    tasks: dict[asyncio.Task[AgentResult | None], str],
    sem: asyncio.Semaphore,
    runner: StepRunner,
    session: str,
) -> None:
    """Mark dependency-blocked steps skipped and launch every ready step."""
    if monitor.should_stop_launching():
        return
    ready, skipped = _classify_pending(plan, monitor, pending)
    monitor.mark_skipped(skipped)
    pending.difference_update(skipped)
    for step_id in ready:
        step = plan.step_by_id(step_id)
        if step is None:
            continue
        task = asyncio.create_task(_run_one(step, monitor, runner, sem, session))
        tasks[task] = step_id
        pending.discard(step_id)


def _error_text(result: AgentResult) -> str:
    """Extract a human-readable error from a failed result."""
    return str(result.output.get("error") or "step failed")


def _on_done(
    plan: ExecutionPlan,
    monitor: RunMonitor,
    task: asyncio.Task[AgentResult | None],
) -> None:
    """Inspect a finished task and request a re-plan on a structural failure."""
    if task.cancelled():
        return
    result = task.result()
    if result is None or result.status != "failed":
        return
    if classify_failure(plan, result.step_id, monitor.step_status) == "structural":
        monitor.request_replan(result.step_id, _error_text(result))


async def _cancel_inflight(
    tasks: dict[asyncio.Task[AgentResult | None], str], monitor: RunMonitor
) -> None:
    """Preemptively cancel all in-flight tasks and mark their steps cancelled."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    monitor.mark_cancelled(set(tasks.values()))
    tasks.clear()


async def execute_plan(
    plan: ExecutionPlan,
    monitor: RunMonitor,
    runner: StepRunner | None = None,
    concurrency: int | None = None,
    session_id: str = "local",
) -> None:
    """Execute the plan's DAG with continuous concurrency, updating ``monitor`` in place.

    Args:
        plan: The validated plan to run.
        monitor: The shared run monitor (status, trace, control events).
        runner: Step executor (defaults to the real agent dispatch); injectable for tests.
        concurrency: Max concurrent steps (defaults to the configured cap).
        session_id: Session id carried into agent calls.
    """
    runner = runner or dispatch_module.dispatch
    limit = concurrency or get_config().orchestrator.concurrency
    sem = asyncio.Semaphore(limit)
    pending = {
        step.id for step in plan.steps if monitor.step_status.get(step.id) == StepStatus.PENDING
    }
    tasks: dict[asyncio.Task[AgentResult | None], str] = {}
    while pending or tasks:
        _launch_ready(plan, monitor, pending, tasks, sem, runner, session_id)
        if not tasks:
            monitor.mark_skipped(set(pending))
            pending.clear()
            break
        done, _ = await asyncio.wait(set(tasks), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            tasks.pop(task, None)
            _on_done(plan, monitor, task)
        if monitor.replan_requested:
            await _cancel_inflight(tasks, monitor)
            break
