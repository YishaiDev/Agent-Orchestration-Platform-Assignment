"""Run monitor: observability, totals, failure classification, and the run-control events.

One :class:`RunMonitor` is created per task and shared with the inner scheduler. It owns the
execution trace, live status/progress, and token/cost totals, and exposes two asyncio events: a
cooperative ``cancel`` (also driven by an optional deadline) and a preemptive ``replan`` raised the
instant a *structural* failure is seen — letting the scheduler cancel in-flight steps and route to
the re-plan decider without draining the rest of the DAG.

Failure classification is the deterministic pre-filter that decides whether the LLM decider is even
consulted: a failed step is *skippable* when independent non-optional work survives its loss, and
*structural* when its cascade removes every remaining non-optional step.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.src.general_utils.agent_base import AgentResult
from app.src.schemas.plan import ExecutionPlan, ExecutionStep, StepStatus, TaskState
from app.src.schemas.run_state import Progress

_TERMINAL_LOST = {StepStatus.FAILED, StepStatus.SKIPPED, StepStatus.CANCELLED}


class TraceEntry(BaseModel):
    """One observable record per executed step for ``GET /tasks/{id}``."""

    step_id: str
    agent: str
    action: str
    status: str
    duration_ms: int
    tokens: int
    input: dict[str, object] = Field(default_factory=dict)
    output: dict[str, object] = Field(default_factory=dict)


def transitive_dependents(plan: ExecutionPlan, step_id: str) -> set[str]:
    """Return all step ids that transitively depend on ``step_id``.

    Args:
        plan: The current plan.
        step_id: The upstream step whose dependents are sought.

    Returns:
        The set of downstream step ids (excluding ``step_id`` itself).
    """
    dependents: set[str] = set()
    frontier = [step_id]
    while frontier:
        current = frontier.pop()
        for step in plan.steps:
            if current in step.dependencies and step.id not in dependents:
                dependents.add(step.id)
                frontier.append(step.id)
    return dependents


def skip_cascade(plan: ExecutionPlan, failed_id: str) -> set[str]:
    """Return the steps to skip when ``failed_id`` fails (its transitive dependents)."""
    return transitive_dependents(plan, failed_id)


def classify_failure(
    plan: ExecutionPlan, failed_id: str, step_status: dict[str, StepStatus]
) -> str:
    """Classify a failed step as ``skippable`` or ``structural``.

    Args:
        plan: The current plan.
        failed_id: The id of the step that failed.
        step_status: Live status per step id.

    Returns:
        ``skippable`` when non-optional work survives the failure; ``structural`` otherwise.
    """
    step = plan.step_by_id(failed_id)
    if step is not None and step.optional:
        return "skippable"
    cascade = {failed_id} | skip_cascade(plan, failed_id)
    survivors = [
        s
        for s in plan.steps
        if s.id not in cascade
        and not s.optional
        and step_status.get(s.id) not in _TERMINAL_LOST
    ]
    return "skippable" if survivors else "structural"


def _now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


class RunMonitor:
    """Per-run observability and control hub shared with the scheduler."""

    def __init__(self, task_id: str, deadline_seconds: float | None = None) -> None:
        """Initialise monitor state and control events.

        Args:
            task_id: The task this monitor tracks.
            deadline_seconds: Optional wall-clock budget after which cancel is requested.
        """
        self.task_id = task_id
        self.state = TaskState.PENDING
        self.created_at = _now()
        self.updated_at = self.created_at
        self.plan: ExecutionPlan | None = None
        self.step_status: dict[str, StepStatus] = {}
        self.results: dict[str, AgentResult] = {}
        self.trace: list[TraceEntry] = []
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.current_step: str | None = None
        self.failed_step_id: str | None = None
        self.failure_error: str | None = None
        self.draft: dict[str, object] | None = None
        self.final_result: dict[str, object] | None = None
        self._cancel = asyncio.Event()
        self._replan = asyncio.Event()
        self._deadline = time.monotonic() + deadline_seconds if deadline_seconds else None

    def _touch(self) -> None:
        """Stamp the last-updated time."""
        self.updated_at = _now()

    def set_state(self, state: TaskState) -> None:
        """Transition the task to ``state``."""
        self.state = state
        self._touch()

    def set_draft(self, draft: dict[str, object]) -> None:
        """Store the latest synthesis draft for the judge node to adjudicate."""
        self.draft = draft
        self._touch()

    def set_final_result(self, result: dict[str, object]) -> None:
        """Store the synthesized final result for the result endpoint."""
        self.final_result = result
        self._touch()

    def attach_plan(self, plan: ExecutionPlan) -> None:
        """Adopt a (re-)validated plan, seeding any new steps as pending."""
        self.plan = plan
        for step in plan.steps:
            self.step_status.setdefault(step.id, StepStatus.PENDING)
        self._touch()

    def start_step(self, step: ExecutionStep) -> None:
        """Mark ``step`` running and record it as the current step."""
        self.step_status[step.id] = StepStatus.RUNNING
        self.current_step = step.id
        self._touch()

    def _entry(self, step: ExecutionStep, result: AgentResult, status: StepStatus) -> TraceEntry:
        """Build the trace entry for a finished step."""
        return TraceEntry(
            step_id=result.step_id,
            agent=result.agent,
            action=step.action,
            status=status.value,
            duration_ms=result.execution_time_ms,
            tokens=result.tokens_used,
            input=dict(step.input),
            output=result.output,
        )

    def record_result(self, step: ExecutionStep, result: AgentResult) -> None:
        """Store a finished step's result and update trace, status, and totals."""
        status = StepStatus.COMPLETED if result.status == "completed" else StepStatus.FAILED
        self.results[result.step_id] = result
        self.step_status[result.step_id] = status
        self.total_tokens += result.tokens_used
        self.total_cost_usd += result.actual_cost_usd or 0.0
        self.trace.append(self._entry(step, result, status))
        self._touch()

    def mark_skipped(self, step_ids: set[str]) -> None:
        """Mark still-pending steps in ``step_ids`` as skipped."""
        for step_id in step_ids:
            if self.step_status.get(step_id) == StepStatus.PENDING:
                self.step_status[step_id] = StepStatus.SKIPPED
        self._touch()

    def mark_cancelled(self, step_ids: set[str]) -> None:
        """Mark pending or running steps in ``step_ids`` as cancelled."""
        for step_id in step_ids:
            if self.step_status.get(step_id) in {StepStatus.PENDING, StepStatus.RUNNING}:
                self.step_status[step_id] = StepStatus.CANCELLED
        self._touch()

    def request_replan(self, failed_id: str, error: str) -> None:
        """Record a structural failure and raise the preemptive re-plan event."""
        self.failed_step_id = failed_id
        self.failure_error = error
        self._replan.set()
        self._touch()

    def clear_replan(self) -> None:
        """Reset the re-plan signal before the next execution round."""
        self._replan = asyncio.Event()
        self.failed_step_id = None
        self.failure_error = None

    def request_cancel(self) -> None:
        """Raise the cooperative cancel event."""
        self._cancel.set()
        self._touch()

    @property
    def cancelled(self) -> bool:
        """Whether a cooperative cancel has been requested."""
        return self._cancel.is_set()

    @property
    def replan_requested(self) -> bool:
        """Whether a structural failure has requested a re-plan."""
        return self._replan.is_set()

    def deadline_exceeded(self) -> bool:
        """Whether the optional wall-clock deadline has passed."""
        return self._deadline is not None and time.monotonic() >= self._deadline

    def should_stop_launching(self) -> bool:
        """Whether the scheduler must stop launching new steps."""
        return self.cancelled or self.replan_requested or self.deadline_exceeded()

    def completed_count(self) -> int:
        """Return the number of steps that completed successfully."""
        return sum(1 for status in self.step_status.values() if status == StepStatus.COMPLETED)

    def completed_step_ids(self) -> list[str]:
        """Return the ids of steps that completed successfully."""
        return [sid for sid, status in self.step_status.items() if status == StepStatus.COMPLETED]

    def progress(self) -> Progress:
        """Return the live progress snapshot."""
        total = len(self.plan.steps) if self.plan else 0
        return Progress(
            total_steps=total,
            completed_steps=self.completed_count(),
            current_step=self.current_step,
        )

    def trace_dicts(self) -> list[dict[str, object]]:
        """Return the execution trace as serializable dicts."""
        return [entry.model_dump() for entry in self.trace]
