"""Plan domain schemas: step/task status sets, the planner draft, and the validated plan.

The planner emits a :class:`PlannerDraft` (``reasoning`` first, then the step list) so a strong
model lays out its decomposition rationale before committing to steps. ``task_id`` and the derived
``parallel_groups`` are filled by the engine, never by the model, which is why the LLM-output schema
and the stored :class:`ExecutionPlan` are kept separate.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class StepStatus(StrEnum):
    """Lifecycle of a single execution step inside the inner scheduler."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskState(StrEnum):
    """Task-level status set mandated by the assignment spec."""

    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionStep(BaseModel):
    """One unit of work routed to a single agent action.

    ``optional`` marks a step whose failure must not fail the task: its branch is skipped and
    independent branches keep running. ``dependencies`` reference upstream step ids whose outputs
    are passed into this step.
    """

    id: str = Field(description="Unique step id, e.g. s1; referenced by dependents.")
    agent: str = Field(description="Target agent name from the registry allowlist.")
    action: str = Field(description="Capability/action the agent should perform.")
    description: str = Field(default="", description="Short human-readable intent for the trace.")
    input: dict[str, object] = Field(
        default_factory=dict, description="Agent-specific input fields for this step."
    )
    dependencies: list[str] = Field(
        default_factory=list, description="Upstream step ids whose outputs feed this step."
    )
    optional: bool = Field(
        default=False, description="If true, this step's failure never fails the task."
    )


class PlannerDraft(BaseModel):
    """Raw planner output: reasoning first, then the proposed steps.

    Deliberately excludes ``task_id`` and ``parallel_groups`` — those are engine-assigned and
    engine-derived, so the model is never asked to fabricate them.
    """

    reasoning: str = Field(description="Decomposition rationale produced before the steps.")
    steps: list[ExecutionStep] = Field(description="The ordered step list forming the work DAG.")


class ExecutionPlan(BaseModel):
    """A validated, engine-owned plan ready for execution.

    ``parallel_groups`` holds the deterministic topological levels (steps in the same group have
    no inter-dependencies and may run concurrently); it is derived during validation, not authored
    by the model.
    """

    reasoning: str
    task_id: str
    steps: list[ExecutionStep]
    parallel_groups: list[list[str]] = Field(default_factory=list)

    def step_ids(self) -> set[str]:
        """Return the set of all step ids in the plan."""
        return {step.id for step in self.steps}

    def step_by_id(self, step_id: str) -> ExecutionStep | None:
        """Return the step with ``step_id``, or None when absent."""
        return next((step for step in self.steps if step.id == step_id), None)


class ReplanDecision(BaseModel):
    """The re-plan decider's verdict, reasoning first.

    ``new_steps`` is populated only when ``decision`` is ``replan`` and covers the unfinished work
    (completed steps stay frozen); their ids are namespaced by the engine before merging.
    """

    reasoning: str = Field(description="Why the verdict was reached, before the decision.")
    decision: Literal["continue", "replan"] = Field(
        description="continue to synthesis, or replan the unfinished part."
    )
    new_steps: list[ExecutionStep] = Field(
        default_factory=list, description="Replacement steps for the unfinished work (replan only)."
    )


class SynthesisVerdict(BaseModel):
    """The synthesis quality judge's verdict, reasoning first.

    ``accept`` ships the draft; ``resynthesize`` requests another synthesis pass guided by
    ``feedback`` (cheap, same data); ``replan`` requests fresh upstream work via ``new_steps``
    (covering the coverage gap, completed steps stay frozen — namespaced by the engine on merge).
    """

    reasoning: str = Field(description="Why the verdict was reached, before the decision.")
    verdict: Literal["accept", "resynthesize", "replan"] = Field(
        description="accept the draft, re-synthesize with feedback, or replan the unfinished work."
    )
    feedback: str = Field(
        default="", description="Scoped fix guidance for re-synthesis or the new steps."
    )
    new_steps: list[ExecutionStep] = Field(
        default_factory=list, description="Replacement steps for the coverage gap (replan only)."
    )
