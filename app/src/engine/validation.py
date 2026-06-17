"""Deterministic plan validation: structure, registry references, and DAG acyclicity.

Validation is the no-LLM guarantee that a plan is *runnable* before any agent is invoked. It also
derives ``parallel_groups`` (topological levels) so the scheduler and the trace agree on which steps
may run concurrently. A bad plan is rejected here and fixed reactively by the bounded re-plan, which
is why there is no always-on LLM judge on the planner.
"""

from __future__ import annotations

from app.src.engine.registry import is_allowed
from app.src.schemas.plan import ExecutionPlan, ExecutionStep, PlannerDraft


class PlanValidationError(Exception):
    """Raised when a draft or merged plan fails deterministic validation.

    Carries the full list of problems so the planner's bounded retry can show the model every
    issue at once instead of one per round-trip.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def _check_structure(steps: list[ExecutionStep]) -> list[str]:
    """Check the plan is non-empty with unique step ids."""
    if not steps:
        return ["plan has no steps"]
    seen: set[str] = set()
    errors: list[str] = []
    for step in steps:
        if step.id in seen:
            errors.append(f"duplicate step id: {step.id}")
        seen.add(step.id)
    return errors


def _check_references(steps: list[ExecutionStep]) -> list[str]:
    """Check every agent/action and dependency reference resolves."""
    ids = {s.id for s in steps}
    errors: list[str] = []
    for step in steps:
        if not is_allowed(step.agent, step.action):
            errors.append(f"step {step.id}: unknown agent/action {step.agent}/{step.action}")
        errors.extend(
            f"step {step.id}: unknown dependency {dep}"
            for dep in step.dependencies
            if dep not in ids
        )
        if step.id in step.dependencies:
            errors.append(f"step {step.id}: depends on itself")
    return errors


def _dependents(steps: list[ExecutionStep]) -> dict[str, list[str]]:
    """Map each step id to the ids of steps that depend on it."""
    dependents: dict[str, list[str]] = {s.id: [] for s in steps}
    for step in steps:
        for dep in step.dependencies:
            if dep in dependents:
                dependents[dep].append(step.id)
    return dependents


def derive_parallel_groups(steps: list[ExecutionStep]) -> list[list[str]]:
    """Compute topological levels via Kahn's algorithm; raise on any cycle.

    Args:
        steps: The plan's steps (references assumed already checked).

    Returns:
        Ordered groups of step ids; steps within a group have no inter-dependencies.

    Raises:
        PlanValidationError: When the dependency graph contains a cycle.
    """
    indegree = {s.id: len(s.dependencies) for s in steps}
    dependents = _dependents(steps)
    groups: list[list[str]] = []
    resolved = 0
    while True:
        ready = sorted(sid for sid, deg in indegree.items() if deg == 0)
        if not ready:
            break
        groups.append(ready)
        resolved += len(ready)
        for sid in ready:
            indegree[sid] = -1
            for child in dependents[sid]:
                indegree[child] -= 1
    if resolved != len(steps):
        raise PlanValidationError(["plan dependencies contain a cycle"])
    return groups


def _collect_errors(steps: list[ExecutionStep]) -> list[str]:
    """Run the structural and referential checks and return all problems."""
    return _check_structure(steps) + _check_references(steps)


def validate_and_finalize(draft: PlannerDraft, task_id: str) -> ExecutionPlan:
    """Validate a planner draft and lift it into an executable plan.

    Args:
        draft: Raw planner output (reasoning + steps).
        task_id: Engine-assigned task id stamped onto the plan.

    Returns:
        A validated ExecutionPlan with derived ``parallel_groups``.

    Raises:
        PlanValidationError: On any structural, referential, or acyclicity failure.
    """
    errors = _collect_errors(draft.steps)
    if errors:
        raise PlanValidationError(errors)
    groups = derive_parallel_groups(draft.steps)
    return ExecutionPlan(
        reasoning=draft.reasoning, task_id=task_id, steps=draft.steps, parallel_groups=groups
    )


def revalidate(plan: ExecutionPlan) -> ExecutionPlan:
    """Re-validate an already-built (e.g. merged) plan and recompute its groups.

    Args:
        plan: The candidate plan to re-check.

    Returns:
        The same plan with freshly derived ``parallel_groups``.

    Raises:
        PlanValidationError: On any structural, referential, or acyclicity failure.
    """
    errors = _collect_errors(plan.steps)
    if errors:
        raise PlanValidationError(errors)
    groups = derive_parallel_groups(plan.steps)
    return plan.model_copy(update={"parallel_groups": groups})
