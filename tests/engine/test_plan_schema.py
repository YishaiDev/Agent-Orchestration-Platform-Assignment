"""Offline tests for the plan/run-state domain schemas (no network, no LLM).

Covers the spec status sets, the reasoning-first planner draft, plan id helpers, and the
initial run-state baseline.

Run standalone: ``python tests/engine/test_plan_schema.py`` or via pytest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")

from app.src.schemas.plan import (  # noqa: E402
    ExecutionPlan,
    ExecutionStep,
    PlannerDraft,
    StepStatus,
    TaskState,
)
from app.src.schemas.run_state import initial_state  # noqa: E402


def test_task_state_matches_spec_set() -> None:
    values = {state.value for state in TaskState}
    assert values == {"pending", "planning", "executing", "completed", "failed", "cancelled"}


def test_step_status_covers_scheduler_lifecycle() -> None:
    values = {status.value for status in StepStatus}
    assert values == {"pending", "running", "completed", "failed", "skipped", "cancelled"}


def test_step_defaults_are_non_optional_with_no_deps() -> None:
    step = ExecutionStep(id="s1", agent="research", action="research")
    assert step.optional is False
    assert step.dependencies == []
    assert step.input == {}


def test_planner_draft_orders_reasoning_first() -> None:
    fields = list(PlannerDraft.model_fields)
    assert fields[0] == "reasoning"
    assert "task_id" not in fields
    assert "parallel_groups" not in fields


def test_execution_plan_id_helpers() -> None:
    steps = [
        ExecutionStep(id="s1", agent="research", action="research"),
        ExecutionStep(id="s2", agent="analysis", action="analyze", dependencies=["s1"]),
    ]
    plan = ExecutionPlan(reasoning="r", task_id="t1", steps=steps)
    assert plan.step_ids() == {"s1", "s2"}
    assert plan.step_by_id("s2").dependencies == ["s1"]
    assert plan.step_by_id("missing") is None


def test_initial_state_baseline() -> None:
    state = initial_state("t1", "do a thing", "", "sess", max_replans=1)
    assert state["task_id"] == "t1"
    assert state["goal"] == "do a thing"
    assert state["replans"] == 0
    assert state["max_replans"] == 1
    assert state["decision"] == ""
    assert state["final_result"] is None


def _main() -> None:
    tests = [
        test_task_state_matches_spec_set,
        test_step_status_covers_scheduler_lifecycle,
        test_step_defaults_are_non_optional_with_no_deps,
        test_planner_draft_orders_reasoning_first,
        test_execution_plan_id_helpers,
        test_initial_state_baseline,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _main()
