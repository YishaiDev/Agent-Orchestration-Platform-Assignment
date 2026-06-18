"""Offline tests for the run monitor (no network, no LLM).

Covers failure classification (optional / localized-required / structural), the skip cascade,
trace + totals + progress accounting, and the cooperative-cancel and preemptive-replan events.

Run standalone: ``python tests/engine/test_monitor.py`` or via pytest.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")

from app.src.engine.monitor import (  # noqa: E402
    RunMonitor,
    classify_failure,
    skip_cascade,
)
from app.src.general_utils.agent_base import AgentResult  # noqa: E402
from app.src.schemas.plan import (  # noqa: E402
    ExecutionPlan,
    ExecutionStep,
    StepStatus,
    TaskState,
)


def _plan(steps: list[ExecutionStep]) -> ExecutionPlan:
    return ExecutionPlan(reasoning="r", task_id="t1", steps=steps)


def _chain_plan() -> ExecutionPlan:
    return _plan(
        [
            ExecutionStep(id="s1", agent="research", action="research"),
            ExecutionStep(id="s2", agent="analysis", action="analyze", dependencies=["s1"]),
        ]
    )


def _branchy_plan() -> ExecutionPlan:
    return _plan(
        [
            ExecutionStep(id="s1", agent="research", action="research"),
            ExecutionStep(id="s2", agent="research", action="research"),
            ExecutionStep(id="s3", agent="analysis", action="analyze", dependencies=["s1"]),
        ]
    )


def _result(
    step_id: str, status: str = "completed", tokens: int = 7, cost: float = 0.01
) -> AgentResult:
    return AgentResult(
        step_id=step_id,
        agent="research",
        status=status,
        output={"content": "x"} if status == "completed" else {"error": "boom"},
        tokens_used=tokens,
        execution_time_ms=12,
        actual_cost_usd=cost,
    )


def test_optional_failure_is_skippable() -> None:
    plan = _plan([ExecutionStep(id="s1", agent="research", action="research", optional=True)])
    assert classify_failure(plan, "s1", {"s1": StepStatus.FAILED}) == "skippable"


def test_required_failure_is_structural_even_with_surviving_branch() -> None:
    plan = _branchy_plan()
    status = {"s1": StepStatus.FAILED, "s2": StepStatus.PENDING, "s3": StepStatus.PENDING}
    assert classify_failure(plan, "s1", status) == "structural"
    assert skip_cascade(plan, "s1") == {"s3"}


def test_optional_failure_losing_crucial_dependent_is_structural() -> None:
    plan = _plan(
        [
            ExecutionStep(id="s1", agent="research", action="research", optional=True),
            ExecutionStep(id="s2", agent="analysis", action="analyze", dependencies=["s1"]),
        ]
    )
    status = {"s1": StepStatus.FAILED, "s2": StepStatus.PENDING}
    assert classify_failure(plan, "s1", status) == "structural"


def test_optional_failure_with_optional_dependent_is_skippable() -> None:
    plan = _plan(
        [
            ExecutionStep(id="s1", agent="research", action="research", optional=True),
            ExecutionStep(
                id="s2", agent="analysis", action="analyze",
                dependencies=["s1"], optional=True,
            ),
        ]
    )
    status = {"s1": StepStatus.FAILED, "s2": StepStatus.PENDING}
    assert classify_failure(plan, "s1", status) == "skippable"


def test_structural_failure_kills_all_remaining() -> None:
    plan = _chain_plan()
    status = {"s1": StepStatus.FAILED, "s2": StepStatus.PENDING}
    assert classify_failure(plan, "s1", status) == "structural"


def test_record_result_tracks_trace_totals_progress() -> None:
    monitor = RunMonitor("t1")
    plan = _chain_plan()
    monitor.attach_plan(plan)
    step = plan.step_by_id("s1")
    monitor.start_step(step)
    monitor.record_result(step, _result("s1"))
    assert monitor.completed_count() == 1
    assert monitor.total_tokens == 7
    assert round(monitor.total_cost_usd, 3) == 0.01
    progress = monitor.progress()
    assert progress.total_steps == 2 and progress.completed_steps == 1
    entry = monitor.trace_dicts()[0]
    assert entry["agent"] == "research"
    assert entry["started_at"] is not None and entry["completed_at"] is not None
    assert entry["tokens_used"] == 7 and entry["execution_time_ms"] == 12


def test_deadline_exceeded_stops_launching() -> None:
    monitor = RunMonitor("t1", deadline_seconds=0.02)
    assert monitor.deadline_exceeded() is False
    time.sleep(0.05)
    assert monitor.deadline_exceeded() is True
    assert monitor.should_stop_launching() is True


def test_request_replan_stops_launching() -> None:
    monitor = RunMonitor("t1")
    assert monitor.should_stop_launching() is False
    monitor.request_replan("s1", "boom")
    assert monitor.replan_requested is True
    assert monitor.should_stop_launching() is True
    monitor.clear_replan()
    assert monitor.replan_requested is False
    assert monitor.failed_step_id is None


def test_request_cancel_and_skip_marks() -> None:
    monitor = RunMonitor("t1")
    monitor.attach_plan(_branchy_plan())
    monitor.request_cancel()
    assert monitor.cancelled is True
    monitor.mark_skipped({"s3"})
    assert monitor.step_status["s3"] == StepStatus.SKIPPED
    monitor.set_state(TaskState.CANCELLED)
    assert monitor.state == TaskState.CANCELLED


def _main() -> None:
    tests = [
        test_optional_failure_is_skippable,
        test_required_failure_is_structural_even_with_surviving_branch,
        test_optional_failure_losing_crucial_dependent_is_structural,
        test_optional_failure_with_optional_dependent_is_skippable,
        test_structural_failure_kills_all_remaining,
        test_record_result_tracks_trace_totals_progress,
        test_deadline_exceeded_stops_launching,
        test_request_replan_stops_launching,
        test_request_cancel_and_skip_marks,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _main()
