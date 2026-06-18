"""Offline tests for the failure-recovery zone (mocked model, no network).

Covers the run monitor's failure classification (optional / localized-required / structural), the
skip cascade, trace + totals + progress accounting, the deadline and cooperative-cancel /
preemptive-replan control events, and the re-plan decider plus the deterministic merge protocol
(completed steps frozen, new ids namespaced, intra-batch deps rewritten, dangling deps rejected).

Run standalone: ``python tests/test_recovery.py`` or via pytest.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from app.src.engine.evaluation import decide_replan, merge_replan  # noqa: E402
from app.src.engine.monitor import (  # noqa: E402
    RunMonitor,
    classify_failure,
    skip_cascade,
)
from app.src.engine.validation import PlanValidationError  # noqa: E402
from app.src.general_utils.agent_base import AgentResult  # noqa: E402
from app.src.schemas.plan import (  # noqa: E402
    ExecutionPlan,
    ExecutionStep,
    ReplanDecision,
    StepStatus,
    TaskState,
)


class _Raw:
    """Minimal model message carrying token usage metadata."""

    def __init__(self) -> None:
        self.usage_metadata = {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5}


class _Runnable:
    """Structured-output runnable returning a scripted decision and recording messages."""

    def __init__(self, model: FakeModel) -> None:
        self._model = model

    def invoke(self, messages: object) -> dict:
        self._model.seen = messages
        return {"parsed": self._model.decision, "raw": _Raw()}


class FakeModel:
    """Fake model returning a scripted ``ReplanDecision`` for ``with_structured_output``."""

    def __init__(self, decision: ReplanDecision) -> None:
        self.decision = decision
        self.seen: object = None

    def with_structured_output(self, schema: type, include_raw: bool = False) -> _Runnable:
        return _Runnable(self)


def _plan(steps: list[ExecutionStep] | None = None) -> ExecutionPlan:
    """Default to the research->analysis chain; otherwise wrap the given steps."""
    if steps is None:
        steps = [
            ExecutionStep(id="s1", agent="research", action="research"),
            ExecutionStep(id="s2", agent="analysis", action="analyze", dependencies=["s1"]),
        ]
    return ExecutionPlan(reasoning="base", task_id="t1", steps=steps)


def _chain_plan() -> ExecutionPlan:
    """A linear research->analysis plan."""
    return _plan()


def _branchy_plan() -> ExecutionPlan:
    """Two independent research steps with one analysis dependent on the first."""
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
    """Build an ``AgentResult`` carrying tokens/cost; ``failed`` carries an error payload."""
    return AgentResult(
        step_id=step_id,
        agent="research",
        status=status,
        output={"content": "x"} if status == "completed" else {"error": "boom"},
        tokens_used=tokens,
        execution_time_ms=12,
        actual_cost_usd=cost,
    )


# --- monitor: failure classification & accounting ---


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


# --- evaluation: re-plan decider & merge ---


def test_decider_returns_continue() -> None:
    decision = ReplanDecision(reasoning="rest reaches goal", decision="continue")
    model = FakeModel(decision)
    out, tokens = asyncio.run(
        decide_replan(_plan(), "study X", {"s1": _result("s1")}, "s2", "boom", model=model)
    )
    assert out.decision == "continue"
    assert tokens == 5
    user = model.seen[1]["content"]
    assert "<goal>" in user and "<failure>" in user and "boom" in user


def test_decider_returns_replan_with_steps() -> None:
    new = [ExecutionStep(id="s2", agent="research", action="research")]
    model = FakeModel(
        ReplanDecision(reasoning="retry differently", decision="replan", new_steps=new)
    )
    out, _ = asyncio.run(
        decide_replan(_plan(), "study X", {"s1": _result("s1")}, "s2", "boom", model=model)
    )
    assert out.decision == "replan"
    assert out.new_steps[0].id == "s2"


def test_merge_namespaces_and_preserves_completed() -> None:
    plan = _plan()
    status = {"s1": StepStatus.COMPLETED, "s2": StepStatus.FAILED}
    new = [
        ExecutionStep(id="a", agent="analysis", action="analyze", dependencies=["s1"]),
        ExecutionStep(id="b", agent="writing", action="write", dependencies=["a"]),
    ]
    decision = ReplanDecision(reasoning="new path", decision="replan", new_steps=new)
    merged = merge_replan(plan, decision, status, round_no=1)
    ids = merged.step_ids()
    assert ids == {"s1", "r1_a", "r1_b"}
    r1_b = merged.step_by_id("r1_b")
    assert r1_b.dependencies == ["r1_a"]
    r1_a = merged.step_by_id("r1_a")
    assert r1_a.dependencies == ["s1"]
    assert merged.parallel_groups == [["s1"], ["r1_a"], ["r1_b"]]


def test_merge_rejects_id_collision() -> None:
    plan = _plan()
    status = {"s1": StepStatus.COMPLETED, "s2": StepStatus.FAILED}
    new = [ExecutionStep(id="s1", agent="research", action="research")]
    decision = ReplanDecision(reasoning="x", decision="replan", new_steps=new)
    merged = merge_replan(plan, decision, status, round_no=1)
    assert "r1_s1" in merged.step_ids()
    assert "s1" in merged.step_ids()


def test_merge_rejects_dangling_new_dependency() -> None:
    plan = _plan()
    status = {"s1": StepStatus.COMPLETED, "s2": StepStatus.FAILED}
    new = [ExecutionStep(id="a", agent="analysis", action="analyze", dependencies=["ghost"])]
    decision = ReplanDecision(reasoning="x", decision="replan", new_steps=new)
    try:
        merge_replan(plan, decision, status, round_no=1)
        raise AssertionError("expected PlanValidationError")
    except PlanValidationError as exc:
        assert any("unknown dependency" in e for e in exc.errors)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
