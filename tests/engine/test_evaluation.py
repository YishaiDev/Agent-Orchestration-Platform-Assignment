"""Offline tests for the re-plan decider and the deterministic merge (mocked model, no network).

Covers the continue/replan verdicts, the goal/failure fencing, and the merge protocol: completed
steps frozen, new ids namespaced under ``r{n}_``, intra-batch dependencies rewritten, references to
completed steps preserved, collisions rejected, and the merged DAG re-validated.

Run standalone: ``python tests/engine/test_evaluation.py`` or via pytest.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")

from app.src.engine.evaluation import decide_replan, merge_replan  # noqa: E402
from app.src.engine.validation import PlanValidationError  # noqa: E402
from app.src.general_utils.agent_base import AgentResult  # noqa: E402
from app.src.schemas.plan import (  # noqa: E402
    ExecutionPlan,
    ExecutionStep,
    ReplanDecision,
    StepStatus,
)


class _Raw:
    def __init__(self) -> None:
        self.usage_metadata = {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5}


class _Runnable:
    def __init__(self, model: FakeModel) -> None:
        self._model = model

    def invoke(self, messages: object) -> dict:
        self._model.seen = messages
        return {"parsed": self._model.decision, "raw": _Raw()}


class FakeModel:
    def __init__(self, decision: ReplanDecision) -> None:
        self.decision = decision
        self.seen: object = None

    def with_structured_output(self, schema: type, include_raw: bool = False) -> _Runnable:
        return _Runnable(self)


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        reasoning="base",
        task_id="t1",
        steps=[
            ExecutionStep(id="s1", agent="research", action="research"),
            ExecutionStep(id="s2", agent="analysis", action="analyze", dependencies=["s1"]),
        ],
    )


def _result(step_id: str) -> AgentResult:
    return AgentResult(
        step_id=step_id, agent="research", status="completed",
        output={"content": "found facts"}, tokens_used=1, execution_time_ms=1,
    )


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


def _main() -> None:
    tests = [
        test_decider_returns_continue,
        test_decider_returns_replan_with_steps,
        test_merge_namespaces_and_preserves_completed,
        test_merge_rejects_id_collision,
        test_merge_rejects_dangling_new_dependency,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _main()
