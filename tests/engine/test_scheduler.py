"""Offline tests for the inner async scheduler (fake runner, no agents, no network).

Covers sequential dependency passing, genuine parallelism (timing), skippable-failure cascade with
surviving siblings, structural-failure preemption of in-flight work (timing + cancelled status),
cooperative cancel, and the concurrency cap.

Run standalone: ``python tests/engine/test_scheduler.py`` or via pytest.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from app.src.engine.monitor import RunMonitor  # noqa: E402
from app.src.engine.scheduler import execute_plan  # noqa: E402
from app.src.general_utils.agent_base import AgentResult  # noqa: E402
from app.src.schemas.plan import (  # noqa: E402
    ExecutionPlan,
    ExecutionStep,
    StepStatus,
    TaskState,
)


class FakeRunner:
    """Configurable step runner: per-step delays and a set of step ids that fail."""

    def __init__(
        self, delays: dict[str, float] | None = None, fails: set[str] | None = None
    ) -> None:
        self.delays = delays or {}
        self.fails = fails or set()
        self.started: list[str] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def __call__(
        self, step: ExecutionStep, results: dict[str, AgentResult], session: str
    ) -> AgentResult:
        self.started.append(step.id)
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            await asyncio.sleep(self.delays.get(step.id, 0.0))
        finally:
            self.concurrent -= 1
        failed = step.id in self.fails
        output = (
            {"error": "boom"}
            if failed
            else {"content": f"out-{step.id}", "deps": sorted(results)}
        )
        return AgentResult(
            step_id=step.id,
            agent=step.agent,
            status="failed" if failed else "completed",
            output=output,
            tokens_used=1,
            execution_time_ms=1,
        )


def _plan(steps: list[ExecutionStep]) -> ExecutionPlan:
    return ExecutionPlan(reasoning="r", task_id="t1", steps=steps)


def _run(plan: ExecutionPlan, runner: FakeRunner, concurrency: int | None = None) -> RunMonitor:
    monitor = RunMonitor("t1")
    monitor.attach_plan(plan)
    monitor.set_state(TaskState.EXECUTING)
    asyncio.run(execute_plan(plan, monitor, runner=runner, concurrency=concurrency))
    return monitor


def _step(
    step_id: str, agent: str = "research", action: str = "research", **kw: object
) -> ExecutionStep:
    return ExecutionStep(id=step_id, agent=agent, action=action, **kw)  # type: ignore[arg-type]


def test_sequential_chain_runs_in_order_and_passes_outputs() -> None:
    plan = _plan(
        [
            _step("s1"),
            _step("s2", agent="analysis", action="analyze", dependencies=["s1"]),
            _step("s3", agent="writing", action="write", dependencies=["s2"]),
        ]
    )
    runner = FakeRunner()
    monitor = _run(plan, runner)
    assert runner.started == ["s1", "s2", "s3"]
    assert monitor.completed_count() == 3
    assert monitor.results["s3"].output["deps"] == ["s1", "s2"]


def test_independent_steps_run_in_parallel() -> None:
    plan = _plan([_step("s1"), _step("s2"), _step("s3")])
    runner = FakeRunner(delays={"s1": 0.1, "s2": 0.1, "s3": 0.1})
    start = time.perf_counter()
    monitor = _run(plan, runner, concurrency=3)
    elapsed = time.perf_counter() - start
    assert monitor.completed_count() == 3
    assert elapsed < 0.25
    assert runner.max_concurrent == 3


def test_skippable_failure_skips_dependents_siblings_survive() -> None:
    plan = _plan(
        [
            _step("s1"),
            _step("s2"),
            _step("s3", agent="analysis", action="analyze", dependencies=["s1"]),
        ]
    )
    monitor = _run(plan, FakeRunner(fails={"s1"}))
    assert monitor.step_status["s1"] == StepStatus.FAILED
    assert monitor.step_status["s3"] == StepStatus.SKIPPED
    assert monitor.step_status["s2"] == StepStatus.COMPLETED
    assert monitor.replan_requested is False


def test_structural_failure_preempts_inflight_optional() -> None:
    plan = _plan(
        [
            _step("s_req"),
            _step("s_opt", optional=True),
        ]
    )
    runner = FakeRunner(delays={"s_opt": 5.0}, fails={"s_req"})
    start = time.perf_counter()
    monitor = _run(plan, runner, concurrency=3)
    elapsed = time.perf_counter() - start
    assert monitor.replan_requested is True
    assert monitor.failed_step_id == "s_req"
    assert monitor.step_status["s_opt"] == StepStatus.CANCELLED
    assert elapsed < 1.0


def test_cooperative_cancel_before_start_launches_nothing() -> None:
    plan = _plan([_step("s1"), _step("s2")])
    monitor = RunMonitor("t1")
    monitor.attach_plan(plan)
    monitor.request_cancel()
    runner = FakeRunner()
    asyncio.run(execute_plan(plan, monitor, runner=runner, concurrency=3))
    assert runner.started == []
    assert monitor.completed_count() == 0


def test_concurrency_cap_is_respected() -> None:
    plan = _plan([_step(f"s{i}") for i in range(4)])
    runner = FakeRunner(delays={f"s{i}": 0.1 for i in range(4)})
    monitor = _run(plan, runner, concurrency=2)
    assert monitor.completed_count() == 4
    assert runner.max_concurrent == 2


def _main() -> None:
    tests = [
        test_sequential_chain_runs_in_order_and_passes_outputs,
        test_independent_steps_run_in_parallel,
        test_skippable_failure_skips_dependents_siblings_survive,
        test_structural_failure_preempts_inflight_optional,
        test_cooperative_cancel_before_start_launches_nothing,
        test_concurrency_cap_is_respected,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _main()
