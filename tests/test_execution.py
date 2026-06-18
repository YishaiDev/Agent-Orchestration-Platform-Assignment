"""Offline tests for the execution zone (fake runner, stubbed agents, no network).

Covers the inner async scheduler (sequential dependency passing, genuine parallelism, skippable
cascade with surviving siblings, structural preemption of in-flight work, cooperative cancel, the
concurrency cap), dispatch routing with upstream-context injection and the context-budget trim, and
the deterministic prompt-token estimator.

Run standalone: ``python tests/test_execution.py`` or via pytest.
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

from app.src.engine import dispatch as dispatch_module  # noqa: E402
from app.src.engine.monitor import RunMonitor  # noqa: E402
from app.src.engine.scheduler import execute_plan  # noqa: E402
from app.src.general_utils.agent_base import AgentResult  # noqa: E402
from app.src.general_utils.tokens import count_prompt_tokens  # noqa: E402
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
    """Wrap steps in an execution plan for scheduler runs."""
    return ExecutionPlan(reasoning="r", task_id="t1", steps=steps)


def _run(plan: ExecutionPlan, runner: FakeRunner, concurrency: int | None = None) -> RunMonitor:
    """Attach a monitor, mark executing, and run the plan to completion."""
    monitor = RunMonitor("t1")
    monitor.attach_plan(plan)
    monitor.set_state(TaskState.EXECUTING)
    asyncio.run(execute_plan(plan, monitor, runner=runner, concurrency=concurrency))
    return monitor


def _step(
    step_id: str, agent: str = "research", action: str = "research", **kw: object
) -> ExecutionStep:
    """Build a single execution step with sensible research defaults."""
    return ExecutionStep(id=step_id, agent=agent, action=action, **kw)  # type: ignore[arg-type]


def _result(step_id: str, output: dict | None = None, status: str = "completed") -> AgentResult:
    """Build a dispatch upstream result with default completed content."""
    return AgentResult(
        step_id=step_id,
        agent="x",
        status=status,
        output=output or {"content": "done"},
        tokens_used=1,
        execution_time_ms=1,
    )


def _ok(step_id: str, **captured: object) -> AgentResult:
    """Build a stub agent result echoing whatever the fake entrypoint captured."""
    return AgentResult(
        step_id=step_id, agent="stub", status="completed", output=dict(captured),
        tokens_used=0, execution_time_ms=0,
    )


def _patch(monkey: dict[str, object]) -> dict[str, object]:
    """Swap dispatch module entrypoints, returning the saved originals."""
    saved = {name: getattr(dispatch_module, name) for name in monkey}
    for name, fn in monkey.items():
        setattr(dispatch_module, name, fn)
    return saved


def _restore(saved: dict[str, object]) -> None:
    """Restore dispatch module entrypoints captured by ``_patch``."""
    for name, fn in saved.items():
        setattr(dispatch_module, name, fn)


# --- scheduler ---


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
            _step("s1", optional=True),
            _step("s2"),
            _step(
                "s3", agent="analysis", action="analyze",
                dependencies=["s1"], optional=True,
            ),
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


# --- dispatch ---


def test_routes_analysis_with_upstream_data() -> None:
    async def fake_analysis(instruction, action, data, sources, step_id, session_id):  # noqa: ANN001
        return _ok(step_id, action=action, data=data, instruction=instruction)

    saved = _patch({"run_analysis_agent": fake_analysis})
    try:
        step = ExecutionStep(id="s2", agent="analysis", action="compare",
                             input={"instruction": "compare them"}, dependencies=["s1"])
        results = {"s1": _result("s1", {"content": "upstream facts"})}
        out = asyncio.run(dispatch_module.dispatch(step, results))
    finally:
        _restore(saved)
    assert out.output["action"] == "compare"
    assert out.output["data"] == [{"content": "upstream facts"}]
    assert out.output["instruction"] == "compare them"


def test_routes_analysis_injects_all_upstream_deps() -> None:
    async def fake_analysis(instruction, action, data, sources, step_id, session_id):  # noqa: ANN001
        return _ok(step_id, data=data)

    saved = _patch({"run_analysis_agent": fake_analysis})
    try:
        step = ExecutionStep(id="s3", agent="analysis", action="analyze",
                             input={"instruction": "analyze both"}, dependencies=["s1", "s2"])
        results = {
            "s1": _result("s1", {"content": "facts one"}),
            "s2": _result("s2", {"content": "facts two"}),
        }
        out = asyncio.run(dispatch_module.dispatch(step, results))
    finally:
        _restore(saved)
    contents = [item["content"] for item in out.output["data"]]
    assert contents == ["facts one", "facts two"]


def test_missing_dependency_in_results_yields_empty_data() -> None:
    async def fake_analysis(instruction, action, data, sources, step_id, session_id):  # noqa: ANN001
        return _ok(step_id, data=data)

    saved = _patch({"run_analysis_agent": fake_analysis})
    try:
        step = ExecutionStep(id="s2", agent="analysis", action="analyze",
                             input={"instruction": "analyze"}, dependencies=["s1"])
        out = asyncio.run(dispatch_module.dispatch(step, {}))
    finally:
        _restore(saved)
    assert out.output["data"] is None


def test_routes_code_with_context_string() -> None:
    async def fake_code(task_input, action, step_id, language, upstream_context):  # noqa: ANN001
        return _ok(step_id, task_input=task_input, language=language, context=upstream_context)

    saved = _patch({"run_code_agent": fake_code})
    try:
        step = ExecutionStep(id="s2", agent="code", action="generate",
                             input={"task_input": "write fn", "language": "python"},
                             dependencies=["s1"])
        results = {"s1": _result("s1", {"content": "spec detail"})}
        out = asyncio.run(dispatch_module.dispatch(step, results))
    finally:
        _restore(saved)
    assert out.output["language"] == "python"
    assert "spec detail" in out.output["context"]
    assert "[s1/x]" in out.output["context"]


def test_failed_upstream_not_injected() -> None:
    async def fake_code(task_input, action, step_id, language, upstream_context):  # noqa: ANN001
        return _ok(step_id, context=upstream_context)

    saved = _patch({"run_code_agent": fake_code})
    try:
        step = ExecutionStep(id="s2", agent="code", action="generate",
                             input={"task_input": "x"}, dependencies=["s1"])
        results = {"s1": _result("s1", {"error": "boom"}, status="failed")}
        out = asyncio.run(dispatch_module.dispatch(step, results))
    finally:
        _restore(saved)
    assert out.output["context"] == ""


def test_context_trimmed_to_budget() -> None:
    huge = {"s1": _result("s1", {"content": "z" * 50_000})}
    captured: dict[str, str] = {}

    def fake_writing(inp, step_id):  # noqa: ANN001 - sync, matching the real entrypoint
        captured["src"] = inp.source_material
        return _ok(step_id)

    saved = _patch({"run_writing_agent": fake_writing})
    try:
        step = ExecutionStep(id="s2", agent="writing", action="write",
                             input={"instruction": "summarize"}, dependencies=["s1"])
        asyncio.run(dispatch_module.dispatch(step, huge))
    finally:
        _restore(saved)
    assert len(captured["src"]) <= 6000


def test_mapping_error_returns_failed_result() -> None:
    async def boom(*a: object, **k: object) -> AgentResult:
        raise ValueError("bad mapping")

    saved = _patch({"run_research_agent": boom})
    try:
        step = ExecutionStep(id="s1", agent="research", action="research",
                             input={"subtopic": "x"})
        out = asyncio.run(dispatch_module.dispatch(step, {}))
    finally:
        _restore(saved)
    assert out.status == "failed"
    assert "bad mapping" in out.output["error"]


# --- token estimator ---


def test_empty_prompt_is_zero() -> None:
    assert count_prompt_tokens([]) == 0
    assert count_prompt_tokens([{"role": "user", "content": ""}]) == 0


def test_counts_real_prompt_length() -> None:
    messages = [{"role": "user", "content": "x" * 40}]
    assert count_prompt_tokens(messages, chars_per_token=4.0) == 10


def test_scales_with_prompt_size() -> None:
    small = [{"role": "user", "content": "short"}]
    large = [{"role": "user", "content": "long " * 500}]
    assert count_prompt_tokens(large) > count_prompt_tokens(small)


def test_aggregates_across_messages() -> None:
    one = [{"role": "system", "content": "a" * 20}]
    two = [{"role": "system", "content": "a" * 20}, {"role": "user", "content": "b" * 20}]
    assert count_prompt_tokens(two) > count_prompt_tokens(one)


def test_chars_per_token_changes_estimate() -> None:
    messages = [{"role": "user", "content": "y" * 40}]
    coarse = count_prompt_tokens(messages, chars_per_token=8.0)
    fine = count_prompt_tokens(messages, chars_per_token=2.0)
    assert fine > coarse


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
