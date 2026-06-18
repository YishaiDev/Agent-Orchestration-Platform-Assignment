"""Offline integration tests for the LangGraph outer loop (fake models + fake runner, no network).

Drives the whole plan -> execute -> evaluate -> synthesize loop with a schema-routing fake model and
a fake step runner: the happy path, structural-failure re-plan recovery (loops back to execute), and
the bounded give-up that synthesizes a failed partial without consulting the decider.

Run standalone: ``python tests/engine/test_graph.py`` or via pytest.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")

from app.src.engine.graph import run_task  # noqa: E402
from app.src.engine.nodes import EngineDeps  # noqa: E402
from app.src.engine.runs import RunRegistry  # noqa: E402
from app.src.engine.synthesizer import Synthesis  # noqa: E402
from app.src.general_utils.agent_base import AgentResult  # noqa: E402
from app.src.schemas.plan import (  # noqa: E402
    ExecutionStep,
    PlannerDraft,
    ReplanDecision,
    SynthesisVerdict,
)


class _Raw:
    def __init__(self) -> None:
        self.usage_metadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


class _Runnable:
    def __init__(self, model: ScriptedModel, schema_name: str) -> None:
        self._model = model
        self._name = schema_name

    def invoke(self, messages: object) -> dict:
        queue = self._model.by_schema[self._name]
        out = queue.pop(0) if len(queue) > 1 else queue[0]
        return {"parsed": out, "raw": _Raw()}


class ScriptedModel:
    """Fake model that returns scripted outputs keyed by the requested schema name."""

    def __init__(self, by_schema: dict[str, list]) -> None:
        self.by_schema = {name: list(items) for name, items in by_schema.items()}

    def with_structured_output(self, schema: type, include_raw: bool = False) -> _Runnable:
        return _Runnable(self, schema.__name__)


class FakeRunner:
    def __init__(self, fails: set[str] | None = None) -> None:
        self.fails = fails or set()
        self.ran: list[str] = []

    async def __call__(self, step: ExecutionStep, results: dict, session: str) -> AgentResult:
        self.ran.append(step.id)
        failed = step.id in self.fails
        return AgentResult(
            step_id=step.id, agent=step.agent, status="failed" if failed else "completed",
            output=(
                {"error": "boom"}
                if failed
                else {"content": f"out-{step.id}", "confidence": 0.9}
            ),
            tokens_used=1, execution_time_ms=1,
        )


def _two_step_draft() -> PlannerDraft:
    return PlannerDraft(
        reasoning="research then analyze",
        steps=[
            ExecutionStep(id="s1", agent="research", action="research"),
            ExecutionStep(id="s2", agent="analysis", action="analyze", dependencies=["s1"]),
        ],
    )


def _single_required_draft() -> PlannerDraft:
    return PlannerDraft(
        reasoning="one required step",
        steps=[ExecutionStep(id="s1", agent="research", action="research")],
    )


def _run(deps: EngineDeps, **kw: object) -> dict:
    return asyncio.run(run_task("t-x", "study X", "", "sess", deps=deps, **kw))


def _accept_verdict() -> SynthesisVerdict:
    return SynthesisVerdict(reasoning="grounded and complete", verdict="accept")


def test_happy_path_completes() -> None:
    model = ScriptedModel(
        {
            "PlannerDraft": [_two_step_draft()],
            "Synthesis": [Synthesis(content="final answer", confidence=0.9)],
            "SynthesisVerdict": [_accept_verdict()],
        }
    )
    registry = RunRegistry()
    deps = EngineDeps(registry=registry, runner=FakeRunner(), planner_model=model,
                      decider_model=model, synth_model=model, judge_model=model, concurrency=3)
    state = _run(deps, max_replans=1)
    assert state["final_result"]["status"] == "completed"
    assert state["final_output"] == "final answer"
    monitor = registry.get("t-x")
    assert monitor.completed_count() == 2


def test_structural_failure_replans_and_recovers() -> None:
    model = ScriptedModel(
        {
            "PlannerDraft": [_single_required_draft()],
            "ReplanDecision": [
                ReplanDecision(
                    reasoning="retry via a fresh step",
                    decision="replan",
                    new_steps=[ExecutionStep(id="a", agent="research", action="research")],
                )
            ],
            "Synthesis": [Synthesis(content="recovered", confidence=0.8)],
            "SynthesisVerdict": [_accept_verdict()],
        }
    )
    registry = RunRegistry()
    runner = FakeRunner(fails={"s1"})
    deps = EngineDeps(registry=registry, runner=runner, planner_model=model,
                      decider_model=model, synth_model=model, judge_model=model, concurrency=3)
    state = _run(deps, max_replans=1)
    assert state["replans"] == 1
    assert state["final_result"]["status"] == "completed"
    assert state["final_output"] == "recovered"
    assert "r1_a" in runner.ran


def test_bounded_replan_gives_up_to_failed() -> None:
    model = ScriptedModel({"PlannerDraft": [_single_required_draft()]})
    registry = RunRegistry()
    deps = EngineDeps(registry=registry, runner=FakeRunner(fails={"s1"}), planner_model=model,
                      decider_model=model, synth_model=model, judge_model=model, concurrency=3)
    state = _run(deps, max_replans=0)
    assert state["replans"] == 0
    assert state["final_result"]["status"] == "failed"
    assert "s1" in state["final_result"]["failed_steps"]


def test_resynthesize_loop_then_accepts() -> None:
    model = ScriptedModel(
        {
            "PlannerDraft": [_two_step_draft()],
            "Synthesis": [
                Synthesis(content="draft one", confidence=0.5),
                Synthesis(content="draft two", confidence=0.9),
            ],
            "SynthesisVerdict": [
                SynthesisVerdict(reasoning="unsupported claim", verdict="resynthesize",
                                 feedback="drop the unsupported claim"),
                _accept_verdict(),
            ],
        }
    )
    registry = RunRegistry()
    deps = EngineDeps(registry=registry, runner=FakeRunner(), planner_model=model,
                      decider_model=model, synth_model=model, judge_model=model, concurrency=3)
    state = _run(deps, max_replans=1, max_resynth=2)
    assert state["resynth_rounds"] == 1
    assert state["final_output"] == "draft two"
    assert state["final_result"]["status"] == "completed"


def test_judge_replan_loops_through_execute() -> None:
    model = ScriptedModel(
        {
            "PlannerDraft": [_single_required_draft()],
            "Synthesis": [
                Synthesis(content="partial", confidence=0.5),
                Synthesis(content="complete", confidence=0.9),
            ],
            "SynthesisVerdict": [
                SynthesisVerdict(
                    reasoning="coverage gap", verdict="replan",
                    feedback="cover the missing entity",
                    new_steps=[ExecutionStep(id="b", agent="analysis", action="analyze")],
                ),
                _accept_verdict(),
            ],
        }
    )
    registry = RunRegistry()
    runner = FakeRunner()
    deps = EngineDeps(registry=registry, runner=runner, planner_model=model,
                      decider_model=model, synth_model=model, judge_model=model, concurrency=3)
    state = _run(deps, max_replans=1)
    assert state["replans"] == 1
    assert "r1_b" in runner.ran
    assert state["final_output"] == "complete"
    assert state["final_result"]["status"] == "completed"


def test_resynthesize_budget_exhausted_degrades() -> None:
    model = ScriptedModel(
        {
            "PlannerDraft": [_two_step_draft()],
            "Synthesis": [Synthesis(content="draft", confidence=0.6)],
            "SynthesisVerdict": [
                SynthesisVerdict(reasoning="still weak", verdict="resynthesize",
                                 feedback="tighten the wording"),
            ],
        }
    )
    registry = RunRegistry()
    deps = EngineDeps(registry=registry, runner=FakeRunner(), planner_model=model,
                      decider_model=model, synth_model=model, judge_model=model, concurrency=3)
    state = _run(deps, max_replans=0, max_resynth=1)
    assert state["resynth_rounds"] == 1
    assert state["final_result"]["status"] == "completed_degraded"


class _CodeRunner(FakeRunner):
    async def __call__(self, step: ExecutionStep, results: dict, session: str) -> AgentResult:
        self.ran.append(step.id)
        return AgentResult(
            step_id=step.id, agent=step.agent, status="completed",
            output={"content": f"out-{step.id}", "code": "print('hi')",
                    "language": "python", "confidence": 0.9},
            tokens_used=1, execution_time_ms=1,
        )


def test_synthesis_failure_falls_back_to_degraded() -> None:
    import app.src.engine.nodes as nodes_mod

    async def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("tool_use_failed: malformed function call")

    model = ScriptedModel(
        {
            "PlannerDraft": [PlannerDraft(reasoning="code it",
                steps=[ExecutionStep(id="s1", agent="code", action="generate")])],
            "SynthesisVerdict": [_accept_verdict()],
        }
    )
    registry = RunRegistry()
    deps = EngineDeps(registry=registry, runner=_CodeRunner(), planner_model=model,
                      decider_model=model, synth_model=model, judge_model=model, concurrency=3)
    original = nodes_mod.synthesize
    nodes_mod.synthesize = _boom  # type: ignore[assignment]
    try:
        state = _run(deps, max_replans=1)
    finally:
        nodes_mod.synthesize = original  # type: ignore[assignment]
    assert state["final_result"]["status"] == "completed_degraded"
    assert state["synth_failed"] is True
    assert "```python" in state["final_output"] and "print('hi')" in state["final_output"]


def test_judge_failure_accepts_degraded() -> None:
    import app.src.engine.nodes as nodes_mod

    async def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("tool_use_failed")

    model = ScriptedModel(
        {
            "PlannerDraft": [_two_step_draft()],
            "Synthesis": [Synthesis(content="real synthesis", confidence=0.9)],
        }
    )
    registry = RunRegistry()
    deps = EngineDeps(registry=registry, runner=FakeRunner(), planner_model=model,
                      decider_model=model, synth_model=model, judge_model=model, concurrency=3)
    original = nodes_mod.judge_synthesis
    nodes_mod.judge_synthesis = _boom  # type: ignore[assignment]
    try:
        state = _run(deps, max_replans=1)
    finally:
        nodes_mod.judge_synthesis = original  # type: ignore[assignment]
    assert state["final_result"]["status"] == "completed_degraded"
    assert state["final_output"] == "real synthesis"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
