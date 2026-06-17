"""Offline tests for the synthesizer (mocked model, no network).

Covers structured synthesis with goal fencing, deterministic provenance/totals assembly, and the
failed/skipped step lists in the final result.

Run standalone: ``python tests/engine/test_synthesizer.py`` or via pytest.
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

from app.src.engine.monitor import RunMonitor  # noqa: E402
from app.src.engine.synthesizer import (  # noqa: E402
    Synthesis,
    build_final_result,
    synthesize,
)
from app.src.general_utils.agent_base import AgentResult  # noqa: E402
from app.src.schemas.plan import (  # noqa: E402
    ExecutionPlan,
    ExecutionStep,
)


class _Raw:
    def __init__(self) -> None:
        self.usage_metadata = {"input_tokens": 4, "output_tokens": 1, "total_tokens": 9}


class _Runnable:
    def __init__(self, model: FakeModel) -> None:
        self._model = model

    def invoke(self, messages: object) -> dict:
        self._model.seen = messages
        return {"parsed": self._model.synthesis, "raw": _Raw()}


class FakeModel:
    def __init__(self, synthesis: Synthesis) -> None:
        self.synthesis = synthesis
        self.seen: object = None

    def with_structured_output(self, schema: type, include_raw: bool = False) -> _Runnable:
        return _Runnable(self)


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        reasoning="r",
        task_id="t1",
        steps=[
            ExecutionStep(id="s1", agent="research", action="research"),
            ExecutionStep(id="s2", agent="analysis", action="analyze", dependencies=["s1"]),
        ],
    )


def _result(
    step_id: str, agent: str, status: str = "completed", conf: float | None = 0.8
) -> AgentResult:
    output: dict = {"content": f"text-{step_id}"}
    if conf is not None:
        output["confidence"] = conf
        output["sources"] = ["http://example.com"]
    if status == "failed":
        output = {"error": "boom"}
    return AgentResult(
        step_id=step_id, agent=agent, status=status, output=output,
        tokens_used=3, execution_time_ms=5,
    )


def test_synthesize_returns_structured_content() -> None:
    model = FakeModel(Synthesis(content="combined answer", confidence=0.77))
    plan = _plan()
    results = {"s1": _result("s1", "research")}
    synthesis, tokens = asyncio.run(synthesize("study X", plan, results, model=model))
    assert synthesis.content == "combined answer"
    assert tokens == 9
    user = model.seen[1]["content"]
    assert "<goal>" in user and "<outputs>" in user


def test_build_final_result_assembles_provenance_and_totals() -> None:
    monitor = RunMonitor("t1")
    plan = _plan()
    monitor.attach_plan(plan)
    monitor.start_step(plan.step_by_id("s1"))
    monitor.record_result(plan.step_by_id("s1"), _result("s1", "research"))
    monitor.start_step(plan.step_by_id("s2"))
    monitor.record_result(plan.step_by_id("s2"), _result("s2", "analysis", status="failed"))
    final = build_final_result(monitor, Synthesis(content="ans", confidence=0.6), "completed")
    assert final.status == "completed"
    assert final.total_tokens == 6
    assert final.failed_steps == ["s2"]
    prov_ids = {p.step_id for p in final.provenance}
    assert prov_ids == {"s1", "s2"}
    s1_prov = next(p for p in final.provenance if p.step_id == "s1")
    assert s1_prov.confidence == 0.8 and s1_prov.sources == ["http://example.com"]


def test_skipped_steps_listed() -> None:
    monitor = RunMonitor("t1")
    monitor.attach_plan(_plan())
    monitor.mark_skipped({"s2"})
    final = build_final_result(monitor, Synthesis(content="ans", confidence=0.5), "completed")
    assert final.skipped_steps == ["s2"]


def _main() -> None:
    tests = [
        test_synthesize_returns_structured_content,
        test_build_final_result_assembles_provenance_and_totals,
        test_skipped_steps_listed,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _main()
