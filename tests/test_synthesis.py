"""Offline tests for the synthesis zone (mocked models, no network).

Covers the synthesizer (structured synthesis with goal fencing, deterministic provenance/totals
assembly, failed/skipped step lists) and the synthesis quality gate (the free deterministic checks
- confidence calibration, empty content, output-format compliance, attribution sanity - plus the
structured LLM judge returning accept / resynthesize / replan). The models are injected.

Run standalone: ``python tests/test_synthesis.py`` or via pytest.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from app.src.engine.monitor import RunMonitor  # noqa: E402
from app.src.engine.synthesis_judge import (  # noqa: E402
    calibrated_confidence,
    check_synthesis,
    judge_synthesis,
)
from app.src.engine.synthesizer import (  # noqa: E402
    Synthesis,
    build_final_result,
    synthesize,
)
from app.src.general_utils.agent_base import AgentResult  # noqa: E402
from app.src.schemas.plan import (  # noqa: E402
    ExecutionPlan,
    ExecutionStep,
    SynthesisVerdict,
)


class _SynthRaw:
    """Synthesizer model message: nine total tokens (asserted by a synthesizer test)."""

    def __init__(self) -> None:
        self.usage_metadata = {"input_tokens": 4, "output_tokens": 1, "total_tokens": 9}


class _JudgeRaw:
    """Judge model message: two total tokens (asserted by the judge tests)."""

    def __init__(self) -> None:
        self.usage_metadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


class _SynthRunnable:
    """Structured-output runnable returning a scripted ``Synthesis`` and recording messages."""

    def __init__(self, model: FakeSynthModel) -> None:
        self._model = model

    def invoke(self, messages: object) -> dict:
        self._model.seen = messages
        return {"parsed": self._model.synthesis, "raw": _SynthRaw()}


class FakeSynthModel:
    """Fake model returning a scripted ``Synthesis`` for ``with_structured_output``."""

    def __init__(self, synthesis: Synthesis) -> None:
        self.synthesis = synthesis
        self.seen: object = None

    def with_structured_output(self, schema: type, include_raw: bool = False) -> _SynthRunnable:
        return _SynthRunnable(self)


class _JudgeRunnable:
    """Structured-output runnable returning a scripted ``SynthesisVerdict``."""

    def __init__(self, verdict: SynthesisVerdict) -> None:
        self._verdict = verdict

    def invoke(self, messages: object) -> dict:
        return {"parsed": self._verdict, "raw": _JudgeRaw()}


class FakeJudgeModel:
    """Fake model returning a scripted ``SynthesisVerdict`` for ``with_structured_output``."""

    def __init__(self, verdict: SynthesisVerdict) -> None:
        self._verdict = verdict

    def with_structured_output(self, schema: type, include_raw: bool = False) -> _JudgeRunnable:
        return _JudgeRunnable(self._verdict)


def _plan() -> ExecutionPlan:
    """A two-step research-then-analyze plan shared by both sub-areas."""
    return ExecutionPlan(
        reasoning="r",
        task_id="t1",
        steps=[
            ExecutionStep(id="s1", agent="research", action="research"),
            ExecutionStep(id="s2", agent="analysis", action="analyze", dependencies=["s1"]),
        ],
    )


def _result(
    step_id: str, agent: str = "research", status: str = "completed", conf: float | None = 0.8
) -> AgentResult:
    """Build an ``AgentResult`` with optional confidence/sources; ``failed`` carries an error."""
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


# --- synthesizer ---


def test_synthesize_returns_structured_content() -> None:
    model = FakeSynthModel(Synthesis(content="combined answer", confidence=0.77))
    plan = _plan()
    results = {"s1": _result("s1", "research")}
    synthesis, tokens = asyncio.run(synthesize("study X", plan, results, model=model))
    assert synthesis.content == "combined answer"
    assert tokens == 9
    user = model.seen[1]["content"]
    assert "<goal>" in user and "<outputs>" in user


def test_code_output_rendered_verbatim_into_prompt() -> None:
    model = FakeSynthModel(Synthesis(content="ans", confidence=0.9))
    code = "import asyncio\n\nasync def main():\n    await asyncio.sleep(1)"
    results = {
        "s1": AgentResult(
            step_id="s1", agent="code", status="completed",
            output={"content": "explanation", "code": code, "language": "python"},
            tokens_used=3, execution_time_ms=5,
        )
    }
    asyncio.run(synthesize("write asyncio code", _plan(), results, model=model))
    user = model.seen[1]["content"]
    assert "```python" in user
    assert "async def main()" in user


def test_build_final_result_assembles_provenance_and_totals() -> None:
    monitor = RunMonitor("t1")
    plan = _plan()
    monitor.attach_plan(plan)
    monitor.start_step(plan.step_by_id("s1"))
    monitor.record_result(plan.step_by_id("s1"), _result("s1", "research"))
    monitor.start_step(plan.step_by_id("s2"))
    monitor.record_result(plan.step_by_id("s2"), _result("s2", "analysis", status="failed"))
    final = build_final_result(
        monitor, Synthesis(content="one two three", confidence=0.6), "completed", "markdown"
    )
    assert final.status == "completed"
    assert final.total_tokens == 6
    assert final.failed_steps == ["s2"]
    assert final.result.content == "one two three"
    assert final.result.format == "markdown"
    assert final.result.word_count == 3
    assert len(final.execution_trace) == 2
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


# --- synthesis quality judge ---


def _judge(verdict: SynthesisVerdict) -> SynthesisVerdict:
    """Run ``judge_synthesis`` with a fake judge model and assert its token capture."""
    out, tokens = asyncio.run(
        judge_synthesis(
            "study X", _plan(), {"s1": _result("s1"), "s2": _result("s2")},
            Synthesis(content="answer", confidence=0.8), [], model=FakeJudgeModel(verdict),
        )
    )
    assert tokens == 2
    return out


def test_calibrated_confidence_caps_by_ratio() -> None:
    assert calibrated_confidence(0.9, 1, 2) == 0.5
    assert calibrated_confidence(0.4, 2, 2) == 0.4
    assert calibrated_confidence(0.9, 0, 0) == 0.0


def test_check_synthesis_flags_empty_content() -> None:
    errors = check_synthesis(Synthesis(content="   ", confidence=0.5), _plan(), 1, None)
    assert any("empty" in e for e in errors)


def test_check_synthesis_passes_clean_draft() -> None:
    errors = check_synthesis(Synthesis(content="a clear answer", confidence=0.5), _plan(), 2, None)
    assert errors == []


def test_check_synthesis_json_format() -> None:
    bad = check_synthesis(Synthesis(content="not json", confidence=0.5), _plan(), 1, "json")
    good = check_synthesis(Synthesis(content='{"a": 1}', confidence=0.5), _plan(), 1, "json")
    assert any("JSON" in e for e in bad)
    assert good == []


def test_check_synthesis_bullet_format() -> None:
    bad = check_synthesis(Synthesis(content="prose only", confidence=0.5), _plan(), 1, "bullets")
    good = check_synthesis(Synthesis(content="- one\n- two", confidence=0.5), _plan(), 1, "bullets")
    assert any("bullet" in e for e in bad)
    assert good == []


def test_check_synthesis_attribution() -> None:
    errors = check_synthesis(
        Synthesis(content="grounded in [s9] only", confidence=0.5), _plan(), 1, None
    )
    assert any("s9" in e for e in errors)


def test_judge_returns_accept() -> None:
    out = _judge(SynthesisVerdict(reasoning="ok", verdict="accept"))
    assert out.verdict == "accept"


def test_judge_returns_resynthesize() -> None:
    out = _judge(SynthesisVerdict(reasoning="weak", verdict="resynthesize", feedback="fix it"))
    assert out.verdict == "resynthesize"
    assert out.feedback == "fix it"


def test_judge_returns_replan() -> None:
    steps = [ExecutionStep(id="b", agent="analysis", action="analyze")]
    out = _judge(SynthesisVerdict(reasoning="gap", verdict="replan", new_steps=steps))
    assert out.verdict == "replan"
    assert out.new_steps[0].id == "b"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
