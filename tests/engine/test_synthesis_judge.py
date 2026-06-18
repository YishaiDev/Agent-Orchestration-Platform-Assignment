"""Offline unit tests for the synthesis quality gate (deterministic checks + LLM judge).

Covers the free no-LLM checks (confidence calibration, empty content, output-format compliance,
attribution sanity) and the structured judge call with a fake model returning each verdict. No
network: the judge model is injected.

Run standalone: ``python tests/engine/test_synthesis_judge.py`` or via pytest.
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

from app.src.engine.synthesis_judge import (  # noqa: E402
    calibrated_confidence,
    check_synthesis,
    judge_synthesis,
)
from app.src.engine.synthesizer import Synthesis  # noqa: E402
from app.src.general_utils.agent_base import AgentResult  # noqa: E402
from app.src.schemas.plan import ExecutionPlan, ExecutionStep, SynthesisVerdict  # noqa: E402


class _Raw:
    def __init__(self) -> None:
        self.usage_metadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


class _Runnable:
    def __init__(self, verdict: SynthesisVerdict) -> None:
        self._verdict = verdict

    def invoke(self, messages: object) -> dict:
        return {"parsed": self._verdict, "raw": _Raw()}


class FakeJudgeModel:
    def __init__(self, verdict: SynthesisVerdict) -> None:
        self._verdict = verdict

    def with_structured_output(self, schema: type, include_raw: bool = False) -> _Runnable:
        return _Runnable(self._verdict)


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        reasoning="two steps",
        task_id="t-x",
        steps=[
            ExecutionStep(id="s1", agent="research", action="research"),
            ExecutionStep(id="s2", agent="analysis", action="analyze", dependencies=["s1"]),
        ],
    )


def _result(step_id: str) -> AgentResult:
    return AgentResult(
        step_id=step_id, agent="research", status="completed",
        output={"content": f"out-{step_id}", "confidence": 0.9},
        tokens_used=1, execution_time_ms=1,
    )


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


def _judge(verdict: SynthesisVerdict) -> SynthesisVerdict:
    out, tokens = asyncio.run(
        judge_synthesis(
            "study X", _plan(), {"s1": _result("s1"), "s2": _result("s2")},
            Synthesis(content="answer", confidence=0.8), [], model=FakeJudgeModel(verdict),
        )
    )
    assert tokens == 2
    return out


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


def _main() -> None:
    tests = [
        test_calibrated_confidence_caps_by_ratio,
        test_check_synthesis_flags_empty_content,
        test_check_synthesis_passes_clean_draft,
        test_check_synthesis_json_format,
        test_check_synthesis_bullet_format,
        test_check_synthesis_attribution,
        test_judge_returns_accept,
        test_judge_returns_resynthesize,
        test_judge_returns_replan,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _main()
