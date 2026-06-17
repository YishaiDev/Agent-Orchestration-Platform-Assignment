"""Offline tests for dispatch routing and upstream-context injection (agents stubbed).

Covers per-agent routing, dependency-output injection into the correct agent field (analysis data,
code/writing context), the deterministic context trim, and the failed-result guard on mapping
errors.

Run standalone: ``python tests/engine/test_dispatch.py`` or via pytest.
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

from app.src.engine import dispatch as dispatch_module  # noqa: E402
from app.src.general_utils.agent_base import AgentResult  # noqa: E402
from app.src.schemas.plan import ExecutionStep  # noqa: E402


def _result(step_id: str, output: dict | None = None, status: str = "completed") -> AgentResult:
    return AgentResult(
        step_id=step_id,
        agent="x",
        status=status,
        output=output or {"content": "done"},
        tokens_used=1,
        execution_time_ms=1,
    )


def _ok(step_id: str, **captured: object) -> AgentResult:
    return AgentResult(
        step_id=step_id, agent="stub", status="completed", output=dict(captured),
        tokens_used=0, execution_time_ms=0,
    )


def _patch(monkey: dict[str, object]) -> dict[str, object]:
    saved = {name: getattr(dispatch_module, name) for name in monkey}
    for name, fn in monkey.items():
        setattr(dispatch_module, name, fn)
    return saved


def _restore(saved: dict[str, object]) -> None:
    for name, fn in saved.items():
        setattr(dispatch_module, name, fn)


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


def _main() -> None:
    tests = [
        test_routes_analysis_with_upstream_data,
        test_routes_code_with_context_string,
        test_failed_upstream_not_injected,
        test_context_trimmed_to_budget,
        test_mapping_error_returns_failed_result,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _main()
