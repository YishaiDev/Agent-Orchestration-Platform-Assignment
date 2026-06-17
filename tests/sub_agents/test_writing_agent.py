"""Offline tests for the Writing Agent reflection loop (mocked Gemini, scripted judge verdicts).

Run standalone: ``python tests/sub_agents/test_writing_agent.py`` or via pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.src.sub_agents.writing.agent import build_writing_graph, initial_state  # noqa: E402
from app.src.sub_agents.writing.schemas import (  # noqa: E402
    ContentOut,
    JudgeVerdict,
    WritingInput,
)

WRITER_CONTENT = "written body text here now"


class _Raw:
    """Minimal stand-in for a model message carrying token usage metadata."""

    def __init__(self, total_tokens: int) -> None:
        self.usage_metadata = {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": total_tokens,
        }


class _Runnable:
    """Structured-output runnable returning scripted parsed values plus token-bearing raw."""

    def __init__(self, model: FakeModel, schema: type) -> None:
        self._model = model
        self._schema = schema

    def invoke(self, _messages: object) -> dict:
        self._model.calls.append(self._schema.__name__)
        if self._schema is JudgeVerdict:
            parsed: object = self._model.next_verdict()
        else:
            parsed = ContentOut(content=WRITER_CONTENT)
        return {"parsed": parsed, "raw": _Raw(self._model.tokens_per_call)}


class FakeModel:
    """Fake chat model: writer schemas return canned content; judge pops scripted verdicts."""

    def __init__(self, verdicts: list[JudgeVerdict] | None = None, tokens_per_call: int = 5):
        self.verdicts = list(verdicts or [])
        self.tokens_per_call = tokens_per_call
        self.calls: list[str] = []

    def with_structured_output(self, schema: type, include_raw: bool = False) -> _Runnable:
        return _Runnable(self, schema)

    def next_verdict(self) -> JudgeVerdict:
        return self.verdicts.pop(0)


def _verdict(kind: str, issues: list[str] | None = None) -> JudgeVerdict:
    return JudgeVerdict(
        edit_ok=kind != "reedit",
        format_ok=kind != "reformat",
        verdict=kind,
        issues=issues or ([] if kind == "return" else ["fix it"]),
    )


def _run(writer: FakeModel, judge: FakeModel, max_revisions: int = 2) -> dict:
    graph = build_writing_graph(writer, judge, max_revisions)
    inp = WritingInput(
        instruction="Write about X", source_material="facts", output_format="markdown"
    )
    return graph.invoke(initial_state(inp, max_words=100))


def test_happy_path_order_and_finalizes() -> None:
    writer = FakeModel()
    judge = FakeModel([_verdict("return")])
    final = _run(writer, judge)
    assert writer.calls == ["ContentOut", "ContentOut", "ContentOut"]
    assert judge.calls == ["JudgeVerdict"]
    assert final["content"] == WRITER_CONTENT
    assert final["word_count"] == 5
    assert final["verdict"] == "return"


def test_reedit_routes_back_to_edit() -> None:
    writer = FakeModel()
    judge = FakeModel([_verdict("reedit"), _verdict("return")])
    final = _run(writer, judge)
    assert final["edit_runs"] == 2
    assert final["format_runs"] == 2


def test_reformat_routes_to_format_only() -> None:
    writer = FakeModel()
    judge = FakeModel([_verdict("reformat"), _verdict("return")])
    final = _run(writer, judge)
    assert final["edit_runs"] == 1
    assert final["format_runs"] == 2


def test_revision_cap_forces_return() -> None:
    writer = FakeModel()
    judge = FakeModel([_verdict("reedit") for _ in range(5)])
    final = _run(writer, judge, max_revisions=2)
    assert final["cycles"] == 3
    assert final["verdict"] == "reedit"
    assert final["issues"]


def test_tokens_accumulated_across_all_nodes() -> None:
    writer = FakeModel(tokens_per_call=5)
    judge = FakeModel([_verdict("return")], tokens_per_call=5)
    final = _run(writer, judge)
    assert final["tokens_used"] == 20


def _main() -> None:
    tests = [
        test_happy_path_order_and_finalizes,
        test_reedit_routes_back_to_edit,
        test_reformat_routes_to_format_only,
        test_revision_cap_forces_return,
        test_tokens_accumulated_across_all_nodes,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _main()
