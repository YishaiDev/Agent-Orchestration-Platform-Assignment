"""Offline tests for the HTTP API (FastAPI TestClient, fake models + runner, no network).

Covers the six endpoints: task submission, live status/trace, result retrieval (incl. the
not-ready 409 and unknown-task 404), cooperative cancel returning completed steps, and the agent
catalog. The full submit->poll->result path drives the real graph with injected fakes.

Run standalone: ``python tests/engine/test_api.py`` or via pytest.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")

from app.src.api.app import create_app  # noqa: E402
from app.src.engine.monitor import RunMonitor  # noqa: E402
from app.src.engine.nodes import EngineDeps  # noqa: E402
from app.src.engine.runs import RunRegistry  # noqa: E402
from app.src.engine.synthesizer import Synthesis  # noqa: E402
from app.src.general_utils.agent_base import AgentResult  # noqa: E402
from app.src.schemas.plan import (  # noqa: E402
    ExecutionPlan,
    ExecutionStep,
    PlannerDraft,
    StepStatus,
    SynthesisVerdict,
    TaskState,
)
from fastapi.testclient import TestClient  # noqa: E402


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
    """Fake model returning scripted outputs keyed by the requested schema name."""

    def __init__(self, by_schema: dict[str, list]) -> None:
        self.by_schema = {name: list(items) for name, items in by_schema.items()}

    def with_structured_output(self, schema: type, include_raw: bool = False) -> _Runnable:
        return _Runnable(self, schema.__name__)


class FakeRunner:
    async def __call__(self, step: ExecutionStep, results: dict, session: str) -> AgentResult:
        return AgentResult(
            step_id=step.id,
            agent=step.agent,
            status="completed",
            output={"content": f"out-{step.id}", "confidence": 0.9},
            tokens_used=1,
            execution_time_ms=1,
        )


def _two_step_draft() -> PlannerDraft:
    return PlannerDraft(
        reasoning="research then analyze",
        steps=[
            ExecutionStep(id="s1", agent="research", action="research"),
            ExecutionStep(id="s2", agent="analysis", action="analyze", dependencies=["s1"]),
        ],
    )


def _deps_factory(model: ScriptedModel):
    """Build a deps factory that injects the fake model and runner per run."""

    def factory(registry: RunRegistry) -> EngineDeps:
        return EngineDeps(
            registry=registry,
            runner=FakeRunner(),
            planner_model=model,
            decider_model=model,
            synth_model=model,
            judge_model=model,
            concurrency=3,
        )

    return factory


def _scripted_app():
    """Build an app whose runs use scripted fakes and a fresh registry."""
    model = ScriptedModel(
        {
            "PlannerDraft": [_two_step_draft()],
            "Synthesis": [Synthesis(content="final answer", confidence=0.9)],
            "SynthesisVerdict": [SynthesisVerdict(reasoning="grounded", verdict="accept")],
        }
    )
    registry = RunRegistry()
    return create_app(registry=registry, deps_factory=_deps_factory(model))


def _poll_until_terminal(client: TestClient, task_id: str, timeout: float = 5.0) -> dict:
    """Poll status until the task reaches a terminal state or the timeout elapses."""
    deadline = time.monotonic() + timeout
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/tasks/{task_id}").json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.02)
    return body


def test_submit_poll_and_result() -> None:
    with TestClient(_scripted_app()) as client:
        created = client.post("/tasks", json={"goal": "study X"})
        assert created.status_code == 202
        assert created.json()["status"] == "planning"
        task_id = created.json()["task_id"]
        status = _poll_until_terminal(client, task_id)
        assert status["status"] == "completed"
        assert status["progress"]["completed_steps"] == 2
        assert len(status["execution_trace"]) == 2
        entry = status["execution_trace"][0]
        assert "tokens_used" in entry and "execution_time_ms" in entry
        result = client.get(f"/tasks/{task_id}/result")
        assert result.status_code == 200
        body = result.json()
        assert body["result"]["content"] == "final answer"
        assert body["result"]["word_count"] == 2
        assert body["result"]["format"] == "markdown"
        assert len(body["execution_trace"]) == 2


def test_concurrent_submissions_tracked_independently() -> None:
    with TestClient(_scripted_app()) as client:
        task_ids = [
            client.post("/tasks", json={"goal": f"study {i}"}).json()["task_id"]
            for i in range(3)
        ]
        assert len(set(task_ids)) == 3
        for task_id in task_ids:
            status = _poll_until_terminal(client, task_id)
            assert status["status"] == "completed"
            result = client.get(f"/tasks/{task_id}/result").json()
            assert result["task_id"] == task_id
            assert result["result"]["content"] == "final answer"


def test_unknown_task_is_404() -> None:
    with TestClient(_scripted_app()) as client:
        assert client.get("/tasks/nope").status_code == 404
        assert client.get("/tasks/nope/result").status_code == 404


def test_result_not_ready_is_409() -> None:
    registry = RunRegistry()
    registry.create("t-pending")
    client = TestClient(create_app(registry=registry))
    assert client.get("/tasks/t-pending/result").status_code == 409


def test_cancel_returns_completed_steps() -> None:
    registry = RunRegistry()
    monitor: RunMonitor = registry.create("t-cancel")
    plan = ExecutionPlan(
        reasoning="r",
        task_id="t-cancel",
        steps=[ExecutionStep(id="s1", agent="research", action="research")],
    )
    monitor.attach_plan(plan)
    monitor.step_status["s1"] = StepStatus.COMPLETED
    client = TestClient(create_app(registry=registry))
    body = client.post("/tasks/t-cancel/cancel").json()
    assert body["status"] == "cancelled"
    assert body["completed_steps"] == ["s1"]
    assert monitor.cancelled is True


def test_agents_catalog_lists_four() -> None:
    client = TestClient(create_app(registry=RunRegistry()))
    body = client.get("/agents").json()
    agents = body["agents"]
    assert {a["name"] for a in agents} == {"research", "analysis", "code", "writing"}
    assert all(a["status"] == "available" for a in agents)


def test_plan_endpoint_exposes_validated_plan() -> None:
    with TestClient(_scripted_app()) as client:
        task_id = client.post("/tasks", json={"goal": "study X"}).json()["task_id"]
        _poll_until_terminal(client, task_id)
        plan = client.get(f"/tasks/{task_id}/plan")
        assert plan.status_code == 200
        body = plan.json()
        assert body["task_id"] == task_id
        assert [s["id"] for s in body["steps"]] == ["s1", "s2"]
        assert body["parallel_groups"] == [["s1"], ["s2"]]


def test_plan_not_ready_is_409() -> None:
    registry = RunRegistry()
    registry.create("t-pending")
    client = TestClient(create_app(registry=registry))
    assert client.get("/tasks/t-pending/plan").status_code == 409


def test_stream_emits_sse_events_until_terminal() -> None:
    registry = RunRegistry()
    monitor = registry.create("t-stream")
    plan = ExecutionPlan(
        reasoning="r",
        task_id="t-stream",
        steps=[ExecutionStep(id="s1", agent="research", action="research")],
    )
    monitor.attach_plan(plan)
    monitor.step_status["s1"] = StepStatus.COMPLETED
    monitor.set_state(TaskState.COMPLETED)
    client = TestClient(create_app(registry=registry))
    resp = client.get("/tasks/t-stream/stream")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "data:" in resp.text
    assert '"status": "completed"' in resp.text


def _main() -> None:
    tests = [
        test_submit_poll_and_result,
        test_concurrent_submissions_tracked_independently,
        test_unknown_task_is_404,
        test_result_not_ready_is_409,
        test_cancel_returns_completed_steps,
        test_agents_catalog_lists_four,
        test_plan_endpoint_exposes_validated_plan,
        test_plan_not_ready_is_409,
        test_stream_emits_sse_events_until_terminal,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _main()
