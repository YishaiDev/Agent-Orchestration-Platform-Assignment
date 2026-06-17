"""Offline tests for the agent registry and deterministic plan validation (no LLM, no network).

Covers the routing allowlist, the ``/agents`` view, topological-level derivation, cycle detection,
and rejection of unknown agents/actions, dangling dependencies, duplicate ids, and empty plans.

Run standalone: ``python tests/engine/test_registry_validation.py`` or via pytest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from app.src.engine.registry import (  # noqa: E402
    agent_names,
    describe_agents,
    is_allowed,
)
from app.src.engine.validation import (  # noqa: E402
    PlanValidationError,
    derive_parallel_groups,
    validate_and_finalize,
)
from app.src.schemas.plan import ExecutionStep, PlannerDraft  # noqa: E402


def _draft(steps: list[ExecutionStep]) -> PlannerDraft:
    return PlannerDraft(reasoning="because", steps=steps)


def test_registry_has_four_agents() -> None:
    assert agent_names() == {"research", "analysis", "code", "writing"}


def test_is_allowed_enforces_capabilities() -> None:
    assert is_allowed("analysis", "compare") is True
    assert is_allowed("analysis", "research") is False
    assert is_allowed("ghost", "anything") is False


def test_describe_agents_shape() -> None:
    described = describe_agents()
    assert len(described) == 4
    sample = next(item for item in described if item["name"] == "code")
    assert "generate" in sample["capabilities"]
    assert sample["status"] == "available"


def test_finalize_derives_parallel_groups() -> None:
    steps = [
        ExecutionStep(id="s1", agent="research", action="research"),
        ExecutionStep(id="s2", agent="research", action="research"),
        ExecutionStep(id="s3", agent="analysis", action="analyze", dependencies=["s1", "s2"]),
    ]
    plan = validate_and_finalize(_draft(steps), "t1")
    assert plan.parallel_groups == [["s1", "s2"], ["s3"]]
    assert plan.task_id == "t1"


def test_unknown_agent_action_rejected() -> None:
    steps = [ExecutionStep(id="s1", agent="analysis", action="research")]
    try:
        validate_and_finalize(_draft(steps), "t1")
        raise AssertionError("expected PlanValidationError")
    except PlanValidationError as exc:
        assert any("unknown agent/action" in e for e in exc.errors)


def test_dangling_dependency_rejected() -> None:
    steps = [ExecutionStep(id="s1", agent="research", action="research", dependencies=["sX"])]
    try:
        validate_and_finalize(_draft(steps), "t1")
        raise AssertionError("expected PlanValidationError")
    except PlanValidationError as exc:
        assert any("unknown dependency" in e for e in exc.errors)


def test_duplicate_ids_rejected() -> None:
    steps = [
        ExecutionStep(id="s1", agent="research", action="research"),
        ExecutionStep(id="s1", agent="research", action="research"),
    ]
    try:
        validate_and_finalize(_draft(steps), "t1")
        raise AssertionError("expected PlanValidationError")
    except PlanValidationError as exc:
        assert any("duplicate step id" in e for e in exc.errors)


def test_empty_plan_rejected() -> None:
    try:
        validate_and_finalize(_draft([]), "t1")
        raise AssertionError("expected PlanValidationError")
    except PlanValidationError as exc:
        assert any("no steps" in e for e in exc.errors)


def test_cycle_detected() -> None:
    steps = [
        ExecutionStep(id="s1", agent="research", action="research", dependencies=["s2"]),
        ExecutionStep(id="s2", agent="research", action="research", dependencies=["s1"]),
    ]
    try:
        derive_parallel_groups(steps)
        raise AssertionError("expected PlanValidationError")
    except PlanValidationError as exc:
        assert any("cycle" in e for e in exc.errors)


def _main() -> None:
    tests = [
        test_registry_has_four_agents,
        test_is_allowed_enforces_capabilities,
        test_describe_agents_shape,
        test_finalize_derives_parallel_groups,
        test_unknown_agent_action_rejected,
        test_dangling_dependency_rejected,
        test_duplicate_ids_rejected,
        test_empty_plan_rejected,
        test_cycle_detected,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _main()
