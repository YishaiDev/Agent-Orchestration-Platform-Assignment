"""Build, compile, and drive the LangGraph outer loop.

The outer loop is a small fixed state machine — ``plan -> execute -> evaluate -> synthesize`` — with
one LLM-routed conditional edge (``evaluate`` back to ``execute`` for a bounded re-plan, else onward
to ``synthesize``). A ``MemorySaver`` checkpointer persists outer-loop state per task. The inner
concurrent step execution stays in the plain-async scheduler that ``execute`` wraps.
"""

from __future__ import annotations

from collections.abc import Hashable
from functools import lru_cache
from typing import Any, cast

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.src.engine.nodes import (
    EngineDeps,
    evaluate_node,
    execute_node,
    judge_node,
    plan_node,
    route_after_evaluate,
    route_after_judge,
    route_after_plan,
    synthesize_node,
)
from app.src.engine.runs import RunRegistry, get_run_registry
from app.src.schemas.config import get_config
from app.src.schemas.run_state import RunState, initial_state

_BRANCH: dict[Hashable, str] = {"execute": "execute", "synthesize": "synthesize"}
_JUDGE_BRANCH: dict[Hashable, str] = {"synthesize": "synthesize", "execute": "execute", "end": END}


def build_graph() -> Any:
    """Build and compile the outer-loop state graph with a MemorySaver checkpointer."""
    graph = StateGraph(RunState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("judge", judge_node)
    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", route_after_plan, _BRANCH)
    graph.add_edge("execute", "evaluate")
    graph.add_conditional_edges("evaluate", route_after_evaluate, _BRANCH)
    graph.add_edge("synthesize", "judge")
    graph.add_conditional_edges("judge", route_after_judge, _JUDGE_BRANCH)
    return graph.compile(checkpointer=MemorySaver())


@lru_cache(maxsize=1)
def get_graph() -> Any:
    """Return the process-wide compiled graph (built once)."""
    return build_graph()


def _recursion_limit(max_replans: int, max_resynth: int) -> int:
    """Compute a safe recursion backstop covering all bounded re-plan and re-synthesis loops."""
    return max(40, (max_replans + 1) * (5 + 2 * max_resynth) + 10)


async def run_task(
    task_id: str,
    goal: str,
    constraints: str,
    session_id: str,
    deps: EngineDeps | None = None,
    max_replans: int | None = None,
    max_resynth: int | None = None,
    output_format: str = "",
    deadline_seconds: float | None = None,
) -> RunState:
    """Drive one task through the outer loop, returning the terminal state.

    Args:
        task_id: Engine-assigned task id.
        goal: The untrusted task goal.
        constraints: Optional untrusted constraints text.
        session_id: Session id carried into agent calls.
        deps: Injected engine dependencies (defaults to production registry + real agents/models).
        max_replans: Re-plan bound (defaults to the configured value).
        max_resynth: Re-synthesis bound (defaults to the configured value).
        output_format: Optional requested output format, checked at synthesis time.
        deadline_seconds: Optional wall-clock budget.

    Returns:
        The terminal RunState (carries ``final_result`` and ``final_output``).
    """
    registry: RunRegistry = deps.registry if deps else get_run_registry()
    deps = deps or EngineDeps(registry=registry)
    orchestrator = get_config().orchestrator
    bound = max_replans if max_replans is not None else orchestrator.max_replans
    bound_resynth = max_resynth if max_resynth is not None else orchestrator.max_resynth
    if registry.get(task_id) is None:
        registry.create(task_id, deadline_seconds)
    state = initial_state(
        task_id, goal, constraints, session_id, bound, bound_resynth, output_format
    )
    config = {
        "configurable": {"deps": deps, "thread_id": task_id},
        "recursion_limit": _recursion_limit(bound, bound_resynth),
    }
    final = await get_graph().ainvoke(state, config=config)
    return cast(RunState, final)
