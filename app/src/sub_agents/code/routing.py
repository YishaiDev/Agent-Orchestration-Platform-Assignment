"""Conditional routing after the judge node."""

from __future__ import annotations

from langgraph.graph import END

from app.src.sub_agents.code.schemas import CodeState


def route_after_judge(state: CodeState) -> str:
    """Decide the next node from the judge signal, bounded by the retry cap.

    Args:
        state: Current code state (reads ``problem`` and ``rounds``).

    Returns:
        ``"refine"`` to regenerate, or ``END`` when the code is accepted or the retry budget
        for the active tier is spent.
    """
    if state["problem"] is None or state["rounds"] >= state["max_rounds"]:
        return END
    return "refine"
