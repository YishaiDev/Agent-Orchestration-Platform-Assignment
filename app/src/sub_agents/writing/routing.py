"""Conditional routing after the judge node."""

from __future__ import annotations

from langgraph.graph import END

from app.src.sub_agents.writing.schemas import WritingState


def route_after_judge(state: WritingState, *, max_revisions: int) -> str:
    """Decide the next node from the judge verdict, bounded by the revision cap.

    Args:
        state: Current writing state (reads ``verdict`` and ``cycles``).
        max_revisions: Maximum reflection cycles before forcing a return.

    Returns:
        ``"edit"`` to re-edit, ``"format"`` to re-format, or ``END`` to finalize.
    """
    if state["verdict"] == "return" or state["cycles"] > max_revisions:
        return END
    return "edit" if state["verdict"] == "reedit" else "format"
