"""Shared reasoning tool: a ``think`` scratchpad for any tool-calling sub-agent.

``think`` is genuinely cross-agent — it has no domain coupling (pure ``str -> str``), so it lives in
``general_utils`` and is attached to the main reasoning model of every ReAct-loop agent. Domain
tools that depend on agent-specific runtime state (e.g. the analysis ``compute`` tool) live in their
own sub-agent package instead.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from app.src.general_utils.streaming import emit_status

logger = logging.getLogger(__name__)


@tool
def think(thought: str = "") -> str:
    """Internal reasoning scratchpad — the user CANNOT see this output.

    Allows you to reason through complex problems step-by-step without
    producing visible output. Useful for multi-step analysis, formula
    derivation, and planning an approach before acting.
    You MUST produce a visible response or tool call after thinking.

    Args:
        thought: Your reasoning, analysis, or formula derivation in plain text.

    Returns:
        The same thought text, unchanged.
    """
    if not thought.strip():
        logger.info("\n========== TOOL: think (empty) ==========\n")
        return (
            "[No thought recorded. Continue your reasoning, then respond to the user "
            "or route to a sub-agent.]"
        )
    logger.info("\n========== TOOL: think ==========\n%s", thought)
    emit_status("Thinking through the problem...")
    nudge = "[The user cannot see the above. Respond to the user or route to a sub-agent now.]"
    return f"{thought}\n\n{nudge}"
