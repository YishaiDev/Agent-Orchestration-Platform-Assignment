"""Shared prompt-construction helpers used across specialized sub-agents.

Fencing untrusted text as data (rather than concatenating it into instructions) is the core
prompt-injection defense: system prompts hold the only authoritative instructions, so text inside
a fence cannot redirect the agent.
"""

from __future__ import annotations


def fence(label: str, body: str) -> str:
    """Wrap untrusted text in a labeled data fence.

    Args:
        label: Fence label (e.g. ``request``).
        body: Untrusted text to fence.

    Returns:
        The fenced block, or an empty string when ``body`` is blank.
    """
    if not body:
        return ""
    return f"<{label}>\n{body}\n</{label}>"


def join_parts(*parts: str) -> str:
    """Join non-empty prompt parts with blank lines.

    Args:
        *parts: Prompt fragments; empty fragments are dropped.

    Returns:
        The fragments joined by blank lines.
    """
    return "\n\n".join(part for part in parts if part)
