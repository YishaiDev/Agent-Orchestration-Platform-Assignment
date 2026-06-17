"""Prompt-token estimation for pre-execution cost figures.

A pre-execution estimate is inherently approximate, so this counts the *real* assembled prompt
with a character-ratio heuristic rather than a flat per-agent constant: the estimate then scales
with instruction, upstream context, and data-preview size. The heuristic is deliberately local
(no provider ``count_tokens`` round-trip): the agents run on the async event loop, and a
synchronous network tokenizer call there would block concurrent steps. Exact token usage is still
captured post-call from ``usage_metadata`` for the actual-cost figure.
"""

from __future__ import annotations

import math

from app.src.general_utils.agent_base import Messages

_DEFAULT_CHARS_PER_TOKEN = 4.0


def _messages_text(messages: Messages) -> str:
    """Concatenate the content of every role/content message into one string."""
    return "\n".join(str(message.get("content", "")) for message in messages)


def count_prompt_tokens(
    messages: Messages, chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN
) -> int:
    """Estimate input tokens for an assembled prompt from its character length.

    Args:
        messages: The role/content messages that seed the model call.
        chars_per_token: Average characters per token for the heuristic (English ~4).

    Returns:
        Estimated input-token count (0 for an empty prompt).
    """
    text = _messages_text(messages)
    if not text:
        return 0
    return max(1, math.ceil(len(text) / chars_per_token))
