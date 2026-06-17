"""Schemas for the Code Agent: the per-step input and the structured code output.

There is no mutable runtime-context dataclass (unlike the autonomous research/analysis agents):
the Code Agent is a single bounded structured call with an optional syntax-correction retry, so
all state lives in the request/response models below.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Action = Literal["generate", "explain", "debug"]
CAPABILITIES: tuple[str, ...] = ("generate", "explain", "debug")
Verdict = Literal["revise", "return"]


def coerce_action(raw: str | None) -> Action:
    """Map a planner-supplied action onto the supported vocabulary.

    The planner is an LLM and may emit off-vocabulary actions; degrade gracefully to ``generate``
    instead of failing the step.

    Args:
        raw: The raw action string from the execution step (may be None).

    Returns:
        A valid ``Action`` (``generate`` for anything unrecognized).
    """
    return raw if raw in CAPABILITIES else "generate"  # type: ignore[return-value]


class CodeInput(BaseModel):
    """Caller-facing input for one code task."""

    action: Action = "generate"
    input: str
    language: str = "python"
    context: str = ""


class CodeOutput(BaseModel):
    """Structured payload the model returns; becomes the agent's ``output`` dict."""

    content: str = Field(description="Plain-language explanation or description of the code.")
    code: str = Field(default="", description="The code itself, no surrounding markdown fence.")
    language: str = Field(default="python", description="Language the code is written in.")


class CodeVerdict(BaseModel):
    """Tier-2 critic verdict for languages without a deterministic parser.

    Mirrors the Writing agent's ``JudgeVerdict`` shape: a decision plus concrete issues fed back to
    the generator on ``revise``.
    """

    verdict: Verdict = Field(
        default="return", description="'revise' to regenerate with fixes, 'return' to accept."
    )
    issues: list[str] = Field(
        default_factory=list, description="Concrete problems the generator must fix when revising."
    )
