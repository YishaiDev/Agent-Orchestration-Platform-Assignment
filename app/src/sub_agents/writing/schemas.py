"""Schemas for the Writing Agent: input, graph state, node outputs, and judge verdict."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

Verdict = Literal["reedit", "reformat", "return"]


class WritingInput(BaseModel):
    """Caller-facing input for one writing task."""

    instruction: str
    source_material: str = ""
    constraints: dict[str, Any] = Field(default_factory=dict)
    output_format: str = "markdown"


class ContentOut(BaseModel):
    """Text payload returned by the generate, edit, and format nodes."""

    content: str


class JudgeVerdict(BaseModel):
    """Critique of edit quality and format correctness, plus a routing verdict."""

    edit_ok: bool
    format_ok: bool
    verdict: Verdict
    issues: list[str] = Field(default_factory=list)


class WritingState(TypedDict):
    """LangGraph state for the generate -> edit -> format -> judge reflection loop.

    Counters use additive reducers so each node returns only its delta.
    """

    instruction: str
    source_material: str
    constraints: dict[str, Any]
    output_format: str
    max_words: int
    draft: str
    edited: str
    content: str
    word_count: int
    tokens_used: Annotated[int, operator.add]
    edit_runs: Annotated[int, operator.add]
    format_runs: Annotated[int, operator.add]
    cycles: Annotated[int, operator.add]
    verdict: str
    issues: list[str]
