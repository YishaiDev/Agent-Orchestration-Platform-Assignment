"""LangGraph node functions for the Writing Agent reflection loop.

Each node makes one structured LLM call and returns only its state delta. Counters
(``tokens_used``, ``edit_runs``, ``format_runs``, ``cycles``) use additive reducers.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from app.src.general_utils.agent_base import invoke_structured
from app.src.sub_agents.writing import prompts
from app.src.sub_agents.writing.schemas import ContentOut, JudgeVerdict, WritingState


def generate_node(state: WritingState, *, model: BaseChatModel) -> dict[str, Any]:
    """Produce the first draft from the instruction and fenced source material."""
    messages = prompts.generate_messages(
        state["instruction"], state["source_material"], state["max_words"]
    )
    draft, tokens = invoke_structured(model, ContentOut, messages)
    return {"draft": draft.content, "tokens_used": tokens}


def edit_node(state: WritingState, *, model: BaseChatModel) -> dict[str, Any]:
    """Improve the latest text for clarity, tone, and constraints, fixing reviewer issues."""
    source = state.get("edited") or state["draft"]
    messages = prompts.edit_messages(
        source,
        state["instruction"],
        state["constraints"],
        state["max_words"],
        state.get("issues", []),
    )
    edited, tokens = invoke_structured(model, ContentOut, messages)
    return {"edited": edited.content, "tokens_used": tokens, "edit_runs": 1}


def format_node(state: WritingState, *, model: BaseChatModel) -> dict[str, Any]:
    """Render the edited text into the requested output format and count words."""
    messages = prompts.format_messages(
        state["edited"], state["output_format"], state.get("issues", [])
    )
    formatted, tokens = invoke_structured(model, ContentOut, messages)
    content = formatted.content
    return {
        "content": content,
        "word_count": len(content.split()),
        "tokens_used": tokens,
        "format_runs": 1,
    }


def judge_node(state: WritingState, *, model: BaseChatModel) -> dict[str, Any]:
    """Critique edit quality and format correctness, emitting a routing verdict."""
    messages = prompts.judge_messages(
        state["content"],
        state["instruction"],
        state["output_format"],
        state["max_words"],
        state["word_count"],
    )
    verdict, tokens = invoke_structured(model, JudgeVerdict, messages)
    return {
        "verdict": verdict.verdict,
        "issues": verdict.issues,
        "tokens_used": tokens,
        "cycles": 1,
    }
