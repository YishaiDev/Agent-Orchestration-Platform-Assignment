"""Writing Agent: a LangGraph reflection loop (generate -> edit -> format -> judge).

The judge routes back to re-edit or re-format until both axes pass or the revision cap is hit,
then the final state is mapped to the platform's uniform ``AgentResult``.
"""

from __future__ import annotations

import time
from functools import partial
from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.src.general_utils.llm import build_chat_model
from app.src.schemas import get_config
from app.src.general_utils.agent_base import AgentResult
from app.src.sub_agents.writing import nodes
from app.src.sub_agents.writing.routing import route_after_judge
from app.src.sub_agents.writing.schemas import WritingInput, WritingState

AGENT_NAME = "writing"


def build_writing_graph(
    writer: BaseChatModel, judge: BaseChatModel, max_revisions: int
) -> CompiledStateGraph:
    """Wire the reflection-loop StateGraph.

    Args:
        writer: Model for generate/edit/format nodes.
        judge: Model for the judge node.
        max_revisions: Cap on reflection cycles before forced return.

    Returns:
        A compiled LangGraph runnable over ``WritingState``.
    """
    graph = StateGraph(WritingState)
    graph.add_node("generate", partial(nodes.generate_node, model=writer))
    graph.add_node("edit", partial(nodes.edit_node, model=writer))
    graph.add_node("format", partial(nodes.format_node, model=writer))
    graph.add_node("judge", partial(nodes.judge_node, model=judge))
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "edit")
    graph.add_edge("edit", "format")
    graph.add_edge("format", "judge")
    graph.add_conditional_edges(
        "judge",
        partial(route_after_judge, max_revisions=max_revisions),
        {"edit": "edit", "format": "format", END: END},
    )
    return graph.compile()


def initial_state(inp: WritingInput, max_words: int) -> WritingState:
    """Build the initial graph state from caller input.

    Args:
        inp: Caller-facing writing input.
        max_words: Resolved word limit for this run.

    Returns:
        A fully-initialized WritingState.
    """
    return {
        "instruction": inp.instruction,
        "source_material": inp.source_material,
        "constraints": inp.constraints,
        "output_format": inp.output_format,
        "max_words": max_words,
        "draft": "",
        "edited": "",
        "content": "",
        "word_count": 0,
        "tokens_used": 0,
        "edit_runs": 0,
        "format_runs": 0,
        "cycles": 0,
        "verdict": "",
        "issues": [],
    }


def assemble_result(
    final: WritingState, step_id: str, elapsed_ms: int
) -> AgentResult:
    """Map the final graph state to the platform's uniform agent output.

    Args:
        final: Terminal graph state.
        step_id: Orchestrator-assigned step identifier.
        elapsed_ms: Wall-clock duration of the full invoke.

    Returns:
        An AgentResult; status is ``completed_degraded`` when returned with open issues.
    """
    degraded = bool(final.get("issues")) and final.get("verdict") != "return"
    status = "completed_degraded" if degraded else "completed"
    return AgentResult(
        step_id=step_id,
        agent=AGENT_NAME,
        status=status,
        output={
            "content": final["content"],
            "format": final["output_format"],
            "word_count": final["word_count"],
            "unresolved_issues": final.get("issues", []) if degraded else [],
        },
        tokens_used=final["tokens_used"],
        execution_time_ms=elapsed_ms,
    )


def _resolve_max_words(inp: WritingInput, default_max_words: int) -> int:
    """Pick the word limit from constraints, falling back to the configured default."""
    raw = inp.constraints.get("max_words")
    return int(raw) if raw else default_max_words


def run_writing_agent(inp: WritingInput, step_id: str = "writing") -> AgentResult:
    """Run the Writing Agent end-to-end, returning a structured result on any failure.

    Args:
        inp: Caller-facing writing input.
        step_id: Orchestrator-assigned step identifier.

    Returns:
        An AgentResult; status ``failed`` (with an ``error`` field) on unrecoverable errors.
    """
    started = time.perf_counter()
    try:
        cfg = get_config().writing_agent
        writer = build_chat_model(
            cfg.model_id, cfg.temperature, get_config().google_api_key.get_secret_value()
        )
        judge = build_chat_model(
            cfg.judge_model_id,
            cfg.judge_temperature,
            get_config().google_api_key.get_secret_value(),
        )
        graph = build_writing_graph(writer, judge, cfg.max_revisions)
        max_words = _resolve_max_words(inp, cfg.default_max_words)
        final = cast(WritingState, graph.invoke(initial_state(inp, max_words)))
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return assemble_result(final, step_id, elapsed_ms)
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return AgentResult(
            step_id=step_id,
            agent=AGENT_NAME,
            status="failed",
            output={"error": str(exc)},
            tokens_used=0,
            execution_time_ms=elapsed_ms,
        )
