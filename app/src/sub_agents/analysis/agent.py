"""Analysis Agent: an autonomous LangChain ``create_agent`` reason/compute loop.

The compiled agent (a LangGraph) reasons over upstream structured data using a private ``think``
scratchpad and a deterministic ``compute`` tool, compacting history between rounds. A final
structured summarization then yields the platform's uniform ``AgentResult`` with grounded findings,
a calibrated confidence, and pre/post cost figures.
"""

from __future__ import annotations

import json
import time
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from app.src.general_utils.agent_base import AgentResult, extract_tokens
from app.src.general_utils.cost import estimate_cost, token_cost
from app.src.general_utils.llm import build_chat_model
from app.src.general_utils.middleware import (
    build_compaction_middleware,
    build_token_cost_middleware,
)
from app.src.general_utils.tools import compute, think
from app.src.schemas.config import AnalysisAgentConfig, ModelPrice, get_config
from app.src.sub_agents.analysis import prompts
from app.src.sub_agents.analysis.schemas import Action, AnalysisContext, AnalysisSummary

AGENT_NAME = "analysis"
_AVG_INPUT_TOKENS = 1100
_AVG_OUTPUT_TOKENS = 400
_PREVIEW_MAX_CHARS = 6000


def build_analysis_agent(
    model: BaseChatModel, summarizer: BaseChatModel, cfg: AnalysisAgentConfig, price: ModelPrice
) -> CompiledStateGraph[Any]:
    """Assemble the autonomous analysis ``create_agent`` (compiles to a LangGraph).

    Args:
        model: Main agentic model that drives the reason/compute loop.
        summarizer: Cheap model for between-rounds compaction.
        cfg: Analysis-agent runtime parameters.
        price: Price table entry for the main model (drives cost capture).

    Returns:
        A compiled agent graph awaiting ``ainvoke`` with an ``AnalysisContext``.
    """
    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        ModelCallLimitMiddleware(run_limit=cfg.recursion_limit, exit_behavior="end"),
        build_compaction_middleware(summarizer, cfg.trigger_messages, cfg.keep_recent),
        build_token_cost_middleware(price),
    ]
    return create_agent(
        model=model,
        tools=[think, compute],
        middleware=middleware,
        context_schema=AnalysisContext,
        name=AGENT_NAME,
    )


def _normalize_dataset(data: Any) -> list[dict[str, Any]]:
    """Coerce upstream data into a list of dict rows for the compute tool."""
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _data_preview(data: Any) -> str:
    """Render a bounded JSON preview of the upstream data for the prompt."""
    try:
        text = json.dumps(data, ensure_ascii=False, default=str, indent=2)
    except (TypeError, ValueError):
        text = str(data)
    if len(text) > _PREVIEW_MAX_CHARS:
        return f"{text[:_PREVIEW_MAX_CHARS]}\n... [truncated]"
    return text


def _transcript(messages: list[BaseMessage]) -> str:
    """Join assistant and tool message text into a single analysis transcript."""
    parts = [
        str(msg.content)
        for msg in messages
        if isinstance(msg, (AIMessage, ToolMessage)) and msg.content
    ]
    return "\n\n".join(parts)


async def _summarize(
    summarizer: BaseChatModel, instruction: str, transcript: str, price: ModelPrice
) -> tuple[AnalysisSummary, int, float]:
    """Run the final structured summarization, returning the summary plus its tokens and cost."""
    messages = prompts.summarize_messages(instruction, transcript)
    runnable = summarizer.with_structured_output(AnalysisSummary, include_raw=True)
    result = cast(dict[str, Any], await runnable.ainvoke(messages))
    raw = result.get("raw")
    meta = getattr(raw, "usage_metadata", None) or {}
    in_tokens = int(meta.get("input_tokens") or 0)
    out_tokens = int(meta.get("output_tokens") or 0)
    cost = token_cost(price, in_tokens, out_tokens)
    return cast(AnalysisSummary, result["parsed"]), extract_tokens(raw), cost


def _status(summary: AnalysisSummary, threshold: float) -> str:
    """Map confidence onto the platform status vocabulary."""
    return "completed_degraded" if summary.confidence < threshold else "completed"


def _assemble_result(
    summary: AnalysisSummary,
    ctx: AnalysisContext,
    summary_tokens: int,
    summary_cost: float,
    est_cost: float,
    elapsed_ms: int,
    threshold: float,
) -> AgentResult:
    """Map the run's outputs into the platform's uniform AgentResult."""
    return AgentResult(
        step_id=ctx.step_id,
        agent=AGENT_NAME,
        status=_status(summary, threshold),
        output={
            "content": summary.content,
            "findings": summary.findings,
            "confidence": summary.confidence,
            "sources": ctx.sources,
        },
        tokens_used=ctx.tokens_used + summary_tokens,
        execution_time_ms=elapsed_ms,
        est_cost_usd=round(est_cost, 6),
        actual_cost_usd=round(ctx.actual_cost_usd + summary_cost, 6),
    )


def _build_context(
    action: Action,
    data: Any,
    sources: list[str],
    step_id: str,
    session_id: str,
    cfg: AnalysisAgentConfig,
) -> AnalysisContext:
    """Construct the per-step runtime context from config and the upstream data."""
    return AnalysisContext(
        session_id=session_id,
        step_id=step_id,
        action=action,
        dataset=_normalize_dataset(data),
        sources=sources,
        max_compute_calls=cfg.max_compute_calls,
    )


async def run_analysis_agent(
    instruction: str,
    action: Action = "analyze",
    data: Any = None,
    sources: list[str] | None = None,
    step_id: str = "analysis",
    session_id: str = "local",
    model: BaseChatModel | None = None,
    summarizer: BaseChatModel | None = None,
) -> AgentResult:
    """Run the Analysis Agent end-to-end, returning a structured result on any failure.

    Args:
        instruction: The untrusted analysis instruction for this step.
        action: One of ``analyze`` / ``compare`` / ``identify_patterns`` (shapes the prompt).
        data: Upstream structured data (dict or list of dicts) to quantify over.
        sources: Optional upstream provenance, passed through to the result.
        step_id: Orchestrator-assigned step identifier (echoed into the result).
        session_id: Session identifier carried in the runtime context.
        model: Optional injected main model (defaults to the configured Gemini model).
        summarizer: Optional injected summarizer model (defaults to the configured Gemini model).

    Returns:
        An AgentResult; status ``failed`` (with an ``error`` field) on unrecoverable errors.
    """
    started = time.perf_counter()
    app_cfg = get_config()
    cfg = app_cfg.analysis_agent
    try:
        model, summarizer = _resolve_models(app_cfg, cfg, model, summarizer)
        price = app_cfg.pricing[cfg.model_id]
        ctx = _build_context(action, data, sources or [], step_id, session_id, cfg)
        agent = build_analysis_agent(model, summarizer, cfg, price)
        final = await agent.ainvoke(
            {"messages": prompts.initial_messages(instruction, action, _data_preview(data))},
            context=ctx,  # type: ignore[call-overload]
        )
        transcript = _transcript(final["messages"])
        summary, summary_tokens, summary_cost = await _summarize(
            summarizer, instruction, transcript, price
        )
        est = estimate_cost(
            price, cfg.max_compute_calls + 2, _AVG_INPUT_TOKENS, _AVG_OUTPUT_TOKENS
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return _assemble_result(
            summary, ctx, summary_tokens, summary_cost, est, elapsed_ms, cfg.confidence_threshold
        )
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


def _resolve_models(
    app_cfg: Any,
    cfg: AnalysisAgentConfig,
    model: BaseChatModel | None,
    summarizer: BaseChatModel | None,
) -> tuple[BaseChatModel, BaseChatModel]:
    """Return the main and summarizer models, building defaults from config when not injected."""
    api_key = app_cfg.google_api_key.get_secret_value()
    model = model or build_chat_model(cfg.model_id, cfg.temperature, api_key)
    summarizer = summarizer or build_chat_model(
        cfg.summarizer_model_id, cfg.summarizer_temperature, api_key
    )
    return model, summarizer
