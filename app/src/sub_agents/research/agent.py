"""Research Agent: an autonomous LangChain ``create_agent`` search loop with grounded citations.

The compiled agent (a LangGraph) searches the web under a configured budget, compacting history
between rounds; a final structured summarization then yields the platform's uniform ``AgentResult``
with grounded sources and pre/post cost figures.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import partial
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from app.src.general_utils.agent_base import AgentResult, Messages, extract_tokens
from app.src.general_utils.cost import estimate_cost, token_cost
from app.src.general_utils.llm import build_chat_model
from app.src.general_utils.middleware import (
    build_compaction_middleware,
    build_token_cost_middleware,
)
from app.src.general_utils.tokens import count_prompt_tokens
from app.src.general_utils.tools import think
from app.src.schemas.config import ModelPrice, ResearchAgentConfig, get_config
from app.src.services.tavily_client import TavilySearch
from app.src.sub_agents.research import prompts
from app.src.sub_agents.research.schemas import ResearchContext, ResearchSummary, SearchFn
from app.src.sub_agents.research.tools import web_search

AGENT_NAME = "research"
_LOW_CONFIDENCE_CAP = 0.3
_TOOL_USE_FAILED = "tool_use_failed"

logger = logging.getLogger("app.sub_agents.research")


def build_research_agent(
    model: BaseChatModel, summarizer: BaseChatModel, cfg: ResearchAgentConfig, price: ModelPrice
) -> CompiledStateGraph[Any]:
    """Assemble the autonomous research ``create_agent`` (compiles to a LangGraph).

    Args:
        model: Main agentic model that drives tool calls.
        summarizer: Cheap model for between-rounds compaction.
        cfg: Research-agent runtime parameters.
        price: Price table entry for the main model (drives cost capture).

    Returns:
        A compiled agent graph awaiting ``ainvoke`` with a ``ResearchContext``.
    """
    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        ModelCallLimitMiddleware(run_limit=cfg.recursion_limit, exit_behavior="end"),
        build_compaction_middleware(summarizer, cfg.trigger_messages, cfg.keep_recent),
        build_token_cost_middleware(price),
    ]
    return create_agent(
        model=model,
        tools=[web_search, think],
        system_prompt=prompts.RESEARCH_SYSTEM,
        middleware=middleware,
        context_schema=ResearchContext,
        name=AGENT_NAME,
    )


def _transcript(messages: list[BaseMessage]) -> str:
    """Join assistant and tool message text into a single findings transcript."""
    parts = [
        str(msg.content)
        for msg in messages
        if isinstance(msg, (AIMessage, ToolMessage)) and msg.content
    ]
    return "\n\n".join(parts)


async def _summarize(
    summarizer: BaseChatModel,
    subtopic: str,
    findings: str,
    sources: list[str],
    price: ModelPrice,
) -> tuple[ResearchSummary, int, float]:
    """Run the final structured summarization, returning the summary plus its tokens and cost."""
    messages = prompts.summarize_messages(subtopic, findings, sources)
    runnable = summarizer.with_structured_output(ResearchSummary, include_raw=True)
    result = cast(dict[str, Any], await runnable.ainvoke(messages))
    raw = result.get("raw")
    meta = getattr(raw, "usage_metadata", None) or {}
    in_tokens = int(meta.get("input_tokens") or 0)
    out_tokens = int(meta.get("output_tokens") or 0)
    cost = token_cost(price, in_tokens, out_tokens)
    return cast(ResearchSummary, result["parsed"]), extract_tokens(raw), cost


def _grounded_confidence(summary: ResearchSummary, sources: list[str]) -> float:
    """Clamp confidence low when no sources were collected."""
    if not sources:
        return min(summary.confidence, _LOW_CONFIDENCE_CAP)
    return summary.confidence


def _assemble_result(
    summary: ResearchSummary,
    ctx: ResearchContext,
    summary_tokens: int,
    summary_cost: float,
    est_cost: float,
    elapsed_ms: int,
) -> AgentResult:
    """Map the run's outputs into the platform's uniform AgentResult."""
    return AgentResult(
        step_id=ctx.step_id,
        agent=AGENT_NAME,
        status="completed",
        output={
            "content": summary.content,
            "sources": ctx.collected_sources,
            "confidence": _grounded_confidence(summary, ctx.collected_sources),
        },
        tokens_used=ctx.tokens_used + summary_tokens,
        execution_time_ms=elapsed_ms,
        est_cost_usd=round(est_cost, 6),
        actual_cost_usd=round(ctx.actual_cost_usd + summary_cost, 6),
    )


def _build_context(
    subtopic: str, step_id: str, session_id: str, searcher: SearchFn, cfg: ResearchAgentConfig
) -> ResearchContext:
    """Construct the per-step runtime context from config and the injected searcher."""
    return ResearchContext(
        searcher=searcher,
        session_id=session_id,
        step_id=step_id,
        max_search_calls=cfg.max_search_calls,
        search_top_k=cfg.search_top_k,
    )


def _is_tool_use_failed(exc: Exception) -> bool:
    """True when an exception is the provider's malformed-tool-call rejection (worth re-asking)."""
    return _TOOL_USE_FAILED in str(exc)


def _attempt_model(
    cfg: ResearchAgentConfig, injected: BaseChatModel | None, attempt: int
) -> BaseChatModel:
    """Return the model for one attempt: the injected one, or a config model whose temperature is
    bumped on retries so a near-greedy loop re-samples a different (well-formed) tool call."""
    if injected is not None:
        return injected
    temperature = min(1.0, cfg.temperature + cfg.tool_retry_temp_bump * attempt)
    return build_chat_model(cfg.model_id, temperature)


async def _invoke_with_tool_retry(
    cfg: ResearchAgentConfig,
    injected_model: BaseChatModel | None,
    summarizer: BaseChatModel,
    price: ModelPrice,
    initial: Messages,
    ctx_factory: Callable[[], ResearchContext],
) -> tuple[dict[str, Any], ResearchContext]:
    """Run the agent loop, re-asking on a malformed tool call.

    The provider intermittently rejects a channel-token-leaked tool call with ``tool_use_failed``.
    Each retry rebuilds the model with a perturbed temperature and a fresh context so a clean call
    can be re-sampled. An injected model (tests) is never rebuilt and never retried.
    """
    last_exc: Exception | None = None
    last_attempt = cfg.tool_retry_attempts
    for attempt in range(last_attempt + 1):
        ctx = ctx_factory()
        model = _attempt_model(cfg, injected_model, attempt)
        agent = build_research_agent(model, summarizer, cfg, price)
        try:
            final = await agent.ainvoke({"messages": initial}, context=ctx)  # type: ignore[call-overload]
            return cast(dict[str, Any], final), ctx
        except Exception as exc:
            last_exc = exc
            exhausted = attempt == last_attempt
            if injected_model is not None or not _is_tool_use_failed(exc) or exhausted:
                raise
            logger.warning(
                "research tool call malformed for step %s (attempt %d/%d); re-asking",
                ctx.step_id, attempt + 1, cfg.tool_retry_attempts + 1,
            )
    raise cast(Exception, last_exc)


async def run_research_agent(
    subtopic: str,
    step_id: str = "research",
    session_id: str = "local",
    searcher: SearchFn | None = None,
    model: BaseChatModel | None = None,
    summarizer: BaseChatModel | None = None,
) -> AgentResult:
    """Run the Research Agent end-to-end, returning a structured result on any failure.

    Args:
        subtopic: The untrusted research subtopic for this step.
        step_id: Orchestrator-assigned step identifier (echoed into the result).
        session_id: Session identifier carried in the runtime context.
        searcher: Optional injected async search function (defaults to Tavily); enables tests.
        model: Optional injected main model (defaults to the configured Gemini model).
        summarizer: Optional injected summarizer model (defaults to the configured Gemini model).

    Returns:
        An AgentResult; status ``failed`` (with an ``error`` field) on unrecoverable errors.
    """
    started = time.perf_counter()
    app_cfg = get_config()
    cfg = app_cfg.research_agent
    try:
        summarizer = summarizer or build_chat_model(
            cfg.summarizer_model_id, cfg.summarizer_temperature
        )
        searcher = searcher or TavilySearch(
            app_cfg.tavily_api_key.get_secret_value() if app_cfg.tavily_api_key else "",
            cfg.tavily_ttl_seconds,
        )
        price = app_cfg.pricing[cfg.model_id]
        initial = prompts.initial_messages(subtopic)
        ctx_factory = partial(_build_context, subtopic, step_id, session_id, searcher, cfg)
        final, ctx = await _invoke_with_tool_retry(
            cfg, model, summarizer, price, initial, ctx_factory
        )
        findings = _transcript(final["messages"])
        summary, summary_tokens, summary_cost = await _summarize(
            summarizer, subtopic, findings, ctx.collected_sources, price
        )
        input_tokens = count_prompt_tokens(initial, app_cfg.estimation.chars_per_token)
        est = estimate_cost(price, cfg.max_search_calls + 2, input_tokens, cfg.avg_output_tokens)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return _assemble_result(summary, ctx, summary_tokens, summary_cost, est, elapsed_ms)
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
