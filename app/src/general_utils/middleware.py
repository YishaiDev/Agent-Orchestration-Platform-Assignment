"""Shared agent middleware: history compaction between rounds + token/cost capture.

Both pieces are reused by every autonomous sub-agent (research, analysis, future code). Compaction
uses LangChain's ``SummarizationMiddleware``: it fires from ``before_model`` only once history
grows past the trigger (after tool output has accumulated), never on the first turn, so the raw
instruction always reaches the first model call intact. The cost middleware accrues into the runtime
context (not the message history) so compaction cannot erase the running totals; it is typed
against a small :class:`UsageContext` protocol that any context with the usage fields satisfies.
"""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    SummarizationMiddleware,
    after_model,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from app.src.general_utils.cost import token_cost
from app.src.schemas.config import ModelPrice


@runtime_checkable
class UsageContext(Protocol):
    """Structural contract the cost middleware needs from any agent's runtime context."""

    tokens_used: int
    actual_cost_usd: float


def build_compaction_middleware(
    summarizer: BaseChatModel, trigger_messages: int, keep_recent: int
) -> AgentMiddleware:
    """Build the between-rounds history-compaction middleware.

    Args:
        summarizer: Cheap model used to summarize older turns.
        trigger_messages: Message count above which summarization fires.
        keep_recent: Number of most-recent messages preserved verbatim.

    Returns:
        A configured SummarizationMiddleware.
    """
    return SummarizationMiddleware(
        model=summarizer,
        trigger=("messages", trigger_messages),
        keep=("messages", keep_recent),
    )


def build_token_cost_middleware(price: ModelPrice) -> AgentMiddleware:
    """Build an ``after_model`` middleware that accrues tokens and cost into the runtime context.

    Args:
        price: Per-1M-token price table for the agent's main model.

    Returns:
        An after_model middleware instance.
    """

    @after_model
    def capture_usage(state: AgentState, runtime: Runtime[UsageContext]) -> None:
        """Add the latest model message's tokens and cost to the runtime context."""
        message = state["messages"][-1]
        if not isinstance(message, AIMessage):
            return None
        meta = getattr(message, "usage_metadata", None) or {}
        ctx = runtime.context
        ctx.tokens_used += int(meta.get("total_tokens") or 0)
        ctx.actual_cost_usd += token_cost(
            price, int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0)
        )
        return None

    return cast(AgentMiddleware[Any, Any, Any], capture_usage)
