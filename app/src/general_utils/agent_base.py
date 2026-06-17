"""Shared building blocks for all specialized sub-agents.

Holds the common output contract (``AgentResult``), null-safe token extraction, and a
retried structured-output invocation used by every agent node.
"""

from __future__ import annotations

from typing import Any, TypeVar, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

Messages = list[dict[str, str]]
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class AgentResult(BaseModel):
    """Uniform agent output matching the platform's Agent Output Format.

    ``est_cost_usd``/``actual_cost_usd`` are additive (default None) and satisfy the spec's
    Nice-to-Have cost estimation without altering the required output shape.
    """

    step_id: str
    agent: str
    status: str
    output: dict[str, Any]
    tokens_used: int
    execution_time_ms: int
    est_cost_usd: float | None = None
    actual_cost_usd: float | None = None


def extract_tokens(message: BaseMessage | None) -> int:
    """Read total token usage from a message, tolerating missing metadata.

    Args:
        message: The raw model message (may be None or lack usage metadata).

    Returns:
        Total tokens used, or 0 when unavailable.
    """
    if message is None:
        return 0
    metadata = getattr(message, "usage_metadata", None) or {}
    return int(metadata.get("total_tokens") or 0)


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(initial=4, max=70),
    reraise=True,
)
def invoke_structured(
    model: BaseChatModel, schema: type[SchemaT], messages: Messages
) -> tuple[SchemaT, int]:
    """Invoke a model for structured output with bounded retry and token capture.

    Args:
        model: The chat model to call.
        schema: Pydantic schema the model must populate.
        messages: Chat messages (role/content dicts).

    Returns:
        A tuple of (parsed schema instance, tokens used).
    """
    runnable = model.with_structured_output(schema, include_raw=True)
    result = cast(dict[str, Any], runnable.invoke(messages))
    parsed = cast(SchemaT, result["parsed"])
    return parsed, extract_tokens(result.get("raw"))
