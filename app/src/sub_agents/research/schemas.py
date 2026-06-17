"""Schemas for the Research Agent: the mutable runtime context and the structured summary.

``ResearchContext`` is the app-runtime state for one research step. It is injected into the
``web_search`` tool (never into the model-visible schema) and accumulates citations, counts, and
cost as the autonomous loop runs.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.src.services.tavily_client import SearchHit

SearchFn = Callable[[str, int], Awaitable[Sequence[SearchHit]]]


@dataclass
class ResearchContext:
    """Per-step runtime state: identity, bounded config, and accumulated results.

    The ``search_*``/``collected_*``/``tokens_used``/``actual_cost_usd`` fields are mutated in
    place by the tool and middleware across the single agent invocation.
    """

    searcher: SearchFn
    session_id: str
    step_id: str
    action: str = "research"
    agent_name: str = "research"
    capabilities: tuple[str, ...] = ("search", "summarize", "cite")
    max_search_calls: int = 5
    search_top_k: int = 5
    search_count: int = 0
    collected_sources: list[str] = field(default_factory=list)
    collected_urls: list[str] = field(default_factory=list)
    tokens_used: int = 0
    actual_cost_usd: float = 0.0


class ResearchSummary(BaseModel):
    """Structured final summary produced after the search loop ends."""

    content: str = Field(
        description="Synthesized findings, grounded only in the collected sources."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence (0-1) that the findings answer the subtopic."
    )
