"""Schemas for the Analysis Agent: the mutable runtime context and the structured summary.

``AnalysisContext`` is the app-runtime state for one analysis step. It is injected into the
``compute`` tool (never into the model-visible schema) and accumulates the compute-call count and
cost as the autonomous reason/compute loop runs. The ``dataset`` holds the upstream structured rows
the agent quantifies over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

Action = Literal["analyze", "compare", "identify_patterns"]
CAPABILITIES: tuple[str, ...] = ("analyze", "compare", "identify_patterns")


@dataclass
class AnalysisContext:
    """Per-step runtime state: identity, bounded compute budget, and accumulated cost.

    The ``compute_count``/``tokens_used``/``actual_cost_usd`` fields are mutated in place by the
    ``compute`` tool and the cost middleware across the single agent invocation.
    """

    session_id: str
    step_id: str
    action: Action = "analyze"
    agent_name: str = "analysis"
    capabilities: tuple[str, ...] = CAPABILITIES
    dataset: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    max_compute_calls: int = 6
    compute_count: int = 0
    tokens_used: int = 0
    actual_cost_usd: float = 0.0


class AnalysisSummary(BaseModel):
    """Structured final summary produced after the reason/compute loop ends."""

    content: str = Field(
        description="Synthesized analysis, grounded in the computed values and provided data."
    )
    findings: list[str] = Field(
        default_factory=list,
        description="Discrete insights: key takeaways, pros/cons, or named patterns.",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence (0-1) that the analysis is well-supported."
    )
