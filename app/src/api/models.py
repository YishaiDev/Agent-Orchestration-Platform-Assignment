"""Request/response models for the HTTP API.

These are the wire shapes only; the engine's domain schemas (plan, result, trace) are reused
verbatim in responses so the API never drifts from what the monitor actually records.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.src.schemas.run_state import Progress


class TaskRequest(BaseModel):
    """Body for ``POST /tasks``: the goal plus optional constraints and bounds.

    ``constraints`` accepts either the spec's free-text string or its JSON object form (e.g.
    ``{"max_words": 1500, "tone": "friendly"}``); the object is normalized to text so the planner
    receives it as fenced data without the engine needing a structured constraints type.
    """

    goal: str = Field(min_length=1, description="The task goal to decompose and execute.")
    constraints: str = Field(
        default="", description="Optional constraints: free text or JSON object."
    )
    output_format: str = Field(
        default="", description="Optional output format hint, e.g. markdown."
    )
    session_id: str = Field(default="local", description="Session id carried into agent calls.")
    max_replans: int | None = Field(default=None, ge=0, description="Override the re-plan bound.")
    deadline_seconds: float | None = Field(
        default=None, gt=0, description="Optional wall-clock budget for the whole run."
    )

    @field_validator("constraints", mode="before")
    @classmethod
    def _normalize_constraints(cls, value: object) -> str:
        """Coerce the spec's JSON-object constraints (or null) into planner-ready text."""
        if isinstance(value, dict):
            return "; ".join(f"{key}: {val}" for key, val in value.items())
        return value if isinstance(value, str) else ""


class TaskCreated(BaseModel):
    """Response for ``POST /tasks``: the assigned id and accepted state."""

    task_id: str
    status: str


class TaskStatusResponse(BaseModel):
    """Response for ``GET /tasks/{id}``: live status, progress, totals, and the trace."""

    task_id: str
    status: str
    created_at: str
    updated_at: str
    progress: Progress
    total_tokens: int
    total_cost_usd: float
    execution_trace: list[dict[str, object]] = Field(default_factory=list)


class CancelResponse(BaseModel):
    """Response for ``POST /tasks/{id}/cancel``: acknowledgement plus completed work so far."""

    task_id: str
    status: str
    completed_steps: list[str] = Field(default_factory=list)


class AgentInfo(BaseModel):
    """One entry of ``GET /agents``: a registered agent and its routable capabilities."""

    name: str
    description: str
    capabilities: list[str]
    status: str


class AgentCatalog(BaseModel):
    """Response for ``GET /agents``: the registered agents under the spec's ``agents`` key."""

    agents: list[AgentInfo] = Field(default_factory=list)
