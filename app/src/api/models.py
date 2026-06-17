"""Request/response models for the HTTP API.

These are the wire shapes only; the engine's domain schemas (plan, result, trace) are reused
verbatim in responses so the API never drifts from what the monitor actually records.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.src.schemas.run_state import Progress


class TaskRequest(BaseModel):
    """Body for ``POST /tasks``: the goal plus optional constraints and bounds."""

    goal: str = Field(min_length=1, description="The task goal to decompose and execute.")
    constraints: str = Field(default="", description="Optional free-text constraints.")
    session_id: str = Field(default="local", description="Session id carried into agent calls.")
    max_replans: int | None = Field(default=None, ge=0, description="Override the re-plan bound.")
    deadline_seconds: float | None = Field(
        default=None, gt=0, description="Optional wall-clock budget for the whole run."
    )


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
    trace: list[dict[str, object]] = Field(default_factory=list)


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
