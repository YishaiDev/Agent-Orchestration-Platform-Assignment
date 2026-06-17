"""FastAPI application exposing the six orchestration endpoints.

The app is a thin HTTP skin over the engine: ``POST /tasks`` spawns a background ``run_task`` and
returns immediately, while the GET endpoints read live state from the shared run registry's monitor
so status, progress, and the trace are visible while a task is still executing.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable

from fastapi import FastAPI, HTTPException

from app.src.api.models import (
    AgentInfo,
    CancelResponse,
    TaskCreated,
    TaskRequest,
    TaskStatusResponse,
)
from app.src.engine.graph import run_task
from app.src.engine.monitor import RunMonitor
from app.src.engine.nodes import EngineDeps
from app.src.engine.registry import describe_agents
from app.src.engine.runs import RunRegistry, get_run_registry

DepsFactory = Callable[[RunRegistry], EngineDeps]


def _default_deps_factory(registry: RunRegistry) -> EngineDeps:
    """Build production engine dependencies (real registry, agents, and models)."""
    return EngineDeps(registry=registry)


def _spawn_run(registry: RunRegistry, deps_factory: DepsFactory, request: TaskRequest) -> str:
    """Create a monitor and launch the background run, returning the new task id."""
    task_id = uuid.uuid4().hex
    registry.create(task_id, request.deadline_seconds)
    coro = run_task(
        task_id,
        request.goal,
        request.constraints,
        request.session_id,
        deps=deps_factory(registry),
        max_replans=request.max_replans,
        output_format=request.output_format,
        deadline_seconds=request.deadline_seconds,
    )
    registry.register_task(task_id, asyncio.create_task(coro))
    return task_id


def _require_monitor(registry: RunRegistry, task_id: str) -> RunMonitor:
    """Fetch a monitor or raise 404 when the task id is unknown."""
    monitor = registry.get(task_id)
    if monitor is None:
        raise HTTPException(status_code=404, detail=f"unknown task {task_id}")
    return monitor


def _status_view(monitor: RunMonitor) -> TaskStatusResponse:
    """Project a monitor into the status response shape."""
    return TaskStatusResponse(
        task_id=monitor.task_id,
        status=monitor.state.value,
        created_at=monitor.created_at.isoformat(),
        updated_at=monitor.updated_at.isoformat(),
        progress=monitor.progress(),
        total_tokens=monitor.total_tokens,
        total_cost_usd=monitor.total_cost_usd,
        trace=monitor.trace_dicts(),
    )


def create_app(
    registry: RunRegistry | None = None, deps_factory: DepsFactory | None = None
) -> FastAPI:
    """Build the FastAPI app bound to a run registry (defaults to the process singleton).

    Args:
        registry: Run registry to back the endpoints; injected for offline tests.
        deps_factory: Builds the engine deps per run; injected with fakes for offline tests.

    Returns:
        The configured FastAPI application.
    """
    registry = registry or get_run_registry()
    make_deps = deps_factory or _default_deps_factory
    app = FastAPI(title="Agent Orchestration Platform", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe for the container healthcheck."""
        return {"status": "ok"}

    @app.post("/tasks", response_model=TaskCreated, status_code=202)
    async def submit_task(request: TaskRequest) -> TaskCreated:
        """Accept a goal, spawn the background run, and return its task id."""
        task_id = _spawn_run(registry, make_deps, request)
        return TaskCreated(task_id=task_id, status="pending")

    @app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
    async def get_status(task_id: str) -> TaskStatusResponse:
        """Return live status, progress, totals, and the execution trace."""
        return _status_view(_require_monitor(registry, task_id))

    @app.get("/tasks/{task_id}/result")
    async def get_result(task_id: str) -> dict[str, object]:
        """Return the synthesized final result, or 409 while the task is still running."""
        monitor = _require_monitor(registry, task_id)
        if monitor.final_result is None:
            raise HTTPException(status_code=409, detail=f"task {task_id} has no result yet")
        return monitor.final_result

    @app.post("/tasks/{task_id}/cancel", response_model=CancelResponse)
    async def cancel_task(task_id: str) -> CancelResponse:
        """Request cooperative cancellation and report the steps completed so far."""
        monitor = _require_monitor(registry, task_id)
        monitor.request_cancel()
        return CancelResponse(
            task_id=task_id,
            status="cancelling",
            completed_steps=monitor.completed_step_ids(),
        )

    @app.get("/agents", response_model=list[AgentInfo])
    async def list_agents() -> list[AgentInfo]:
        """Return the registered agents with their capabilities and status."""
        return [AgentInfo.model_validate(entry) for entry in describe_agents()]

    return app


app = create_app()
