"""Run registry: the shared, in-process store of live monitors and background run tasks.

The LangGraph state is checkpointed and serializable, but the :class:`RunMonitor` is a live mutable
object with asyncio events — so it lives here, keyed by task id, rather than in graph state. Both
the graph nodes and the API read the same registry: that is how ``GET /tasks/{id}`` returns live
status
while the run is still executing.
"""

from __future__ import annotations

import asyncio

from app.src.engine.monitor import RunMonitor


class RunRegistry:
    """In-process registry of per-task monitors and their background asyncio tasks."""

    def __init__(self) -> None:
        self._monitors: dict[str, RunMonitor] = {}
        self._tasks: dict[str, asyncio.Task[object]] = {}

    def create(self, task_id: str, deadline_seconds: float | None = None) -> RunMonitor:
        """Create and store a monitor for ``task_id``."""
        monitor = RunMonitor(task_id, deadline_seconds)
        self._monitors[task_id] = monitor
        return monitor

    def get(self, task_id: str) -> RunMonitor | None:
        """Return the monitor for ``task_id``, or None when unknown."""
        return self._monitors.get(task_id)

    def register_task(self, task_id: str, task: asyncio.Task[object]) -> None:
        """Record the background task driving ``task_id``."""
        self._tasks[task_id] = task

    def get_task(self, task_id: str) -> asyncio.Task[object] | None:
        """Return the background task for ``task_id``, or None when absent."""
        return self._tasks.get(task_id)

    def ids(self) -> list[str]:
        """Return all known task ids."""
        return list(self._monitors)


_registry = RunRegistry()


def get_run_registry() -> RunRegistry:
    """Return the process-wide run registry singleton."""
    return _registry
