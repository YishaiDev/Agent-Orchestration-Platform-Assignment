"""Run state for the LangGraph outer loop plus the live progress view the Monitor maintains.

``RunState`` is the channel schema threaded through ``plan -> execute -> evaluate -> synthesize``.
Nodes return only the keys they change (last-value semantics); the inner async scheduler owns
concurrency, so no additive reducers are needed at this level.
"""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field


class Progress(BaseModel):
    """Live progress snapshot surfaced through ``GET /tasks/{id}``."""

    total_steps: int = 0
    completed_steps: int = 0
    current_step: str | None = None


class ProvenanceEntry(BaseModel):
    """Per-step attribution attached to the final result."""

    step_id: str
    agent: str
    action: str
    status: str
    confidence: float | None = None
    sources: list[str] = Field(default_factory=list)


class ResultPayload(BaseModel):
    """The spec's nested ``result`` object: the final content, its format, and word count."""

    content: str
    format: str = "markdown"
    word_count: int = 0


class FinalResult(BaseModel):
    """The synthesized answer in the spec's Final Result shape (``GET /tasks/{id}/result``).

    The spec keys (``result``/``execution_trace``/``total_tokens``/``total_time_ms``) are matched
    exactly; provenance, confidence, and the failed/skipped lists are retained as additive fields
    so attribution and partial-run accounting are not lost.
    """

    task_id: str
    status: str
    result: ResultPayload
    execution_trace: list[dict[str, object]] = Field(default_factory=list)
    total_tokens: int = 0
    total_time_ms: int = 0
    confidence: float = 0.0
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
    failed_steps: list[str] = Field(default_factory=list)
    skipped_steps: list[str] = Field(default_factory=list)
    total_cost_usd: float = 0.0


class RunState(TypedDict, total=False):
    """Outer-loop state machine channels for one task run.

    ``replan_requested`` is raised by the scheduler's preemptive early-exit; ``decision`` is the
    re-plan decider's verdict (``continue`` / ``replan``) that the router branches on.
    ``synth_failed`` is set when the synthesizer LLM call fails terminally and the engine falls back
    to a deterministic assembly, so the judge accepts the degraded draft instead of re-judging it.
    """

    task_id: str
    session_id: str
    goal: str
    constraints: str
    output_format: str
    replans: int
    max_replans: int
    resynth_rounds: int
    max_resynth: int
    synth_feedback: str
    decision: str
    synth_failed: bool
    final_output: str
    final_result: dict[str, object] | None
    error: str | None


def initial_state(
    task_id: str,
    goal: str,
    constraints: str,
    session_id: str,
    max_replans: int,
    max_resynth: int = 2,
    output_format: str = "",
) -> RunState:
    """Build the starting :class:`RunState` for a fresh task run.

    The live plan, results, and step statuses live on the run's :class:`~app.src.engine.monitor`
    (not in graph state), keeping the checkpointed state small and msgpack-serializable.

    Args:
        task_id: Engine-assigned task identifier.
        goal: The untrusted task goal (fenced as data downstream).
        constraints: Optional untrusted constraints text.
        session_id: Session identifier carried into agent calls.
        max_replans: Upper bound on bounded re-plans for this run.
        max_resynth: Upper bound on bounded re-synthesis passes for this run.
        output_format: Optional requested output format, checked at synthesis time.

    Returns:
        A RunState initialised to the pre-planning baseline.
    """
    return RunState(
        task_id=task_id,
        session_id=session_id,
        goal=goal,
        constraints=constraints,
        output_format=output_format,
        replans=0,
        max_replans=max_replans,
        resynth_rounds=0,
        max_resynth=max_resynth,
        synth_feedback="",
        decision="",
        final_output="",
        final_result=None,
        error=None,
    )
