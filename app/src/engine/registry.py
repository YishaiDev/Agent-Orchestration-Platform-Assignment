"""Agent registry: the post-LLM allowlist and capability catalog.

This is the single routing authority. The planner may *suggest* any ``(agent, action)`` pair, but
only pairs present here are dispatchable — validation rejects anything else before execution, which
is the structural defense against a hijacked plan invoking an unintended agent. Capability tuples
are imported from each agent's own schema so the catalog never drifts from the implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.src.sub_agents.analysis.schemas import CAPABILITIES as ANALYSIS_CAPABILITIES
from app.src.sub_agents.code.schemas import CAPABILITIES as CODE_CAPABILITIES

RESEARCH_CAPABILITIES: tuple[str, ...] = ("research",)
WRITING_CAPABILITIES: tuple[str, ...] = ("write",)


@dataclass(frozen=True)
class AgentSpec:
    """One registered agent: its routable capabilities and the input shape it expects."""

    name: str
    description: str
    capabilities: tuple[str, ...]
    input_hint: str


_REGISTRY: dict[str, AgentSpec] = {
    "research": AgentSpec(
        name="research",
        description="Grounded web research: searches, summarizes, and cites sources.",
        capabilities=RESEARCH_CAPABILITIES,
        input_hint='{"subtopic": "the focused question to research"}',
    ),
    "analysis": AgentSpec(
        name="analysis",
        description="Quantitative analysis, comparison, and pattern identification over data.",
        capabilities=ANALYSIS_CAPABILITIES,
        input_hint='{"instruction": "what to analyze", "data": <optional upstream data>}',
    ),
    "code": AgentSpec(
        name="code",
        description="Generates, explains, or debugs code (no execution).",
        capabilities=CODE_CAPABILITIES,
        input_hint='{"task_input": "spec/code/error", "language": "python"}',
    ),
    "writing": AgentSpec(
        name="writing",
        description="Synthesizes polished prose from source material with format control.",
        capabilities=WRITING_CAPABILITIES,
        input_hint='{"instruction": "what to write", "output_format": "markdown"}',
    ),
}


def get_registry() -> dict[str, AgentSpec]:
    """Return the agent registry keyed by agent name."""
    return _REGISTRY


def agent_names() -> set[str]:
    """Return the set of registered agent names."""
    return set(_REGISTRY)


def is_allowed(agent: str, action: str) -> bool:
    """Report whether ``action`` is a registered capability of ``agent``.

    Args:
        agent: Candidate agent name.
        action: Candidate action/capability.

    Returns:
        True only when the agent exists and lists the action as a capability.
    """
    spec = _REGISTRY.get(agent)
    return spec is not None and action in spec.capabilities


def describe_agents() -> list[dict[str, object]]:
    """Return the ``GET /agents`` view: name, description, capabilities, and status.

    Returns:
        One dict per agent; agents are stateless, so status is always ``available``.
    """
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "capabilities": list(spec.capabilities),
            "status": "available",
        }
        for spec in _REGISTRY.values()
    ]


def capabilities_catalog() -> str:
    """Render the registry as a compact catalog for the planner prompt.

    Returns:
        A newline-delimited catalog of each agent, its actions, and its input hint.
    """
    lines = [
        f"- {spec.name}: actions={list(spec.capabilities)}; "
        f"input={spec.input_hint} — {spec.description}"
        for spec in _REGISTRY.values()
    ]
    return "\n".join(lines)
