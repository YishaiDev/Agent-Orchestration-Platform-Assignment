"""Application configuration: runtime parameters from ``app/config.yaml`` plus secrets from env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


class WritingAgentConfig(BaseModel):
    """Runtime parameters for the Writing Agent."""

    model_id: str
    temperature: float = 0.4
    judge_model_id: str
    judge_temperature: float = 0.0
    default_max_words: int = 1500
    enforce_word_limit: bool = True
    max_revisions: int = 2


class ResearchAgentConfig(BaseModel):
    """Runtime parameters for the Research Agent (autonomous search loop)."""

    model_id: str
    temperature: float = 0.0
    summarizer_model_id: str
    summarizer_temperature: float = 0.0
    max_search_calls: int = 5
    recursion_limit: int = 12
    search_top_k: int = 5
    trigger_messages: int = 16
    keep_recent: int = 6
    llm_cache: str = "memory"
    tavily_ttl_seconds: int = 3600


class AnalysisAgentConfig(BaseModel):
    """Runtime parameters for the Analysis Agent (autonomous reason/compute loop)."""

    model_id: str
    temperature: float = 0.2
    summarizer_model_id: str
    summarizer_temperature: float = 0.0
    recursion_limit: int = 10
    max_compute_calls: int = 6
    confidence_threshold: float = 0.5
    trigger_messages: int = 16
    keep_recent: int = 6


class CodeAgentConfig(BaseModel):
    """Runtime parameters for the Code Agent (parser gate plus LLM-critic fallback)."""

    model_id: str
    temperature: float = 0.2
    default_language: str = "python"
    max_syntax_retries: int = 4
    review_model_id: str = "gemini-2.5-flash-lite"
    review_temperature: float = 0.0
    max_review_retries: int = 1


class OrchestratorConfig(BaseModel):
    """Runtime parameters for the engine: planner/decider/synthesizer models and run bounds."""

    planner_model_id: str
    planner_temperature: float = 0.2
    decider_model_id: str
    decider_temperature: float = 0.0
    synthesizer_model_id: str
    synthesizer_temperature: float = 0.3
    judge_model_id: str
    judge_temperature: float = 0.0
    max_replans: int = 1
    max_resynth: int = 2
    concurrency: int = 3
    max_steps: int = 12
    step_timeout_seconds: int = 120
    planner_max_attempts: int = 2
    context_char_budget: int = 6000


class ModelPrice(BaseModel):
    """USD price per 1M tokens for one model id."""

    input: float
    output: float


class AppConfig(BaseSettings):
    """Top-level config: secrets from ``.env``, runtime params injected from YAML."""

    google_api_key: SecretStr = Field(alias="GOOGLE_API_KEY")
    tavily_api_key: SecretStr | None = Field(default=None, alias="TAVILY_API_KEY")
    writing_agent: WritingAgentConfig
    research_agent: ResearchAgentConfig
    analysis_agent: AnalysisAgentConfig
    code_agent: CodeAgentConfig
    orchestrator: OrchestratorConfig
    pricing: dict[str, ModelPrice] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", populate_by_name=True
    )


def _read_yaml() -> dict[str, Any]:
    """Load the raw YAML config file.

    Returns:
        Parsed YAML as a dict (empty if the file is blank).
    """
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _map_writing_agent(block: dict[str, Any]) -> WritingAgentConfig:
    """Flatten the nested ``writing_agent`` YAML block into its config model.

    Args:
        block: The ``writing_agent`` mapping from YAML.

    Returns:
        A populated WritingAgentConfig.
    """
    model = block.get("model", {})
    judge = block.get("judge_model", {})
    return WritingAgentConfig(
        model_id=model.get("id"),
        temperature=model.get("temperature", 0.4),
        judge_model_id=judge.get("id"),
        judge_temperature=judge.get("temperature", 0.0),
        default_max_words=block.get("default_max_words", 1500),
        enforce_word_limit=block.get("enforce_word_limit", True),
        max_revisions=block.get("max_revisions", 2),
    )


def _map_research_agent(block: dict[str, Any]) -> ResearchAgentConfig:
    """Flatten the nested ``research_agent`` YAML block into its config model.

    Args:
        block: The ``research_agent`` mapping from YAML.

    Returns:
        A populated ResearchAgentConfig.
    """
    model = block.get("model", {})
    summarizer = block.get("summarizer_model", {})
    summarization = block.get("summarization", {})
    caching = block.get("caching", {})
    return ResearchAgentConfig(
        model_id=model.get("id"),
        temperature=model.get("temperature", 0.0),
        summarizer_model_id=summarizer.get("id"),
        summarizer_temperature=summarizer.get("temperature", 0.0),
        max_search_calls=block.get("max_search_calls", 5),
        recursion_limit=block.get("recursion_limit", 12),
        search_top_k=block.get("search_top_k", 5),
        trigger_messages=summarization.get("trigger_messages", 16),
        keep_recent=summarization.get("keep_recent", 6),
        llm_cache=caching.get("llm_cache", "memory"),
        tavily_ttl_seconds=caching.get("tavily_ttl_seconds", 3600),
    )


def _map_analysis_agent(block: dict[str, Any]) -> AnalysisAgentConfig:
    """Flatten the nested ``analysis_agent`` YAML block into its config model.

    Args:
        block: The ``analysis_agent`` mapping from YAML.

    Returns:
        A populated AnalysisAgentConfig.
    """
    model = block.get("model", {})
    summarizer = block.get("summarizer_model", {})
    summarization = block.get("summarization", {})
    return AnalysisAgentConfig(
        model_id=model.get("id"),
        temperature=model.get("temperature", 0.2),
        summarizer_model_id=summarizer.get("id"),
        summarizer_temperature=summarizer.get("temperature", 0.0),
        recursion_limit=block.get("recursion_limit", 10),
        max_compute_calls=block.get("max_compute_calls", 6),
        confidence_threshold=block.get("confidence_threshold", 0.5),
        trigger_messages=summarization.get("trigger_messages", 16),
        keep_recent=summarization.get("keep_recent", 6),
    )


def _map_code_agent(block: dict[str, Any]) -> CodeAgentConfig:
    """Flatten the nested ``code_agent`` YAML block into its config model.

    Args:
        block: The ``code_agent`` mapping from YAML.

    Returns:
        A populated CodeAgentConfig.
    """
    model = block.get("model", {})
    review = block.get("review_model", {})
    return CodeAgentConfig(
        model_id=model.get("id"),
        temperature=model.get("temperature", 0.2),
        default_language=block.get("default_language", "python"),
        max_syntax_retries=block.get("max_syntax_retries", 4),
        review_model_id=review.get("id", "gemini-2.5-flash-lite"),
        review_temperature=review.get("temperature", 0.0),
        max_review_retries=block.get("max_review_retries", 1),
    )


def _map_orchestrator(block: dict[str, Any]) -> OrchestratorConfig:
    """Flatten the nested ``orchestrator`` YAML block into its config model.

    Args:
        block: The ``orchestrator`` mapping from YAML.

    Returns:
        A populated OrchestratorConfig.
    """
    planner = block.get("planner_model", {})
    decider = block.get("decider_model", {})
    synthesizer = block.get("synthesizer_model", {})
    judge = block.get("judge_model", {})
    bounds = block.get("bounds", {})
    return OrchestratorConfig(
        planner_model_id=planner.get("id"),
        planner_temperature=planner.get("temperature", 0.2),
        decider_model_id=decider.get("id"),
        decider_temperature=decider.get("temperature", 0.0),
        synthesizer_model_id=synthesizer.get("id"),
        synthesizer_temperature=synthesizer.get("temperature", 0.3),
        judge_model_id=judge.get("id"),
        judge_temperature=judge.get("temperature", 0.0),
        max_replans=bounds.get("max_replans", 1),
        max_resynth=bounds.get("max_resynth", 2),
        concurrency=bounds.get("concurrency", 3),
        max_steps=bounds.get("max_steps", 12),
        step_timeout_seconds=bounds.get("step_timeout_seconds", 120),
        planner_max_attempts=bounds.get("planner_max_attempts", 2),
        context_char_budget=bounds.get("context_char_budget", 6000),
    )


def _map_pricing(block: dict[str, Any]) -> dict[str, ModelPrice]:
    """Map the ``pricing`` YAML block into per-model price models."""
    return {model_id: ModelPrice(**prices) for model_id, prices in block.items()}


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache the application config (secrets + runtime params).

    Returns:
        The singleton AppConfig instance.
    """
    raw = _read_yaml()
    return AppConfig(  # type: ignore[call-arg]
        writing_agent=_map_writing_agent(raw.get("writing_agent", {})),
        research_agent=_map_research_agent(raw.get("research_agent", {})),
        analysis_agent=_map_analysis_agent(raw.get("analysis_agent", {})),
        code_agent=_map_code_agent(raw.get("code_agent", {})),
        orchestrator=_map_orchestrator(raw.get("orchestrator", {})),
        pricing=_map_pricing(raw.get("pricing", {})),
    )
