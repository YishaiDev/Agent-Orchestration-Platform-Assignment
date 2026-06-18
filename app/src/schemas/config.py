"""Application configuration: runtime parameters from ``app/config.yaml`` plus secrets from env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
LLM_CONFIG_PATH = Path(__file__).resolve().parents[2] / "llm_config.yml"
DEFAULT_PROVIDER = "groq"


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
    avg_output_tokens: int = 350
    tool_retry_attempts: int = 2
    tool_retry_temp_bump: float = 0.3


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
    avg_output_tokens: int = 400


class CodeAgentConfig(BaseModel):
    """Runtime parameters for the Code Agent (parser gate plus LLM-critic fallback)."""

    model_id: str
    temperature: float = 0.2
    default_language: str = "python"
    max_syntax_retries: int = 4
    review_model_id: str
    review_temperature: float = 0.0
    max_review_retries: int = 1
    avg_output_tokens: int = 450


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


class EstimationConfig(BaseModel):
    """Runtime parameters for pre-execution cost estimation."""

    chars_per_token: float = 4.0


class ModelPrice(BaseModel):
    """USD price per 1M tokens for one model id."""

    input: float
    output: float


class AppConfig(BaseSettings):
    """Top-level config: secrets from ``.env``, runtime params injected from YAML."""

    provider: str = DEFAULT_PROVIDER
    google_api_key: SecretStr | None = Field(default=None, alias="GOOGLE_API_KEY")
    groq_api_key: SecretStr | None = Field(default=None, alias="GROQ_API_KEY")
    tavily_api_key: SecretStr | None = Field(default=None, alias="TAVILY_API_KEY")
    writing_agent: WritingAgentConfig
    research_agent: ResearchAgentConfig
    analysis_agent: AnalysisAgentConfig
    code_agent: CodeAgentConfig
    orchestrator: OrchestratorConfig
    estimation: EstimationConfig = Field(default_factory=EstimationConfig)
    pricing: dict[str, ModelPrice] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", populate_by_name=True
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    """Load a raw YAML file into a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML as a dict (empty if the file is blank).
    """
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _active_tiers(llm: dict[str, Any], provider: str) -> dict[str, str]:
    """Return the tier -> model-id map for the active provider.

    Args:
        llm: Parsed ``llm_config.yml`` mapping.
        provider: Active provider name (e.g. ``groq`` or ``gemini``).

    Returns:
        Mapping of tier name (``big``/``small``) to a concrete model id.

    Raises:
        KeyError: If the provider is missing from ``llm_config.yml``.
    """
    providers = llm.get("providers", {})
    if provider not in providers:
        raise KeyError(f"provider '{provider}' not found in llm_config.yml")
    tiers: dict[str, str] = providers[provider].get("tiers", {})
    return tiers


def _merged_pricing(llm: dict[str, Any]) -> dict[str, ModelPrice]:
    """Merge per-model pricing across all providers (keyed by model id).

    Args:
        llm: Parsed ``llm_config.yml`` mapping.

    Returns:
        A flat mapping of model id to its price model.
    """
    merged: dict[str, ModelPrice] = {}
    for spec in llm.get("providers", {}).values():
        for model_id, prices in spec.get("pricing", {}).items():
            merged[model_id] = ModelPrice(**prices)
    return merged


def _tier_id(sub_block: dict[str, Any], tiers: dict[str, str]) -> str:
    """Resolve a model sub-block's ``tier`` into a concrete model id.

    Args:
        sub_block: A model mapping such as ``{tier: big, temperature: 0.2}``.
        tiers: The active provider's tier -> model-id map.

    Returns:
        The concrete model id for the requested tier.

    Raises:
        KeyError: If the requested tier is not defined for the active provider.
    """
    tier = sub_block.get("tier", "big")
    if tier not in tiers:
        raise KeyError(f"tier '{tier}' not defined for the active provider")
    return tiers[tier]


def _map_writing_agent(
    block: dict[str, Any], tiers: dict[str, str]
) -> WritingAgentConfig:
    """Flatten the nested ``writing_agent`` YAML block into its config model.

    Args:
        block: The ``writing_agent`` mapping from YAML.
        tiers: The active provider's tier -> model-id map.

    Returns:
        A populated WritingAgentConfig.
    """
    model = block.get("model", {})
    judge = block.get("judge_model", {})
    return WritingAgentConfig(
        model_id=_tier_id(model, tiers),
        temperature=model.get("temperature", 0.4),
        judge_model_id=_tier_id(judge, tiers),
        judge_temperature=judge.get("temperature", 0.0),
        default_max_words=block.get("default_max_words", 1500),
        enforce_word_limit=block.get("enforce_word_limit", True),
        max_revisions=block.get("max_revisions", 2),
    )


def _map_research_agent(
    block: dict[str, Any], tiers: dict[str, str]
) -> ResearchAgentConfig:
    """Flatten the nested ``research_agent`` YAML block into its config model.

    Args:
        block: The ``research_agent`` mapping from YAML.
        tiers: The active provider's tier -> model-id map.

    Returns:
        A populated ResearchAgentConfig.
    """
    model = block.get("model", {})
    summarizer = block.get("summarizer_model", {"tier": "small"})
    summarization = block.get("summarization", {})
    caching = block.get("caching", {})
    return ResearchAgentConfig(
        model_id=_tier_id(model, tiers),
        temperature=model.get("temperature", 0.0),
        summarizer_model_id=_tier_id(summarizer, tiers),
        summarizer_temperature=summarizer.get("temperature", 0.0),
        max_search_calls=block.get("max_search_calls", 5),
        recursion_limit=block.get("recursion_limit", 12),
        search_top_k=block.get("search_top_k", 5),
        trigger_messages=summarization.get("trigger_messages", 16),
        keep_recent=summarization.get("keep_recent", 6),
        llm_cache=caching.get("llm_cache", "memory"),
        tavily_ttl_seconds=caching.get("tavily_ttl_seconds", 3600),
        avg_output_tokens=block.get("avg_output_tokens", 350),
        tool_retry_attempts=block.get("tool_retry_attempts", 2),
        tool_retry_temp_bump=block.get("tool_retry_temp_bump", 0.3),
    )


def _map_analysis_agent(
    block: dict[str, Any], tiers: dict[str, str]
) -> AnalysisAgentConfig:
    """Flatten the nested ``analysis_agent`` YAML block into its config model.

    Args:
        block: The ``analysis_agent`` mapping from YAML.
        tiers: The active provider's tier -> model-id map.

    Returns:
        A populated AnalysisAgentConfig.
    """
    model = block.get("model", {})
    summarizer = block.get("summarizer_model", {"tier": "small"})
    summarization = block.get("summarization", {})
    return AnalysisAgentConfig(
        model_id=_tier_id(model, tiers),
        temperature=model.get("temperature", 0.2),
        summarizer_model_id=_tier_id(summarizer, tiers),
        summarizer_temperature=summarizer.get("temperature", 0.0),
        recursion_limit=block.get("recursion_limit", 10),
        max_compute_calls=block.get("max_compute_calls", 6),
        confidence_threshold=block.get("confidence_threshold", 0.5),
        trigger_messages=summarization.get("trigger_messages", 16),
        keep_recent=summarization.get("keep_recent", 6),
        avg_output_tokens=block.get("avg_output_tokens", 400),
    )


def _map_code_agent(block: dict[str, Any], tiers: dict[str, str]) -> CodeAgentConfig:
    """Flatten the nested ``code_agent`` YAML block into its config model.

    Args:
        block: The ``code_agent`` mapping from YAML.
        tiers: The active provider's tier -> model-id map.

    Returns:
        A populated CodeAgentConfig.
    """
    model = block.get("model", {})
    review = block.get("review_model", {"tier": "small"})
    return CodeAgentConfig(
        model_id=_tier_id(model, tiers),
        temperature=model.get("temperature", 0.2),
        default_language=block.get("default_language", "python"),
        max_syntax_retries=block.get("max_syntax_retries", 4),
        review_model_id=_tier_id(review, tiers),
        review_temperature=review.get("temperature", 0.0),
        max_review_retries=block.get("max_review_retries", 1),
        avg_output_tokens=block.get("avg_output_tokens", 450),
    )


def _map_orchestrator(
    block: dict[str, Any], tiers: dict[str, str]
) -> OrchestratorConfig:
    """Flatten the nested ``orchestrator`` YAML block into its config model.

    Args:
        block: The ``orchestrator`` mapping from YAML.
        tiers: The active provider's tier -> model-id map.

    Returns:
        A populated OrchestratorConfig.
    """
    planner = block.get("planner_model", {})
    decider = block.get("decider_model", {})
    synthesizer = block.get("synthesizer_model", {})
    judge = block.get("judge_model", {})
    bounds = block.get("bounds", {})
    return OrchestratorConfig(
        planner_model_id=_tier_id(planner, tiers),
        planner_temperature=planner.get("temperature", 0.2),
        decider_model_id=_tier_id(decider, tiers),
        decider_temperature=decider.get("temperature", 0.0),
        synthesizer_model_id=_tier_id(synthesizer, tiers),
        synthesizer_temperature=synthesizer.get("temperature", 0.3),
        judge_model_id=_tier_id(judge, tiers),
        judge_temperature=judge.get("temperature", 0.0),
        max_replans=bounds.get("max_replans", 1),
        max_resynth=bounds.get("max_resynth", 2),
        concurrency=bounds.get("concurrency", 3),
        max_steps=bounds.get("max_steps", 12),
        step_timeout_seconds=bounds.get("step_timeout_seconds", 120),
        planner_max_attempts=bounds.get("planner_max_attempts", 2),
        context_char_budget=bounds.get("context_char_budget", 6000),
    )


def _map_estimation(block: dict[str, Any]) -> EstimationConfig:
    """Flatten the ``estimation`` YAML block into its config model.

    Args:
        block: The ``estimation`` mapping from YAML.

    Returns:
        A populated EstimationConfig (defaults applied when keys are absent).
    """
    return EstimationConfig(chars_per_token=block.get("chars_per_token", 4.0))


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache the application config (secrets + runtime params).

    Model ids are resolved from ``llm_config.yml`` using the active ``provider``
    and each agent's tier (``big``/``small``); pricing is merged across providers.

    Returns:
        The singleton AppConfig instance.
    """
    raw = _read_yaml(CONFIG_PATH)
    llm = _read_yaml(LLM_CONFIG_PATH)
    provider = raw.get("provider", DEFAULT_PROVIDER)
    tiers = _active_tiers(llm, provider)
    return AppConfig(
        provider=provider,
        writing_agent=_map_writing_agent(raw.get("writing_agent", {}), tiers),
        research_agent=_map_research_agent(raw.get("research_agent", {}), tiers),
        analysis_agent=_map_analysis_agent(raw.get("analysis_agent", {}), tiers),
        code_agent=_map_code_agent(raw.get("code_agent", {}), tiers),
        orchestrator=_map_orchestrator(raw.get("orchestrator", {}), tiers),
        estimation=_map_estimation(raw.get("estimation", {})),
        pricing=_merged_pricing(llm),
    )
