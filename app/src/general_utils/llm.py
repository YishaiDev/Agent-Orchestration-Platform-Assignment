"""LLM factory: provider-agnostic chat model init via LangChain ``init_chat_model``.

The active provider is selected in ``config.yaml`` (``provider:``). Supported providers:
``groq`` (``GROQ_API_KEY``) and ``gemini`` / Google (``GOOGLE_API_KEY``). Groq is the default
because its free tier is far larger than Gemini's, which lets multi-step runs complete without
hitting a daily request cap.
"""

from __future__ import annotations

from typing import Any, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from app.src.schemas.config import get_config


def _resolve_provider_key() -> tuple[str, str, str]:
    """Resolve the LangChain provider, api-key kwarg name, and key for the active provider.

    The provider is taken from ``config.yaml`` (``provider:``); the model id never decides it.

    Returns:
        A ``(model_provider, key_kwarg, api_key)`` tuple for ``init_chat_model``.

    Raises:
        ValueError: If the provider is unknown or its required API key is not configured.
    """
    cfg = get_config()
    if cfg.provider == "groq":
        if cfg.groq_api_key is None:
            raise ValueError("GROQ_API_KEY is required for Groq models but is not set.")
        return "groq", "api_key", cfg.groq_api_key.get_secret_value()
    if cfg.provider == "gemini":
        if cfg.google_api_key is None:
            raise ValueError("GOOGLE_API_KEY is required for Gemini models but is not set.")
        return "google_genai", "google_api_key", cfg.google_api_key.get_secret_value()
    raise ValueError(f"Unknown provider '{cfg.provider}' (expected 'groq' or 'gemini').")


def build_chat_model(
    model_id: str, temperature: float, max_retries: int = 3
) -> BaseChatModel:
    """Build a chat model through LangChain's provider-agnostic initializer.

    The provider and API key come from the active ``provider`` in config, so callers stay
    provider-agnostic and only one line in ``config.yaml`` decides which backend is used.

    Args:
        model_id: Model id for the active provider (e.g. ``llama-3.3-70b-versatile``,
            ``gemini-2.5-flash``).
        temperature: Sampling temperature.
        max_retries: Provider-level retries with exponential backoff on transient errors
            (e.g. 429 rate limits and 503 UNAVAILABLE "high demand" spikes).

    Returns:
        An initialized chat model ready for ``with_structured_output``.

    Raises:
        ValueError: If the provider is unknown or its required API key is not configured.
    """
    provider, key_kwarg, api_key = _resolve_provider_key()
    kwargs: dict[str, Any] = {key_kwarg: api_key}
    model = init_chat_model(
        model_id,
        model_provider=provider,
        temperature=temperature,
        max_retries=max_retries,
        **kwargs,
    )
    return cast(BaseChatModel, model)
