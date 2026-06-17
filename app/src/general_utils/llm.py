"""LLM factory: provider-agnostic chat model init via LangChain ``init_chat_model`` (Gemini)."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel


def build_chat_model(
    model_id: str, temperature: float, api_key: str, max_retries: int = 6
) -> BaseChatModel:
    """Build a Gemini chat model through LangChain's provider-agnostic initializer.

    Args:
        model_id: Gemini model id (e.g. ``gemini-2.5-flash``).
        temperature: Sampling temperature.
        api_key: Google Generative AI API key.
        max_retries: Provider-level retries with backoff on transient errors (e.g. 429).

    Returns:
        An initialized chat model ready for ``with_structured_output``.
    """
    return init_chat_model(
        model_id,
        model_provider="google_genai",
        temperature=temperature,
        google_api_key=api_key,
        max_retries=max_retries,
    )
