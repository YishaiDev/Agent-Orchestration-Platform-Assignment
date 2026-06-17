"""Cost helpers: per-1M-token pricing into pre-execution estimates and post-execution actuals."""

from __future__ import annotations

from app.src.schemas.config import ModelPrice

_PER_MILLION = 1_000_000


def token_cost(price: ModelPrice, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost for a token count under a model's price table.

    Args:
        price: Per-1M-token input/output prices for the model.
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens produced.

    Returns:
        Cost in USD.
    """
    return (input_tokens * price.input + output_tokens * price.output) / _PER_MILLION


def estimate_cost(
    price: ModelPrice, expected_calls: int, avg_input_tokens: int, avg_output_tokens: int
) -> float:
    """Estimate USD cost before execution from expected call volume and average token sizes.

    Args:
        price: Per-1M-token prices for the model.
        expected_calls: Anticipated number of model calls (initial + tool turns + summary).
        avg_input_tokens: Assumed average prompt tokens per call.
        avg_output_tokens: Assumed average completion tokens per call.

    Returns:
        Estimated cost in USD.
    """
    return token_cost(price, avg_input_tokens, avg_output_tokens) * expected_calls
