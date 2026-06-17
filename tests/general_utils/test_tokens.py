"""Offline tests for the prompt-token estimator (no network, deterministic).

Covers empty prompts, the character-ratio heuristic, monotonic growth with prompt size, and
sensitivity to the configured ``chars_per_token``.

Run standalone: ``python tests/general_utils/test_tokens.py`` or via pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.src.general_utils.tokens import count_prompt_tokens  # noqa: E402


def test_empty_prompt_is_zero() -> None:
    assert count_prompt_tokens([]) == 0
    assert count_prompt_tokens([{"role": "user", "content": ""}]) == 0


def test_counts_real_prompt_length() -> None:
    messages = [{"role": "user", "content": "x" * 40}]
    assert count_prompt_tokens(messages, chars_per_token=4.0) == 10


def test_scales_with_prompt_size() -> None:
    small = [{"role": "user", "content": "short"}]
    large = [{"role": "user", "content": "long " * 500}]
    assert count_prompt_tokens(large) > count_prompt_tokens(small)


def test_aggregates_across_messages() -> None:
    one = [{"role": "system", "content": "a" * 20}]
    two = [{"role": "system", "content": "a" * 20}, {"role": "user", "content": "b" * 20}]
    assert count_prompt_tokens(two) > count_prompt_tokens(one)


def test_chars_per_token_changes_estimate() -> None:
    messages = [{"role": "user", "content": "y" * 40}]
    coarse = count_prompt_tokens(messages, chars_per_token=8.0)
    fine = count_prompt_tokens(messages, chars_per_token=2.0)
    assert fine > coarse


def _main() -> None:
    tests = [
        test_empty_prompt_is_zero,
        test_counts_real_prompt_length,
        test_scales_with_prompt_size,
        test_aggregates_across_messages,
        test_chars_per_token_changes_estimate,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _main()
