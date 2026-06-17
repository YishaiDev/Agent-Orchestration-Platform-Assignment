"""Master test runner: discovers and runs every test under ``tests/`` via pytest."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent


def main() -> int:
    """Run the full test suite.

    Returns:
        The pytest exit code (0 on success).
    """
    return pytest.main([str(TESTS_DIR), "-v"])


if __name__ == "__main__":
    sys.exit(main())
