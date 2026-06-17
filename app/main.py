"""Entry point: start the orchestration HTTP API with uvicorn.

Run from the ``app`` directory with ``uv run python main.py``. Host/port are env-overridable
(``HOST`` / ``PORT``) so the container and local runs share one entry point. The repo root is put on
``sys.path`` so the ``app`` package imports identically whether or not ``PYTHONPATH`` is set (the
container sets it; a local ``uv run`` does not).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import uvicorn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestration.api")


def main() -> None:
    """Launch the API server, binding host/port from the environment."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    logger.info("starting orchestration API on %s:%d", host, port)
    uvicorn.run("app.src.api.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
