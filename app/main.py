"""Entry point: start the orchestration HTTP API with uvicorn.

Run with ``uv run python app/main.py``. Host/port are env-overridable (``HOST`` / ``PORT``) so the
container and local runs share one entry point.
"""

from __future__ import annotations

import logging
import os

import uvicorn

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
