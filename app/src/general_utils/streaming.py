"""Status streaming hook shared by agent tools.

Provides a single ``emit_status`` sink so tools can surface progress without depending on a UI
layer. The current implementation logs at INFO; a real status channel (SSE, websocket, queue) can
replace the body without touching callers.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def emit_status(message: str) -> None:
    """Emit a short human-readable progress message for the current agent step.

    Args:
        message: Status text to surface (e.g. ``"Calculating..."``).
    """
    logger.info("STATUS: %s", message)
