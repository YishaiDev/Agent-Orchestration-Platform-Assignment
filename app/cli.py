"""Local CLI: run a single goal through the full orchestration loop and print the result.

Useful for debugging without the HTTP layer. Run with::

    uv run python app/cli.py "Compare X and Y and write a brief"

Hits real agents and Gemini, so it needs ``GOOGLE_API_KEY`` (and ``TAVILY_API_KEY`` for research).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid

from app.src.engine.graph import run_task
from app.src.engine.runs import get_run_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestration.cli")


def _parse_args() -> argparse.Namespace:
    """Parse the goal and optional run bounds from the command line."""
    parser = argparse.ArgumentParser(description="Run one goal through the orchestration engine.")
    parser.add_argument("goal", help="The task goal to plan and execute.")
    parser.add_argument("--constraints", default="", help="Optional free-text constraints.")
    parser.add_argument(
        "--output-format", default="", help="Optional output format hint, e.g. markdown."
    )
    parser.add_argument("--max-replans", type=int, default=None, help="Override the re-plan bound.")
    parser.add_argument(
        "--deadline", type=float, default=None, help="Optional wall-clock budget in seconds."
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> dict[str, object]:
    """Drive one task and return its monitor's final result."""
    task_id = uuid.uuid4().hex
    await run_task(
        task_id,
        args.goal,
        args.constraints,
        session_id="cli",
        max_replans=args.max_replans,
        deadline_seconds=args.deadline,
        output_format=args.output_format,
    )
    monitor = get_run_registry().get(task_id)
    return monitor.final_result or {} if monitor else {}


def main() -> None:
    """Parse args, run the goal, and print the final result as JSON."""
    args = _parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
