"""Label-based eval for the synthesis quality judge.

The component under test is itself an LLM-as-judge, so this eval is label-based rather than
LLM-graded: each dataset example is a hand-built synthesis scenario with a KNOWN-CORRECT expected
verdict. The harness reconstructs the plan + agent outputs + draft, runs the real deterministic
checks (``check_synthesis``) and the live judge (``judge_synthesis``), and scores PASS when the
judge's verdict matches the ground-truth label. One LLM call per example keeps it under free-tier
quota; no second judge is needed because the labels ARE the ground truth.

Examples:
    uv run python evals/judges/synthesis_judge_eval.py --all
    uv run python evals/judges/synthesis_judge_eval.py --id SY-01
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT.parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
load_dotenv(ROOT / ".env", override=True)
load_dotenv(ROOT.parent / ".env")

import yaml  # noqa: E402

from app.src.engine.synthesis_judge import check_synthesis, judge_synthesis  # noqa: E402
from app.src.engine.synthesizer import Synthesis  # noqa: E402
from app.src.general_utils.agent_base import AgentResult  # noqa: E402
from app.src.schemas.plan import ExecutionPlan, ExecutionStep  # noqa: E402

DATASET_PATH = ROOT / "evals" / "datasets" / "synthesis_judge.yaml"
REPORTS_DIR = ROOT / "evals" / "reports"


def load_examples(path: Path) -> list[dict]:
    """Load all examples from the dataset YAML."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle).get("examples", [])


def load_example(example_id: str, path: Path) -> dict:
    """Find a single example by id, raising if absent."""
    match = next((ex for ex in load_examples(path) if ex.get("id") == example_id), None)
    if match is None:
        raise ValueError(f"No example with id '{example_id}' in {path}")
    return match


def _build_plan(example: dict) -> ExecutionPlan:
    """Reconstruct an ExecutionPlan from the example's step specs."""
    steps = [
        ExecutionStep(
            id=spec["id"], agent=spec["agent"], action=spec["action"],
            dependencies=spec.get("dependencies", []),
        )
        for spec in example["steps"]
    ]
    return ExecutionPlan(reasoning="eval scenario", task_id=example["id"], steps=steps)


def _build_results(example: dict) -> dict[str, AgentResult]:
    """Reconstruct the completed-step results dict from the example's outputs."""
    results: dict[str, AgentResult] = {}
    for out in example["outputs"]:
        results[out["step_id"]] = AgentResult(
            step_id=out["step_id"], agent="agent", status="completed",
            output={"content": out["content"], "confidence": out.get("confidence"),
                    "sources": out.get("sources", [])},
            tokens_used=1, execution_time_ms=1,
        )
    return results


async def _judge_example(example: dict) -> dict:
    """Run the deterministic checks and the live judge over one scenario."""
    plan = _build_plan(example)
    results = _build_results(example)
    draft_spec = example["draft"]
    draft = Synthesis(content=draft_spec["content"], confidence=draft_spec["confidence"])
    det_errors = check_synthesis(draft, plan, len(results), example.get("output_format"))
    verdict, tokens = await judge_synthesis(
        example["goal"], plan, results, draft, det_errors
    )
    return _collect_row(example, verdict, det_errors, tokens)


def _collect_row(example: dict, verdict, det_errors: list[str], tokens: int) -> dict:
    """Assemble the per-example result row for reporting."""
    expected = example["expected_verdict"]
    got = verdict.verdict
    return {
        "id": example["id"],
        "category": example.get("category", "N/A"),
        "expected": expected,
        "got": got,
        "status": "PASS" if got == expected else "FAIL",
        "det_errors": det_errors,
        "reasoning": verdict.reasoning,
        "feedback": verdict.feedback,
        "new_steps": [s.model_dump() for s in verdict.new_steps],
        "tokens": tokens,
    }


def evaluate_example(example: dict) -> dict:
    """Run one example end-to-end and print a one-line status."""
    row = asyncio.run(_judge_example(example))
    print(
        f"[{row['status']}] {row['id']} | {row['category']} "
        f"| expected={row['expected']} got={row['got']} det_errors={len(row['det_errors'])}"
    )
    return row


def _summary_line(results: list[dict]) -> str:
    """Build the totals/pass-rate summary line."""
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    rate = f"{passed / total * 100:.0f}%" if total else "0%"
    return f"Total: {total} | Passed: {passed} | Failed: {total - passed} | Pass rate: {rate}"


def _format_result_block(r: dict) -> list[str]:
    """Render one result as markdown lines."""
    det = "; ".join(r["det_errors"]) or "none"
    steps = ", ".join(s["id"] + "/" + s["agent"] for s in r["new_steps"]) or "none"
    return [
        f"## [{r['status']}] {r['id']} — {r['category']}",
        f"- expected: `{r['expected']}` | got: `{r['got']}` | tokens: {r['tokens']}",
        f"- deterministic checks: {det}",
        f"- judge reasoning: {r['reasoning'][:300]}",
        f"- judge feedback: {(r['feedback'] or 'N/A')[:300]}",
        f"- authored new_steps: {steps}",
        "",
    ]


def write_markdown_report(results: list[dict]) -> Path:
    """Write a timestamped markdown report and return its path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = REPORTS_DIR / f"synthesis_judge_eval_{timestamp}.md"
    lines = [f"# Synthesis Judge Eval — {timestamp}", "", f"**{_summary_line(results)}**", ""]
    for r in results:
        lines += _format_result_block(r)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def print_summary(results: list[dict]) -> None:
    """Print a console summary table."""
    print("\n" + "=" * 70)
    print("SYNTHESIS JUDGE EVAL SUMMARY")
    print("=" * 70)
    for r in results:
        print(
            f"  [{r['status']}] {r['id']} ({r['category']}): "
            f"expected={r['expected']} got={r['got']}"
        )
    print("-" * 70)
    print(_summary_line(results))
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the synthesis judge (label-based)")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    parser.add_argument("--id", type=str, default=None, help="Run a single example by id")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="Dataset YAML path")
    return parser.parse_args()


def main() -> None:
    """Entry point: run one or all examples and write a report."""
    args = parse_args()
    dataset = Path(args.dataset)
    if args.id:
        results = [evaluate_example(load_example(args.id, dataset))]
    else:
        results = [evaluate_example(ex) for ex in load_examples(dataset)]
    print_summary(results)
    report = write_markdown_report(results)
    print(f"\nReport written to: {report}")


if __name__ == "__main__":
    main()
