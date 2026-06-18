"""LLM-as-judge eval for the Writing Agent.

Runs the real Writing Agent (live Gemini) on each dataset example, then asks a Gemini judge to
return PASS/FAIL + Reason + Suggested Fix against the example's expected criteria and the agent's
observability signals (status, word_count, tokens, time). Writes a markdown report.

Examples:
    uv run python evals/judges/writing_judge.py --all
    uv run python evals/judges/writing_judge.py --id W-02
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(APP_DIR / ".env", override=True)
load_dotenv(REPO_ROOT / ".env")

import yaml  # noqa: E402
from tenacity import retry, stop_after_attempt, wait_exponential_jitter  # noqa: E402

from app.src.general_utils.agent_base import AgentResult  # noqa: E402
from app.src.general_utils.llm import build_chat_model  # noqa: E402
from app.src.schemas import get_config  # noqa: E402
from app.src.sub_agents.writing.agent import run_writing_agent  # noqa: E402
from app.src.sub_agents.writing.schemas import WritingInput  # noqa: E402

DATASET_PATH = APP_DIR / "evals" / "datasets" / "writing_agent.yaml"
REPORTS_DIR = APP_DIR / "evals" / "reports"
MAX_WORKERS = 1  # free-tier Gemini is 5 req/min; sequential + backoff stays under quota

JUDGE_PROMPT = """You are an evaluation judge for a Writing Agent.

The agent generates content, edits/improves it, formats it, and self-critiques in a loop.

## Task instruction given to the agent:
"{instruction}"

## Constraints:
{constraints}

## Requested output format:
{output_format}

## Expected outcome (GROUND TRUTH criteria):
{expected_output}

## Agent run signals:
- status: {status}            (completed | completed_degraded | failed)
- word_count: {word_count}
- tokens_used: {tokens_used}
- execution_time_ms: {execution_time_ms}

## Agent's actual output content:
<output>
{content}
</output>

JUDGING RULES:
- PASS if the content satisfies the expected outcome and respects the requested format.
- Word limit is SOFT: small overage (<= ~20%) is acceptable; a large overage is a FAIL.
- status "failed" or empty content is always a FAIL.
- status "completed_degraded" is acceptable (PASS) if the content still meets the criteria.
- Minor wording imperfections do NOT cause a FAIL when the criteria are met.
- The output must not contain leftover instructions, fence tags, or meta-commentary.

Respond in EXACTLY this format (no deliberation):
PASS or FAIL
Reason: <one sentence>
Suggested Fix: <one sentence, or "N/A" if PASS>
"""


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


def build_input(example: dict) -> WritingInput:
    """Map a dataset example to a WritingInput."""
    return WritingInput(
        instruction=example["instruction"],
        source_material=example.get("source_material", "") or "",
        constraints=example.get("constraints", {}) or {},
        output_format=example.get("output_format", "markdown"),
    )


def build_judge_model():
    """Build the judge model from config (deterministic)."""
    cfg = get_config()
    return build_chat_model(cfg.writing_agent.judge_model_id, 0.0)


def apply_model_override(model_id: str) -> None:
    """Retarget the writer + judge models on the cached config singleton.

    Mutates the in-memory config only (prod ``config.yaml`` is untouched), so both the
    Writing Agent and the judge use ``model_id`` for this eval run. Used to dodge an
    exhausted per-model daily free-tier quota.

    Args:
        model_id: Gemini model id to use for writer, judge nodes, and the eval judge.
    """
    writing = get_config().writing_agent
    writing.model_id = model_id
    writing.judge_model_id = model_id


def _as_text(content: object) -> str:
    """Coerce a model message content (str or list of parts) to plain text."""
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)


@retry(stop=stop_after_attempt(6), wait=wait_exponential_jitter(initial=4, max=70), reraise=True)
def _invoke_judge(model, prompt: str) -> str:
    """Invoke the judge model with bounded retry (honors free-tier 429 backoff)."""
    return _as_text(model.invoke(prompt).content)


def judge(example: dict, result: AgentResult, model) -> str:
    """Ask the judge model to score one agent run."""
    prompt = JUDGE_PROMPT.format(
        instruction=example["instruction"],
        constraints=example.get("constraints", {}),
        output_format=example.get("output_format", "markdown"),
        expected_output=example.get("expected_output", "N/A").strip(),
        status=result.status,
        word_count=result.output.get("word_count", "N/A"),
        tokens_used=result.tokens_used,
        execution_time_ms=result.execution_time_ms,
        content=result.output.get("content", result.output.get("error", "")),
    )
    return _invoke_judge(model, prompt)


def parse_verdict(verdict: str) -> str:
    """Extract PASS or FAIL (last occurrence wins)."""
    status = "UNKNOWN"
    for line in verdict.strip().splitlines():
        upper = line.strip().upper()
        if upper.startswith("PASS"):
            status = "PASS"
        elif upper.startswith("FAIL"):
            status = "FAIL"
    return status


def parse_verdict_details(verdict: str) -> dict:
    """Split verdict text into status, reason, and suggested_fix."""
    reason = re.search(r"Reason:\s*(.+)", verdict, re.IGNORECASE)
    fix = re.search(r"Suggested Fix:\s*(.+)", verdict, re.IGNORECASE)
    return {
        "status": parse_verdict(verdict),
        "reason": reason.group(1).strip() if reason else "N/A",
        "suggested_fix": fix.group(1).strip() if fix else "N/A",
    }


def evaluate_example(example: dict, judge_model) -> dict:
    """Run the agent and judge for one example."""
    example_id = example.get("id", "unknown")
    result = run_writing_agent(build_input(example), step_id=example_id)
    verdict = judge(example, result, judge_model)
    details = parse_verdict_details(verdict)
    print(f"[{details['status']}] {example_id} | {example['category']} "
          f"| words={result.output.get('word_count', 'NA')} tokens={result.tokens_used}")
    return {
        "id": example_id,
        "category": example.get("category", "N/A"),
        "status": result.status,
        "word_count": result.output.get("word_count", "N/A"),
        "tokens_used": result.tokens_used,
        "execution_time_ms": result.execution_time_ms,
        "content": result.output.get("content", result.output.get("error", "")),
        "verdict_status": details["status"],
        "reason": details["reason"],
        "suggested_fix": details["suggested_fix"],
    }


def run_parallel(examples: list[dict], judge_model) -> list[dict]:
    """Evaluate all examples concurrently, preserving input order."""
    results: list[dict | None] = [None] * len(examples)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_example, ex, judge_model): i
                   for i, ex in enumerate(examples)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [r for r in results if r is not None]


def _summary_line(results: list[dict]) -> str:
    """Build the totals/pass-rate summary line."""
    total = len(results)
    passed = sum(1 for r in results if r["verdict_status"] == "PASS")
    rate = f"{passed / total * 100:.0f}%" if total else "0%"
    return f"Total: {total} | Passed: {passed} | Failed: {total - passed} | Pass rate: {rate}"


def write_markdown_report(results: list[dict]) -> Path:
    """Write a timestamped markdown report and return its path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = REPORTS_DIR / f"writing_eval_{timestamp}.md"
    lines = [f"# Writing Agent Eval — {timestamp}", "", f"**{_summary_line(results)}**", ""]
    for r in results:
        lines += _format_result_block(r)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _format_result_block(r: dict) -> list[str]:
    """Render one result as markdown lines."""
    preview = r["content"].replace("\n", " ")[:300]
    return [
        f"## [{r['verdict_status']}] {r['id']} — {r['category']}",
        f"- run status: `{r['status']}` | words: {r['word_count']} "
        f"| tokens: {r['tokens_used']} | time: {r['execution_time_ms']}ms",
        f"- reason: {r['reason']}",
        f"- suggested fix: {r['suggested_fix']}",
        f"- output preview: {preview}",
        "",
    ]


def print_summary(results: list[dict]) -> None:
    """Print a console summary table."""
    print("\n" + "=" * 70)
    print("WRITING AGENT EVAL SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  [{r['verdict_status']}] {r['id']} ({r['category']}): {r['reason']}")
    print("-" * 70)
    print(_summary_line(results))
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the Writing Agent with an LLM judge")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    parser.add_argument("--id", type=str, default=None, help="Run a single example by id")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="Dataset YAML path")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override writer+judge model id (e.g. gemini-2.5-flash) for this run"
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: run one or all examples and write a report."""
    args = parse_args()
    if args.model:
        apply_model_override(args.model)
        print(f"[override] writer+judge model -> {args.model}")
    judge_model = build_judge_model()
    dataset = Path(args.dataset)
    if args.id:
        results = [evaluate_example(load_example(args.id, dataset), judge_model)]
    else:
        results = run_parallel(load_examples(dataset), judge_model)
    print_summary(results)
    report = write_markdown_report(results)
    print(f"\nReport written to: {report}")


if __name__ == "__main__":
    main()
