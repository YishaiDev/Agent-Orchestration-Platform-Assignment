"""LLM-as-judge eval for the Analysis Agent.

Runs the real Analysis Agent (live Gemini, deterministic compute tool) on each dataset example, then
asks a Gemini judge to return PASS/FAIL + Reason + Suggested Fix against the example's expected
criteria and the agent's observability signals (status, findings, confidence, tokens, time, cost).
Writes a markdown report.

Examples:
    uv run python evals/judges/analysis_judge.py --all
    uv run python evals/judges/analysis_judge.py --id A-01
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
from app.src.sub_agents.analysis.agent import run_analysis_agent  # noqa: E402

DATASET_PATH = APP_DIR / "evals" / "datasets" / "analysis_agent.yaml"
REPORTS_DIR = APP_DIR / "evals" / "reports"
MAX_WORKERS = 1  # free-tier Gemini is 5 req/min; sequential + backoff stays under quota

JUDGE_PROMPT = """You are an evaluation judge for a data-Analysis Agent.

The agent reasons over a small structured dataset using a deterministic compute tool (exact
arithmetic and aggregation), then writes a grounded analysis with discrete findings and a calibrated
confidence score (0-1).

## Action requested: {action}   (analyze | compare | identify_patterns)

## Instruction the agent was given:
"{instruction}"

## The dataset the agent analyzed:
{data}

## Expected outcome (GROUND TRUTH criteria):
{expected_output}

## Agent run signals:
- status: {status}            (completed | completed_degraded | failed)
- findings: {findings}
- confidence: {confidence}     (agent's self-reported 0-1)
- tokens_used: {tokens_used}
- execution_time_ms: {execution_time_ms}
- est_cost_usd: {est_cost_usd} | actual_cost_usd: {actual_cost_usd}

## Agent's actual analysis content:
<output>
{content}
</output>

JUDGING RULES:
- PASS only if the analysis satisfies the expected outcome AND every numeric claim is consistent
  with the dataset (grounded, exact, not fabricated).
- For QUANTITATIVE cases: the specific expected numbers must appear and be correct. Wrong or
  approximated figures where an exact value was expected are a FAIL.
- For COMPARE: the recommendation and trade-offs must follow from the listed fields.
- For IDENTIFY_PATTERNS: named trends/outliers must be supported by the actual values.
- For the GROUNDING-HONESTY case: PASS requires the agent to state the answer cannot be derived from
  the data and to report LOW confidence. A confident fabricated figure is a FAIL (hallucination).
- status "failed" or empty content is always a FAIL.
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


def build_judge_model():
    """Build the judge model from config (deterministic)."""
    cfg = get_config()
    return build_chat_model(cfg.analysis_agent.model_id, 0.0)


def _as_text(content: object) -> str:
    """Coerce a model message content (str or list of parts) to plain text."""
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)


@retry(stop=stop_after_attempt(6), wait=wait_exponential_jitter(initial=4, max=70), reraise=True)
def _invoke_judge(model, prompt: str) -> str:
    """Invoke the judge model with bounded retry (honors free-tier 429 backoff)."""
    return _as_text(model.invoke(prompt).content)


def _run_agent(example: dict) -> AgentResult:
    """Run the async Analysis Agent to completion in a fresh event loop."""
    return asyncio.run(
        run_analysis_agent(
            example["instruction"],
            action=example.get("action", "analyze"),
            data=example.get("data"),
            step_id=example.get("id", "analysis"),
        )
    )


def judge(example: dict, result: AgentResult, model) -> str:
    """Ask the judge model to score one agent run."""
    findings = result.output.get("findings", []) or []
    prompt = JUDGE_PROMPT.format(
        action=example.get("action", "analyze"),
        instruction=example["instruction"].strip(),
        data=json.dumps(example.get("data"), indent=2, default=str),
        expected_output=example.get("expected_output", "N/A").strip(),
        status=result.status,
        findings="; ".join(str(f) for f in findings) if findings else "(none)",
        confidence=result.output.get("confidence", "N/A"),
        tokens_used=result.tokens_used,
        execution_time_ms=result.execution_time_ms,
        est_cost_usd=result.est_cost_usd,
        actual_cost_usd=result.actual_cost_usd,
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
    result = _run_agent(example)
    verdict = judge(example, result, judge_model)
    details = parse_verdict_details(verdict)
    findings = result.output.get("findings", []) or []
    print(f"[{details['status']}] {example_id} | {example['category']} "
          f"| findings={len(findings)} conf={result.output.get('confidence', 'NA')} "
          f"tokens={result.tokens_used}")
    return {
        "id": example_id,
        "category": example.get("category", "N/A"),
        "action": example.get("action", "analyze"),
        "status": result.status,
        "finding_count": len(findings),
        "findings": findings,
        "confidence": result.output.get("confidence", "N/A"),
        "tokens_used": result.tokens_used,
        "execution_time_ms": result.execution_time_ms,
        "est_cost_usd": result.est_cost_usd,
        "actual_cost_usd": result.actual_cost_usd,
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
    path = REPORTS_DIR / f"analysis_eval_{timestamp}.md"
    lines = [f"# Analysis Agent Eval — {timestamp}", "", f"**{_summary_line(results)}**", ""]
    for r in results:
        lines += _format_result_block(r)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _format_result_block(r: dict) -> list[str]:
    """Render one result as markdown lines."""
    preview = r["content"].replace("\n", " ")[:300]
    findings = "; ".join(str(f) for f in r["findings"]) if r["findings"] else "(none)"
    return [
        f"## [{r['verdict_status']}] {r['id']} — {r['category']} ({r['action']})",
        f"- run status: `{r['status']}` | findings: {r['finding_count']} "
        f"| confidence: {r['confidence']} | tokens: {r['tokens_used']} "
        f"| time: {r['execution_time_ms']}ms",
        f"- cost: est ${r['est_cost_usd']} | actual ${r['actual_cost_usd']}",
        f"- findings: {findings}",
        f"- reason: {r['reason']}",
        f"- suggested fix: {r['suggested_fix']}",
        f"- output preview: {preview}",
        "",
    ]


def print_summary(results: list[dict]) -> None:
    """Print a console summary table."""
    print("\n" + "=" * 70)
    print("ANALYSIS AGENT EVAL SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  [{r['verdict_status']}] {r['id']} ({r['category']}): {r['reason']}")
    print("-" * 70)
    print(_summary_line(results))
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the Analysis Agent with an LLM judge")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    parser.add_argument("--id", type=str, default=None, help="Run a single example by id")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="Dataset YAML path")
    return parser.parse_args()


def main() -> None:
    """Entry point: run one or all examples and write a report."""
    args = parse_args()
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
