"""Deterministic eval for the pre-execution cost estimate (no LLM, no network, no quota).

Drives each updated agent's REAL prompt builder, counts the assembled prompt with
``general_utils/tokens.py::count_prompt_tokens``, and recomputes ``est_cost_usd`` exactly as the
agent does at its call site. It then scores three deterministic metrics that together prove the
estimate now tracks prompt size rather than a flat per-agent constant:

    M1 populated  — every estimate is > 0
    M2 scales     — within an agent, the large-prompt estimate exceeds the small-prompt one
    M3 sensitive  — doubling ``chars_per_token`` lowers the token estimate (config is honored)

Examples:
    uv run python evals/judges/cost_estimation_eval.py --all
    uv run python evals/judges/cost_estimation_eval.py --id COD-L
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from app.src.general_utils.cost import estimate_cost  # noqa: E402
from app.src.general_utils.tokens import count_prompt_tokens  # noqa: E402
from app.src.schemas import get_config  # noqa: E402
from app.src.sub_agents.analysis import prompts as analysis_prompts  # noqa: E402
from app.src.sub_agents.analysis.agent import _data_preview  # noqa: E402
from app.src.sub_agents.code import prompts as code_prompts  # noqa: E402
from app.src.sub_agents.code.schemas import CodeInput, coerce_action  # noqa: E402
from app.src.sub_agents.code.validation import has_validator  # noqa: E402
from app.src.sub_agents.research import prompts as research_prompts  # noqa: E402

DATASET_PATH = APP_DIR / "evals" / "datasets" / "cost_estimation.yaml"
REPORTS_DIR = APP_DIR / "evals" / "reports"


def load_examples(path: Path) -> list[dict]:
    """Load all examples from the dataset YAML."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle).get("examples", [])


def _assemble(agent: str, payload: dict, chars_per_token: float) -> tuple[list[dict], int]:
    """Build the agent's real prompt and return it with its estimated token count."""
    if agent == "research":
        messages = research_prompts.initial_messages(payload["subtopic"])
    elif agent == "analysis":
        preview = _data_preview(payload.get("data"))
        messages = analysis_prompts.initial_messages(
            payload["instruction"], payload["action"], preview
        )
    else:
        messages = code_prompts.build_messages(_code_input(payload))
    return messages, count_prompt_tokens(messages, chars_per_token)


def _code_input(payload: dict) -> CodeInput:
    """Construct a validated CodeInput from a code scenario payload."""
    return CodeInput(
        action=coerce_action(payload["action"]),
        input=payload["input"],
        language=payload["language"],
        context=payload.get("context", ""),
    )


def _estimate_code(payload: dict, input_tokens: int, cfg) -> float:
    """Mirror the Code Agent call site: one generator call plus bounded reviewer calls."""
    code_cfg = cfg.code_agent
    parser = has_validator(payload["language"])
    rev_calls = code_cfg.max_syntax_retries if parser else 2 * code_cfg.max_review_retries + 1
    generator = estimate_cost(cfg.pricing[code_cfg.model_id], 1, input_tokens, code_cfg.avg_output_tokens)
    reviewer = estimate_cost(
        cfg.pricing[code_cfg.review_model_id], rev_calls, input_tokens, code_cfg.avg_output_tokens
    )
    return generator + reviewer


def _estimate(agent: str, payload: dict, input_tokens: int, cfg) -> tuple[float, str]:
    """Recompute est_cost_usd exactly as the named agent does, returning (cost, price_id)."""
    if agent == "research":
        rc = cfg.research_agent
        cost = estimate_cost(cfg.pricing[rc.model_id], rc.max_search_calls + 2, input_tokens, rc.avg_output_tokens)
        return cost, rc.model_id
    if agent == "analysis":
        ac = cfg.analysis_agent
        cost = estimate_cost(cfg.pricing[ac.model_id], ac.max_compute_calls + 2, input_tokens, ac.avg_output_tokens)
        return cost, ac.model_id
    return _estimate_code(payload, input_tokens, cfg), cfg.code_agent.model_id


def evaluate_example(example: dict, cfg) -> dict:
    """Estimate cost for one scenario, recording prompt tokens and the config-sensitivity probe."""
    agent, payload = example["agent"], example["payload"]
    base_cpt = cfg.estimation.chars_per_token
    _, input_tokens = _assemble(agent, payload, base_cpt)
    _, coarse_tokens = _assemble(agent, payload, base_cpt * 2)
    cost, price_id = _estimate(agent, payload, input_tokens, cfg)
    record = {
        "id": example.get("id", "?"),
        "agent": agent,
        "size": example.get("size", "?"),
        "input_tokens": input_tokens,
        "coarse_tokens": coarse_tokens,
        "est_cost_usd": round(cost, 6),
        "price_id": price_id,
    }
    print(f"  {record['id']:<7} {agent:<9} {record['size']:<6} "
          f"in_tok={input_tokens:<5} est=${record['est_cost_usd']}")
    return record


def _metric_populated(records: list[dict]) -> tuple[bool, str]:
    """M1: every estimate is strictly positive."""
    bad = [r["id"] for r in records if r["est_cost_usd"] <= 0]
    return (not bad, "all estimates > 0" if not bad else f"non-positive: {bad}")


def _metric_scales(records: list[dict]) -> tuple[bool, str]:
    """M2: per agent, the large-prompt estimate and token count exceed the small one."""
    failures = []
    for agent in sorted({r["agent"] for r in records}):
        sized = {r["size"]: r for r in records if r["agent"] == agent}
        small, large = sized.get("small"), sized.get("large")
        if not (small and large):
            continue
        if not (large["input_tokens"] > small["input_tokens"] and large["est_cost_usd"] > small["est_cost_usd"]):
            failures.append(agent)
    return (not failures, "large > small for every agent" if not failures else f"no scaling: {failures}")


def _metric_sensitive(records: list[dict]) -> tuple[bool, str]:
    """M3: a coarser chars_per_token yields fewer estimated tokens (config is honored)."""
    bad = [r["id"] for r in records if not r["coarse_tokens"] < r["input_tokens"]]
    return (not bad, "coarser ratio lowers token count" if not bad else f"insensitive: {bad}")


def score_metrics(records: list[dict]) -> list[dict]:
    """Run all deterministic metrics over the records."""
    checks = [("M1 populated", _metric_populated), ("M2 scales", _metric_scales),
              ("M3 sensitive", _metric_sensitive)]
    results = []
    for name, func in checks:
        passed, detail = func(records)
        results.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})
    return results


def _summary_line(metrics: list[dict]) -> str:
    """Build the totals/pass-rate summary line over the metrics."""
    total = len(metrics)
    passed = sum(1 for m in metrics if m["status"] == "PASS")
    rate = f"{passed / total * 100:.0f}%" if total else "0%"
    return f"Metrics: {total} | Passed: {passed} | Failed: {total - passed} | Pass rate: {rate}"


def write_markdown_report(records: list[dict], metrics: list[dict]) -> Path:
    """Write a timestamped markdown report and return its path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = REPORTS_DIR / f"cost_estimation_eval_{timestamp}.md"
    lines = [f"# Cost-Estimation Eval — {timestamp}", "", f"**{_summary_line(metrics)}**", ""]
    lines += _metrics_table(metrics) + [""] + _records_table(records)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _metrics_table(metrics: list[dict]) -> list[str]:
    """Render the metric verdicts as a markdown table."""
    rows = ["## Metrics", "", "| Metric | Status | Detail |", "| --- | --- | --- |"]
    rows += [f"| {m['name']} | {m['status']} | {m['detail']} |" for m in metrics]
    return rows


def _records_table(records: list[dict]) -> list[str]:
    """Render the per-scenario estimates as a markdown table."""
    rows = ["## Per-scenario estimates", "",
            "| id | agent | size | input_tokens | coarse_tokens | est_cost_usd | price |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    rows += [f"| {r['id']} | {r['agent']} | {r['size']} | {r['input_tokens']} | "
             f"{r['coarse_tokens']} | {r['est_cost_usd']} | {r['price_id']} |" for r in records]
    return rows


def print_summary(metrics: list[dict]) -> None:
    """Print a console summary of the metric verdicts."""
    print("\n" + "=" * 70)
    print("COST-ESTIMATION EVAL SUMMARY")
    print("=" * 70)
    for m in metrics:
        print(f"  [{m['status']}] {m['name']}: {m['detail']}")
    print("-" * 70)
    print(_summary_line(metrics))
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Deterministic eval for the cost estimate")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    parser.add_argument("--id", type=str, default=None, help="Run a single example by id")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="Dataset YAML path")
    return parser.parse_args()


def _select(examples: list[dict], example_id: str | None) -> list[dict]:
    """Filter examples to a single id when requested."""
    if not example_id:
        return examples
    chosen = [ex for ex in examples if ex.get("id") == example_id]
    if not chosen:
        raise ValueError(f"No example with id '{example_id}'")
    return chosen


def main() -> None:
    """Entry point: estimate every scenario, score metrics, write a report."""
    args = parse_args()
    cfg = get_config()
    examples = _select(load_examples(Path(args.dataset)), args.id)
    print(f"Estimating {len(examples)} scenario(s) at chars_per_token={cfg.estimation.chars_per_token}:")
    records = [evaluate_example(ex, cfg) for ex in examples]
    metrics = score_metrics(records)
    print_summary(metrics)
    report = write_markdown_report(records, metrics)
    print(f"\nReport written to: {report}")


if __name__ == "__main__":
    main()
