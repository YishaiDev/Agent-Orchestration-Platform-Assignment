"""End-to-end orchestration eval (offline, deterministic, no network/quota).

Drives the full LangGraph outer loop (plan -> execute -> evaluate -> synthesize -> judge) via
``run_task`` with scripted models + a timing-aware fake step runner, one scenario per documented
DECISIONS.md claim. Orchestration correctness is deterministic, so the "judge" here is a set of
assertions over the terminal state + monitor (not an LLM) — the right tool for control-flow, and it
mirrors the platform's own deterministic-first philosophy. Writes a markdown report.

Examples:
    uv run python evals/judges/orchestration_eval.py --all
    uv run python evals/judges/orchestration_eval.py --id O-02
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from app.src.engine.nodes import EngineDeps  # noqa: E402
from app.src.engine.runs import RunRegistry  # noqa: E402
from app.src.engine.synthesizer import Synthesis  # noqa: E402
from app.src.general_utils.agent_base import AgentResult  # noqa: E402
from app.src.schemas.plan import (  # noqa: E402
    ExecutionStep,
    PlannerDraft,
    ReplanDecision,
    SynthesisVerdict,
)

DATASET_PATH = APP_DIR / "evals" / "datasets" / "orchestration.yaml"
REPORTS_DIR = APP_DIR / "evals" / "reports"
_STEP_DELAY = 0.15
_PARALLEL_BUDGET_MS = 300


class _Raw:
    """Minimal model message carrying token usage metadata."""

    def __init__(self) -> None:
        self.usage_metadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


class _Runnable:
    """Structured-output runnable: pops the next scripted output for the requested schema."""

    def __init__(self, model: ScriptedModel, schema_name: str) -> None:
        self._model = model
        self._name = schema_name

    def invoke(self, messages: object) -> dict:
        queue = self._model.by_schema[self._name]
        out = queue.pop(0) if len(queue) > 1 else queue[0]
        return {"parsed": out, "raw": _Raw()}


class ScriptedModel:
    """Fake model returning scripted outputs keyed by the requested schema name."""

    def __init__(self, by_schema: dict[str, list]) -> None:
        self.by_schema = {name: list(items) for name, items in by_schema.items()}

    def with_structured_output(self, schema: type, include_raw: bool = False) -> _Runnable:
        return _Runnable(self, schema.__name__)


class TimingRunner:
    """Fake step runner: records call order + overlap, optionally sleeps, fails a chosen set."""

    def __init__(self, fails: set[str] | None = None, delay: float = 0.0) -> None:
        self.fails = fails or set()
        self.delay = delay
        self.ran: list[str] = []
        self.max_concurrent = 0
        self._active = 0

    async def __call__(self, step: ExecutionStep, results: dict, session: str) -> AgentResult:
        self.ran.append(step.id)
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        if self.delay:
            await asyncio.sleep(self.delay)
        self._active -= 1
        failed = step.id in self.fails
        out = {"error": "boom"} if failed else {"content": f"out-{step.id}", "confidence": 0.9}
        return AgentResult(
            step_id=step.id, agent=step.agent, status="failed" if failed else "completed",
            output=out,
            tokens_used=1, execution_time_ms=int(self.delay * 1000),
        )


def _step(sid: str, agent: str, action: str, deps: list[str] | None = None) -> ExecutionStep:
    """Build one execution step."""
    return ExecutionStep(id=sid, agent=agent, action=action, dependencies=deps or [])


def _accept() -> SynthesisVerdict:
    """A synthesis verdict that accepts the draft as-is."""
    return SynthesisVerdict(reasoning="grounded and complete", verdict="accept")


def _deps(registry: RunRegistry, runner: TimingRunner, model: ScriptedModel) -> EngineDeps:
    """Wire one scripted model into all four engine roles plus the fake runner."""
    return EngineDeps(
        registry=registry, runner=runner, planner_model=model, decider_model=model,
        synth_model=model, judge_model=model, concurrency=3,
    )


def _run(task_id: str, runner: TimingRunner, model: ScriptedModel, **kw: object) -> dict:
    """Run one task end-to-end through the outer loop, returning the terminal state."""
    from app.src.engine.graph import run_task

    registry = RunRegistry()
    deps = _deps(registry, runner, model)
    state = asyncio.run(run_task(task_id, "study X", "", "sess", deps=deps, **kw))
    return {"state": state, "monitor": registry.get(task_id)}


def _observe(out: dict, runner: TimingRunner, elapsed_ms: int) -> dict:
    """Collect the observability signals surfaced in the report."""
    state, monitor = out["state"], out["monitor"]
    final = state.get("final_result") or {}
    return {
        "status": final.get("status", "?"),
        "completed": monitor.completed_count(),
        "total": len(monitor.plan.steps) if monitor.plan else 0,
        "failed_steps": final.get("failed_steps", []),
        "skipped_steps": final.get("skipped_steps", []),
        "replans": state.get("replans", 0),
        "resynth_rounds": state.get("resynth_rounds", 0),
        "tokens": monitor.total_tokens,
        "trace_len": len(monitor.trace),
        "ran": runner.ran,
        "max_concurrent": runner.max_concurrent,
        "elapsed_ms": elapsed_ms,
        "final_output": state.get("final_output", ""),
    }


def scenario_o01() -> tuple[dict, list]:
    """Sequential dependency chain: the dependent never starts before its upstream completes."""
    model = ScriptedModel({
        "PlannerDraft": [PlannerDraft(reasoning="r", steps=[
            _step("s1", "research", "research"),
            _step("s2", "analysis", "analyze", ["s1"])])],
        "Synthesis": [Synthesis(content="answer", confidence=0.9)],
        "SynthesisVerdict": [_accept()]})
    runner = TimingRunner()
    obs = _timed("o01", runner, model)
    checks = [("status == completed", obs["status"] == "completed"),
              ("both completed (2/2)", obs["completed"] == 2),
              ("s1 ran before s2", obs["ran"].index("s1") < obs["ran"].index("s2"))]
    return obs, checks


def scenario_o02() -> tuple[dict, list]:
    """Parallel execution is genuinely concurrent: 3 independent 150ms steps beat the serial sum."""
    model = ScriptedModel({
        "PlannerDraft": [PlannerDraft(reasoning="r", steps=[
            _step("a", "research", "research"), _step("b", "research", "research"),
            _step("c", "research", "research")])],
        "Synthesis": [Synthesis(content="answer", confidence=0.9)],
        "SynthesisVerdict": [_accept()]})
    runner = TimingRunner(delay=_STEP_DELAY)
    obs = _timed("o02", runner, model)
    checks = [("status == completed", obs["status"] == "completed"),
              ("all three completed (3/3)", obs["completed"] == 3),
              (f"wall-clock < {_PARALLEL_BUDGET_MS}ms", obs["elapsed_ms"] < _PARALLEL_BUDGET_MS),
              ("all 3 ran concurrently", obs["max_concurrent"] == 3)]
    return obs, checks


def scenario_o03() -> tuple[dict, list]:
    """Skip-and-continue: a fails, its dependent a2 is skipped, independent b survives (partial)."""
    model = ScriptedModel({
        "PlannerDraft": [PlannerDraft(reasoning="r", steps=[
            _step("a", "research", "research"), _step("a2", "analysis", "analyze", ["a"]),
            _step("b", "research", "research")])],
        "Synthesis": [Synthesis(content="partial answer", confidence=0.7)],
        "SynthesisVerdict": [_accept()]})
    runner = TimingRunner(fails={"a"})
    obs = _timed("o03", runner, model)
    checks = [("status == completed (partial)", obs["status"] == "completed"),
              ("a in failed_steps", "a" in obs["failed_steps"]),
              ("a2 in skipped_steps", "a2" in obs["skipped_steps"]),
              ("b completed", "b" in obs["ran"] and obs["completed"] == 1)]
    return obs, checks


def scenario_o04() -> tuple[dict, list]:
    """Structural failure triggers a bounded re-plan and recovers via a namespaced fresh step."""
    model = ScriptedModel({
        "PlannerDraft": [PlannerDraft(reasoning="r", steps=[_step("s1", "research", "research")])],
        "ReplanDecision": [ReplanDecision(reasoning="retry fresh", decision="replan",
                                          new_steps=[_step("a", "research", "research")])],
        "Synthesis": [Synthesis(content="recovered", confidence=0.8)],
        "SynthesisVerdict": [_accept()]})
    runner = TimingRunner(fails={"s1"})
    obs = _timed("o04", runner, model, max_replans=1)
    checks = [("replans == 1", obs["replans"] == 1),
              ("status == completed", obs["status"] == "completed"),
              ("namespaced recovery step ran", any(r.startswith("r1_") for r in obs["ran"]))]
    return obs, checks


def scenario_o05() -> tuple[dict, list]:
    """Bounded re-plan budget exhausted (max_replans=0): the task fails cleanly with s1 recorded."""
    model = ScriptedModel({
        "PlannerDraft": [PlannerDraft(reasoning="r", steps=[_step("s1", "research", "research")])],
        "Synthesis": [Synthesis(content="", confidence=0.0)], "SynthesisVerdict": [_accept()]})
    runner = TimingRunner(fails={"s1"})
    obs = _timed("o05", runner, model, max_replans=0)
    checks = [("replans == 0", obs["replans"] == 0),
              ("status == failed", obs["status"] == "failed"),
              ("s1 in failed_steps", "s1" in obs["failed_steps"])]
    return obs, checks


def scenario_o06() -> tuple[dict, list]:
    """Post-synthesis judge re-synthesizes once (feedback) then accepts; the second draft ships."""
    model = ScriptedModel({
        "PlannerDraft": [PlannerDraft(reasoning="r", steps=[
            _step("s1", "research", "research"), _step("s2", "analysis", "analyze", ["s1"])])],
        "Synthesis": [Synthesis(content="draft one", confidence=0.5),
                      Synthesis(content="draft two", confidence=0.9)],
        "SynthesisVerdict": [SynthesisVerdict(reasoning="weak", verdict="resynthesize",
                                              feedback="tighten it"), _accept()]})
    runner = TimingRunner()
    obs = _timed("o06", runner, model, max_resynth=2)
    checks = [("resynth_rounds == 1", obs["resynth_rounds"] == 1),
              ("status == completed", obs["status"] == "completed"),
              ("final output is the second draft", obs["final_output"] == "draft two")]
    return obs, checks


def _timed(task_id: str, runner: TimingRunner, model: ScriptedModel, **kw: object) -> dict:
    """Run a scenario and capture wall-clock elapsed (for the concurrency check)."""
    started = time.perf_counter()
    out = _run(task_id, runner, model, **kw)
    return _observe(out, runner, int((time.perf_counter() - started) * 1000))


SCENARIOS: dict[str, Callable[[], tuple[dict, list]]] = {
    "O-01": scenario_o01, "O-02": scenario_o02, "O-03": scenario_o03,
    "O-04": scenario_o04, "O-05": scenario_o05, "O-06": scenario_o06,
}


def load_cases(path: Path) -> list[dict]:
    """Load the descriptive scenario registry from YAML."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def evaluate_case(case: dict) -> dict:
    """Run one scenario and grade its deterministic assertions."""
    case_id = case["id"]
    obs, checks = SCENARIOS[case_id]()
    passed = all(ok for _, ok in checks)
    print(f"[{'PASS' if passed else 'FAIL'}] {case_id} — {case['title']}")
    return {"case": case, "obs": obs, "checks": checks, "passed": passed}


def _check_lines(row: dict) -> str:
    """Render the per-assertion check list for one scenario."""
    return "\n".join(f"  - {'✅' if ok else '❌'} {label}" for label, ok in row["checks"])


def _obs_line(obs: dict) -> str:
    """Render the observability signal line for one scenario."""
    return (
        f"- signals: status=`{obs['status']}` | steps {obs['completed']}/{obs['total']} "
        f"| failed={obs['failed_steps']} | skipped={obs['skipped_steps']} "
        f"| replans={obs['replans']} | resynth={obs['resynth_rounds']} "
        f"| max_concurrent={obs['max_concurrent']} | tokens={obs['tokens']} "
        f"| trace={obs['trace_len']} entries | wall={obs['elapsed_ms']}ms"
    )


def _row_md(row: dict) -> str:
    """Render one scenario's full report block."""
    case, verdict = row["case"], "PASS" if row["passed"] else "FAIL"
    return (
        f"## [{verdict}] {case['id']} — {case['title']}\n"
        f"- claim: {case['claim']}\n"
        f"- scenario: {case['scenario']}\n"
        f"{_obs_line(row['obs'])}\n"
        f"- checks:\n{_check_lines(row)}\n"
    )


def _summary_line(rows: list[dict]) -> str:
    """Build the totals/pass-rate header."""
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    rate = round(100 * passed / total) if total else 0
    return f"**Total: {total} | Passed: {passed} | Failed: {total - passed} | Pass rate: {rate}%**"


def write_report(rows: list[dict], stamp: str) -> Path:
    """Write the markdown eval report and return its path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"orchestration_eval_{stamp}.md"
    body = "\n".join(_row_md(r) for r in rows)
    header = f"# Orchestration Eval (end-to-end, offline) — {stamp}\n\n{_summary_line(rows)}\n\n"
    path.write_text(header + body, encoding="utf-8")
    return path


def _select(cases: list[dict], only: str | None) -> list[dict]:
    """Filter cases to a single id when requested."""
    if only is None:
        return cases
    return [c for c in cases if c["id"] == only]


def main() -> int:
    """Run the orchestration eval and write a report."""
    parser = argparse.ArgumentParser(description="End-to-end orchestration eval.")
    parser.add_argument("--all", action="store_true", help="run every scenario")
    parser.add_argument("--id", help="run a single scenario by id, e.g. O-02")
    args = parser.parse_args()
    cases = _select(load_cases(DATASET_PATH), args.id)
    if not cases:
        print(f"No scenario matched id={args.id!r}")
        return 1
    rows = [evaluate_case(case) for case in cases]
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = write_report(rows, stamp)
    print(f"\n{_summary_line(rows)}\nReport: {path}")
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
