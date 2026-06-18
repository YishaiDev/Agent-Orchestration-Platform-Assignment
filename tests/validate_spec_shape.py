"""Ad-hoc end-to-end spec-conformance check (scripted models, no network).

Drives the real FastAPI app through submit -> poll -> result -> cancel -> agents and asserts each
response matches the assignment's required input/output shapes. Run: ``uv run python
tests/validate_spec_shape.py`` from the ``app`` directory (or project root with PYTHONPATH set).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "engine"))

from fastapi.testclient import TestClient  # noqa: E402
from test_api import _poll_until_terminal, _scripted_app  # noqa: E402

_ok = True


def check(name: str, cond: object) -> None:
    global _ok
    _ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def main() -> int:
    with TestClient(_scripted_app()) as c:
        created = c.post(
            "/tasks",
            json={
                "goal": "Compare X and Y and write a brief",
                "constraints": {"max_words": 800, "tone": "professional"},
                "output_format": "markdown",
            },
        )
        print("== POST /tasks ==")
        check("HTTP 202", created.status_code == 202)
        cj = created.json()
        check("task_id (str)", isinstance(cj.get("task_id"), str))
        check("status == 'planning'", cj.get("status") == "planning")
        tid = cj["task_id"]

        st = _poll_until_terminal(c, tid)
        print("== GET /tasks/{id} ==")
        check("task_id matches", st.get("task_id") == tid)
        check("status terminal=completed", st.get("status") == "completed")
        check("progress.total_steps int", isinstance(st["progress"]["total_steps"], int))
        check("progress.completed_steps int", isinstance(st["progress"]["completed_steps"], int))
        check("progress.current_step key", "current_step" in st["progress"])
        check("execution_trace list", isinstance(st.get("execution_trace"), list))

        rj = c.get(f"/tasks/{tid}/result").json()
        print("== GET /tasks/{id}/result (Final Result Format) ==")
        for key, typ in [
            ("task_id", str),
            ("status", str),
            ("result", dict),
            ("execution_trace", list),
            ("total_tokens", int),
            ("total_time_ms", int),
        ]:
            check(f"{key} ({typ.__name__})", isinstance(rj.get(key), typ))
        res = rj.get("result", {})
        for key, typ in [("content", str), ("format", str), ("word_count", int)]:
            check(f"result.{key} ({typ.__name__})", isinstance(res.get(key), typ))
        check("result.format echoes output_format", res.get("format") == "markdown")

        print("== execution_trace[] vs Agent Output Format ==")
        req = ["step_id", "agent", "status", "output", "tokens_used", "execution_time_ms"]
        for e in rj["execution_trace"]:
            miss = [k for k in req if k not in e]
            check(f"{e.get('step_id')}/{e.get('agent')}: {req}", not miss)
            check(f"{e.get('step_id')}: output.content", "content" in e.get("output", {}))
            check(
                f"{e.get('step_id')}: started_at+completed_at",
                e.get("started_at") and e.get("completed_at"),
            )

        pj = c.get(f"/tasks/{tid}/plan").json()
        print("== GET /tasks/{id}/plan (Execution Plan Format) ==")
        check("task_id matches", pj.get("task_id") == tid)
        check("steps list", isinstance(pj.get("steps"), list))
        check("parallel_groups list", isinstance(pj.get("parallel_groups"), list))
        for s in pj.get("steps", []):
            miss = [k for k in ["id", "agent", "action", "dependencies"] if k not in s]
            check(f"{s.get('id')}: id/agent/action/dependencies", not miss)

        cz = c.post(f"/tasks/{tid}/cancel").json()
        print("== POST /tasks/{id}/cancel ==")
        check("status == 'cancelled'", cz.get("status") == "cancelled")
        check("completed_steps list", isinstance(cz.get("completed_steps"), list))

        ag = c.get("/agents").json()
        print("== GET /agents ==")
        check("spec shape {'agents': [...]}", isinstance(ag, dict) and "agents" in ag)
        rows = ag.get("agents") if isinstance(ag, dict) else ag
        names = {a["name"] for a in rows}
        check(">= 4 agents", len(names) >= 4)
        check(
            "research/analysis/code/writing present",
            {"research", "analysis", "code", "writing"}.issubset(names),
        )
        for a in rows:
            check(
                f"{a.get('name')}: name/description/capabilities/status",
                all(k in a for k in ["name", "description", "capabilities", "status"]),
            )

    print("\n=== OVERALL:", "ALL CONFORM" if _ok else "DEVIATIONS FOUND", "===")
    return 0 if _ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
