# Tests

**127 tests, fully offline** — scripted fake models and fake runners, no network or provider quota.
Each file maps to one zone of the app and is independently runnable; `run_all_tests.py` runs them all.

| File | Zone |
|------|------|
| `test_planning.py` | planner, registry allowlist, DAG validation, plan/run-state schemas |
| `test_execution.py` | async scheduler, dispatch routing + context injection, token estimator |
| `test_recovery.py` | monitor failure classification, re-plan decider + merge |
| `test_synthesis.py` | synthesizer (provenance/totals) + quality judge |
| `test_graph.py` | outer loop `plan → execute → evaluate → synthesize → judge` end to end |
| `test_api.py` | HTTP endpoints + response spec-shape conformance |
| `test_research_agent.py` / `test_analysis_agent.py` | ReAct sub-agents |
| `test_code_agent.py` / `test_writing_agent.py` | reflection-loop sub-agents |

Each file inlines only the fakes it needs (no `conftest.py`) and puts the repo root on `sys.path`.

```powershell
cd app
uv run python ../tests/run_all_tests.py     # full suite
uv run pytest ../tests/test_planning.py -q    # one zone
uv run python ../tests/test_planning.py       # one file, standalone
```
