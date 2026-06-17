"""Offline tests for the Research Agent (mocked Gemini + Tavily, no network).

Covers the autonomous search loop, both iteration budgets, grounded citations, history
compaction, the Tavily TTL cache, tool-schema secrecy, cost accounting, genuine async
concurrency, and the spec Agent Output Format.

Run standalone: ``python tests/sub_agents/test_research_agent.py`` or via pytest.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from app.src.general_utils.cost import token_cost  # noqa: E402
from app.src.schemas.config import ModelPrice, ResearchAgentConfig, get_config  # noqa: E402
from app.src.services.tavily_client import SearchHit, TavilySearch  # noqa: E402
from app.src.sub_agents.research.agent import (  # noqa: E402
    build_research_agent,
    run_research_agent,
)
from app.src.sub_agents.research.schemas import ResearchContext, ResearchSummary  # noqa: E402
from app.src.sub_agents.research.tools import web_search  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402
from pydantic import Field  # noqa: E402

_MAIN_PRICE = ModelPrice(input=0.30, output=2.50)
_HITS = [
    SearchHit("Python", "https://www.python.org/about", "Python is a language."),
    SearchHit("Real Python", "https://realpython.com/intro", "Tutorials for Python."),
]


def _usage(in_tok: int, out_tok: int) -> dict:
    return {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok}


def _tool_msg(call_id: str, query: str = "q", in_tok: int = 200, out_tok: int = 50) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "args": {"query": query}, "id": call_id}],
        usage_metadata=_usage(in_tok, out_tok),
    )


def _final_msg(text: str = "Done.", in_tok: int = 200, out_tok: int = 50) -> AIMessage:
    return AIMessage(content=text, usage_metadata=_usage(in_tok, out_tok))


def _tool_turns(count: int) -> list[AIMessage]:
    return [_tool_msg(f"call_{i}", f"query_{i}") for i in range(count)]


class ScriptedModel(FakeMessagesListChatModel):
    """Main model double: cycles a scripted response list; tolerates tool binding."""

    def bind_tools(self, tools: object, **kwargs: object) -> ScriptedModel:
        return self


class _StructuredFake:
    """Stand-in for ``model.with_structured_output(...)`` returning a fixed summary + usage."""

    def __init__(self, content: str, confidence: float, in_tok: int, out_tok: int) -> None:
        self._summary = ResearchSummary(content=content, confidence=confidence)
        self._raw = AIMessage(content="", usage_metadata=_usage(in_tok, out_tok))

    async def ainvoke(self, _messages: object) -> dict:
        return {"parsed": self._summary, "raw": self._raw}


class FakeSummarizer(FakeMessagesListChatModel):
    """Summarizer double: real model for compaction, scripted structured final summary."""

    responses: list = Field(default_factory=lambda: [AIMessage(content="COMPACTED")])
    sink: dict = Field(default_factory=lambda: {"compactions": 0})
    summary_content: str = "Synthesized summary of findings."
    summary_confidence: float = 0.9
    summary_in: int = 100
    summary_out: int = 40

    def bind_tools(self, tools: object, **kwargs: object) -> FakeSummarizer:
        return self

    def with_structured_output(self, schema: object, **kwargs: object) -> _StructuredFake:
        return _StructuredFake(
            self.summary_content, self.summary_confidence, self.summary_in, self.summary_out
        )

    def invoke(self, model_input: object, config: object = None, **kwargs: object) -> AIMessage:
        self.sink["compactions"] += 1
        return super().invoke(model_input, config, **kwargs)

    async def ainvoke(
        self, model_input: object, config: object = None, **kwargs: object
    ) -> AIMessage:
        self.sink["compactions"] += 1
        return await super().ainvoke(model_input, config, **kwargs)


class CountingSearcher:
    """Async searcher double that records call count and returns canned hits."""

    def __init__(self, hits: list[SearchHit] | None = None, delay: float = 0.0) -> None:
        self._hits = _HITS if hits is None else hits
        self._delay = delay
        self.calls = 0

    async def __call__(self, query: str, top_k: int) -> list[SearchHit]:
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._hits[:top_k]


def _cfg(**overrides: object) -> ResearchAgentConfig:
    base = dict(
        model_id="gemini-3.5-flash",
        summarizer_model_id="gemini-2.5-flash",
        max_search_calls=5,
        recursion_limit=12,
        search_top_k=5,
        trigger_messages=16,
        keep_recent=6,
    )
    base.update(overrides)
    return ResearchAgentConfig(**base)


def _context(searcher: CountingSearcher, cfg: ResearchAgentConfig) -> ResearchContext:
    return ResearchContext(
        searcher=searcher,
        session_id="sess",
        step_id="step-1",
        max_search_calls=cfg.max_search_calls,
        search_top_k=cfg.search_top_k,
    )


async def _drive_agent(
    responses: list[AIMessage], cfg: ResearchAgentConfig, searcher: CountingSearcher
) -> tuple[dict, ResearchContext]:
    model = ScriptedModel(responses=responses)
    summarizer = FakeSummarizer()
    agent = build_research_agent(model, summarizer, cfg, _MAIN_PRICE)
    ctx = _context(searcher, cfg)
    final = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]}, context=ctx)
    return final, ctx


def test_web_search_stops_at_max_search_calls() -> None:
    cfg = _cfg(max_search_calls=2, recursion_limit=10)
    searcher = CountingSearcher()
    responses = _tool_turns(6) + [_final_msg()]
    _, ctx = asyncio.run(_drive_agent(responses, cfg, searcher))
    assert ctx.search_count == 2
    assert searcher.calls == 2


def test_recursion_limit_halts_no_progress_loop() -> None:
    cfg = _cfg(max_search_calls=10, recursion_limit=3)
    searcher = CountingSearcher()
    final, ctx = asyncio.run(_drive_agent(_tool_turns(20), cfg, searcher))
    assert ctx.search_count <= 3
    assert len(final["messages"]) < 25


def test_sources_are_grounded_hosts() -> None:
    cfg = _cfg()
    _, ctx = asyncio.run(_drive_agent([_tool_msg("c0"), _final_msg()], cfg, CountingSearcher()))
    assert ctx.collected_sources == ["python.org", "realpython.com"]
    assert ctx.collected_urls == [
        "https://www.python.org/about",
        "https://realpython.com/intro",
    ]


def test_compaction_fires_and_agent_still_returns() -> None:
    cfg = _cfg(trigger_messages=2, keep_recent=2, max_search_calls=5)
    model = ScriptedModel(responses=_tool_turns(3) + [_final_msg()])
    summarizer = FakeSummarizer()
    agent = build_research_agent(model, summarizer, cfg, _MAIN_PRICE)
    ctx = _context(CountingSearcher(), cfg)
    final = asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": "go"}]}, context=ctx)
    )
    assert summarizer.sink["compactions"] >= 1
    assert final["messages"]


def test_tavily_ttl_cache_skips_second_network_call() -> None:
    asyncio.run(_assert_tavily_cached())


async def _assert_tavily_cached() -> None:
    search = TavilySearch(api_key="unused", ttl_seconds=3600)
    counter = {"n": 0}

    async def fake_search(query: str, max_results: int, search_depth: str) -> dict:
        counter["n"] += 1
        return {"results": [{"title": "T", "url": "https://x.io/a", "content": "c"}]}

    search._client.search = fake_search  # type: ignore[method-assign]
    first = await search("same query", 3)
    second = await search("same query", 3)
    assert counter["n"] == 1
    assert first == second


def test_tavily_key_and_runtime_absent_from_tool_schema() -> None:
    arg_names = set(web_search.args.keys())
    assert "query" in arg_names
    assert "runtime" not in arg_names
    assert "api_key" not in arg_names
    assert "searcher" not in arg_names


def test_actual_cost_matches_price_table() -> None:
    result = asyncio.run(_run_full(step_id="step-cost"))
    main = get_config().pricing["gemini-3.5-flash"]
    expected = token_cost(main, 400, 100) + token_cost(main, 100, 40)
    assert result.actual_cost_usd == round(expected, 6)
    assert result.tokens_used == 500 + 140


async def _run_full(step_id: str = "step-1") -> object:
    return await run_research_agent(
        "the history of python",
        step_id=step_id,
        searcher=CountingSearcher(),
        model=ScriptedModel(responses=[_tool_msg("c0"), _final_msg()]),
        summarizer=FakeSummarizer(),
    )


def _new_full_call() -> object:
    return run_research_agent(
        "concurrent subtopic",
        step_id="step-c",
        searcher=CountingSearcher(delay=0.2),
        model=ScriptedModel(responses=[_tool_msg("c0"), _final_msg()]),
        summarizer=FakeSummarizer(),
    )


async def _gather_two() -> float:
    started = time.perf_counter()
    await asyncio.gather(_new_full_call(), _new_full_call())
    return time.perf_counter() - started


def test_two_research_calls_run_concurrently() -> None:
    elapsed = asyncio.run(_gather_two())
    assert elapsed < 0.35


def test_result_matches_spec_output_format() -> None:
    result = asyncio.run(_run_full(step_id="step-42"))
    payload = result.model_dump()
    assert set(payload) == {
        "step_id",
        "agent",
        "status",
        "output",
        "tokens_used",
        "execution_time_ms",
        "est_cost_usd",
        "actual_cost_usd",
    }
    assert payload["step_id"] == "step-42"
    assert payload["agent"] == "research"
    assert payload["status"] == "completed"
    assert set(payload["output"]) == {"content", "sources", "confidence"}
    assert isinstance(payload["output"]["confidence"], float)
    assert payload["output"]["sources"] == ["python.org", "realpython.com"]
    assert payload["est_cost_usd"] is not None


def _main() -> None:
    tests = [
        test_web_search_stops_at_max_search_calls,
        test_recursion_limit_halts_no_progress_loop,
        test_sources_are_grounded_hosts,
        test_compaction_fires_and_agent_still_returns,
        test_tavily_ttl_cache_skips_second_network_call,
        test_tavily_key_and_runtime_absent_from_tool_schema,
        test_actual_cost_matches_price_table,
        test_two_research_calls_run_concurrently,
        test_result_matches_spec_output_format,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _main()
