"""The ``web_search`` tool: ToolRuntime-injected, config-bounded, source-accumulating search.

The Tavily-backed ``searcher`` and all counters live in the injected ``ResearchContext`` and are
never exposed in the model-visible tool schema.
"""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlparse

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from app.src.services.tavily_client import SearchHit
from app.src.sub_agents.research.schemas import ResearchContext


def _host(url: str) -> str:
    """Reduce a URL to its bare host (dropping a leading ``www.``)."""
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def _record_sources(ctx: ResearchContext, hits: Sequence[SearchHit]) -> None:
    """Append new URLs and de-duplicated hosts from ``hits`` into the runtime context."""
    for hit in hits:
        if not hit.url or hit.url in ctx.collected_urls:
            continue
        ctx.collected_urls.append(hit.url)
        host = _host(hit.url)
        if host and host not in ctx.collected_sources:
            ctx.collected_sources.append(host)


def _format_hits(hits: Sequence[SearchHit]) -> str:
    """Render hits as numbered snippets for the model, or a no-results notice."""
    if not hits:
        return "No results found for that query."
    blocks = [f"[{i}] {hit.title} ({hit.url})\n{hit.content}" for i, hit in enumerate(hits, 1)]
    return "\n\n".join(blocks)


@tool
async def web_search(query: str, runtime: ToolRuntime[ResearchContext]) -> str:
    """Search the web for information and return ranked result snippets with their sources.

    Args:
        query: A focused search query for one facet of the research subtopic.

    Returns:
        Formatted result snippets, or a notice to stop and summarize when the budget is reached.
    """
    ctx = runtime.context
    if ctx.search_count >= ctx.max_search_calls:
        return "Search budget reached. Do not search again; summarize your findings now."
    ctx.search_count += 1
    hits = list(await ctx.searcher(query, ctx.search_top_k))
    _record_sources(ctx, hits)
    return _format_hits(hits)
