"""Async Tavily search client: egress-capped (top-K, search-only), TTL-cached, never raises.

The agent only ever performs bounded result-list searches here. There is no URL-fetch path, so a
malicious snippet cannot induce server-side request forgery through this boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from tavily import AsyncTavilyClient

from app.src.general_utils.caching import AsyncTTLCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchHit:
    """One search result: a title, its source URL, and a short content snippet."""

    title: str
    url: str
    content: str


def _to_hits(results: list[dict[str, Any]], top_k: int) -> list[SearchHit]:
    """Convert raw Tavily result dicts into capped, typed search hits."""
    hits = [
        SearchHit(
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            content=str(item.get("content", "")),
        )
        for item in results
    ]
    return hits[:top_k]


class TavilySearch:
    """Callable async search wrapper bounded by ``top_k`` and backed by a TTL cache."""

    def __init__(self, api_key: str, ttl_seconds: int) -> None:
        """Initialize the client.

        Args:
            api_key: Tavily API key (held only here, never exposed to the model or tools).
            ttl_seconds: Result cache lifetime.
        """
        self._client = AsyncTavilyClient(api_key=api_key)
        self._cache = AsyncTTLCache(ttl_seconds)

    async def __call__(self, query: str, top_k: int) -> list[SearchHit]:
        """Search the web for ``query``, returning at most ``top_k`` hits (cached, never raises)."""
        key = (query, top_k)
        cached = self._cache.get(key)
        if cached is not None:
            return cast(list[SearchHit], cached)
        hits = await self._run(query, top_k)
        self._cache.set(key, hits)
        return hits

    async def _run(self, query: str, top_k: int) -> list[SearchHit]:
        """Execute the bounded Tavily search, degrading to an empty list on any error."""
        try:
            response = await self._client.search(
                query=query, max_results=top_k, search_depth="basic"
            )
        except Exception as exc:
            logger.warning("Tavily search failed for %r: %s", query, exc)
            return []
        return _to_hits(response.get("results", []), top_k)
