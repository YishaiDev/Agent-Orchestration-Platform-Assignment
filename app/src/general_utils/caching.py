"""Caching: a LangChain-native LLM response cache plus a small async TTL cache for tool results."""

from __future__ import annotations

import logging
import time
from collections.abc import Hashable
from typing import Any

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

logger = logging.getLogger(__name__)


def setup_llm_cache(kind: str) -> None:
    """Install a process-wide LangChain LLM response cache (exact-match).

    Args:
        kind: ``memory`` for an in-process cache, ``sqlite`` for an on-disk cache. Unknown
            values disable caching.
    """
    if kind == "memory":
        set_llm_cache(InMemoryCache())
    elif kind == "sqlite":
        from langchain_community.cache import SQLiteCache

        set_llm_cache(SQLiteCache(database_path="llm_cache.db"))
    else:
        logger.warning("Unknown llm_cache kind %r; LLM response cache disabled", kind)


class AsyncTTLCache:
    """A minimal time-to-live cache for awaited tool results (e.g. search responses)."""

    def __init__(self, ttl_seconds: int) -> None:
        """Initialize the cache.

        Args:
            ttl_seconds: Entry lifetime in seconds.
        """
        self._ttl = ttl_seconds
        self._store: dict[Hashable, tuple[float, Any]] = {}

    def get(self, key: Hashable) -> Any | None:
        """Return a live cached value for ``key`` or None when missing/expired."""
        item = self._store.get(key)
        if item is None:
            return None
        stored_at, value = item
        if time.monotonic() - stored_at > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: Hashable, value: Any) -> None:
        """Store ``value`` under ``key`` with the current timestamp."""
        self._store[key] = (time.monotonic(), value)
