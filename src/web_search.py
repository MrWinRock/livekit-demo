"""Web search providers for the agent.

Currently uses DuckDuckGo (no API key required). A Tavily provider is
scaffolded for future use — flip the WEB_SEARCH_PROVIDER env var to "tavily"
and install `tavily-python` to enable it.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    title: str
    snippet: str
    url: str

    def to_summary(self) -> str:
        return f"{self.title}: {self.snippet} ({self.url})"


class SearchProvider(Protocol):
    async def search(self, query: str, max_results: int = 3) -> list[SearchResult]: ...


class DuckDuckGoProvider:
    """Free web search via the `ddgs` library. No API key needed."""

    async def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
        from ddgs import DDGS

        def _sync_search() -> list[SearchResult]:
            with DDGS() as ddg:
                rows = ddg.text(query, max_results=max_results)
                return [
                    SearchResult(
                        title=row.get("title", ""),
                        snippet=row.get("body", ""),
                        url=row.get("href", ""),
                    )
                    for row in rows
                ]

        return await asyncio.to_thread(_sync_search)


class TavilyProvider:
    """LLM-optimized search via Tavily.

    Requires `tavily-python` to be installed and `TAVILY_API_KEY` env var set.
    Not active by default — enable with WEB_SEARCH_PROVIDER=tavily.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "TAVILY_API_KEY env var not set; cannot use Tavily provider"
            )

    async def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
        from tavily import TavilyClient

        def _sync_search() -> list[SearchResult]:
            client = TavilyClient(api_key=self.api_key)
            response = client.search(query=query, max_results=max_results)
            return [
                SearchResult(
                    title=row.get("title", ""),
                    snippet=row.get("content", ""),
                    url=row.get("url", ""),
                )
                for row in response.get("results", [])
            ]

        return await asyncio.to_thread(_sync_search)


class FallbackProvider:
    """Try `primary`; if it raises, fall back to `secondary`.

    Used to keep the agent working when a preferred provider (e.g. Tavily)
    has a runtime failure: missing package, expired key, network error, etc.
    """

    def __init__(self, primary: SearchProvider, secondary: SearchProvider) -> None:
        self.primary = primary
        self.secondary = secondary

    async def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
        try:
            return await self.primary.search(query, max_results=max_results)
        except Exception as exc:
            logger.warning(
                "Primary search provider failed (%s); falling back to secondary",
                exc,
            )
            return await self.secondary.search(query, max_results=max_results)


def get_default_provider() -> SearchProvider:
    """Pick a provider from the WEB_SEARCH_PROVIDER env var.

    - "tavily" → Tavily with DuckDuckGo as a runtime fallback.
      If Tavily cannot even be constructed (missing key/package),
      degrades to DuckDuckGo only and logs a warning.
    - anything else (including "ddg" or unset) → DuckDuckGo.
    """
    name = os.environ.get("WEB_SEARCH_PROVIDER", "ddg").lower()
    if name == "tavily":
        try:
            primary = TavilyProvider()
        except Exception as exc:
            logger.warning(
                "Could not initialize Tavily provider (%s); using DuckDuckGo",
                exc,
            )
            return DuckDuckGoProvider()
        return FallbackProvider(primary, DuckDuckGoProvider())
    return DuckDuckGoProvider()
