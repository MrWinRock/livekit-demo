from unittest.mock import patch

import pytest

from web_search import (
    DuckDuckGoProvider,
    FallbackProvider,
    SearchResult,
    TavilyProvider,
    get_default_provider,
)


class _FakeDDGS:
    """Stand-in for ddgs.DDGS used in DuckDuckGoProvider tests."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.last_query: str | None = None

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def text(self, query: str, max_results: int = 3):
        self.last_query = query
        return self.rows[:max_results]


def test_search_result_summary_includes_all_fields():
    r = SearchResult(title="Foo", snippet="Bar baz", url="https://example.com")
    summary = r.to_summary()
    assert "Foo" in summary
    assert "Bar baz" in summary
    assert "https://example.com" in summary


@pytest.mark.asyncio
async def test_ddg_provider_returns_parsed_results():
    fake_rows = [
        {
            "title": "USB-C info",
            "body": "USB-C is a connector standard.",
            "href": "https://example.com/1",
        },
        {
            "title": "Cable specs",
            "body": "100W power delivery.",
            "href": "https://example.com/2",
        },
    ]
    fake = _FakeDDGS(fake_rows)
    with patch("ddgs.DDGS", fake):
        provider = DuckDuckGoProvider()
        results = await provider.search("USB-C", max_results=2)

    assert len(results) == 2
    assert results[0].title == "USB-C info"
    assert results[0].snippet == "USB-C is a connector standard."
    assert results[0].url == "https://example.com/1"


@pytest.mark.asyncio
async def test_ddg_provider_respects_max_results():
    fake_rows = [
        {"title": f"r{i}", "body": "", "href": f"https://example.com/{i}"}
        for i in range(10)
    ]
    fake = _FakeDDGS(fake_rows)
    with patch("ddgs.DDGS", fake):
        provider = DuckDuckGoProvider()
        results = await provider.search("anything", max_results=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_ddg_provider_handles_empty_results():
    fake = _FakeDDGS([])
    with patch("ddgs.DDGS", fake):
        provider = DuckDuckGoProvider()
        results = await provider.search("nonsense", max_results=3)
    assert results == []


@pytest.mark.asyncio
async def test_ddg_provider_handles_missing_fields():
    fake_rows = [{}]
    fake = _FakeDDGS(fake_rows)
    with patch("ddgs.DDGS", fake):
        provider = DuckDuckGoProvider()
        results = await provider.search("anything")
    assert len(results) == 1
    assert results[0].title == ""
    assert results[0].snippet == ""
    assert results[0].url == ""


def test_default_provider_is_ddg_when_unset(monkeypatch):
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    assert isinstance(get_default_provider(), DuckDuckGoProvider)


def test_default_provider_is_ddg_for_explicit_ddg(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "ddg")
    assert isinstance(get_default_provider(), DuckDuckGoProvider)


def test_default_provider_wraps_tavily_in_fallback(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-test")
    provider = get_default_provider()
    assert isinstance(provider, FallbackProvider)
    assert isinstance(provider.primary, TavilyProvider)
    assert isinstance(provider.secondary, DuckDuckGoProvider)


def test_default_provider_falls_back_to_ddg_when_tavily_unconfigured(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider = get_default_provider()
    assert isinstance(provider, DuckDuckGoProvider)


def test_tavily_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        TavilyProvider()


class _StubProvider:
    def __init__(self, results=None, raises=None):
        self.results = results or []
        self.raises = raises
        self.calls = 0

    async def search(self, query, max_results=3):
        self.calls += 1
        if self.raises:
            raise self.raises
        return self.results


@pytest.mark.asyncio
async def test_fallback_returns_primary_when_primary_succeeds():
    primary_results = [SearchResult("primary", "snippet", "https://primary")]
    secondary_results = [SearchResult("secondary", "snippet", "https://secondary")]
    primary = _StubProvider(results=primary_results)
    secondary = _StubProvider(results=secondary_results)

    provider = FallbackProvider(primary, secondary)
    out = await provider.search("anything")

    assert out == primary_results
    assert primary.calls == 1
    assert secondary.calls == 0


@pytest.mark.asyncio
async def test_fallback_uses_secondary_when_primary_raises():
    secondary_results = [SearchResult("secondary", "snippet", "https://secondary")]
    primary = _StubProvider(raises=RuntimeError("tavily down"))
    secondary = _StubProvider(results=secondary_results)

    provider = FallbackProvider(primary, secondary)
    out = await provider.search("anything")

    assert out == secondary_results
    assert primary.calls == 1
    assert secondary.calls == 1
