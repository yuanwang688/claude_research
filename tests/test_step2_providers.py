"""Step 2: Search providers — async protocol and mock/DuckDuckGo implementations."""
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from deep_research.providers.base import AsyncSearchProvider, SearchResult
from deep_research.providers.mock import MockSearchProvider


# ---------- SearchResult ----------

def test_search_result_model():
    r = SearchResult(url="https://example.com", title="Title", snippet="Snippet")
    assert r.full_text is None


def test_search_result_with_full_text():
    r = SearchResult(
        url="https://example.com",
        title="T",
        snippet="S",
        full_text="Full article text here.",
    )
    assert r.full_text == "Full article text here."


# ---------- MockSearchProvider ----------

async def test_mock_returns_results():
    provider = MockSearchProvider()
    results = await provider.search("test query")
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


async def test_mock_respects_max_results():
    provider = MockSearchProvider()
    results = await provider.search("test query", max_results=1)
    assert len(results) == 1


async def test_mock_rerank_is_identity():
    provider = MockSearchProvider()
    results = await provider.search("test")
    reranked = await provider.rerank("test", results)
    assert reranked == results


async def test_mock_custom_results():
    custom = [SearchResult(url="https://custom.com", title="Custom", snippet="Custom snippet")]
    provider = MockSearchProvider(results=custom)
    results = await provider.search("anything")
    assert results[0].url == "https://custom.com"


async def test_mock_is_async():
    """Confirm mock is awaitable and yields control (asyncio.sleep(0))."""
    provider = MockSearchProvider()
    # Should complete without blocking
    results = await asyncio.wait_for(provider.search("test"), timeout=1.0)
    assert results


# ---------- AsyncSearchProvider (sync wrapper) ----------

async def test_async_wrapper_delegates_to_sync():
    class SyncProvider:
        def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
            return [SearchResult(url="https://sync.com", title="Sync", snippet="Sync snippet")]

    provider = AsyncSearchProvider(SyncProvider())
    results = await provider.search("test")
    assert len(results) == 1
    assert results[0].url == "https://sync.com"


async def test_async_wrapper_rerank_is_identity():
    class SyncProvider:
        def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
            return []

    provider = AsyncSearchProvider(SyncProvider())
    results = [SearchResult(url="https://x.com", title="X", snippet="X")]
    reranked = await provider.rerank("test", results)
    assert reranked == results


async def test_async_wrapper_passes_max_results():
    received: dict = {}

    class SyncProvider:
        def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
            received["max_results"] = max_results
            return []

    provider = AsyncSearchProvider(SyncProvider())
    await provider.search("test", max_results=3)
    assert received["max_results"] == 3


# ---------- DuckDuckGoSearchProvider ----------

async def test_duckduckgo_provider_wraps_ddgs():
    from deep_research.providers.duckduckgo import DuckDuckGoSearchProvider

    mock_rows = [
        {"href": "https://ddg.com/1", "title": "DDG Result", "body": "DDG body text"},
    ]

    with patch("ddgs.DDGS") as MockDDGS:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text = MagicMock(return_value=iter(mock_rows))
        MockDDGS.return_value = mock_ctx

        provider = DuckDuckGoSearchProvider()
        results = await provider.search("test query", max_results=5)

    assert len(results) == 1
    assert results[0].url == "https://ddg.com/1"
    assert results[0].title == "DDG Result"
    assert results[0].snippet == "DDG body text"
    mock_ctx.text.assert_called_once_with("test query", max_results=5)


async def test_duckduckgo_handles_missing_fields():
    from deep_research.providers.duckduckgo import DuckDuckGoSearchProvider

    mock_rows = [{"href": "", "title": "", "body": ""}]

    with patch("ddgs.DDGS") as MockDDGS:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text = MagicMock(return_value=iter(mock_rows))
        MockDDGS.return_value = mock_ctx

        provider = DuckDuckGoSearchProvider()
        results = await provider.search("test")

    assert len(results) == 1
    assert results[0].url == ""
