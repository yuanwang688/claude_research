import asyncio

from .base import SearchResult


class MockSearchProvider:
    """Async mock provider for unit tests. Never makes network calls."""

    def __init__(self, results: list[SearchResult] | None = None):
        self._results = results if results is not None else [
            SearchResult(
                url="https://example.com/article-1",
                title="Example Article One",
                snippet="This is a relevant snippet about the research topic.",
            ),
            SearchResult(
                url="https://example.com/article-2",
                title="Example Article Two",
                snippet="Another relevant snippet with complementary information.",
            ),
            SearchResult(
                url="https://example.com/article-3",
                title="Example Article Three",
                snippet="A third source providing additional context and data.",
            ),
        ]

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        await asyncio.sleep(0)
        return self._results[:max_results]

    async def rerank(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        await asyncio.sleep(0)
        return results
