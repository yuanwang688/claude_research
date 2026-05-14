import asyncio
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    full_text: str | None = None


@runtime_checkable
class SearchProvider(Protocol):
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...

    async def rerank(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        return results


class SyncSearchProvider(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...


class AsyncSearchProvider:
    """Wraps a synchronous provider for async compatibility via run_in_executor."""

    def __init__(self, sync_provider: SyncSearchProvider):
        self._inner = sync_provider

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._inner.search, query, max_results
        )

    async def rerank(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        return results
