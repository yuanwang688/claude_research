from .base import AsyncSearchProvider, SearchResult


class _SyncDuckDuckGoProvider:
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    SearchResult(
                        url=r.get("href", ""),
                        title=r.get("title", ""),
                        snippet=r.get("body", ""),
                    )
                )
        return results


class DuckDuckGoSearchProvider(AsyncSearchProvider):
    """DuckDuckGo search provider (free, no API key). Wrapped for async compatibility."""

    def __init__(self):
        super().__init__(_SyncDuckDuckGoProvider())
