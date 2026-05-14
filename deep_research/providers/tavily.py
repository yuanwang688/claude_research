import httpx

from .base import SearchResult


class TavilySearchProvider:
    """Native async Tavily search provider via httpx."""

    _BASE_URL = "https://api.tavily.com"

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._BASE_URL}/search",
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        return [
            SearchResult(
                url=r["url"],
                title=r.get("title", ""),
                snippet=r.get("content", ""),
            )
            for r in data.get("results", [])
        ]

    async def rerank(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        return results  # Tavily reranks internally during search
