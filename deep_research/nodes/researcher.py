from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import Configuration
from ..prompts import SUMMARIZE_PROMPT
from ..state import Finding, Source, WebResearchState
from ..utils.compression import count_tokens


def _format_results(results) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.title}\nURL: {r.url}\n{r.snippet}")
    return "\n\n".join(parts)


async def web_research_node(state: WebResearchState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    query = state["query"]

    async with cfg.semaphore:
        results = await cfg.search_provider.search(
            query, max_results=cfg.max_results_per_query
        )

    if not results:
        return {"findings": [], "sources": {}}

    summary_msg = await cfg.fast_llm.ainvoke([
        SystemMessage(SUMMARIZE_PROMPT),
        HumanMessage(f"Query: {query}\n\nSearch results:\n{_format_results(results)}"),
    ])

    finding = Finding(
        query=query,
        summary=summary_msg.content,
        source_urls=[r.url for r in results],
        loop_number=state["research_loop_count"],
        token_count=count_tokens(summary_msg.content),
    )
    sources = {
        r.url: Source(url=r.url, title=r.title, snippet=r.snippet)
        for r in results
    }

    return {"findings": [finding], "sources": sources}
