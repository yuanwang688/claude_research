"""Quick live test: real Anthropic LLMs + DuckDuckGo search."""
import asyncio
import os

from langchain_anthropic import ChatAnthropic

from deep_research import ClarificationNeeded, Complete, Config, DeepResearchAgent, DraftReady, GapReview, PlanReady
from deep_research.events import ResearchUpdate
from deep_research.providers.duckduckgo import DuckDuckGoSearchProvider


async def main():
    fast_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=os.environ["ANTHROPIC_API_KEY"])
    powerful_llm = ChatAnthropic(model="claude-sonnet-4-6", api_key=os.environ["ANTHROPIC_API_KEY"])

    agent = DeepResearchAgent(
        fast_llm=fast_llm,
        powerful_llm=powerful_llm,
        search_provider=DuckDuckGoSearchProvider(),
        config=Config(
            max_research_loops=2,
            breadth=3,
            enable_clarification=False,
            enable_gap_review=False,
            enable_draft_review=False,
        ),
    )

    query = "How to set up a python virtual environment?"
    print(f"Query: {query}\n{'='*60}")

    async for event in agent.astream(query):
        if isinstance(event, PlanReady):
            print(f"\n[PlanReady] {len(event.sub_questions)} sub-questions:")
            for sq in event.sub_questions:
                print(f"  • {sq.question}")
        elif isinstance(event, ResearchUpdate):
            print(f"\n[ResearchUpdate] loop={event.loop_count}, sources={event.sources_count}, findings={event.findings_count}")
        elif isinstance(event, ClarificationNeeded):
            print(f"\n[ClarificationNeeded] {event.questions}")
            await agent.resume({})
        elif isinstance(event, GapReview):
            print(f"\n[GapReview] gaps={event.gaps}")
            await agent.resume("approve")
        elif isinstance(event, DraftReady):
            print(f"\n[DraftReady] (draft length={len(event.draft)})")
            await agent.resume("approve")
        elif isinstance(event, Complete):
            print(f"\n{'='*60}")
            print(f"[Complete] loops={event.result.metadata['loops_run']}, "
                  f"sources={event.result.metadata['total_sources']}, "
                  f"elapsed={event.result.metadata['elapsed_seconds']}s")
            print(f"\n{event.result.final_report}")


async def _run():
    await main()
    # Let the event loop drain pending callbacks so httpx connection
    # pools close cleanly and don't emit ResourceWarning on exit.
    await asyncio.sleep(0)

asyncio.run(_run())
