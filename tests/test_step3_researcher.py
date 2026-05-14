"""Step 3: query_generator_node, web_research_node, fan-out via Send."""
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from deep_research.config import Config, Configuration
from deep_research.nodes.query_gen import dispatch_web_research, query_generator_node
from deep_research.nodes.researcher import web_research_node
from deep_research.providers.mock import MockSearchProvider
from deep_research.state import (
    OverallState,
    SearchQueriesOutput,
    SubQuestion,
    WebResearchState,
)
from langgraph.types import Send


# ---------- Helpers ----------

def make_runnable_config(config: Config | None = None, mock_llm=None, search=None):
    from tests.conftest import MockChatModel

    if config is None:
        config = Config(enable_clarification=False, enable_gap_review=False, enable_draft_review=False)
    cfg = Configuration(
        fast_llm=mock_llm or MockChatModel(),
        powerful_llm=mock_llm or MockChatModel(),
        search_provider=search or MockSearchProvider(),
        config=config,
    )
    return {"configurable": {"thread_id": config.thread_id, "_configuration": cfg}}


def base_state(**overrides) -> dict:
    state = {
        "original_query": "test query",
        "messages": [],
        "research_brief": "Test research brief",
        "research_plan": [],
        "findings": [],
        "sources": {},
        "research_loop_count": 0,
        "max_research_loops": 3,
        "is_sufficient": False,
        "knowledge_gaps": [],
        "follow_up_queries": [],
        "draft_report": None,
        "user_feedback": None,
        "final_report": None,
    }
    state.update(overrides)
    return state


# ---------- query_generator_node ----------

async def test_query_gen_uses_plan_on_first_loop(mock_llm):
    plan = [
        SubQuestion(id="q1", question="What is X?", evidence_type="qualitative"),
        SubQuestion(id="q2", question="How does Y work?", evidence_type="technical"),
    ]
    state = base_state(research_plan=plan, follow_up_queries=[])
    runnable_config = make_runnable_config(mock_llm=mock_llm)

    expected = SearchQueriesOutput(queries=["search query 1", "search query 2"])
    with patch(
        "deep_research.nodes.query_gen.astructured_output",
        new=AsyncMock(return_value=expected),
    ):
        result = await query_generator_node(state, runnable_config)

    assert result["follow_up_queries"] == ["search query 1", "search query 2"]


async def test_query_gen_uses_follow_up_on_subsequent_loop(mock_llm):
    state = base_state(
        follow_up_queries=["gap 1", "gap 2"],
        research_loop_count=1,
    )
    runnable_config = make_runnable_config(mock_llm=mock_llm)

    expected = SearchQueriesOutput(queries=["refined query for gap 1", "refined query for gap 2"])
    with patch(
        "deep_research.nodes.query_gen.astructured_output",
        new=AsyncMock(return_value=expected),
    ) as mock_so:
        result = await query_generator_node(state, runnable_config)

    # Verify the prompt context included the follow-up queries
    call_messages = mock_so.call_args[0][2]  # third positional arg = messages
    prompt_text = " ".join(m.content for m in call_messages)
    assert "gap 1" in prompt_text
    assert result["follow_up_queries"] == ["refined query for gap 1", "refined query for gap 2"]


async def test_query_gen_falls_back_to_original_query(mock_llm):
    """If neither plan nor follow_ups, use original_query."""
    state = base_state(follow_up_queries=[], research_plan=[])
    runnable_config = make_runnable_config(mock_llm=mock_llm)

    expected = SearchQueriesOutput(queries=["fallback query"])
    with patch(
        "deep_research.nodes.query_gen.astructured_output",
        new=AsyncMock(return_value=expected),
    ) as mock_so:
        result = await query_generator_node(state, runnable_config)

    call_messages = mock_so.call_args[0][2]
    prompt_text = " ".join(m.content for m in call_messages)
    assert "test query" in prompt_text


# ---------- dispatch_web_research ----------

def test_dispatch_creates_send_per_query():
    state = base_state(follow_up_queries=["q1", "q2", "q3"])
    sends = dispatch_web_research(state)
    assert len(sends) == 3
    assert all(isinstance(s, Send) for s in sends)
    assert sends[0].node == "web_research_node"
    assert sends[0].arg["query"] == "q1"
    assert sends[1].arg["query_index"] == 1


def test_dispatch_falls_back_to_plan():
    plan = [
        SubQuestion(id="q1", question="Plan Q1", evidence_type="qualitative"),
        SubQuestion(id="q2", question="Plan Q2", evidence_type="technical"),
    ]
    state = base_state(follow_up_queries=[], research_plan=plan)
    sends = dispatch_web_research(state)
    assert len(sends) == 2
    assert sends[0].arg["query"] == "Plan Q1"


def test_dispatch_falls_back_to_original_query():
    state = base_state(follow_up_queries=[], research_plan=[])
    sends = dispatch_web_research(state)
    assert len(sends) == 1
    assert sends[0].arg["query"] == "test query"


def test_dispatch_passes_loop_count():
    state = base_state(follow_up_queries=["q1"], research_loop_count=2)
    sends = dispatch_web_research(state)
    assert sends[0].arg["research_loop_count"] == 2


# ---------- web_research_node ----------

async def test_researcher_returns_finding_and_sources(mock_llm):
    ws: WebResearchState = {
        "query": "AI impact on jobs",
        "query_index": 0,
        "research_loop_count": 0,
    }
    runnable_config = make_runnable_config(mock_llm=mock_llm)
    result = await web_research_node(ws, runnable_config)

    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding.query == "AI impact on jobs"
    assert finding.loop_number == 0
    assert finding.token_count > 0
    assert len(result["sources"]) > 0


async def test_researcher_uses_llm_for_summary(mock_llm):
    """Verify the LLM is called and its content ends up in the finding summary."""
    mock_llm.responses = ["This is the LLM summary of search results."]
    ws: WebResearchState = {
        "query": "test",
        "query_index": 0,
        "research_loop_count": 1,
    }
    runnable_config = make_runnable_config(mock_llm=mock_llm)
    result = await web_research_node(ws, runnable_config)
    assert result["findings"][0].summary == "This is the LLM summary of search results."


async def test_researcher_empty_results_returns_empty():
    """If search returns no results, findings and sources should be empty."""
    from tests.conftest import MockChatModel

    config = Config()
    search = MockSearchProvider(results=[])
    runnable_config = make_runnable_config(config=config, mock_llm=MockChatModel(), search=search)
    ws: WebResearchState = {"query": "obscure topic", "query_index": 0, "research_loop_count": 0}

    result = await web_research_node(ws, runnable_config)
    assert result["findings"] == []
    assert result["sources"] == {}


async def test_researcher_sources_keyed_by_url(mock_llm):
    ws: WebResearchState = {"query": "test", "query_index": 0, "research_loop_count": 0}
    runnable_config = make_runnable_config(mock_llm=mock_llm)
    result = await web_research_node(ws, runnable_config)

    for url, src in result["sources"].items():
        assert url == src.url


async def test_researcher_respects_semaphore(mock_llm):
    """Semaphore should not block single sequential call."""
    import asyncio

    ws: WebResearchState = {"query": "test", "query_index": 0, "research_loop_count": 0}
    runnable_config = make_runnable_config(mock_llm=mock_llm)
    # Should complete without deadlock
    result = await asyncio.wait_for(web_research_node(ws, runnable_config), timeout=5.0)
    assert result["findings"]


# ---------- Full graph fan-out integration ----------

async def test_full_graph_fanout_produces_n_findings(mock_llm, mock_search):
    """Full graph run: 2 queries → 2 web_research_node calls → 2 findings."""
    from deep_research import Config, DeepResearchAgent

    queries = SearchQueriesOutput(queries=["query A", "query B"])
    with patch(
        "deep_research.nodes.query_gen.astructured_output",
        new=AsyncMock(return_value=queries),
    ), patch(
        "deep_research.nodes.reflection.astructured_output",
        new=AsyncMock(
            return_value=__import__(
                "deep_research.state", fromlist=["ReflectionOutput"]
            ).ReflectionOutput(
                is_sufficient=True,
                confidence=0.9,
                covered_topics=["topic A", "topic B"],
                missing_topics=[],
                follow_up_queries=[],
            )
        ),
    ):
        agent = DeepResearchAgent(
            llm=mock_llm,
            search_provider=mock_search,
            config=Config(
                max_research_loops=1,
                enable_clarification=False,
                enable_gap_review=False,
                enable_draft_review=False,
            ),
        )
        async for _ in agent.astream("test"):
            pass

        state = await agent.aget_state()

    assert len(state["findings"]) == 2
    queries_run = {f.query for f in state["findings"]}
    assert queries_run == {"query A", "query B"}
