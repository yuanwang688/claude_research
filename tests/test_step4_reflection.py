"""Step 4: reflection_node, route_after_reflection, research loop control."""
from unittest.mock import AsyncMock, patch

import pytest

from deep_research.config import Config, Configuration
from deep_research.nodes.reflection import (
    _parse_gap_response,
    reflection_node,
    route_after_reflection,
)
from deep_research.providers.mock import MockSearchProvider
from deep_research.state import (
    Finding,
    OverallState,
    ReflectionOutput,
    SearchQueriesOutput,
)


# ---------- Helpers ----------

def make_runnable_config(config: Config | None = None, mock_llm=None):
    from tests.conftest import MockChatModel

    if config is None:
        config = Config(enable_clarification=False, enable_gap_review=False, enable_draft_review=False)
    cfg = Configuration(
        fast_llm=mock_llm or MockChatModel(),
        powerful_llm=mock_llm or MockChatModel(),
        search_provider=MockSearchProvider(),
        config=config,
    )
    return {"configurable": {"thread_id": config.thread_id, "_configuration": cfg}}


def make_finding(loop: int = 0) -> Finding:
    return Finding(
        query="test query",
        summary="summary text",
        source_urls=["https://example.com"],
        loop_number=loop,
        token_count=20,
    )


def base_state(**overrides) -> dict:
    state = {
        "original_query": "test",
        "messages": [],
        "research_brief": "Test brief",
        "research_plan": [],
        "findings": [make_finding()],
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


# ---------- _parse_gap_response ----------

def test_parse_gap_approve_returns_defaults():
    defaults = ["q1", "q2"]
    assert _parse_gap_response("approve", defaults) == defaults
    assert _parse_gap_response("yes", defaults) == defaults
    assert _parse_gap_response("", defaults) == defaults
    assert _parse_gap_response(None, defaults) == defaults


def test_parse_gap_list_input():
    result = _parse_gap_response(["custom q1", "custom q2"], [])
    assert result == ["custom q1", "custom q2"]


def test_parse_gap_text_split_to_lines():
    text = "query about EU\nquery about US regulations"
    result = _parse_gap_response(text, [])
    assert result == ["query about EU", "query about US regulations"]


def test_parse_gap_falls_back_when_empty_lines():
    defaults = ["default"]
    result = _parse_gap_response("   \n  ", defaults)
    assert result == defaults


# ---------- route_after_reflection ----------

def test_route_insufficient_under_max_loops():
    state = base_state(is_sufficient=False, research_loop_count=1, max_research_loops=3)
    assert route_after_reflection(state) == "query_generator_node"


def test_route_sufficient_goes_to_writer():
    state = base_state(is_sufficient=True, research_loop_count=1, max_research_loops=3)
    assert route_after_reflection(state) == "writer_node"


def test_route_loop_exhausted_goes_to_writer():
    state = base_state(is_sufficient=False, research_loop_count=3, max_research_loops=3)
    assert route_after_reflection(state) == "writer_node"


def test_route_loop_exceeded_goes_to_writer():
    state = base_state(is_sufficient=False, research_loop_count=5, max_research_loops=3)
    assert route_after_reflection(state) == "writer_node"


# ---------- reflection_node ----------

async def test_reflection_sufficient_sets_state(mock_llm):
    state = base_state()
    runnable_config = make_runnable_config(mock_llm=mock_llm)

    mock_output = ReflectionOutput(
        is_sufficient=True,
        confidence=0.95,
        covered_topics=["topic A"],
        missing_topics=[],
        follow_up_queries=[],
    )
    with patch(
        "deep_research.nodes.reflection.astructured_output",
        new=AsyncMock(return_value=mock_output),
    ):
        result = await reflection_node(state, runnable_config)

    assert result["is_sufficient"] is True
    assert result["knowledge_gaps"] == []
    assert result["follow_up_queries"] == []
    assert result["research_loop_count"] == 1  # incremented


async def test_reflection_insufficient_sets_follow_ups(mock_llm):
    state = base_state(research_loop_count=0, max_research_loops=3)
    config = Config(enable_clarification=False, enable_gap_review=False, enable_draft_review=False)
    runnable_config = make_runnable_config(config=config, mock_llm=mock_llm)

    mock_output = ReflectionOutput(
        is_sufficient=False,
        confidence=0.4,
        covered_topics=[],
        missing_topics=["EU regulations", "market data"],
        follow_up_queries=["EU AI regulations 2024", "AI market size statistics"],
    )
    with patch(
        "deep_research.nodes.reflection.astructured_output",
        new=AsyncMock(return_value=mock_output),
    ):
        result = await reflection_node(state, runnable_config)

    assert result["is_sufficient"] is False
    assert result["knowledge_gaps"] == ["EU regulations", "market data"]
    assert result["follow_up_queries"] == ["EU AI regulations 2024", "AI market size statistics"]
    assert result["research_loop_count"] == 1


async def test_reflection_at_max_loops_clears_follow_ups(mock_llm):
    """Even if insufficient, follow_ups should be empty when at max loops."""
    state = base_state(research_loop_count=2, max_research_loops=3)
    config = Config(enable_clarification=False, enable_gap_review=False, enable_draft_review=False)
    runnable_config = make_runnable_config(config=config, mock_llm=mock_llm)

    mock_output = ReflectionOutput(
        is_sufficient=False,
        confidence=0.3,
        covered_topics=[],
        missing_topics=["gap"],
        follow_up_queries=["more queries"],
    )
    with patch(
        "deep_research.nodes.reflection.astructured_output",
        new=AsyncMock(return_value=mock_output),
    ):
        result = await reflection_node(state, runnable_config)

    # research_loop_count becomes 3, which equals max_research_loops → no more loops
    assert result["research_loop_count"] == 3
    assert result["follow_up_queries"] == []


async def test_reflection_increments_loop_count(mock_llm):
    for initial_count in range(3):
        state = base_state(research_loop_count=initial_count)
        runnable_config = make_runnable_config(mock_llm=mock_llm)
        mock_output = ReflectionOutput(
            is_sufficient=True, confidence=0.9,
            covered_topics=[], missing_topics=[], follow_up_queries=[]
        )
        with patch(
            "deep_research.nodes.reflection.astructured_output",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await reflection_node(state, runnable_config)
        assert result["research_loop_count"] == initial_count + 1


# ---------- Loop termination integration ----------

async def test_agent_terminates_after_max_loops(mock_llm, mock_search):
    """Agent should stop after max_research_loops even if never sufficient."""
    from deep_research import Config, DeepResearchAgent

    call_count = {"n": 0}

    async def mock_reflection(*args, **kwargs):
        call_count["n"] += 1
        return ReflectionOutput(
            is_sufficient=False,
            confidence=0.2,
            covered_topics=[],
            missing_topics=["gap"],
            follow_up_queries=["follow up query"],
        )

    with patch(
        "deep_research.nodes.query_gen.astructured_output",
        new=AsyncMock(return_value=SearchQueriesOutput(queries=["q1"])),
    ), patch(
        "deep_research.nodes.reflection.astructured_output",
        new=AsyncMock(side_effect=mock_reflection),
    ):
        agent = DeepResearchAgent(
            llm=mock_llm,
            search_provider=mock_search,
            config=Config(
                max_research_loops=2,
                breadth=1,
                enable_clarification=False,
                enable_gap_review=False,
                enable_draft_review=False,
            ),
        )
        async for _ in agent.astream("test"):
            pass
        state = await agent.aget_state()

    assert state["research_loop_count"] == 2
    assert call_count["n"] == 2


async def test_agent_terminates_when_sufficient(mock_llm, mock_search):
    """Agent stops looping as soon as is_sufficient=True."""
    from deep_research import Config, DeepResearchAgent

    call_count = {"n": 0}

    async def mock_reflection(*args, **kwargs):
        call_count["n"] += 1
        return ReflectionOutput(
            is_sufficient=True,
            confidence=0.95,
            covered_topics=["everything"],
            missing_topics=[],
            follow_up_queries=[],
        )

    with patch(
        "deep_research.nodes.query_gen.astructured_output",
        new=AsyncMock(return_value=SearchQueriesOutput(queries=["q1"])),
    ), patch(
        "deep_research.nodes.reflection.astructured_output",
        new=AsyncMock(side_effect=mock_reflection),
    ):
        agent = DeepResearchAgent(
            llm=mock_llm,
            search_provider=mock_search,
            config=Config(
                max_research_loops=5,
                breadth=1,
                enable_clarification=False,
                enable_gap_review=False,
                enable_draft_review=False,
            ),
        )
        async for _ in agent.astream("test"):
            pass
        state = await agent.aget_state()

    assert state["is_sufficient"] is True
    assert call_count["n"] == 1  # stopped after first loop
