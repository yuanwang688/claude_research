"""Step 1: Scaffold + state — verify imports and basic agent lifecycle."""
import pytest

from deep_research import (
    Config,
    DeepResearchAgent,
    AgentEvent,
    ClarificationNeeded,
    Complete,
    DraftReady,
    GapReview,
)
from deep_research.state import (
    Finding,
    OverallState,
    Source,
    SubQuestion,
    WebResearchState,
    merge_source_dicts,
)
from deep_research.config import Configuration
from deep_research.events import ResearchResult


# ---------- Import smoke tests ----------

def test_top_level_imports():
    import deep_research  # noqa: F401


def test_event_types_importable():
    assert issubclass(ClarificationNeeded, AgentEvent)
    assert issubclass(Complete, AgentEvent)
    assert issubclass(DraftReady, AgentEvent)
    assert issubclass(GapReview, AgentEvent)


def test_state_models():
    sq = SubQuestion(id="q1", question="Test?", evidence_type="qualitative")
    assert sq.depends_on == []

    src = Source(url="https://example.com", title="T", snippet="S")
    assert src.relevance_score == 0.0

    f = Finding(
        query="test",
        summary="summary",
        source_urls=["https://example.com"],
        loop_number=1,
        token_count=10,
    )
    assert f.sub_question_id is None


def test_merge_source_dicts_dedup_by_relevance():
    a = {"http://x.com": Source(url="http://x.com", title="A", snippet="A", relevance_score=0.5)}
    b = {"http://x.com": Source(url="http://x.com", title="B", snippet="B", relevance_score=0.8)}
    merged = merge_source_dicts(a, b)
    assert merged["http://x.com"].title == "B"  # higher score wins


def test_merge_source_dicts_no_overwrite_with_lower():
    a = {"http://x.com": Source(url="http://x.com", title="A", snippet="A", relevance_score=0.9)}
    b = {"http://x.com": Source(url="http://x.com", title="B", snippet="B", relevance_score=0.2)}
    merged = merge_source_dicts(a, b)
    assert merged["http://x.com"].title == "A"  # original kept


# ---------- Config tests ----------

def test_config_defaults():
    cfg = Config()
    assert cfg.max_research_loops == 3
    assert cfg.breadth == 3
    assert cfg.max_concurrent_searches == 5
    assert cfg.enable_clarification is True
    assert cfg.thread_id  # auto-generated


def test_config_thread_id_unique():
    c1 = Config()
    c2 = Config()
    assert c1.thread_id != c2.thread_id


# ---------- Agent instantiation tests ----------

def test_agent_with_single_llm(mock_llm, mock_search):
    agent = DeepResearchAgent(llm=mock_llm, search_provider=mock_search)
    assert agent.thread_id


def test_agent_with_two_tier_llm(mock_llm, mock_search):
    agent = DeepResearchAgent(
        fast_llm=mock_llm,
        powerful_llm=mock_llm,
        search_provider=mock_search,
    )
    assert agent.thread_id


def test_agent_requires_llm(mock_search):
    with pytest.raises(ValueError, match="llm"):
        DeepResearchAgent(search_provider=mock_search)


def test_agent_respects_config(mock_llm, mock_search):
    cfg = Config(max_research_loops=5, breadth=4, thread_id="fixed-id")
    agent = DeepResearchAgent(llm=mock_llm, search_provider=mock_search, config=cfg)
    assert agent.thread_id == "fixed-id"


# ---------- astream scaffold test ----------

async def test_astream_does_not_crash(mock_llm, mock_search):
    """Step 1 goal: astream runs the empty graph without raising."""
    agent = DeepResearchAgent(
        llm=mock_llm,
        search_provider=mock_search,
        config=Config(
            enable_clarification=False,
            enable_gap_review=False,
            enable_draft_review=False,
        ),
    )
    events = []
    async for event in agent.astream("test query"):
        events.append(event)
    # Empty graph (START->END): no events expected, no crash expected
    assert isinstance(events, list)


async def test_aget_state_after_stream(mock_llm, mock_search):
    """aget_state returns a dict after the graph runs."""
    from unittest.mock import AsyncMock, patch
    from deep_research.state import ReflectionOutput, ReportOutline, SearchQueriesOutput

    with patch(
        "deep_research.nodes.query_gen.astructured_output",
        new=AsyncMock(return_value=SearchQueriesOutput(queries=["q1"])),
    ), patch(
        "deep_research.nodes.reflection.astructured_output",
        new=AsyncMock(return_value=ReflectionOutput(
            is_sufficient=True, confidence=0.9,
            covered_topics=["all"], missing_topics=[], follow_up_queries=[],
        )),
    ), patch(
        "deep_research.nodes.writer.astructured_output",
        new=AsyncMock(return_value=ReportOutline(sections=["Summary"])),
    ):
        agent = DeepResearchAgent(
            llm=mock_llm,
            search_provider=mock_search,
            config=Config(
                enable_clarification=False,
                enable_gap_review=False,
                enable_draft_review=False,
            ),
        )
        async for _ in agent.astream("test"):
            pass
        state = await agent.aget_state()
    assert isinstance(state, dict)
