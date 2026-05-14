"""Step 5: writer_node — parallel section drafting, citation assembly, draft feedback."""
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from deep_research.config import Config, Configuration
from deep_research.nodes.writer import writer_node
from deep_research.providers.mock import MockSearchProvider
from deep_research.state import (
    Finding,
    OverallState,
    ReportOutline,
    Source,
)


# ---------- Helpers ----------

def make_runnable_config(config: Config | None = None, mock_llm=None):
    from tests.conftest import MockChatModel

    if config is None:
        config = Config(
            enable_clarification=False,
            enable_gap_review=False,
            enable_draft_review=False,
        )
    cfg = Configuration(
        fast_llm=mock_llm or MockChatModel(),
        powerful_llm=mock_llm or MockChatModel(),
        search_provider=MockSearchProvider(),
        config=config,
    )
    return {"configurable": {"thread_id": config.thread_id, "_configuration": cfg}}


def make_finding(query="test query", loop=0) -> Finding:
    return Finding(
        query=query,
        summary="Summary of findings about " + query,
        source_urls=["https://example.com/a", "https://example.com/b"],
        loop_number=loop,
        token_count=30,
    )


def base_state(**overrides) -> dict:
    sources = {
        "https://example.com/a": Source(url="https://example.com/a", title="Source A", snippet="Snippet A"),
        "https://example.com/b": Source(url="https://example.com/b", title="Source B", snippet="Snippet B"),
    }
    state = {
        "original_query": "What is AI?",
        "messages": [],
        "research_brief": "Comprehensive overview of AI and its impacts.",
        "research_plan": [],
        "findings": [make_finding("AI overview"), make_finding("AI economics")],
        "sources": sources,
        "research_loop_count": 1,
        "max_research_loops": 3,
        "is_sufficient": True,
        "knowledge_gaps": [],
        "follow_up_queries": [],
        "draft_report": None,
        "user_feedback": None,
        "final_report": None,
    }
    state.update(overrides)
    return state


# ---------- writer_node unit tests ----------

async def test_writer_produces_final_report(mock_llm):
    state = base_state()
    runnable_config = make_runnable_config(mock_llm=mock_llm)

    mock_outline = ReportOutline(sections=["Introduction", "Key Findings", "Conclusion"])
    with patch(
        "deep_research.nodes.writer.astructured_output",
        new=AsyncMock(return_value=mock_outline),
    ):
        result = await writer_node(state, runnable_config)

    assert "final_report" in result
    assert result["final_report"] is not None
    assert len(result["final_report"]) > 0


async def test_writer_report_contains_section_headers(mock_llm):
    state = base_state()
    mock_llm.responses = ["Section content here."]
    runnable_config = make_runnable_config(mock_llm=mock_llm)

    mock_outline = ReportOutline(sections=["Background", "Analysis"])
    with patch(
        "deep_research.nodes.writer.astructured_output",
        new=AsyncMock(return_value=mock_outline),
    ):
        result = await writer_node(state, runnable_config)

    report = result["final_report"]
    assert "## Background" in report
    assert "## Analysis" in report


async def test_writer_includes_sources_appendix(mock_llm):
    state = base_state()
    runnable_config = make_runnable_config(mock_llm=mock_llm)

    mock_outline = ReportOutline(sections=["Overview"])
    with patch(
        "deep_research.nodes.writer.astructured_output",
        new=AsyncMock(return_value=mock_outline),
    ):
        result = await writer_node(state, runnable_config)

    report = result["final_report"]
    assert "## Sources" in report
    assert "https://example.com/a" in report or "https://example.com/b" in report


async def test_writer_parallel_sections_all_present(mock_llm):
    """All sections from the outline appear as headers in the report."""
    state = base_state()
    mock_llm.responses = ["Content for this section."]
    runnable_config = make_runnable_config(mock_llm=mock_llm)

    sections = ["Background", "Analysis", "Implications"]
    mock_outline = ReportOutline(sections=sections)
    with patch(
        "deep_research.nodes.writer.astructured_output",
        new=AsyncMock(return_value=mock_outline),
    ):
        result = await writer_node(state, runnable_config)

    report = result["final_report"]
    for section in sections:
        assert f"## {section}" in report, f"Missing section: {section}"


async def test_writer_uses_fallback_sections_when_outline_empty(mock_llm):
    """If outline returns empty sections, writer uses default section titles."""
    state = base_state()
    runnable_config = make_runnable_config(mock_llm=mock_llm)

    mock_outline = ReportOutline(sections=[])
    with patch(
        "deep_research.nodes.writer.astructured_output",
        new=AsyncMock(return_value=mock_outline),
    ):
        result = await writer_node(state, runnable_config)

    report = result["final_report"]
    # At least one default section should appear
    assert any(h in report for h in ["## Overview", "## Key Findings", "## Conclusion"])


async def test_writer_sets_draft_report(mock_llm):
    state = base_state()
    runnable_config = make_runnable_config(mock_llm=mock_llm)

    mock_outline = ReportOutline(sections=["Summary"])
    with patch(
        "deep_research.nodes.writer.astructured_output",
        new=AsyncMock(return_value=mock_outline),
    ):
        result = await writer_node(state, runnable_config)

    assert result["draft_report"] == result["final_report"]


async def test_writer_incorporates_user_feedback(mock_llm):
    """User feedback should appear in the outline prompt context."""
    state = base_state(user_feedback="Please cover the EU regulatory angle.")
    runnable_config = make_runnable_config(mock_llm=mock_llm)

    captured = {}

    async def capture_call(llm, schema, messages, **kwargs):
        captured["messages"] = messages
        return ReportOutline(sections=["Overview"])

    with patch("deep_research.nodes.writer.astructured_output", new=capture_call):
        await writer_node(state, runnable_config)

    prompt_text = " ".join(m.content for m in captured["messages"])
    assert "EU regulatory" in prompt_text


# ---------- Full graph integration: writer produces final_report ----------

async def test_agent_complete_event_has_report(mock_llm, mock_search):
    from deep_research import Complete, Config, DeepResearchAgent
    from deep_research.state import ReflectionOutput, SearchQueriesOutput

    with patch(
        "deep_research.nodes.query_gen.astructured_output",
        new=AsyncMock(return_value=SearchQueriesOutput(queries=["q1"])),
    ), patch(
        "deep_research.nodes.reflection.astructured_output",
        new=AsyncMock(
            return_value=ReflectionOutput(
                is_sufficient=True,
                confidence=0.9,
                covered_topics=["all"],
                missing_topics=[],
                follow_up_queries=[],
            )
        ),
    ), patch(
        "deep_research.nodes.writer.astructured_output",
        new=AsyncMock(return_value=ReportOutline(sections=["Summary"])),
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
        events = []
        async for event in agent.astream("What is AI?"):
            events.append(event)

    complete_events = [e for e in events if isinstance(e, Complete)]
    assert len(complete_events) == 1
    result = complete_events[0].result
    assert result.final_report
    assert "## Summary" in result.final_report
    assert isinstance(result.sources, dict)
    assert result.metadata["loops_run"] >= 1
