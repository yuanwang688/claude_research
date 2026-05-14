"""Step 6: clarifier_node, planner_node, full pipeline (all auto_approve)."""
from unittest.mock import AsyncMock, patch

import pytest

from deep_research.config import Config, Configuration
from deep_research.nodes.clarifier import _synthesize_brief, clarifier_node
from deep_research.nodes.planner import planner_node
from deep_research.providers.mock import MockSearchProvider
from deep_research.state import (
    ClarifyingQuestionsOutput,
    ResearchPlanOutput,
    SubQuestion,
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


def base_state(**overrides) -> dict:
    state = {
        "original_query": "What is the economic impact of AI?",
        "messages": [],
        "research_brief": "",
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


# ---------- _synthesize_brief ----------

async def test_synthesize_brief_no_questions(mock_llm):
    result = await _synthesize_brief(mock_llm, "original query", [], {})
    assert result == "original query"


async def test_synthesize_brief_calls_llm_with_qa(mock_llm):
    mock_llm.responses = ["A focused research brief about AI economics."]
    result = await _synthesize_brief(
        mock_llm,
        "AI economics",
        ["Which sector?", "What time range?"],
        {"Which sector?": "All sectors", "What time range?": "2020-2024"},
    )
    assert result == "A focused research brief about AI economics."


async def test_synthesize_brief_returns_query_if_llm_returns_empty(mock_llm):
    mock_llm.responses = [""]
    result = await _synthesize_brief(
        mock_llm, "fallback query", ["Q?"], {"Q?": "A"}
    )
    assert result == "fallback query"


# ---------- clarifier_node ----------

async def test_clarifier_skips_interrupt_when_disabled(mock_llm):
    """With enable_clarification=False, no interrupt should be raised."""
    state = base_state()
    config = Config(enable_clarification=False, enable_gap_review=False, enable_draft_review=False)
    runnable_config = make_runnable_config(config=config, mock_llm=mock_llm)

    mock_output = ClarifyingQuestionsOutput(
        questions=["What sector?"],
        draft_research_plan="Research AI economics across sectors.",
        estimated_scope="broad",
    )
    with patch(
        "deep_research.nodes.clarifier.astructured_output",
        new=AsyncMock(return_value=mock_output),
    ):
        result = await clarifier_node(state, runnable_config)

    # Interrupt skipped — brief defaults to original_query
    assert result["research_brief"] == state["original_query"]


async def test_clarifier_skips_interrupt_when_no_questions(mock_llm):
    """If the LLM asks 0 questions, no interrupt should fire."""
    state = base_state()
    config = Config(enable_clarification=True, enable_gap_review=False, enable_draft_review=False)
    runnable_config = make_runnable_config(config=config, mock_llm=mock_llm)

    mock_output = ClarifyingQuestionsOutput(
        questions=[],
        draft_research_plan="Research AI.",
        estimated_scope="narrow",
    )
    with patch(
        "deep_research.nodes.clarifier.astructured_output",
        new=AsyncMock(return_value=mock_output),
    ):
        result = await clarifier_node(state, runnable_config)

    assert result["research_brief"] == state["original_query"]


async def test_clarifier_sets_research_brief(mock_llm):
    state = base_state()
    config = Config(enable_clarification=False)
    runnable_config = make_runnable_config(config=config, mock_llm=mock_llm)

    mock_output = ClarifyingQuestionsOutput(
        questions=[],
        draft_research_plan="Plan.",
        estimated_scope="medium",
    )
    with patch(
        "deep_research.nodes.clarifier.astructured_output",
        new=AsyncMock(return_value=mock_output),
    ):
        result = await clarifier_node(state, runnable_config)

    assert "research_brief" in result
    assert result["research_brief"]


# ---------- planner_node ----------

async def test_planner_returns_sub_questions(mock_llm):
    state = base_state(research_brief="Research AI economics comprehensively.")
    runnable_config = make_runnable_config(mock_llm=mock_llm)

    sub_questions = [
        SubQuestion(id="q1", question="GDP impact?", evidence_type="statistical"),
        SubQuestion(id="q2", question="Job displacement?", evidence_type="qualitative"),
    ]
    mock_output = ResearchPlanOutput(
        sub_questions=sub_questions,
        research_brief_confirmed="Research AI economics comprehensively.",
    )
    with patch(
        "deep_research.nodes.planner.astructured_output",
        new=AsyncMock(return_value=mock_output),
    ):
        result = await planner_node(state, runnable_config)

    assert len(result["research_plan"]) == 2
    assert result["research_plan"][0].id == "q1"
    assert result["research_plan"][1].evidence_type == "qualitative"


async def test_planner_falls_back_to_original_query(mock_llm):
    """If research_brief is empty, planner should use original_query."""
    state = base_state(research_brief="")
    runnable_config = make_runnable_config(mock_llm=mock_llm)

    mock_output = ResearchPlanOutput(
        sub_questions=[SubQuestion(id="q1", question="Q?", evidence_type="other")],
        research_brief_confirmed="test",
    )
    with patch(
        "deep_research.nodes.planner.astructured_output",
        new=AsyncMock(return_value=mock_output),
    ) as mock_so:
        await planner_node(state, runnable_config)

    call_messages = mock_so.call_args[0][2]
    prompt_text = " ".join(m.content for m in call_messages)
    assert state["original_query"] in prompt_text


# ---------- End-to-end: full pipeline auto_approve ----------

async def test_full_pipeline_auto_approve(mock_llm, mock_search):
    """Full graph with all HITL disabled produces final_report."""
    from deep_research import Complete, Config, DeepResearchAgent
    from deep_research.state import ReflectionOutput, ReportOutline, SearchQueriesOutput

    clarifier_out = ClarifyingQuestionsOutput(
        questions=[],
        draft_research_plan="Plan.",
        estimated_scope="medium",
    )
    planner_out = ResearchPlanOutput(
        sub_questions=[SubQuestion(id="q1", question="Sub-question?", evidence_type="other")],
        research_brief_confirmed="Brief confirmed.",
    )
    queries_out = SearchQueriesOutput(queries=["search query 1"])
    reflection_out = ReflectionOutput(
        is_sufficient=True, confidence=0.9,
        covered_topics=["everything"], missing_topics=[], follow_up_queries=[],
    )
    outline_out = ReportOutline(sections=["Summary"])

    with patch("deep_research.nodes.clarifier.astructured_output", new=AsyncMock(return_value=clarifier_out)), \
         patch("deep_research.nodes.planner.astructured_output", new=AsyncMock(return_value=planner_out)), \
         patch("deep_research.nodes.query_gen.astructured_output", new=AsyncMock(return_value=queries_out)), \
         patch("deep_research.nodes.reflection.astructured_output", new=AsyncMock(return_value=reflection_out)), \
         patch("deep_research.nodes.writer.astructured_output", new=AsyncMock(return_value=outline_out)):

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


def test_run_sync_returns_result(mock_llm, mock_search):
    """Sync run() wrapper returns a ResearchResult."""
    from deep_research import Config, DeepResearchAgent
    from deep_research.state import ReflectionOutput, ReportOutline, SearchQueriesOutput

    with patch("deep_research.nodes.clarifier.astructured_output", new=AsyncMock(return_value=ClarifyingQuestionsOutput(questions=[], draft_research_plan="p", estimated_scope="narrow"))), \
         patch("deep_research.nodes.planner.astructured_output", new=AsyncMock(return_value=ResearchPlanOutput(sub_questions=[], research_brief_confirmed="b"))), \
         patch("deep_research.nodes.query_gen.astructured_output", new=AsyncMock(return_value=SearchQueriesOutput(queries=["q1"]))), \
         patch("deep_research.nodes.reflection.astructured_output", new=AsyncMock(return_value=ReflectionOutput(is_sufficient=True, confidence=0.9, covered_topics=[], missing_topics=[], follow_up_queries=[]))), \
         patch("deep_research.nodes.writer.astructured_output", new=AsyncMock(return_value=ReportOutline(sections=["Overview"]))):

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
        result = agent.run("test query", auto_approve=True)

    from deep_research.events import ResearchResult
    assert isinstance(result, ResearchResult)
    assert result.final_report
