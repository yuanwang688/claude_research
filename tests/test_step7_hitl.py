"""Step 7: HITL astream/resume coordination and typed event sequence."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from deep_research import (
    ClarificationNeeded,
    Complete,
    Config,
    DeepResearchAgent,
    DraftReady,
    GapReview,
    PlanReady,
)
from deep_research.events import ResearchUpdate
from deep_research.state import (
    ClarifyingQuestionsOutput,
    ReflectionOutput,
    ReportOutline,
    ResearchPlanOutput,
    SearchQueriesOutput,
    SubQuestion,
)


# ---------- Helpers ----------

def _all_patches(
    clarifier_questions=None,
    reflection_sufficient=True,
    outline_sections=None,
):
    clarifier_questions = clarifier_questions or []
    outline_sections = outline_sections or ["Summary"]

    return [
        patch(
            "deep_research.nodes.clarifier.astructured_output",
            new=AsyncMock(
                return_value=ClarifyingQuestionsOutput(
                    questions=clarifier_questions,
                    draft_research_plan="Plan.",
                    estimated_scope="medium",
                )
            ),
        ),
        patch(
            "deep_research.nodes.planner.astructured_output",
            new=AsyncMock(
                return_value=ResearchPlanOutput(
                    sub_questions=[
                        SubQuestion(id="q1", question="Sub Q?", evidence_type="other")
                    ],
                    research_brief_confirmed="Brief.",
                )
            ),
        ),
        patch(
            "deep_research.nodes.query_gen.astructured_output",
            new=AsyncMock(return_value=SearchQueriesOutput(queries=["search q"])),
        ),
        patch(
            "deep_research.nodes.reflection.astructured_output",
            new=AsyncMock(
                return_value=ReflectionOutput(
                    is_sufficient=reflection_sufficient,
                    confidence=0.9,
                    covered_topics=["all"] if reflection_sufficient else [],
                    missing_topics=[] if reflection_sufficient else ["gap"],
                    follow_up_queries=[] if reflection_sufficient else ["follow-up q"],
                )
            ),
        ),
        patch(
            "deep_research.nodes.writer.astructured_output",
            new=AsyncMock(return_value=ReportOutline(sections=outline_sections)),
        ),
    ]


def make_agent(mock_llm, mock_search, **cfg_kwargs):
    base = dict(
        max_research_loops=2,
        enable_clarification=False,
        enable_gap_review=False,
        enable_draft_review=False,
    )
    base.update(cfg_kwargs)
    return DeepResearchAgent(
        llm=mock_llm,
        search_provider=mock_search,
        config=Config(**base),
    )


# ---------- Event ordering tests ----------

async def test_plan_ready_event_emitted(mock_llm, mock_search):
    """PlanReady is yielded after planner_node completes."""
    patches = _all_patches()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        agent = make_agent(mock_llm, mock_search)
        events = []
        async for event in agent.astream("test"):
            events.append(event)

    event_types = [type(e).__name__ for e in events]
    assert "PlanReady" in event_types


async def test_plan_ready_contains_sub_questions(mock_llm, mock_search):
    patches = _all_patches()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        agent = make_agent(mock_llm, mock_search)
        plan_events = []
        async for event in agent.astream("test"):
            if isinstance(event, PlanReady):
                plan_events.append(event)

    assert len(plan_events) == 1
    assert len(plan_events[0].sub_questions) == 1
    assert plan_events[0].sub_questions[0].id == "q1"


async def test_research_update_event_emitted(mock_llm, mock_search):
    """ResearchUpdate is yielded after each reflection_node pass."""
    patches = _all_patches()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        agent = make_agent(mock_llm, mock_search)
        updates = []
        async for event in agent.astream("test"):
            if isinstance(event, ResearchUpdate):
                updates.append(event)

    assert len(updates) >= 1
    assert updates[0].loop_count >= 1
    assert updates[0].sources_count >= 0


async def test_complete_event_is_last(mock_llm, mock_search):
    """Complete is always the last event yielded."""
    patches = _all_patches()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        agent = make_agent(mock_llm, mock_search)
        events = []
        async for event in agent.astream("test"):
            events.append(event)

    assert len(events) >= 1
    assert isinstance(events[-1], Complete)


# ---------- Clarification HITL ----------

async def test_clarification_needed_interrupt(mock_llm, mock_search):
    """ClarificationNeeded is yielded when clarifier asks questions."""
    patches = _all_patches(clarifier_questions=["Which sector?", "What time range?"])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        agent = make_agent(mock_llm, mock_search, enable_clarification=True)

        events = []
        gen = agent.astream("AI economics")
        # Consume events until interrupted
        async for event in gen:
            events.append(event)
            if isinstance(event, ClarificationNeeded):
                await agent.resume({"Which sector?": "All", "What time range?": "2020-2024"})
            if isinstance(event, Complete):
                break

    clarification_events = [e for e in events if isinstance(e, ClarificationNeeded)]
    assert len(clarification_events) == 1
    assert "Which sector?" in clarification_events[0].questions


async def test_clarification_resume_continues_graph(mock_llm, mock_search):
    """After resume(), graph continues and eventually yields Complete."""
    patches = _all_patches(clarifier_questions=["Narrow or broad?"])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        agent = make_agent(mock_llm, mock_search, enable_clarification=True)
        events = []
        async for event in agent.astream("test"):
            events.append(event)
            if isinstance(event, ClarificationNeeded):
                await agent.resume({"Narrow or broad?": "broad"})

    assert any(isinstance(e, Complete) for e in events)


# ---------- Gap review HITL ----------

async def test_gap_review_interrupt(mock_llm, mock_search):
    """GapReview event is yielded when reflection finds gaps and CP2 is enabled."""
    call_count = {"n": 0}

    async def reflection_side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ReflectionOutput(
                is_sufficient=False, confidence=0.3,
                covered_topics=[], missing_topics=["gap topic"],
                follow_up_queries=["gap follow-up"],
            )
        return ReflectionOutput(
            is_sufficient=True, confidence=0.9,
            covered_topics=["all"], missing_topics=[], follow_up_queries=[],
        )

    patches = _all_patches()
    with patches[0], patches[1], patches[2], \
         patch("deep_research.nodes.reflection.astructured_output",
               new=AsyncMock(side_effect=reflection_side_effect)), \
         patches[4]:
        agent = make_agent(mock_llm, mock_search,
                           max_research_loops=3,
                           enable_gap_review=True)
        events = []
        async for event in agent.astream("test"):
            events.append(event)
            if isinstance(event, GapReview):
                await agent.resume("approve")

    gap_events = [e for e in events if isinstance(e, GapReview)]
    assert len(gap_events) == 1
    assert "gap topic" in gap_events[0].gaps


# ---------- Draft review HITL ----------

async def test_draft_ready_interrupt(mock_llm, mock_search):
    """DraftReady event is yielded when CP3 is enabled."""
    patches = _all_patches()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        agent = make_agent(mock_llm, mock_search, enable_draft_review=True)
        events = []
        async for event in agent.astream("test"):
            events.append(event)
            if isinstance(event, DraftReady):
                await agent.resume("approve")

    draft_events = [e for e in events if isinstance(e, DraftReady)]
    assert len(draft_events) == 1
    assert draft_events[0].draft


async def test_draft_feedback_triggers_more_research(mock_llm, mock_search):
    """Non-approve feedback loops back to research."""
    writer_call_count = {"n": 0}

    async def writer_so_side_effect(*args, **kwargs):
        writer_call_count["n"] += 1
        return ReportOutline(sections=["Summary"])

    patches = _all_patches()
    with patches[0], patches[1], patches[2], patches[3], \
         patch("deep_research.nodes.writer.astructured_output",
               new=AsyncMock(side_effect=writer_so_side_effect)):
        agent = make_agent(mock_llm, mock_search,
                           max_research_loops=3, enable_draft_review=True)
        events = []
        draft_count = {"n": 0}
        async for event in agent.astream("test"):
            events.append(event)
            if isinstance(event, DraftReady):
                draft_count["n"] += 1
                if draft_count["n"] == 1:
                    await agent.resume("Please add more detail about EU regulations")
                else:
                    await agent.resume("approve")

    # LangGraph replays the node from the beginning on each resume, so
    # astructured_output fires once per (invocation + replay) pair.
    # With 1 feedback round, expect at least 2 calls (1 original + 1 replay
    # after feedback + 1 original in second research loop + 1 replay on approve).
    assert writer_call_count["n"] > 1
    assert any(isinstance(e, Complete) for e in events)


# ---------- Queue coordination ----------

async def test_resume_before_anext_works(mock_llm, mock_search):
    """resume() called before the next __anext__() still unblocks correctly."""
    patches = _all_patches(clarifier_questions=["Q?"])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        agent = make_agent(mock_llm, mock_search, enable_clarification=True)
        gen = agent.astream("test")

        event1 = await gen.__anext__()
        assert isinstance(event1, ClarificationNeeded)

        # Resume BEFORE calling __anext__()
        await agent.resume({"Q?": "A"})

        # Now iterate to completion
        events = [event1]
        async for event in gen:
            events.append(event)

    assert any(isinstance(e, Complete) for e in events)
