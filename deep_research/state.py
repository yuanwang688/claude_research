import operator
from datetime import UTC, datetime
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


def merge_source_dicts(a: dict, b: dict) -> dict:
    """Merge source dicts; keep higher relevance_score on URL conflict."""
    merged = dict(a)
    for url, source in b.items():
        if url not in merged or source.relevance_score > merged[url].relevance_score:
            merged[url] = source
    return merged


class SubQuestion(BaseModel):
    id: str
    question: str
    evidence_type: Literal["statistical", "qualitative", "comparative", "technical", "other"]
    depends_on: list[str] = []


class Source(BaseModel):
    url: str
    title: str
    snippet: str
    full_text: str | None = None
    relevance_score: float = 0.0
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Finding(BaseModel):
    query: str
    sub_question_id: str | None = None
    summary: str
    source_urls: list[str]
    loop_number: int
    token_count: int


class ClarifyingQuestionsOutput(BaseModel):
    questions: list[str]
    draft_research_plan: str
    estimated_scope: Literal["narrow", "medium", "broad"]


class ResearchPlanOutput(BaseModel):
    sub_questions: list[SubQuestion]
    research_brief_confirmed: str


class SearchQueriesOutput(BaseModel):
    queries: list[str]


class ReflectionOutput(BaseModel):
    is_sufficient: bool
    confidence: float
    covered_topics: list[str]
    missing_topics: list[str]
    follow_up_queries: list[str]


class ReportOutline(BaseModel):
    sections: list[str]


class WebResearchState(TypedDict):
    query: str
    query_index: int
    research_loop_count: int


class OverallState(TypedDict):
    # Input
    original_query: str
    messages: Annotated[list[BaseMessage], add_messages]

    # Set by clarifier (immutable after)
    research_brief: str

    # Set by planner (immutable after)
    research_plan: list[SubQuestion]

    # Accumulated via reducers
    findings: Annotated[list[Finding], operator.add]
    sources: Annotated[dict[str, Source], merge_source_dicts]

    # Loop control
    research_loop_count: int
    max_research_loops: int
    is_sufficient: bool
    knowledge_gaps: list[str]
    follow_up_queries: list[str]

    # Report lifecycle
    draft_report: str | None
    user_feedback: str | None
    final_report: str | None

    # Plan review
    plan_feedback: str | None  # set when user rejects the plan; cleared on approval
