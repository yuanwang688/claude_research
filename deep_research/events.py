from typing import Any

from pydantic import BaseModel

from .state import Source, SubQuestion


class AgentEvent(BaseModel):
    """Base class for all typed events yielded by agent.astream()."""


class ClarificationNeeded(AgentEvent):
    questions: list[str]
    draft_plan: str
    estimated_scope: str


class PlanReady(AgentEvent):
    sub_questions: list[SubQuestion]


class ResearchUpdate(AgentEvent):
    loop_count: int
    sources_count: int
    findings_count: int


class GapReview(AgentEvent):
    gaps: list[str]
    proposed_queries: list[str]
    confidence: float


class DraftReady(AgentEvent):
    draft: str


class ResearchResult(BaseModel):
    final_report: str
    sources: dict[str, Source]
    metadata: dict[str, Any] = {}


class Complete(AgentEvent):
    result: ResearchResult
