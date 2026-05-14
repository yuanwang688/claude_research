from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from ..config import Configuration
from ..prompts import REFLECTION_PROMPT
from ..state import OverallState, ReflectionOutput
from ..utils.compression import compress_findings
from ..utils.structured_output import astructured_output


def _parse_gap_response(user_response, default_queries: list[str]) -> list[str]:
    """Map user CP2 response to a list of follow-up queries."""
    if not user_response:
        return default_queries
    if isinstance(user_response, list):
        return user_response
    text = str(user_response).strip()
    if text.lower() in ("approve", "yes", "ok", ""):
        return default_queries
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines if lines else default_queries


async def reflection_node(state: OverallState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    compressed = compress_findings(
        state.get("findings", []), cfg.max_findings_tokens
    )

    output = await astructured_output(
        cfg.fast_llm,
        ReflectionOutput,
        [
            SystemMessage(REFLECTION_PROMPT),
            HumanMessage(
                f"Research Brief: {state.get('research_brief') or state['original_query']}\n\n"
                f"Findings so far:\n{compressed}\n\n"
                f"Loop: {state['research_loop_count']} of {state['max_research_loops']}"
            ),
        ],
    )

    loop_count = state["research_loop_count"] + 1
    can_loop = loop_count < state["max_research_loops"]

    if not output.is_sufficient and can_loop:
        if cfg.enable_gap_review:
            user_response = interrupt({
                "type": "gap_review",
                "gaps": output.missing_topics,
                "proposed_queries": output.follow_up_queries,
                "confidence": output.confidence,
            })
            follow_up = _parse_gap_response(user_response, output.follow_up_queries)
        else:
            follow_up = output.follow_up_queries
    else:
        follow_up = []

    return {
        "is_sufficient": output.is_sufficient,
        "knowledge_gaps": output.missing_topics,
        "follow_up_queries": follow_up,
        "research_loop_count": loop_count,
    }


def route_after_reflection(
    state: OverallState,
) -> Literal["query_generator_node", "writer_node"]:
    loop_exhausted = state["research_loop_count"] >= state["max_research_loops"]
    if state["is_sufficient"] or loop_exhausted:
        return "writer_node"
    return "query_generator_node"
