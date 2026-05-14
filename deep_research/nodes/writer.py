import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from ..config import Configuration
from ..prompts import META_ANALYSIS_PROMPT, SECTION_PROMPT
from ..state import OverallState, ReportOutline
from ..utils.citations import assemble_report, relevant_findings
from ..utils.compression import compress_findings
from ..utils.structured_output import astructured_output


async def writer_node(state: OverallState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    compressed = compress_findings(
        state.get("findings", []), cfg.max_findings_tokens
    )
    brief = state.get("research_brief") or state["original_query"]
    user_feedback = state.get("user_feedback")

    # 1. Identify report sections
    outline_prompt = compressed
    if user_feedback:
        outline_prompt = f"User feedback on previous draft: {user_feedback}\n\n{compressed}"

    outline = await astructured_output(
        cfg.powerful_llm,
        ReportOutline,
        [
            SystemMessage(META_ANALYSIS_PROMPT),
            HumanMessage(outline_prompt),
        ],
    )

    sections = outline.sections or ["Overview", "Key Findings", "Conclusion"]

    # 2. Draft sections in parallel
    section_responses = await asyncio.gather(*[
        cfg.powerful_llm.ainvoke([
            SystemMessage(
                SECTION_PROMPT.format(section=section, research_brief=brief)
            ),
            HumanMessage(
                f"Section to write: {section}\n\n"
                f"Research findings:\n{relevant_findings(section, state.get('findings', []))}"
            ),
        ])
        for section in sections
    ])

    draft = assemble_report(
        sections,
        [r.content for r in section_responses],
        state.get("sources", {}),
    )

    if cfg.enable_draft_review:
        feedback = interrupt({"type": "draft_ready", "draft": draft})
        if feedback and str(feedback).strip().lower() != "approve":
            return {
                "draft_report": draft,
                "user_feedback": str(feedback),
                "is_sufficient": False,
                "follow_up_queries": [],
            }

    return {"final_report": draft, "draft_report": draft}
