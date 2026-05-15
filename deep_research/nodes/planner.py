from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from ..config import Configuration
from ..prompts import PLANNER_PROMPT
from ..state import OverallState, ResearchPlanOutput
from ..utils.structured_output import astructured_output

_APPROVE_TOKENS = {"approve", "yes", "ok", "looks good", "lgtm", ""}


async def planner_node(state: OverallState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    brief = state.get("research_brief") or state["original_query"]

    output = await astructured_output(
        cfg.powerful_llm,
        ResearchPlanOutput,
        [
            SystemMessage(PLANNER_PROMPT.format(research_brief=brief)),
            HumanMessage(brief),
        ],
    )

    if not cfg.enable_plan_review:
        return {"research_plan": output.sub_questions, "plan_feedback": None}

    feedback = interrupt({
        "type": "plan_review",
        "sub_questions": [sq.model_dump() for sq in output.sub_questions],
    })

    raw = str(feedback).strip().lower() if feedback else ""
    plan_feedback = None if raw in _APPROVE_TOKENS else str(feedback).strip()

    return {"research_plan": output.sub_questions, "plan_feedback": plan_feedback}
