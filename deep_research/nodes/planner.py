from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import Configuration
from ..prompts import PLANNER_PROMPT
from ..state import OverallState, ResearchPlanOutput
from ..utils.structured_output import astructured_output


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

    return {"research_plan": output.sub_questions}
