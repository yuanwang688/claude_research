from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from ..config import Configuration
from ..state import OverallState, SearchQueriesOutput
from ..utils.structured_output import astructured_output

_QUERY_GEN_PROMPT = """\
You are a web search query generator for a research assistant.

Given a research brief and a list of topics or questions, generate specific, targeted
web search queries optimised for search engines.

Guidelines:
- Use keywords and short phrases, not full sentences
- Vary phrasing to capture different angles on the same topic
- Make each query distinct — do not repeat the same idea twice
- Be specific enough to return useful results
"""


async def query_generator_node(state: OverallState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)

    follow_ups = state.get("follow_up_queries") or []
    if follow_ups:
        context = "Knowledge gaps to address:\n" + "\n".join(f"- {t}" for t in follow_ups)
    else:
        plan = state.get("research_plan") or []
        topics = [sq.question for sq in plan] if plan else [state["original_query"]]
        context = "Research questions:\n" + "\n".join(f"- {t}" for t in topics)

    brief = state.get("research_brief") or state["original_query"]

    output = await astructured_output(
        cfg.fast_llm,
        SearchQueriesOutput,
        [
            SystemMessage(_QUERY_GEN_PROMPT),
            HumanMessage(
                f"Research brief: {brief}\n\n"
                f"{context}\n\n"
                f"Generate up to {cfg.breadth} specific search queries."
            ),
        ],
    )

    return {"follow_up_queries": output.queries}


def dispatch_web_research(state: OverallState) -> list[Send]:
    """Conditional edge: fan-out one Send per search query."""
    queries = (
        state.get("follow_up_queries")
        or [sq.question for sq in (state.get("research_plan") or [])]
        or [state["original_query"]]
    )
    return [
        Send(
            "web_research_node",
            {
                "query": q,
                "query_index": i,
                "research_loop_count": state["research_loop_count"],
            },
        )
        for i, q in enumerate(queries)
    ]
