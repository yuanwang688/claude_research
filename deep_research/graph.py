from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes.clarifier import clarifier_node
from .nodes.planner import planner_node
from .nodes.query_gen import dispatch_web_research, query_generator_node
from .nodes.reflection import reflection_node, route_after_reflection
from .nodes.researcher import web_research_node
from .nodes.writer import writer_node
from .state import OverallState


def _route_after_planner(state: OverallState) -> str:
    """If the user rejected the plan, loop back to clarifier to refine the brief."""
    return "clarifier_node" if state.get("plan_feedback") else "query_generator_node"


def _route_after_writer(state: OverallState) -> str:
    """If the writer captured feedback that implies missing research, loop back."""
    if state.get("user_feedback") and not state.get("final_report"):
        return "query_generator_node"
    return "__end__"


def build_graph(config, llms: dict, search_provider, prompts=None):
    """Compile the full research agent StateGraph."""
    builder = StateGraph(OverallState)

    builder.add_node("clarifier_node", clarifier_node)
    builder.add_node("planner_node", planner_node)
    builder.add_node("query_generator_node", query_generator_node)
    builder.add_node("web_research_node", web_research_node)
    builder.add_node("reflection_node", reflection_node)
    builder.add_node("writer_node", writer_node)

    builder.add_edge(START, "clarifier_node")
    builder.add_edge("clarifier_node", "planner_node")
    builder.add_conditional_edges(
        "planner_node",
        _route_after_planner,
        {
            "clarifier_node": "clarifier_node",
            "query_generator_node": "query_generator_node",
        },
    )
    builder.add_conditional_edges("query_generator_node", dispatch_web_research)
    builder.add_edge("web_research_node", "reflection_node")
    builder.add_conditional_edges(
        "reflection_node",
        route_after_reflection,
        {
            "query_generator_node": "query_generator_node",
            "writer_node": "writer_node",
        },
    )
    builder.add_conditional_edges(
        "writer_node",
        _route_after_writer,
        {
            "query_generator_node": "query_generator_node",
            "__end__": END,
        },
    )

    checkpointer = config.checkpointer or MemorySaver()
    return builder.compile(checkpointer=checkpointer)
