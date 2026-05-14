from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from ..config import Configuration
from ..prompts import CLARIFIER_PROMPT, SYNTHESIZE_BRIEF_PROMPT
from ..state import ClarifyingQuestionsOutput, OverallState
from ..utils.structured_output import astructured_output


async def _synthesize_brief(
    llm,
    original_query: str,
    questions: list[str],
    answers: dict,
) -> str:
    """Produce a concise research brief from the Q&A exchange."""
    if not questions or not answers:
        return original_query

    qa_text = "\n".join(
        f"Q: {q}\nA: {answers.get(q, '(no answer)')}" for q in questions
    )
    response = await llm.ainvoke([
        SystemMessage(SYNTHESIZE_BRIEF_PROMPT),
        HumanMessage(
            f"Original query: {original_query}\n\n"
            f"Clarification Q&A:\n{qa_text}"
        ),
    ])
    return response.content.strip() or original_query


async def clarifier_node(state: OverallState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    query = state["original_query"]

    output = await astructured_output(
        cfg.powerful_llm,
        ClarifyingQuestionsOutput,
        [
            SystemMessage(CLARIFIER_PROMPT),
            HumanMessage(query),
        ],
    )

    if not output.questions or not cfg.enable_clarification:
        brief = query
    else:
        answers = interrupt({
            "type": "clarification_needed",
            "questions": output.questions,
            "draft_plan": output.draft_research_plan,
            "estimated_scope": output.estimated_scope,
        })
        # answers is a dict mapping question -> answer string
        if not isinstance(answers, dict):
            answers = {}
        brief = await _synthesize_brief(
            cfg.powerful_llm, query, output.questions, answers
        )

    return {"research_brief": brief}
