from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from ..config import Configuration
from ..prompts import (
    CLARIFIER_PROMPT,
    PLAN_REVIEW_CLARIFIER_PROMPT,
    PLAN_REVIEW_SYNTHESIZE_BRIEF_PROMPT,
    SYNTHESIZE_BRIEF_PROMPT,
)
from ..state import ClarifyingQuestionsOutput, OverallState
from ..utils.structured_output import astructured_output


async def _synthesize_brief(
    llm,
    original_query: str,
    questions: list[str],
    answers: dict,
) -> str:
    """Produce a concise research brief from the initial Q&A exchange."""
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


async def _synthesize_brief_from_plan_feedback(
    llm,
    original_query: str,
    current_brief: str,
    plan_feedback: str,
    questions: list[str],
    answers: dict,
) -> str:
    """Update the research brief after the user rejected the initial plan."""
    qa_text = ""
    if questions and answers:
        qa_text = "\n\nFollow-up Q&A:\n" + "\n".join(
            f"Q: {q}\nA: {answers.get(q, '(no answer)')}" for q in questions
        )

    response = await llm.ainvoke([
        SystemMessage(
            PLAN_REVIEW_SYNTHESIZE_BRIEF_PROMPT.format(
                research_brief=current_brief or original_query,
                plan_feedback=plan_feedback,
            )
        ),
        HumanMessage(
            f"Original query: {original_query}"
            + (f"\n\nPrevious brief: {current_brief}" if current_brief else "")
            + f"\n\nPlan feedback: {plan_feedback}"
            + qa_text
        ),
    ])
    return response.content.strip() or current_brief or original_query


async def clarifier_node(state: OverallState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    query = state["original_query"]
    plan_feedback = state.get("plan_feedback")

    if plan_feedback:
        # Re-triggered because user rejected the plan — refine the brief.
        current_brief = state.get("research_brief") or query
        system_prompt = PLAN_REVIEW_CLARIFIER_PROMPT.format(
            research_brief=current_brief,
            plan_feedback=plan_feedback,
        )
        output = await astructured_output(
            cfg.powerful_llm,
            ClarifyingQuestionsOutput,
            [
                SystemMessage(system_prompt),
                HumanMessage(
                    f"Original query: {query}\n\n"
                    f"Current brief: {current_brief}\n\n"
                    f"Plan feedback: {plan_feedback}"
                ),
            ],
        )

        if output.questions and cfg.enable_clarification:
            answers = interrupt({
                "type": "clarification_needed",
                "questions": output.questions,
                "draft_plan": output.draft_research_plan,
                "estimated_scope": output.estimated_scope,
            })
            if not isinstance(answers, dict):
                answers = {}
        else:
            answers = {}

        brief = await _synthesize_brief_from_plan_feedback(
            cfg.powerful_llm, query, current_brief, plan_feedback,
            output.questions, answers,
        )
    else:
        # First run — standard clarification flow.
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
            if not isinstance(answers, dict):
                answers = {}
            brief = await _synthesize_brief(
                cfg.powerful_llm, query, output.questions, answers
            )

    return {"research_brief": brief}
