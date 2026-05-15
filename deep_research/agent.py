from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator

from langchain_core.messages import HumanMessage

from .config import Config, Configuration
from .events import (
    AgentEvent,
    ClarificationNeeded,
    Complete,
    DraftReady,
    GapReview,
    PlanReady,
    PlanReview,
    ResearchResult,
    ResearchUpdate,
)
from .graph import build_graph

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from .providers.base import SearchProvider


class PromptOverrides:
    def __init__(self, **kwargs: str):
        self._overrides = kwargs

    def get(self, name: str, default: str) -> str:
        return self._overrides.get(name, default)


class DeepResearchAgent:
    def __init__(
        self,
        search_provider: SearchProvider,
        fast_llm: BaseChatModel | None = None,
        powerful_llm: BaseChatModel | None = None,
        llm: BaseChatModel | None = None,
        config: Config | None = None,
        prompts: PromptOverrides | None = None,
    ):
        if config is None:
            config = Config()

        if llm is not None:
            fast_llm = llm
            powerful_llm = llm

        if fast_llm is None or powerful_llm is None:
            raise ValueError(
                "Provide either 'llm' or both 'fast_llm' and 'powerful_llm'."
            )

        self._config = config
        self._search_provider = search_provider
        self._prompts = prompts

        configuration = Configuration(
            fast_llm=fast_llm,
            powerful_llm=powerful_llm,
            search_provider=search_provider,
            config=config,
        )

        self._graph = build_graph(
            config=config,
            llms={"fast": fast_llm, "powerful": powerful_llm},
            search_provider=search_provider,
            prompts=prompts,
        )
        self._resume_queue: asyncio.Queue = asyncio.Queue()
        self._runnable_config: dict = {
            "configurable": {
                "thread_id": config.thread_id,
                "_configuration": configuration,
            }
        }

    # ---------- Primary async interface ----------

    async def astream(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        """Yield AgentEvent objects. Pauses at HITL gates until resume() is called."""
        from langgraph.types import Command

        initial_state = {
            "original_query": query,
            "messages": [HumanMessage(content=query)],
            "findings": [],
            "sources": {},
            "research_loop_count": 0,
            "max_research_loops": self._config.max_research_loops,
            "is_sufficient": False,
            "knowledge_gaps": [],
            "follow_up_queries": [],
            "draft_report": None,
            "user_feedback": None,
            "final_report": None,
            "research_brief": "",
            "research_plan": [],
            "plan_feedback": None,
        }

        start_time = time.monotonic()
        input_or_command: Any = initial_state

        while True:
            # Track which nodes fired in this graph.astream() pass
            planner_fired = False
            reflection_fired = False

            async for chunk in self._graph.astream(
                input_or_command,
                config=self._runnable_config,
                stream_mode="updates",
            ):
                if "planner_node" in chunk:
                    planner_fired = True
                if "reflection_node" in chunk:
                    reflection_fired = True

            # Single aget_state call after the stream completes
            state = await self._graph.aget_state(config=self._runnable_config)
            values = state.values

            # Emit progress events now that we have the full accumulated state.
            # When plan_review is enabled the plan is shown interactively via PlanReview,
            # so skip the informational PlanReady to avoid duplication.
            if planner_fired and not self._config.enable_plan_review:
                plan = values.get("research_plan", [])
                if plan:
                    yield PlanReady(sub_questions=plan)

            if reflection_fired:
                yield ResearchUpdate(
                    loop_count=values.get("research_loop_count", 0),
                    sources_count=len(values.get("sources", {})),
                    findings_count=len(values.get("findings", [])),
                )

            if state.next:
                # Graph is paused at an interrupt() call.
                interrupt_payload = state.tasks[0].interrupts[0].value
                event = self._interrupt_to_event(interrupt_payload)
                yield event
                # Caller calls resume() before iterating again; queue has value.
                resume_value = await self._resume_queue.get()
                input_or_command = Command(resume=resume_value)
            else:
                # Graph ran to completion.
                final_report = values.get("final_report")
                if final_report:
                    elapsed = time.monotonic() - start_time
                    yield Complete(
                        result=ResearchResult(
                            final_report=final_report,
                            sources=values.get("sources", {}),
                            metadata={
                                "loops_run": values.get("research_loop_count", 0),
                                "total_sources": len(values.get("sources", {})),
                                "elapsed_seconds": round(elapsed, 2),
                            },
                        )
                    )
                break

    async def resume(self, user_input: Any) -> None:
        """Unblock the current HITL gate with user_input."""
        await self._resume_queue.put(user_input)

    # ---------- Sync convenience ----------

    def run(self, query: str, auto_approve: bool = False) -> ResearchResult:
        """Blocking end-to-end run. Use auto_approve=True for headless/batch usage."""
        return asyncio.run(self._run_async(query, auto_approve))

    async def _run_async(self, query: str, auto_approve: bool = False) -> ResearchResult:
        async for event in self.astream(query):
            if isinstance(event, Complete):
                return event.result
            if auto_approve and isinstance(event, (ClarificationNeeded, PlanReview, GapReview, DraftReady)):
                await self.resume("approve")
        raise RuntimeError("Agent completed without producing a result.")

    # ---------- Inspection / persistence ----------

    async def aget_state(self):
        state = await self._graph.aget_state(config=self._runnable_config)
        return state.values

    @property
    def thread_id(self) -> str:
        return self._config.thread_id

    # ---------- Internal helpers ----------

    def _interrupt_to_event(self, payload: dict) -> AgentEvent:
        interrupt_type = payload.get("type", "")

        if interrupt_type == "clarification_needed":
            return ClarificationNeeded(
                questions=payload.get("questions", []),
                draft_plan=payload.get("draft_plan", ""),
                estimated_scope=payload.get("estimated_scope", "medium"),
            )
        if interrupt_type == "gap_review":
            return GapReview(
                gaps=payload.get("gaps", []),
                proposed_queries=payload.get("proposed_queries", []),
                confidence=payload.get("confidence", 0.5),
            )
        if interrupt_type == "plan_review":
            from .state import SubQuestion
            sqs = [SubQuestion(**sq) for sq in payload.get("sub_questions", [])]
            return PlanReview(sub_questions=sqs)

        if interrupt_type == "draft_ready":
            return DraftReady(draft=payload.get("draft", ""))

        # Unknown interrupt — surface raw payload so callers can handle it
        return AgentEvent()
