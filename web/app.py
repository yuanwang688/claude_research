from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from deep_research import (
    ClarificationNeeded,
    Complete,
    Config,
    DeepResearchAgent,
    DraftReady,
    GapReview,
    PlanReady,
)
from deep_research.events import ResearchUpdate
from deep_research.providers.duckduckgo import DuckDuckGoSearchProvider

app = FastAPI(title="Deep Research Agent")

# In-memory job store: job_id -> {agent, queue, task}
_jobs: dict[str, dict] = {}

_STATIC = Path(__file__).parent / "static"


# ---------- Request models ----------

class ResearchRequest(BaseModel):
    query: str
    api_key: str
    provider: str = "anthropic"
    fast_model: str = "claude-haiku-4-5-20251001"
    powerful_model: str = "claude-sonnet-4-6"
    max_loops: int = 2
    breadth: int = 3
    enable_clarification: bool = False
    enable_gap_review: bool = False
    enable_draft_review: bool = False
    search_provider: str = "duckduckgo"
    tavily_api_key: str = ""


class ResumeRequest(BaseModel):
    value: Any


# ---------- Helpers ----------

def _make_agent(req: ResearchRequest) -> DeepResearchAgent:
    if req.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        fast_llm = ChatAnthropic(model=req.fast_model, api_key=req.api_key)
        powerful_llm = ChatAnthropic(model=req.powerful_model, api_key=req.api_key)
    else:
        from langchain_openai import ChatOpenAI
        fast_llm = ChatOpenAI(model=req.fast_model, api_key=req.api_key)
        powerful_llm = ChatOpenAI(model=req.powerful_model, api_key=req.api_key)

    if req.search_provider == "tavily" and req.tavily_api_key:
        from deep_research.providers.tavily import TavilySearchProvider
        search = TavilySearchProvider(api_key=req.tavily_api_key)
    else:
        search = DuckDuckGoSearchProvider()

    return DeepResearchAgent(
        fast_llm=fast_llm,
        powerful_llm=powerful_llm,
        search_provider=search,
        config=Config(
            max_research_loops=req.max_loops,
            breadth=req.breadth,
            enable_clarification=req.enable_clarification,
            enable_gap_review=req.enable_gap_review,
            enable_draft_review=req.enable_draft_review,
        ),
    )


async def _run_job(job_id: str, query: str) -> None:
    job = _jobs[job_id]
    queue: asyncio.Queue = job["queue"]
    agent: DeepResearchAgent = job["agent"]

    try:
        async for event in agent.astream(query):
            if isinstance(event, PlanReady):
                await queue.put({
                    "type": "plan_ready",
                    "sub_questions": [
                        {"id": sq.id, "question": sq.question}
                        for sq in event.sub_questions
                    ],
                })
            elif isinstance(event, ResearchUpdate):
                await queue.put({
                    "type": "research_update",
                    "loop_count": event.loop_count,
                    "sources_count": event.sources_count,
                    "findings_count": event.findings_count,
                })
            elif isinstance(event, ClarificationNeeded):
                await queue.put({
                    "type": "clarification_needed",
                    "questions": event.questions,
                    "draft_plan": event.draft_plan,
                    "estimated_scope": event.estimated_scope,
                })
            elif isinstance(event, GapReview):
                await queue.put({
                    "type": "gap_review",
                    "gaps": event.gaps,
                    "proposed_queries": event.proposed_queries,
                    "confidence": event.confidence,
                })
            elif isinstance(event, DraftReady):
                await queue.put({"type": "draft_ready", "draft": event.draft})
            elif isinstance(event, Complete):
                await queue.put({
                    "type": "complete",
                    "final_report": event.result.final_report,
                    "sources": {
                        url: {
                            "title": src.title,
                            "url": src.url,
                            "snippet": src.snippet,
                            "relevance_score": src.relevance_score,
                        }
                        for url, src in event.result.sources.items()
                    },
                    "metadata": event.result.metadata,
                })
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        await queue.put({"type": "error", "message": str(exc)})
    finally:
        await queue.put(None)  # end-of-stream sentinel


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((_STATIC / "index.html").read_text())


@app.post("/api/research")
async def start_research(req: ResearchRequest) -> dict:
    job_id = str(uuid.uuid4())
    agent = _make_agent(req)
    queue: asyncio.Queue = asyncio.Queue()
    _jobs[job_id] = {"agent": agent, "queue": queue, "task": None}
    task = asyncio.create_task(_run_job(job_id, req.query))
    _jobs[job_id]["task"] = task
    return {"job_id": job_id}


@app.get("/api/research/{job_id}/events")
async def research_events(job_id: str, request: Request) -> StreamingResponse:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    queue: asyncio.Queue = _jobs[job_id]["queue"]

    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    yield "event: ping\ndata: {}\n\n"
                    continue
                if event is None:
                    yield "event: done\ndata: {}\n\n"
                    break
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            task = _jobs[job_id].get("task")
            if task and not task.done():
                task.cancel()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/research/{job_id}/resume")
async def resume_research(job_id: str, body: ResumeRequest) -> dict:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    await _jobs[job_id]["agent"].resume(body.value)
    return {"ok": True}
