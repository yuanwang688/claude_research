# MVP Design: Interactive Deep Research Agent

**Target:** Python library (importable, not a service)
**Orchestration:** LangGraph
**LLM:** Provider-agnostic via `BaseChatModel`
**Concurrency:** async-first (`asyncio`) — `astream()` is the primary interface; sync `run()` is a convenience wrapper
**Use case:** General web research

---

## Part 1: Questions and Considerations

These are the design-level questions that must be answered before writing a line of code.
Answering them wrong costs weeks of refactoring.

### Q1: What does "interactive" mean at the API boundary?

A Python library has no HTTP layer, so "ask the user a question and wait for their answer"
must be expressed differently than in a web app. Two options:

**Option A — Async event/resume loop ✅ CHOSEN**
```python
async for event in agent.astream(query):
    if isinstance(event, ClarificationNeeded):
        answers = {q: await async_input(q) for q in event.questions}
        await agent.resume(answers)
```
- Caller drives the loop; agent yields typed `AgentEvent` objects; caller feeds answers back.
- Works in async CLI, Jupyter (with `nest_asyncio`), FastAPI/SSE backends, and async test harnesses.
- LangGraph's `interrupt()` + `Command(resume=value)` maps directly onto this pattern.
- A sync `run()` convenience wrapper uses `asyncio.run()` for scripts that don't need streaming.
- Implemented internally via `asyncio.Queue`: `astream()` awaits the queue at each gate;
  `resume(value)` puts the value in, unblocking the generator. See AD-9 for the sequence diagram.

~~**Option B — Callback registration**~~
~~Rejected: requires threading for async callers, harder to test, harder to adapt to FastAPI.~~

### Q2: How should HITL checkpoints be controlled?

Not every caller wants every checkpoint. A batch script wants `auto_approve=True`.
An end-user wants all three checkpoints. A developer testing the research loop wants
to skip the clarifier but keep the reflection gate.

**Decision:** Each checkpoint is individually toggleable via `Config`. Disabled
checkpoints auto-approve with a sensible default (e.g., approve the plan as-is).

### Q3: How do we stay provider-agnostic without sacrificing structured output?

Structured output (`.with_structured_output(SomeModel)`) is the cleanest way to get
typed node outputs. But behavior varies across providers:
- Anthropic: excellent, uses tool_use under the hood
- OpenAI: excellent with function calling
- Local/open models: hit-or-miss

**Decision:** Wrap structured output calls in a helper that catches parse failures and
retries with an explicit JSON-in-prompt fallback. This keeps node logic clean while
handling provider quirks in one place.

### Q4: How do we handle context growth?

A research session with depth=3, breadth=4 generates ~12 web research calls.
Each returns ~1000 tokens of extracted content. That's 12k tokens of raw findings
before any compression — plus state overhead. This fits in Claude/GPT-4 context windows
today, but:
- Cost scales linearly with every node that reads all findings
- Some local models have 8k-32k context limits

**Decision for MVP:** Store findings as a list of `Finding` objects. Before passing to any
node, compress findings older than the last round using a summarization call.
The `MemoryStore` interface is the extension point: swap the default (in-memory list
+ compression) for a vector store without touching any node logic.

### Q5: How do we prevent citation hallucination?

The Writer node must only cite sources it actually retrieved, not sources the LLM
"knows about" from pretraining.

**Decision:** Maintain a canonical `sources: dict[str, Source]` keyed by URL throughout
the session. The Writer node receives this dict explicitly and is instructed to only use
citation keys from it. The `Source` object carries the raw snippet, so the Writer can
quote it directly rather than paraphrase from memory.

### Q6: What is the unit of parallelism?

The research loop can fan out at two levels:
- **Query-level:** run N search queries concurrently per DAG node
- **DAG-node-level:** run independent DAG nodes concurrently

MVP starts with query-level fan-out (simpler, higher impact). DAG-node-level fan-out
is the "Advanced Pattern" extension described in §9.

### Q7: How do we handle streaming vs. batch results from web search?

Web search APIs (Tavily, etc.) are synchronous HTTP calls. They return quickly.
Full-page scraping (Firecrawl) is slower. The `SearchProvider` protocol returns
`list[SearchResult]` — the provider decides internally whether to scrape.

**Decision:** `SearchProvider` is an async protocol — `async def search()` is the primary
method. Providers use `httpx.AsyncClient` (or equivalent) for non-blocking HTTP. This means:
- Fan-out in `web_research_node` uses `asyncio.gather()` with no thread pool overhead
- Slow scrapers (Firecrawl, full-page) don't block other concurrent research branches
- `MockSearchProvider` uses `asyncio.sleep(0)` to stay async-compatible in tests

Sync providers can be wrapped: `AsyncSearchProvider(sync_provider)` runs the blocking
`search()` in `asyncio.get_event_loop().run_in_executor()`. This wrapper ships in MVP so
third-party sync providers work out of the box.

### Q8: Where does the research brief live?

The research brief is the distilled output of the clarifier Q&A — the internal
specification that all downstream nodes should use as their "north star." It must be:
- Available in every node's prompt
- Immutable after the clarifier sets it (nodes should not change it)
- Human-readable (it will be surfaced at Checkpoint 1)

**Decision:** `research_brief: str` is a top-level field in `OverallState`.
Every prompt template has a `{research_brief}` slot. Nodes cannot write to it
after `clarifier_node` sets it (enforced by convention, not type system — documenting
this is sufficient for a library).

### Q9: What happens when the user rejects the draft report?

After the Writer produces a draft, the user may say "go deeper on section 2" or
"you missed the EU regulatory angle." This is substantively different from the
reflection loop (which addresses gaps in raw research) — it's feedback on the
report structure and emphasis.

**Decision for MVP:** Draft feedback is stored as `user_feedback: str` in state.
The writer re-runs with the feedback injected into its prompt. If the feedback implies
missing research (e.g., "you missed EU regulations"), the writer node sets
`is_sufficient=False` to re-enter the research loop. This is the simplest
correct behavior and is already extensible — a more sophisticated version would
parse the feedback into targeted sub-queries.

### Q10: How do we test this without burning search API credits?

Every run makes real LLM + search API calls. Tests that call real APIs are:
- Slow (10-60 seconds per research loop)
- Expensive
- Non-deterministic (search results change)

**Decision:** The `SearchProvider` protocol enables a `MockSearchProvider` for unit tests.
LLM calls are tested against real models in integration tests, gated behind a flag.
Each node is individually testable by constructing a partial `OverallState` and calling
the node function directly (LangGraph nodes are plain Python functions).

---

## Part 2: Key Architecture Decisions

### AD-1: LangGraph `StateGraph` with `interrupt()`

LangGraph is the orchestration layer. Key features used:
- `StateGraph` with typed `OverallState` — all state in one place, no hidden globals
- `interrupt()` — native suspend/resume for HITL gates
- `Send` API — fan-out parallel web research calls
- `MemorySaver` / `SqliteSaver` — state persistence across interrupts
- Conditional edges — route between research loop and writer

Extension path: LangGraph subgraphs let you swap out entire node clusters (e.g., replace
the single Researcher with a multi-agent Co-STORM cluster) without touching the outer graph.

### AD-2: Two-tier LLM configuration

| Tier | Default nodes | Rationale |
|---|---|---|
| `fast_llm` | query_gen, web_research, reflection | High call frequency; latency matters |
| `powerful_llm` | clarifier, planner, writer | Low call frequency; quality matters |

Both are `BaseChatModel` instances, passed at construction. In the simplest case,
the same model can serve both tiers. Different models can be used to optimize cost/quality.

### AD-3: `SearchProvider` async protocol (not a class)

```python
class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    full_text: str | None = None  # populated by scrapers, None for API-only providers

class SearchProvider(Protocol):
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...
    async def rerank(                                          # optional — providers may no-op
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        return results  # default: identity

class AsyncSearchProvider:
    """Wraps a synchronous SearchProvider for async compatibility."""
    def __init__(self, sync_provider: SyncSearchProvider):
        self._inner = sync_provider

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._inner.search, query, max_results)
```

MVP ships with `TavilySearchProvider` (native async via `httpx`) and
`DuckDuckGoSearchProvider` (wrapped with `AsyncSearchProvider`).
Extension: `FirecrawlSearchProvider`, `InternalDocSearchProvider`, etc.
Composite providers: `HybridSearchProvider([tavily, firecrawl])` runs both concurrently
with `asyncio.gather()` and deduplicates by URL.

### AD-4: Pydantic schemas for all node outputs

Every node returns a Pydantic model, not a raw dict. This gives:
- Type safety at node boundaries
- Easy serialization to JSON (for logging, replay, debugging)
- Schema versioning (add fields with defaults = backward compatible)

All schemas live in `deep_research/schemas.py`. Node logic is in `deep_research/nodes/`.

### AD-5: `OverallState` with typed accumulator fields

```python
class OverallState(TypedDict):
    # Set once
    original_query: str
    research_brief: str
    research_plan: list[SubQuestion]

    # Accumulated across loops (operator.add = list concatenation)
    findings: Annotated[list[Finding], operator.add]
    sources: Annotated[dict[str, Source], merge_dicts]  # dedup by URL

    # Loop control
    research_loop_count: int
    max_research_loops: int
    is_sufficient: bool
    knowledge_gaps: list[str]
    follow_up_queries: list[str]

    # Report lifecycle
    draft_report: str | None
    user_feedback: str | None
    final_report: str | None

    # Conversation (for clarifier)
    messages: Annotated[list[BaseMessage], add_messages]
```

`merge_dicts` is a custom reducer that merges two `dict[str, Source]` dicts,
deduplicating by URL (keeping the higher-scored version on conflict).

### AD-6: Three toggleable HITL checkpoints

| Checkpoint | Trigger | What user sees | Default |
|---|---|---|---|
| CP1: Clarification | After clarifier_node | Questions + draft plan | enabled |
| CP2: Research gap | After reflection when insufficient | Gaps + proposed queries | enabled |
| CP3: Draft review | After writer_node produces draft | Full draft report | enabled |

Each is controlled by a `Config` boolean. When disabled, the checkpoint auto-approves.

### AD-7: Prompt templates as first-class objects

All prompts live in `deep_research/prompts/`. Each is a `PromptTemplate` with named slots.
This makes prompt engineering visible, version-controllable, and overridable by the caller:

```python
agent = DeepResearchAgent(
    ...,
    prompts=PromptOverrides(
        clarifier="You are a research assistant specializing in biomedical literature..."
    )
)
```

Extension path: prompts can be swapped for domain-specific variants without forking the library.

### AD-8: `Config` dataclass as the single knob panel

```python
@dataclass
class Config:
    # Research loop
    max_research_loops: int = 3
    breadth: int = 3               # search queries per loop
    max_results_per_query: int = 5

    # Checkpoints
    enable_clarification: bool = True
    enable_gap_review: bool = True
    enable_draft_review: bool = True

    # Context management
    max_findings_tokens: int = 8000  # compress older findings beyond this

    # Persistence
    thread_id: str = field(default_factory=lambda: str(uuid4()))
    checkpointer: BaseCheckpointSaver | None = None  # None = InMemorySaver
```

### AD-9: `DeepResearchAgent` as the public facade

All internal complexity is hidden behind one class. The caller never imports
LangGraph types, node functions, or state schemas directly.

```python
class DeepResearchAgent:
    def __init__(
        self,
        fast_llm: BaseChatModel,
        powerful_llm: BaseChatModel,
        search_provider: SearchProvider,
        config: Config = Config(),
        prompts: PromptOverrides | None = None,
    ): ...

    # --- Primary async interface ---
    async def astream(self, query: str) -> AsyncIterator[AgentEvent]:
        """Yields AgentEvent objects. Pauses at HITL gates until resume() is called."""
        ...

    async def resume(self, user_input: Any) -> None:
        """Unblocks the current HITL gate with user_input. Must be awaited."""
        ...

    # --- Sync convenience wrappers ---
    def run(self, query: str, auto_approve: bool = False) -> ResearchResult:
        """Blocking end-to-end run. Wraps astream() with asyncio.run()."""
        return asyncio.run(self._run_async(query, auto_approve))

    # --- Inspection ---
    async def aget_state(self) -> OverallState: ...   # async state inspection
    @property
    def thread_id(self) -> str: ...                   # current session ID

    # --- Persistence ---
    async def asave(self, path: str) -> None: ...
    @classmethod
    async def aload(cls, path: str) -> "DeepResearchAgent": ...
```

**Resume coordination mechanism:** `astream()` and `resume()` communicate via an
`asyncio.Queue` owned by the agent instance. When `astream()` hits a LangGraph interrupt
it yields the event and then awaits `self._resume_queue.get()`. Calling `resume(value)`
puts `value` into the queue, unblocking the generator which then calls
`graph.astream(Command(resume=value), config)` to restart graph execution from the
checkpoint. This keeps the caller's `async for` loop simple and linear — no callbacks,
no threads, no separate polling step.

```
caller                     agent.astream()              LangGraph graph
  │                              │                            │
  │── async for event ──────────>│── graph.astream(input) ──>│
  │                              │                            │── interrupt() ──>│
  │<── yield ClarificationNeeded │<── yield interrupt event ──│                  │
  │                              │── await _resume_queue.get()│  (paused)        │
  │── await agent.resume(answers)│                            │                  │
  │                    _resume_queue.put(answers)             │                  │
  │                              │<── got answers             │                  │
  │                              │── graph.astream(Command(resume=answers)) ────>│
  │<── yield PlanReady ──────────│<── yield next event ───────│                  │
```

---

## Part 3: MVP Scope

### In scope

- Clarifier node with structured Q&A (CP1)
- Planner node producing flat list of sub-questions (not full DAG yet)
- Query generator expanding sub-questions into search queries
- Parallel web research via fan-out `Send` pattern
- Reflection node with gap analysis and sufficiency decision (CP2)
- Writer node with parallel section drafting (CP3)
- `TavilySearchProvider` (primary) + `DuckDuckGoSearchProvider` (free fallback)
- Provider-agnostic via `BaseChatModel`
- In-memory state with optional `SqliteSaver` for persistence
- All three HITL checkpoints, individually toggleable
- `auto_approve=True` mode for headless/batch usage
- `MockSearchProvider` + example test suite
- Findings compression (token-budget-based)
- Source deduplication and citation management

### Deliberately out of scope for MVP

| Feature | Why deferred | Extension point |
|---|---|---|
| Full DAG with dependency tracking | Flat list covers 90% of use cases; DAG adds significant planner complexity | `research_plan: ResearchPlan` field is typed to accept either |
| Cross-encoder reranking | Adds model dependency; Tavily reranks internally | `SearchProvider.rerank()` optional method |
| External vector store memory | In-memory + compression is sufficient for <50 sources | Swap `MemoryStore` implementation |
| Multi-perspective question asking (STORM) | Good extension for v2; requires perspective discovery pre-step | `PerspectiveDiscovery` hook before planner |
| RL-based search optimization | Research-stage; requires training infrastructure | Node functions are plain Python, easy to wrap |
| Multi-agent Co-STORM discourse | Complex; better as opt-in mode | LangGraph subgraph replacing `researcher_node` |
| Web scraping / Firecrawl | Tavily returns snippets; full-page scraping is an enhancement | `SearchProvider` already has `full_text` field |
| Streaming token output | Node outputs are complete, not streamed mid-generation | LangGraph supports `.astream_events()` when needed |

---

## Part 4: Directory Structure

```
deep_research/
│
├── __init__.py                 # Public exports: DeepResearchAgent, Config, event types
│
├── agent.py                    # DeepResearchAgent class; compiles and runs the graph
├── graph.py                    # build_graph() — StateGraph definition, edges, checkpointer
├── state.py                    # OverallState TypedDict + all Pydantic schemas
├── config.py                   # Config dataclass
├── events.py                   # AgentEvent union type + all event subtypes
│
├── nodes/
│   ├── __init__.py
│   ├── clarifier.py            # clarifier_node: Q&A → research_brief
│   ├── planner.py              # planner_node: research_brief → sub-questions
│   ├── query_gen.py            # query_generator_node: sub-questions → search queries
│   ├── researcher.py           # web_research_node: query → findings + sources
│   ├── reflection.py           # reflection_node: findings → sufficiency assessment
│   └── writer.py               # writer_node: findings + sources → draft/final report
│
├── providers/
│   ├── __init__.py
│   ├── base.py                 # SearchProvider Protocol + SearchResult model
│   ├── tavily.py               # TavilySearchProvider
│   ├── duckduckgo.py           # DuckDuckGoSearchProvider (free, no API key)
│   └── mock.py                 # MockSearchProvider for testing
│
├── prompts/
│   ├── __init__.py
│   ├── clarifier.py
│   ├── planner.py
│   ├── reflection.py
│   └── writer.py
│
└── utils/
    ├── __init__.py
    ├── compression.py          # findings token-budget compression
    ├── citations.py            # source dedup, citation key generation
    └── structured_output.py    # provider-agnostic structured output helper with retry
```

---

## Part 5: Node Graph

```
START
  │
  ▼
[clarifier_node]
  │  produces: research_brief, messages
  │
  ▼ interrupt(CP1) if enable_clarification
  │  user provides: answers to clarifying questions + optional plan edits
  │
  ▼
[planner_node]
  │  produces: research_plan (list[SubQuestion])
  │
  ▼
[query_generator_node]
  │  produces: search_queries (list[str])
  │
  ▼ Send fan-out (one per query)
  │
  ├─ [web_research_node] ─┐
  ├─ [web_research_node] ─┤  each produces: Finding + Source entries
  └─ [web_research_node] ─┘
                          │
                    (fan-in via Annotated[list, operator.add])
                          │
                          ▼
                  [reflection_node]
                          │  produces: is_sufficient, knowledge_gaps, follow_up_queries
                          │
                          ├─ is_sufficient=True ────────────────────────────────┐
                          │                                                     │
                          ├─ loop_count >= max ─────────────────────────────────┤
                          │                                                     │
                          └─ is_sufficient=False                                │
                                    │                                           │
                          interrupt(CP2) if enable_gap_review                   │
                                    │  user: approve gaps / edit queries        │
                                    │                                           │
                                    ▼                                           │
                          [query_generator_node]  ← loop back with gaps         │
                                                                                │
                                                                                ▼
                                                                       [writer_node]
                                                                                │
                                                                       interrupt(CP3) if enable_draft_review
                                                                                │  user: approve / give feedback
                                                                                │
                                                            ┌───────────────────┴──────────────────────┐
                                                            │ feedback="approve"                       │ feedback=<text>
                                                            ▼                                          ▼
                                                          END                           set is_sufficient=False
                                                     (final_report)                    set user_feedback=<text>
                                                                                       → [query_generator_node]
```

---

## Part 6: Public API Design

### Construction

```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from deep_research import DeepResearchAgent, Config
from deep_research.providers import TavilySearchProvider

# Minimal — same model for both tiers
agent = DeepResearchAgent(
    llm=ChatAnthropic(model="claude-opus-4-7"),
    search_provider=TavilySearchProvider(api_key="..."),
)

# Full — two-tier models, custom config
agent = DeepResearchAgent(
    fast_llm=ChatAnthropic(model="claude-haiku-4-5-20251001"),
    powerful_llm=ChatAnthropic(model="claude-opus-4-7"),
    search_provider=TavilySearchProvider(api_key="..."),
    config=Config(
        max_research_loops=3,
        breadth=3,
        enable_clarification=True,
        enable_gap_review=True,
        enable_draft_review=False,  # skip draft review, auto-publish
    ),
)
```

### Interactive async streaming (primary interface)

```python
import asyncio
from deep_research.events import (
    ClarificationNeeded, PlanReady, ResearchUpdate,
    GapReview, DraftReady, Complete
)

async def main():
    async for event in agent.astream("What are the economic impacts of AI on the labor market?"):
        match event:
            case ClarificationNeeded(questions=qs, draft_plan=plan):
                print(f"Draft plan:\n{plan}\n")
                answers = {}
                for q in qs:
                    answers[q] = input(f"  {q}\n  > ")   # input() is fine in CLI
                await agent.resume(answers)

            case PlanReady(sub_questions=sqs):
                print(f"Research plan: {len(sqs)} sub-questions")

            case ResearchUpdate(loop_count=n, sources_count=s):
                print(f"Loop {n}: {s} sources indexed so far")

            case GapReview(gaps=gaps, proposed_queries=pqs):
                print(f"Gaps identified: {gaps}")
                feedback = input("Approve follow-up queries? (yes/edit/stop): ")
                await agent.resume(feedback)

            case DraftReady(draft=d):
                print(d)
                feedback = input("Feedback (or 'approve'): ")
                await agent.resume(feedback)

            case Complete(result=r):
                print(r.final_report)
                print(f"\nSources ({len(r.sources)}):")
                for url in r.sources:
                    print(f"  {url}")

asyncio.run(main())
```

### FastAPI / SSE backend (zero adaptation needed)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from deep_research.events import AgentEvent
import json

app = FastAPI()

@app.post("/research/start")
async def start_research(query: str):
    agent = DeepResearchAgent(...)

    async def event_stream():
        async for event in agent.astream(query):
            yield f"data: {json.dumps(event.model_dump())}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/research/{thread_id}/resume")
async def resume_research(thread_id: str, user_input: dict):
    agent = await DeepResearchAgent.aload(thread_id)
    await agent.resume(user_input)
    return {"status": "resumed"}
```

### Headless / batch usage (sync convenience)

```python
# Fully automated — all checkpoints auto-approved, blocks until complete
result = agent.run(
    query="Summarize recent advances in protein folding prediction",
    auto_approve=True,
)
print(result.final_report)
print(result.sources)       # dict[url, Source]
print(result.metadata)      # loops_run, total_sources, elapsed_seconds, etc.
```

### Jupyter notebook usage

```python
# nest_asyncio lets asyncio.run() work inside a running event loop (Jupyter)
import nest_asyncio
nest_asyncio.apply()

result = agent.run("What caused the 2008 financial crisis?", auto_approve=True)
print(result.final_report)
```

### Session persistence

```python
from deep_research import Config
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def main():
    checkpointer = AsyncSqliteSaver.from_conn_string("./research_sessions.db")
    config = Config(
        checkpointer=checkpointer,
        thread_id="session-42",   # stable ID → session resumes from last checkpoint
    )
    agent = DeepResearchAgent(..., config=config)
    # ... astream as normal; state is persisted after every node
```

---

## Part 7: State Schema and Pydantic Models

```python
# state.py

class SubQuestion(BaseModel):
    id: str
    question: str
    evidence_type: Literal["statistical", "qualitative", "comparative", "technical", "other"]
    depends_on: list[str] = []     # IDs of questions this depends on — unused in MVP, ready for DAG

class Source(BaseModel):
    url: str
    title: str
    snippet: str
    full_text: str | None = None
    relevance_score: float = 0.0
    accessed_at: datetime = Field(default_factory=datetime.utcnow)

class Finding(BaseModel):
    query: str
    sub_question_id: str | None
    summary: str
    source_urls: list[str]         # keys into OverallState.sources
    loop_number: int
    token_count: int               # for compression budget tracking

class ClarifyingQuestionsOutput(BaseModel):
    questions: list[str]           # 0-5 items
    draft_research_plan: str
    estimated_scope: Literal["narrow", "medium", "broad"]

class ResearchPlanOutput(BaseModel):
    sub_questions: list[SubQuestion]
    research_brief_confirmed: str  # planner restates the brief to confirm understanding

class SearchQueriesOutput(BaseModel):
    queries: list[str]             # breadth * len(sub_questions), typically 3-15

class ReflectionOutput(BaseModel):
    is_sufficient: bool
    confidence: float              # 0.0 - 1.0
    covered_topics: list[str]
    missing_topics: list[str]
    follow_up_queries: list[str]

class OverallState(TypedDict):
    # Input
    original_query: str
    messages: Annotated[list[BaseMessage], add_messages]

    # Set by clarifier (immutable after)
    research_brief: str

    # Set by planner (immutable after)
    research_plan: list[SubQuestion]

    # Accumulated (reducer: list append / dict merge)
    findings: Annotated[list[Finding], operator.add]
    sources: Annotated[dict[str, Source], merge_source_dicts]

    # Loop control
    research_loop_count: int
    max_research_loops: int
    is_sufficient: bool
    knowledge_gaps: list[str]
    follow_up_queries: list[str]

    # Report
    draft_report: str | None
    user_feedback: str | None
    final_report: str | None
```

---

## Part 8: Key Node Skeletons

All node functions are `async def`. LangGraph handles async nodes transparently — no
special registration needed. LLM calls use `await llm.ainvoke()`, search calls use
`await provider.search()`.

### clarifier_node

```python
async def clarifier_node(state: OverallState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    query = state["messages"][-1].content

    output = await astructured_output(          # async variant of the structured-output helper
        cfg.powerful_llm,
        ClarifyingQuestionsOutput,
        [SystemMessage(CLARIFIER_PROMPT), HumanMessage(query)],
    )

    if not output.questions or not cfg.enable_clarification:
        brief = await asynthesize_brief(query, [], {})
    else:
        answers = interrupt({                   # interrupt() is sync even in async nodes
            "type": "clarification_needed",     # LangGraph handles the suspend internally
            "questions": output.questions,
            "draft_plan": output.draft_research_plan,
            "estimated_scope": output.estimated_scope,
        })
        brief = await asynthesize_brief(query, output.questions, answers)

    return {"research_brief": brief}
```

### web_research_node (async I/O — the main speed win)

```python
async def web_research_node(state: WebResearchState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    query = state["query"]

    # Search and summarize run concurrently if provider supports it
    results = await cfg.search_provider.search(query, max_results=cfg.max_results_per_query)

    # Summarize results into a Finding (async LLM call)
    summary = await cfg.fast_llm.ainvoke([
        SystemMessage(SUMMARIZE_PROMPT),
        HumanMessage(f"Query: {query}\n\nResults:\n" + format_results(results)),
    ])

    finding = Finding(
        query=query,
        summary=summary.content,
        source_urls=[r.url for r in results],
        loop_number=state["research_loop_count"],
        token_count=count_tokens(summary.content),
    )
    sources = {r.url: Source(url=r.url, title=r.title, snippet=r.snippet) for r in results}

    return {"findings": [finding], "sources": sources}
```

Because each `web_research_node` invocation is `async`, LangGraph's `Send` fan-out runs
all N instances as concurrent coroutines — no thread pool, no `concurrent.futures`.
For `breadth=4`, all 4 search + summarize calls overlap naturally.

### reflection_node

```python
async def reflection_node(state: OverallState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    compressed = compress_findings(state["findings"], cfg.max_findings_tokens)

    output = await astructured_output(
        cfg.fast_llm,
        ReflectionOutput,
        [SystemMessage(REFLECTION_PROMPT), HumanMessage(f"""
Research Brief: {state['research_brief']}
Findings: {compressed}
Loop: {state['research_loop_count']} of {state['max_research_loops']}
""")],
    )

    if not output.is_sufficient and state["research_loop_count"] < state["max_research_loops"]:
        if cfg.enable_gap_review:
            user_response = interrupt({
                "type": "gap_review",
                "gaps": output.missing_topics,
                "proposed_queries": output.follow_up_queries,
                "confidence": output.confidence,
            })
            follow_up = parse_gap_response(user_response, output.follow_up_queries)
        else:
            follow_up = output.follow_up_queries
    else:
        follow_up = []

    return {
        "is_sufficient": output.is_sufficient,
        "knowledge_gaps": output.missing_topics,
        "follow_up_queries": follow_up,
        "research_loop_count": state["research_loop_count"] + 1,
    }
```

### writer_node (parallel section generation via asyncio.gather)

```python
async def writer_node(state: OverallState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    compressed = compress_findings(state["findings"], cfg.max_findings_tokens)

    # 1. Meta-analysis — identify report sections
    themes_output = await astructured_output(
        cfg.powerful_llm, ReportOutline,
        [SystemMessage(META_ANALYSIS_PROMPT), HumanMessage(compressed)],
    )

    # 2. Parallel section drafting — asyncio.gather replaces thread pool
    section_responses = await asyncio.gather(*[
        cfg.powerful_llm.ainvoke([
            SystemMessage(SECTION_PROMPT),
            HumanMessage(f"Section: {theme}\nResearch: {relevant_findings(theme, state)}"),
        ])
        for theme in themes_output.sections
    ])

    # 3. Assemble + cite
    draft = assemble_report(
        themes_output.sections,
        [r.content for r in section_responses],
        state["sources"],
    )

    if cfg.enable_draft_review:
        feedback = interrupt({"type": "draft_ready", "draft": draft})
        if feedback and feedback.strip().lower() != "approve":
            return {"user_feedback": feedback, "is_sufficient": False, "draft_report": draft}

    return {"final_report": draft, "draft_report": draft}
```

### Routing function (conditional edge — stays sync, pure logic)

```python
def route_after_reflection(state: OverallState) -> Literal["query_generator", "writer"]:
    loop_exhausted = state["research_loop_count"] >= state["max_research_loops"]
    if state["is_sufficient"] or loop_exhausted:
        return "writer"
    return "query_generator"
```

### Fan-out dispatcher (stays sync — returns Send objects, not coroutines)

```python
def dispatch_web_research(state: OverallState) -> list[Send]:
    queries = state["follow_up_queries"] or expand_plan_to_queries(state["research_plan"])
    return [
        Send("web_research_node", {"query": q, "query_index": i,
                                   "research_loop_count": state["research_loop_count"]})
        for i, q in enumerate(queries)
    ]
```

---

## Part 9: Extension Points to Advanced Techniques

Every extension point from the research report maps onto a clean seam in this design.

| Advanced Feature | Extension Point | What Changes |
|---|---|---|
| **DAG-based planning** | `SubQuestion.depends_on` field is already present; planner generates a flat list in MVP | Planner prompt updated to produce dependency edges; graph adds topological traversal logic before fan-out |
| **Multi-perspective questioning (STORM)** | Add `perspective_discovery_node` before `planner_node` in graph | New node; no changes to existing nodes |
| **Cross-encoder reranking** | `SearchProvider` has optional `rerank()` method | Implement in `TavilySearchProvider` or wrap any provider with `RerankingSearchProvider` |
| **External vector store memory** | Replace `compress_findings()` in `utils/compression.py` with vector store read/write | Node logic unchanged; swap utility function |
| **Co-STORM discourse mode** | Replace `researcher_node` + `reflection_node` with a LangGraph subgraph | Outer graph unchanged; inner cluster replaced |
| **RL search optimization** | Node functions are plain Python callables | Wrap node functions with RL policy; reward signal from sufficiency + user approval |
| **Full-page scraping** | `SearchResult.full_text` field is already present; `TavilySearchProvider` populates it | Add `FirecrawlSearchProvider` that populates `full_text` |
| **Feedback-as-training signal** | Log all `interrupt()` payloads + user responses already in state | Add a `TrainingLogger` that writes (interrupt_payload, user_response) pairs to a dataset |
| **Streaming tokens to caller** | MVP yields complete node outputs; token streaming requires `graph.astream_events()` with `kind="on_chat_model_stream"` filter | Only `agent.py` changes; node functions and public event types are stable |
| **True async search fan-out at DAG level** | MVP fans out at query level; DAG node fan-out needs topological scheduler | Add scheduler before `dispatch_web_research`; all nodes stay `async def` |
| **Domain-specific prompts** | `PromptOverrides` accepted at construction | Swap prompt strings; node logic unchanged |
| **Multi-user / production** | Swap `InMemorySaver` for `PostgresSaver` | Only `Config.checkpointer` changes |

---

## Part 10: Implementation Order

Build and validate each layer before the next. Each step is independently runnable.
All node functions are `async def` from Step 3 onward — don't write sync nodes and convert later.

**Step 1 — Scaffold + state**
- Directory structure, `state.py`, `config.py`, `events.py`
- `DeepResearchAgent.__init__` and `build_graph()` returning an empty graph
- `AsyncSearchProvider` wrapper for sync providers
- Goal: `import deep_research` works; `asyncio.run(agent.astream("test"))` yields nothing but doesn't crash

**Step 2 — Async search provider**
- `SearchProvider` async protocol + `AsyncMockSearchProvider` (uses `asyncio.sleep(0)`)
- `DuckDuckGoSearchProvider` wrapped with `AsyncSearchProvider`
- Test: `asyncio.run(provider.search("test"))` returns `list[SearchResult]`

**Step 3 — Researcher node (the core async loop)**
- `async def web_research_node` — `await provider.search()` + `await llm.ainvoke()`
- `query_generator_node` (can be sync or async)
- `dispatch_web_research` fan-out via `Send` — LangGraph runs each as a concurrent coroutine
- Test: `asyncio.run(graph.ainvoke(seed_state))` → inspect `state["findings"]`; confirm N coroutines ran concurrently

**Step 4 — Reflection + loop control**
- `async def reflection_node` with `await astructured_output(...)`
- `route_after_reflection` (sync, pure logic)
- `max_research_loops` cap
- Test: agent runs 1–3 loops and terminates correctly

**Step 5 — Writer node**
- `async def writer_node` with `asyncio.gather()` parallel section drafting
- Source citation from `state["sources"]`
- Test: agent produces a coherent Markdown report with valid citation keys

**Step 6 — Clarifier + planner**
- `async def clarifier_node` and `async def planner_node`
- Full pipeline runs end-to-end, all `auto_approve=True` (no HITL yet)
- Test: `asyncio.run(agent.run("test query", auto_approve=True))` returns `ResearchResult`

**Step 7 — HITL + `astream()` + `resume()`**
- `interrupt()` calls in clarifier, reflection, writer
- `agent.astream()` async generator with `asyncio.Queue` resume coordination
- `async def agent.resume(value)` puts value into queue
- Test: async CLI demo — `async for event in agent.astream(q)` with manual resume calls

**Step 8 — `TavilySearchProvider` + polish**
- Native async Tavily via `httpx.AsyncClient`
- Findings compression; source deduplication; `AsyncSqliteSaver` persistence
- `run()` sync wrapper using `asyncio.run()`
- Test: full end-to-end with real Tavily + real LLM; measure concurrent search speedup vs sequential

---

## Part 11: Open Questions Before Implementation

These require decisions but don't block scaffold work:

1. ~~**Synchronous or async as primary?**~~ **Resolved: async-first.** `astream()` is
   primary; `run()` wraps with `asyncio.run()`. Jupyter users add `nest_asyncio.apply()`.

2. **Which free search provider for development?** DuckDuckGo has no API key requirement
   but rate-limits. Tavily has a free tier (1000 calls/month). Recommendation: ship both,
   default to DuckDuckGo in tests, Tavily in production.

3. **How many clarifying questions is the right default?** The report says 2-5. Empirically,
   3 is the sweet spot: enough to disambiguate, not enough to annoy. The LLM should be
   instructed to ask 0 if the query is already specific.

4. **Should the planner be skippable?** For short/simple queries, a planner that decomposes
   into 5 sub-questions is overkill. A heuristic: if `estimated_scope == "narrow"` from the
   clarifier, skip the planner and go straight to query generation. Add this routing later,
   not in the first pass.

5. **Compression strategy for findings:** Token-budget approach (compress once findings exceed
   N tokens) vs. round-based (always compress after each loop). Round-based is simpler and
   more predictable. Use round-based in MVP.

6. **Async event loop ownership:** `asyncio.run()` in `run()` creates and closes a new event
   loop each call. If the caller is already running an event loop (FastAPI, Jupyter after
   `nest_asyncio`), calling `run()` will raise `RuntimeError: This event loop is already running`.
   Document clearly: in async contexts, use `await agent._run_async()` directly; `run()` is only
   for top-level scripts. A `nest_asyncio`-compatible helper can be added if there is demand.

7. **Concurrency limit for fan-out:** With `breadth=5` and a slow LLM provider, 5 concurrent
   `ainvoke` calls may saturate rate limits. Add `Config.max_concurrent_searches: int = 5` and
   wrap fan-out with `asyncio.Semaphore(cfg.max_concurrent_searches)` in the researcher node.
   Default of 5 matches Tavily's free-tier rate limits.
