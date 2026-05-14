# Research Report: Building an Interactive Deep Research Agent

**Date:** May 13, 2026
**Status:** Complete

---

## 1. Executive Summary

An **interactive deep research agent** is an AI system that autonomously conducts multi-step web research, produces a structured report, and engages in a bidirectional dialogue with the user — asking clarifying questions before and during research, and iterating on both the search strategy and the report draft based on user feedback. This pattern has matured rapidly from a niche research prototype (2023) into a production engineering discipline with well-understood architectural primitives (2025–2026).

This report synthesizes academic papers, open-source implementations, and practitioner guides to produce a concrete blueprint for building one.

---

## 2. Taxonomy of Approaches

A 2025 survey ("Deep Research Agents: A Systematic Examination And Roadmap") classifies agents across three axes:

### 2.1 User Intent Handling (Pre-Research)

| Strategy | Description | Examples |
|---|---|---|
| **Planning-only** | Directly generates a research plan with no clarification | Grok, H2O, Manus |
| **Intent-to-Planning** | Actively asks clarifying questions *before* planning | OpenAI Deep Research |
| **Unified Intent-Planning** | Generates a draft plan *and* asks the user to confirm/revise it | Gemini Deep Research |

For an interactive agent, **Unified Intent-Planning** is generally the best UX: the user sees what the agent intends, can redirect it before expensive research begins, and gets a sense of scope.

### 2.2 Workflow Type

| Type | Description |
|---|---|
| **Static** | Fixed pipeline (plan → search → write). Efficient, predictable, but brittle for open-ended queries. |
| **Dynamic** | Adaptive DAG or ReAct loop. The agent replans mid-flight based on what it finds. |

Most high-quality 2025–2026 systems use a **dynamic workflow** — the research plan is a living artifact.

### 2.3 Agent Composition

| Type | Trade-off |
|---|---|
| **Single-agent** | Simpler, end-to-end RL-trainable, lower coordination overhead |
| **Multi-agent** | Specialization (planner/researcher/writer), parallelism, better for large research tasks |

---

## 3. Core Architecture

The canonical architecture has **six layers**:

```
┌──────────────────────────────────────────────────────────┐
│                    Orchestrator / Graph                   │
│  (LangGraph StateGraph or equivalent)                    │
├──────────────────────────────────────────────────────────┤
│  [1] Clarifier    [2] Planner    [3] Researcher           │
│  Intent Q&A       DAG / outline  Fan-out search + RAG    │
├──────────────────────────────────────────────────────────┤
│  [4] Reflection   [5] HITL Gate  [6] Writer               │
│  Gap analysis     Interrupt()    Report synthesis         │
└──────────────────────────────────────────────────────────┘
         ▲                    │
         └────── user input ──┘
```

### 3.1 The Clarifier Node

Before any research begins, the agent asks 2–5 targeted questions to disambiguate intent. OpenAI's internal guidance notes that users "rarely provide sufficient context in a research request." The clarifier's job is to produce a **research brief** — a precise internal specification used as the north star throughout the pipeline.

**Key patterns:**
- Use structured output (`llm.with_structured_output(ClarifyingQuestions)`) to enumerate exactly N questions, not open-ended chat.
- Show the user a **draft research plan alongside the questions** (Unified Intent-Planning), so they understand what they're clarifying.
- Compress the Q&A into a dense research brief that travels through downstream state — don't pass raw dialogue.

### 3.2 The Planner Node

The planner converts the research brief into a **directed acyclic graph (DAG)** of research sub-questions. Each node in the DAG is an independent, scoped question; edges encode dependencies (i.e., "answering Q3 requires the answer to Q1").

**Key patterns:**
- Generate 5–15 sub-questions for a typical research task.
- Tag each with an estimated "evidence type" (statistical, qualitative, comparative, etc.) to guide searcher behavior.
- Pause and surface the plan to the user as **Checkpoint 1** (described in §4).

### 3.3 The Researcher Node (Iterative Search Loop)

This is the core loop. For each DAG node, one or more Researcher instances:

1. **Formulate queries** — decompose the sub-question into 2–4 search queries using keyword extraction + prior findings.
2. **Execute searches** — fan-out to web search APIs (Tavily, Google Search, Bing), scraping APIs (Firecrawl), or both.
3. **Rerank and chunk results** — cross-encoder reranking gives more accurate semantic relevance than vector similarity alone.
4. **Extract learnings** — summarize and cite what was found; flag gaps.
5. **Recurse** — if depth remains (configurable) and gaps exist, generate follow-up queries and loop.

This is a **ReAct-style loop**: Reason → Act (search) → Observe (results) → Reason again.

**Key parameters:**
- `breadth`: number of parallel queries per iteration (typically 3–5)
- `depth`: number of recursive research cycles (typically 2–4)
- `max_research_loops`: hard cap on total iterations

### 3.4 The Reflection / Gap-Analysis Node

After each research pass, a Reflection node assesses whether the accumulated knowledge is sufficient to answer the research brief.

**Output schema:**
```python
class ReflectionOutput(BaseModel):
    is_sufficient: bool
    knowledge_gaps: list[str]
    follow_up_queries: list[str]
    confidence_score: float
```

If `is_sufficient=False`, the follow-up queries are routed back to the Researcher with gap context. This is **Checkpoint 2** in the HITL design (see §4).

### 3.5 The Writer Node

Once research is sufficient (or the loop cap is hit), a Writer agent synthesizes findings into the final report:

1. **Meta-analysis** — identify overarching themes across all sub-question answers.
2. **Parallel section writing** — dispatch one LLM call per section (divide-and-conquer avoids single-prompt token limits).
3. **Assembly** — stitch sections in order, add citations from `sources_gathered`.
4. **Optional:** generate an executive summary + draft follow-up questions.

**Implementation note:** Use a more capable/expensive model for the Writer than for Researchers — the quality investment pays off at the synthesis step, not the search step.

---

## 4. Human-in-the-Loop (HITL) Design

The interactive loop is the distinguishing feature. The 2026 consensus has shifted toward **"human-above-the-loop"** — the human sets direction at key gates while the agent handles execution autonomously between gates.

### 4.1 LangGraph Interrupt Pattern

LangGraph's `interrupt()` function is the standard primitive for HITL in Python. It pauses graph execution at a node, surfaces a payload to the user, and resumes with their input when provided.

```python
from langgraph.types import interrupt

def checkpoint_node(state: OverallState):
    user_input = interrupt({
        "type": "plan_review",
        "research_plan": state["research_plan"],
        "clarifying_questions": state["open_questions"],
    })
    return {"user_feedback": user_input, "plan_approved": True}
```

State is persisted via a `checkpointer` (e.g., `InMemorySaver` or PostgreSQL-backed for production), keyed by `thread_id`. This allows async interrupts — the user doesn't have to respond immediately.

### 4.2 The Two Natural Checkpoints

**Checkpoint 1 — Before Research Begins**
- Surface the research plan (DAG) and any remaining clarifying questions.
- User can: approve, modify scope (add/remove sub-questions), change emphasis, add constraints ("focus on US market only").
- Trigger: after Planner completes, before first Researcher fan-out.

**Checkpoint 2 — After Each Research Round**
- Surface the reflection assessment: what was found, what gaps remain, proposed follow-up queries.
- User can: declare research sufficient, approve/edit follow-up queries, add new angles they thought of.
- Trigger: after Reflection node, when `is_sufficient=False`.

**Optional Checkpoint 3 — After Draft Report**
- Surface the draft report sections.
- User can: request deeper coverage on section X, flag factual concerns, ask for a different angle, request more/fewer citations.
- Trigger: after Writer's first draft, before final output.

### 4.3 Co-STORM's Collaborative Discourse Model

Stanford's **Co-STORM** (EMNLP 2024) offers a richer but more complex alternative: rather than discrete checkpoint gates, the user *participates in an ongoing multi-agent discourse*.

Architecture:
- **CoStormExperts**: Domain specialist agents that answer questions and raise follow-ups.
- **Moderator**: Generates "thought-provoking questions inspired by information discovered but not yet addressed," preventing conversational stagnation.
- **Human**: Observes the agent discourse and can inject utterances at any turn to steer the direction.
- **Dynamic Mind Map**: A hierarchical knowledge tree updated in real-time using LLM-based insertion + vector similarity matching. Nodes auto-expand when crowded. This is the shared conceptual space between human and system.

**Result:** 70% of participants preferred Co-STORM over search engines; 78% over a RAG chatbot (EMNLP 2024 evaluation).

**Trade-off:** Higher engagement but also higher cognitive load. Better for exploratory research where the user doesn't know what they don't know ("unknown unknowns"). The checkpoint-gate pattern is better when the user has a clear research goal.

---

## 5. State Management and Memory

Long research sessions produce more information than fits in a context window. Three strategies:

| Strategy | When to Use |
|---|---|
| **Context window** | Short research tasks (<10 searches). Pass everything. |
| **Intermediate compression** | Medium tasks. Summarize/chunk findings before passing to the next node. |
| **External structured store** | Long tasks, Co-STORM-style. Persist to a knowledge base (vector DB or mind map). Retrieve relevant chunks per node. |

**LangGraph State Schema Pattern:**

```python
class OverallState(TypedDict):
    messages: Annotated[list, add_messages]              # conversation history
    research_brief: str                                  # compressed from clarifier Q&A
    research_plan: ResearchPlan                          # DAG of sub-questions
    search_queries: Annotated[list, operator.add]        # cumulative
    web_research_results: Annotated[list, operator.add]  # cumulative
    sources_gathered: Annotated[list, operator.add]      # citations
    research_loop_count: int
    max_research_loops: int
    is_sufficient: bool
    knowledge_gaps: list[str]
    draft_report: str | None
    user_feedback: str | None
```

Use `Annotated[list, operator.add]` for fields that accumulate across parallel branches (fan-out/fan-in pattern).

---

## 6. Key Open-Source Implementations

### 6.1 `dzhng/deep-research` (TypeScript, ~500 LOC)
The simplest credible implementation. Upfront clarifying questions → breadth/depth-parameterized recursive search loop → markdown report. No orchestration framework — pure recursion. Good for understanding the core pattern without framework overhead.

### 6.2 `langchain-ai/open_deep_research` (Python, LangGraph)
Production-quality LangGraph implementation. Ranked #6 on Deep Research Bench (score 0.4344). Supports Tavily, Anthropic native search, OpenAI native search, and MCP-compatible tools. Multi-model pipeline (different models for summarization, research, and report generation). Reference implementation for the LangGraph HITL pattern.

### 6.3 `assafelovic/gpt-researcher` (Python)
Planner-executor-publisher pattern. Strong community (7K+ stars). Supports multi-agent mode via LangGraph/AG2. Produces reports >2,000 words from >20 sources. Good for batch/automated research without heavy HITL requirements.

### 6.4 `stanford-oval/storm` (Python)
Best-in-class for document depth and citation quality. STORM = automated; Co-STORM = interactive collaborative mode. If user engagement and exploration are priorities (vs. efficiency), Co-STORM's mind map + moderator architecture is the most sophisticated prior art available.

### 6.5 `qx-labs/agents-deep-research` (Python, OpenAI Agents SDK)
Uses OpenAI Agents SDK with a Clarifier → Planner → Research Manager → Researcher → Writer pipeline. Compatible with 10+ LLM providers. Notable for implementing a Clarifier Agent as a first-class node that blocks until Q&A is complete.

---

## 7. Blueprint: How to Build One

### 7.1 Tech Stack
- **Orchestration:** LangGraph (Python) — best HITL support, state persistence, fan-out/fan-in
- **LLMs:** Fast model (e.g., Claude Haiku 4.5) for search/reflection; powerful model (Claude Opus 4.7) for clarifier and writer
- **Search:** Tavily API (structured) + Firecrawl (deep scraping for full-page content)
- **State persistence:** LangGraph InMemorySaver (dev) → PostgreSQL checkpointer (production)
- **Reranking:** Cross-encoder reranker (e.g., Cohere Rerank or a local Qwen3-Reranker) for search result quality

### 7.2 Node Graph

```
START
  │
  ▼
[clarifier_node]          ← ask 2-5 clarifying questions, produce research_brief
  │
  ▼ interrupt() ────────────────────────────────→ USER (answer questions)
  │
  ▼
[planner_node]            ← generate DAG of sub-questions from research_brief
  │
  ▼ interrupt() ────────────────────────────────→ USER (review/approve plan)
  │
  ▼
[query_generator_node]    ← expand each DAG node into search queries
  │
  ▼ (Send fan-out)
[web_research_node × N]   ← parallel search + scrape + chunk + rerank per query
  │
  ▼ (fan-in)
[reflection_node]         ← assess sufficiency, identify gaps, propose follow-ups
  │
  ├─ is_sufficient=True ──────────────────────────────────────────────┐
  │                                                                    │
  ├─ loop_count >= max ───────────────────────────────────────────────┤
  │                                                                    │
  └─ is_sufficient=False ──┐                                          │
                           │                                          │
                 interrupt() ──────────────────→ USER (approve gaps)  │
                           │                                          │
                           ▼                                          │
                  [query_generator_node]  ← loop back                 │
                                                                      │
                                                                      ▼
                                                           [writer_node]
                                                              ├─ meta-analysis
                                                              ├─ parallel section drafting
                                                              └─ assembly + citations
                                                                      │
                                                           interrupt() ──→ USER (review draft)
                                                                      │
                                                                      ▼
                                                                    END
```

### 7.3 Clarifier Node Design

```python
class ClarifyingQuestions(BaseModel):
    questions: list[str]          # 2-5 targeted questions
    draft_research_plan: str      # show intent early (Unified Intent-Planning)
    estimated_scope: str          # "narrow / medium / broad"

def clarifier_node(state: OverallState) -> OverallState:
    response = fast_llm.with_structured_output(ClarifyingQuestions).invoke([
        SystemMessage(CLARIFIER_SYSTEM_PROMPT),
        HumanMessage(state["messages"][-1].content)
    ])
    user_answers = interrupt({
        "clarifying_questions": response.questions,
        "draft_plan": response.draft_research_plan,
    })
    research_brief = compress_to_brief(
        original_query=state["messages"][-1].content,
        questions=response.questions,
        answers=user_answers,
    )
    return {"research_brief": research_brief}
```

**Clarifier prompt principles:**
- Ask about **scope** (depth, breadth, time range), **audience** (technical vs. executive), **format** (report length, citation style), and **constraints** (geographies, sectors, excluded topics).
- Never ask more than 5 questions — prioritize ruthlessly.
- If the query is already specific enough, generate 0 questions and proceed.

### 7.4 Reflection Node Design

```python
class ReflectionOutput(BaseModel):
    is_sufficient: bool
    missing_topics: list[str]
    follow_up_queries: list[SearchQuery]
    confidence: float             # 0-1

def reflection_node(state: OverallState) -> OverallState:
    output = fast_llm.with_structured_output(ReflectionOutput).invoke([
        SystemMessage(REFLECTION_PROMPT),
        HumanMessage(f"""
Research Brief: {state['research_brief']}
Findings so far: {compress_findings(state['web_research_results'])}
Loop count: {state['research_loop_count']}
""")
    ])
    if not output.is_sufficient and state['research_loop_count'] < state['max_research_loops']:
        # Optional: surface to user
        approved = interrupt({"gaps": output.missing_topics, "proposed_queries": output.follow_up_queries})
    return {
        "is_sufficient": output.is_sufficient,
        "knowledge_gaps": output.missing_topics,
        "follow_up_queries": output.follow_up_queries,
        "research_loop_count": state['research_loop_count'] + 1,
    }
```

### 7.5 Writer Node Pattern (Parallel Section Generation)

```python
def writer_node(state: OverallState) -> OverallState:
    # 1. Meta-analysis: identify themes
    themes = powerful_llm.invoke(META_ANALYSIS_PROMPT + compress_all_findings(state))

    # 2. Parallel section drafts
    sections = asyncio.gather(*[
        powerful_llm.ainvoke(SECTION_PROMPT.format(theme=t, findings=relevant_findings(t, state)))
        for t in themes
    ])

    # 3. Assemble
    draft = assemble_report(themes, sections, state['sources_gathered'])

    # 4. Gate: let user review before finalizing
    feedback = interrupt({"draft_report": draft})
    if feedback and feedback != "approve":
        # Re-enter research loop with targeted follow-up
        return {"user_feedback": feedback, "is_sufficient": False}

    return {"final_report": draft}
```

---

## 8. Advanced Patterns (2025–2026)

### 8.1 DAG-Based Research Planning (Egnyte Pattern)
Rather than a flat list of sub-questions, model the research plan as a DAG where dependencies are explicit. The Master agent does a topological traversal: all questions with satisfied dependencies are dispatched concurrently; the agent cycles back until all nodes are processed. This enables maximum parallelism while respecting logical ordering.

### 8.2 Multi-Perspective Question Asking (STORM Pattern)
Before generating search queries, use a Perspective Discovery step: have the LLM survey existing articles/documents on related topics to identify the *different viewpoints* that exist. Then generate search queries from each perspective. This produces broader, less biased coverage.

### 8.3 Cross-Encoder Reranking
After initial retrieval, pass (query, document) pairs through a cross-encoder that evaluates them jointly — much more accurate than vector similarity alone for semantic relevance. Qwen3-Reranker and Cohere Rerank are the 2026 standard choices.

### 8.4 Feedback-as-Training Signal
Per the 2026 trend toward "humans-above-the-loop": log every user correction at a HITL checkpoint, especially when the user edits follow-up queries or rejects a research direction. These corrections are high-quality training signal for future fine-tuning of the planner and reflection nodes.

### 8.5 RL-Incentivized Search (Emerging)
Search-R1 and related systems use GRPO (Group Relative Policy Optimization) to train the agent's search strategy end-to-end, allowing it to learn when to search vs. reason, how to refine failed queries, and when research is sufficient — without hand-engineered heuristics. This is still research-stage for production use but is the direction of travel.

---

## 9. Failure Modes and Mitigations

| Failure Mode | Cause | Mitigation |
|---|---|---|
| **Research drift** | Agent pursues interesting tangents, ignores brief | Include research_brief in every Researcher prompt; use it as eval criterion in reflection |
| **Citation hallucination** | LLM invents sources | Only cite from `sources_gathered`; pass raw URLs + snippets to Writer, not summaries |
| **Context overflow** | Long sessions blow the context window | Intermediate compression after each research round; external vector store for >50 sources |
| **Clarification fatigue** | Too many questions irritates users | Hard cap at 5 questions; if query is precise, skip clarification entirely |
| **Loop non-termination** | Reflection never declares sufficiency | `max_research_loops` hard cap; force-declare sufficient after cap |
| **Parallel branch divergence** | Fan-out branches accumulate contradictory findings | Deduplicate sources by URL before reflection; use `operator.add` with dedup wrapper |

---

## 10. Evaluation

There is now a standard benchmark: **Deep Research Bench** (GitHub: `Ayanami0730/deep_research_bench`). As of August 2025, top scores:

| System | Score |
|---|---|
| OpenAI Deep Research (o3) | ~0.55 |
| Gemini Deep Research | ~0.50 |
| LangChain Open Deep Research | 0.4344 (#6) |
| GPT Researcher | ~0.38 |

For your own agent, evaluate on:
- **Coverage**: Does the report address all aspects of the research brief?
- **Citation fidelity**: Do citations exist and support the claims made?
- **Coherence**: Is the report logically structured and free of contradictions?
- **User satisfaction at gates**: Were clarifying questions useful? Did HITL checkpoints add value?

---

## 11. Recommended Reading

- [Deep Research Agents: A Systematic Examination And Roadmap (2025)](https://arxiv.org/html/2506.18096v2) — best taxonomy/survey
- [Co-STORM: Into the Unknown Unknowns (EMNLP 2024)](https://arxiv.org/abs/2408.15232) — best human-in-the-loop prior art
- [LangGraph 101: Build a Deep Research Agent](https://towardsdatascience.com/langgraph-101-lets-build-a-deep-research-agent/) — best hands-on tutorial
- [LangGraph 201: Adding Human Oversight](https://towardsdatascience.com/langgraph-201-adding-human-oversight-to-your-deep-research-agent/) — HITL interrupt patterns
- [dzhng/deep-research](https://github.com/dzhng/deep-research) — simplest full implementation to read
- [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) — production LangGraph reference
- [stanford-oval/storm](https://github.com/stanford-oval/storm) — STORM + Co-STORM source

---

## Sources

- [An Open and Reproducible Deep Research Agent (arXiv 2512.13059)](https://arxiv.org/html/2512.13059)
- [Deep Research Agents: A Systematic Examination And Roadmap (arXiv 2506.18096)](https://arxiv.org/html/2506.18096v2)
- [Deep Research: A Survey of Autonomous Research Agents (arXiv 2508.12752)](https://arxiv.org/html/2508.12752v1)
- [Co-STORM: Into the Unknown Unknowns (ACL/EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.554/)
- [Co-STORM DeepWiki Architecture](https://deepwiki.com/stanford-oval/storm/3-co-storm-collaborative-system)
- [stanford-oval/storm (GitHub)](https://github.com/stanford-oval/storm)
- [langchain-ai/open_deep_research (GitHub)](https://github.com/langchain-ai/open_deep_research)
- [dzhng/deep-research (GitHub)](https://github.com/dzhng/deep-research)
- [assafelovic/gpt-researcher (GitHub)](https://github.com/assafelovic/gpt-researcher)
- [qx-labs/agents-deep-research (GitHub)](https://github.com/qx-labs/agents-deep-research)
- [LangGraph 101: Build a Deep Research Agent (Towards Data Science)](https://towardsdatascience.com/langgraph-101-lets-build-a-deep-research-agent/)
- [LangGraph 201: Adding Human Oversight (Towards Data Science)](https://towardsdatascience.com/langgraph-201-adding-human-oversight-to-your-deep-research-agent/)
- [Inside the Architecture of a Deep Research Agent (Egnyte)](https://www.egnyte.com/blog/post/inside-the-architecture-of-a-deep-research-agent)
- [Context Engineering Deep Dive: Building a Deep Research Agent (Promptingguide.ai)](https://www.promptingguide.ai/agents/context-engineering-deep-dive)
- [The Rise of Agent-Based Deep Research (Aaron Tay, Substack)](https://aarontay.substack.com/p/the-rise-of-agent-based-deep-research)
- [DavidZWZ/Awesome-Deep-Research (GitHub)](https://github.com/DavidZWZ/Awesome-Deep-Research)
- [Deep Research Bench (GitHub)](https://github.com/Ayanami0730/deep_research_bench)
- [From Web Search towards Agentic Deep ReSearch (arXiv 2506.18959)](https://arxiv.org/html/2506.18959v1)
- [Human-in-the-Loop Agentic AI (OneReach.ai, 2026)](https://onereach.ai/blog/human-in-the-loop-agentic-ai-systems/)
- [2026: From Human-in-Loop to Humans-Above-Loop (Diginomica)](https://diginomica.com/2026-year-move-human-in-loop-to-humans-above-loop)
- [Multi-Agent Deep Research Architecture (Trilogy AI)](https://trilogyai.substack.com/p/multi-agent-deep-research-architecture)
