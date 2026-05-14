# /solve — Research, Report, Implement

You are helping solve the following problem: **$ARGUMENTS**

Work through three phases in order. Do not skip ahead to implementation without explicit user approval.

---

## Phase 1 — Research

Spawn multiple subagents **in parallel** to investigate different angles simultaneously. Each subagent should be given a focused question. Cover at minimum:

- **Existing solutions & prior art**: What approaches already exist? What do production systems typically look like?
- **Architectures & tech stacks**: What are the main architectural patterns? What technology choices are commonly used and why?
- **Trade-offs & constraints**: What are the key dimensions along which solutions differ (performance, complexity, cost, scalability, maintainability)? What constraints typically drive decisions?
- **Failure modes & gotchas**: What do teams commonly get wrong? What problems only emerge at scale or in production?

Synthesize subagent findings into a concise internal summary before proceeding.

---

## Phase 2 — Report

Present a structured report to the user. Use clear markdown headers. Be educational — assume the user wants to understand the space, not just get an answer.

The report should cite any sources it is based on so the user can quickly check.

### Report structure:

**1. Problem framing**
Restate the problem in precise terms. Call out any ambiguities or assumptions you are making.

**2. Solution landscape**
Describe the main solution approaches (2–4 options). For each:
- What it is and how it works
- Key strengths
- Key weaknesses / when it breaks down
- Representative examples or tools

**3. Key decision factors**
List the questions/constraints that most determine which solution is right. Frame these as "If X, then prefer Y" where possible.

**4. Recommendation**
State a specific recommendation with clear reasoning. If the right answer genuinely depends on unknowns, say so explicitly and list what you need to know.

**5. Open questions for the user**
List any clarifying questions whose answers would change the recommendation or implementation approach. Number them.

---

After delivering the report, **stop and wait**. Ask the user to:
- Answer any open questions
- Confirm the chosen direction (or redirect to a different option)
- Say "proceed" or similar to move to implementation

Do not begin Phase 3 until the user explicitly confirms.

---

## Phase 3 — Implementation

Once the user has confirmed direction:

1. Enter plan mode and present a concrete implementation plan:
   - Files to create or modify
   - Key decisions and their rationale
   - Any prerequisites or setup steps
   - Estimated scope / complexity

2. Wait for plan approval before writing any code.

3. Implement incrementally — complete one logical unit, confirm it works, then continue. Do not write everything at once.
