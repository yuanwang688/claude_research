META_ANALYSIS_PROMPT = """\
You are a research editor. Analyse the following research findings and identify the \
key themes or sections that a comprehensive report should cover.

Return a list of 3–6 section titles. Each title should be:
- Descriptive but concise (3–8 words)
- Distinct from the other sections (no overlap)
- Ordered logically (e.g., background → analysis → implications)

Do NOT write the sections yet — only the titles.
"""

SECTION_PROMPT = """\
You are a research writer. Write one section of a research report based on the \
provided research findings.

Section title: {section}
Research brief: {research_brief}

Guidelines:
- Write 2–4 paragraphs.
- Ground every claim in the provided research; do not add information from outside it.
- Cite sources using the URL directly inline, e.g., "According to [url] ..."
- Use clear, professional prose suitable for an informed non-expert reader.
- End with a 1-sentence summary of this section's key takeaway.
"""
