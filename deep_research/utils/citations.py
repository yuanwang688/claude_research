from __future__ import annotations

from ..state import Finding, Source


def build_citation_map(sources: dict[str, Source]) -> dict[str, str]:
    """Map URL -> short citation key, e.g. 'example.com-1'."""
    keys: dict[str, str] = {}
    domain_counts: dict[str, int] = {}

    for url in sources:
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lstrip("www.")
        except Exception:
            domain = "source"
        count = domain_counts.get(domain, 0) + 1
        domain_counts[domain] = count
        keys[url] = f"{domain}-{count}"

    return keys


def relevant_findings(section_topic: str, findings: list[Finding]) -> str:
    """Return a text block of findings relevant to a report section topic.

    For MVP, returns all findings (filtering added in a future iteration).
    """
    if not findings:
        return "No findings available."
    parts = []
    for f in findings:
        parts.append(f"Query: {f.query}\nSummary: {f.summary}")
    return "\n\n".join(parts)


def assemble_report(
    sections: list[str],
    section_texts: list[str],
    sources: dict[str, Source],
) -> str:
    """Combine section drafts into a final Markdown report with a sources appendix."""
    parts: list[str] = []
    for title, body in zip(sections, section_texts):
        # Strip any leading heading lines the LLM may have included to avoid duplicates
        lines = body.strip().splitlines()
        while lines and lines[0].lstrip().startswith("#"):
            lines.pop(0)
        body = "\n".join(lines).strip()
        parts.append(f"## {title}\n\n{body}")

    report = "\n\n".join(parts)

    if sources:
        citation_map = build_citation_map(sources)
        source_lines = ["\n\n## Sources\n"]
        for url, key in citation_map.items():
            src = sources[url]
            source_lines.append(f"- [{key}] {src.title} — {url}")
        report += "\n".join(source_lines)

    return report
