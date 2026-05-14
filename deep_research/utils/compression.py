from __future__ import annotations

from typing import TYPE_CHECKING

from ..state import Finding

if TYPE_CHECKING:
    pass

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))

except ImportError:
    def count_tokens(text: str) -> int:
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4


def compress_findings(findings: list[Finding], max_tokens: int) -> str:
    """Round-based compression: include all findings within token budget.

    Findings from later loops are prioritised (kept verbatim); earlier ones are
    summarised when the budget is exceeded.
    """
    if not findings:
        return ""

    lines: list[str] = []
    total = 0

    # Iterate newest-first so recent findings are included verbatim first
    for f in reversed(findings):
        entry = f"[Loop {f.loop_number}] Query: {f.query}\nSummary: {f.summary}\nSources: {', '.join(f.source_urls)}"
        tokens = count_tokens(entry)
        if total + tokens <= max_tokens:
            lines.append(entry)
            total += tokens
        else:
            # Budget exhausted — add a brief placeholder for the remainder
            remaining = len(findings) - len(lines)
            if remaining > 0:
                lines.append(
                    f"[{remaining} earlier finding(s) omitted to stay within token budget]"
                )
            break

    return "\n\n".join(reversed(lines))
