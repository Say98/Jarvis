"""Output size limits and normalization helpers for tools."""
from __future__ import annotations

MAX_TOOL_CONTENT_CHARS = 16_000  # safe upper bound for any single tool output


def truncate(text: str, limit: int = MAX_TOOL_CONTENT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = text[: limit - 200]
    return (
        head + f"\n\n[... truncated {len(text) - len(head)} chars ...]",
        True,
    )
