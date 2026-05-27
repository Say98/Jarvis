"""Context-window management: token budgeting and message trimming."""
from __future__ import annotations

from jarvis.core.schemas import ChatMessage


def estimate_tokens(text: str) -> int:
    """Cheap heuristic: ~4 chars/token. Good enough for budgeting."""
    return max(1, len(text) // 4)


def message_tokens(m: ChatMessage) -> int:
    return estimate_tokens(m.content) + 4  # role overhead


class ContextManager:
    """Trim a list of messages to fit a token budget.

    Strategy:
      - ALWAYS keep all messages with role='system'.
      - ALWAYS keep the last `keep_recent` non-system messages.
      - Drop older middle messages until under budget.
    """

    def __init__(self, max_tokens: int = 6000, keep_recent: int = 6) -> None:
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent

    def fit(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        if not messages:
            return messages

        system = [m for m in messages if m.role == "system"]
        rest = [m for m in messages if m.role != "system"]

        recent = rest[-self.keep_recent :] if self.keep_recent > 0 else []
        older = rest[: -self.keep_recent] if self.keep_recent > 0 else rest

        def total(msgs: list[ChatMessage]) -> int:
            return sum(message_tokens(m) for m in msgs)

        kept_older = list(older)
        while total(system + kept_older + recent) > self.max_tokens and kept_older:
            kept_older.pop(0)

        # If still over budget, drop oldest recents (keep at least 1)
        while (
            total(system + kept_older + recent) > self.max_tokens
            and len(recent) > 1
        ):
            recent.pop(0)

        return system + kept_older + recent
