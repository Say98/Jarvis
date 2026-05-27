"""In-memory short-term conversation buffer."""
from __future__ import annotations

from collections import deque

from jarvis.core.schemas import ChatMessage


class ShortTermMemory:
    def __init__(self, max_messages: int = 30) -> None:
        self._buf: deque[ChatMessage] = deque(maxlen=max_messages)

    def add(self, message: ChatMessage) -> None:
        self._buf.append(message)

    def extend(self, messages: list[ChatMessage]) -> None:
        for m in messages:
            self.add(m)

    def all(self) -> list[ChatMessage]:
        return list(self._buf)

    def clear(self) -> None:
        self._buf.clear()
