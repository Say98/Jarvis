"""LLM provider abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod

from jarvis.core.schemas import ChatMessage


class LLMProvider(ABC):
    """Abstract LLM provider. Implementations must be stateless / re-entrant."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Return a single completion string for the given chat messages."""

    def health_check(self) -> bool:  # pragma: no cover - optional
        """Return True if the backend is reachable."""
        return True
