"""LLM provider abstraction with structured-output support."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel

from jarvis.core.schemas import ChatMessage
from jarvis.llm.retry import validate_to_model
from jarvis.logging_setup import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


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
        json_mode: bool = False,
    ) -> str:
        """Return a completion string."""

    def health_check(self) -> bool:  # pragma: no cover
        return True

    # ----------------------------------------------------- structured output

    def generate_json(
        self,
        messages: list[ChatMessage],
        schema: Type[T],
        *,
        max_retries: int = 3,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Generate output validated against a Pydantic schema.

        On validation failure the error is fed back to the model for self-correction.
        """
        schema_json = json.dumps(schema.model_json_schema())
        sys_hint = ChatMessage(
            role="system",
            content=(
                "You MUST reply with a single JSON object that validates against "
                f"this JSON schema:\n{schema_json}\n"
                "No prose, no markdown fences, no commentary outside JSON."
            ),
        )
        convo = [sys_hint, *messages]

        last_error: str | None = None
        for attempt in range(1, max_retries + 1):
            if last_error is not None:
                convo = convo + [
                    ChatMessage(
                        role="user",
                        content=(
                            f"Your previous reply failed validation: {last_error}\n"
                            "Reply again with ONLY a valid JSON object matching the schema."
                        ),
                    )
                ]
            raw = self.generate(
                convo,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
            )
            try:
                return validate_to_model(raw, schema)
            except ValueError as e:
                last_error = str(e)
                log.warning(
                    "generate_json attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    last_error,
                )

        raise LLMError(
            f"generate_json failed after {max_retries} attempts. Last error: {last_error}"
        )
