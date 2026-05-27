"""LLM-backed classifier deciding what is worth storing in long-term memory."""
from __future__ import annotations

from jarvis.core.prompts import (
    MEMORY_CLASSIFIER_SYSTEM,
    memory_classifier_user_prompt,
)
from jarvis.core.schemas import ChatMessage, MemoryClassification
from jarvis.llm.base import LLMProvider
from jarvis.logging_setup import get_logger

log = get_logger(__name__)


class MemoryClassifier:
    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def classify(self, snippet: str, source: str = "agent") -> MemoryClassification:
        msgs = [
            ChatMessage(role="system", content=MEMORY_CLASSIFIER_SYSTEM),
            ChatMessage(
                role="user",
                content=memory_classifier_user_prompt(snippet, source),
            ),
        ]
        try:
            return self.llm.generate_json(
                msgs, MemoryClassification, temperature=0.0, max_retries=2
            )
        except Exception as e:
            log.warning("Memory classification failed: %s", e)
            return MemoryClassification(store=False)
