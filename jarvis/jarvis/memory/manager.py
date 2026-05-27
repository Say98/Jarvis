"""Unified memory manager combining short-term, long-term and persistent stores."""
from __future__ import annotations

import uuid
from typing import Any

from jarvis.config.settings import MemoryConfig
from jarvis.core.schemas import AgentStep, ChatMessage
from jarvis.logging_setup import get_logger
from jarvis.memory.long_term import LongTermMemory
from jarvis.memory.short_term import ShortTermMemory
from jarvis.memory.store import EventStore

log = get_logger(__name__)


class MemoryManager:
    def __init__(self, cfg: MemoryConfig, session_id: str | None = None) -> None:
        self.cfg = cfg
        self.session_id = session_id or str(uuid.uuid4())
        self.short_term = ShortTermMemory(cfg.short_term_max_messages)
        self.long_term = LongTermMemory(
            chroma_path=cfg.chroma_path,
            collection=cfg.long_term_collection,
            embedding_model=cfg.embedding_model,
        )
        self.store = EventStore(cfg.sqlite_path)
        self.store.create_session(self.session_id)

    # Short-term ---------------------------------------------------------
    def add_message(self, message: ChatMessage) -> None:
        self.short_term.add(message)
        self.store.log_event(
            self.session_id,
            "message",
            {"role": message.role, "content": message.content, "name": message.name},
        )

    def conversation(self) -> list[ChatMessage]:
        return self.short_term.all()

    # Long-term ----------------------------------------------------------
    def remember(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        try:
            self.long_term.add(text, metadata or {})
        except Exception as e:  # pragma: no cover
            log.warning("Long-term memory add failed: %s", e)

    def recall(self, query: str, k: int = 5) -> list[str]:
        try:
            return self.long_term.query(query, k=k)
        except Exception as e:  # pragma: no cover
            log.warning("Long-term memory query failed: %s", e)
            return []

    # Agent steps --------------------------------------------------------
    def log_step(self, step: AgentStep) -> None:
        self.store.log_event(self.session_id, "step", step.model_dump(mode="json"))
