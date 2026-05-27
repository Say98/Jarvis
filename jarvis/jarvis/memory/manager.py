"""Memory manager: short-term, filtered long-term (typed), SQLite event log."""
from __future__ import annotations

import uuid
from typing import Any

from jarvis.config.settings import MemoryConfig
from jarvis.core.schemas import (
    AgentStep,
    ChatMessage,
    MemoryClassification,
    MemoryItem,
    MemoryType,
    Plan,
)
from jarvis.logging_setup import get_logger
from jarvis.memory.classifier import MemoryClassifier
from jarvis.memory.long_term import LongTermMemory
from jarvis.memory.short_term import ShortTermMemory
from jarvis.memory.store import EventStore

log = get_logger(__name__)


class MemoryManager:
    def __init__(
        self,
        cfg: MemoryConfig,
        classifier: MemoryClassifier | None = None,
        session_id: str | None = None,
    ) -> None:
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
        self.classifier = classifier

    # --- short term --------------------------------------------------------

    def add_message(self, message: ChatMessage) -> None:
        self.short_term.add(message)
        self.store.log_event(
            self.session_id,
            "message",
            {"role": message.role, "content": message.content, "name": message.name},
        )

    def conversation(self) -> list[ChatMessage]:
        return self.short_term.all()

    # --- long term ---------------------------------------------------------

    def remember(
        self,
        text: str,
        type: MemoryType,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> None:
        """Direct write — bypasses classifier."""
        try:
            self.long_term.add(
                text,
                metadata={
                    "type": type.value,
                    "tags": ",".join(tags or []),
                    "source": source or "",
                },
            )
        except Exception as e:  # pragma: no cover
            log.warning("Long-term memory add failed: %s", e)

    def consider(self, snippet: str, source: str = "agent") -> bool:
        """Run the classifier; store only if it deems the snippet useful."""
        if self.classifier is None:
            return False
        decision: MemoryClassification = self.classifier.classify(snippet, source)
        if not decision.store or decision.type is None:
            return False
        text = decision.summary or snippet
        self.remember(text=text, type=decision.type, tags=decision.tags, source=source)
        log.info("Stored memory (%s): %s", decision.type.value, text[:100])
        return True

    def recall(
        self,
        query: str,
        k: int = 5,
        types: list[MemoryType] | None = None,
    ) -> list[MemoryItem]:
        where: dict[str, Any] | None = None
        if types:
            where = {"type": {"$in": [t.value for t in types]}}
        try:
            hits = self.long_term.query(query, k=k, where=where)
        except Exception as e:  # pragma: no cover
            log.warning("Long-term recall failed: %s", e)
            return []
        items: list[MemoryItem] = []
        for doc, meta, dist in hits:
            try:
                mtype = MemoryType(meta.get("type", "fact"))
            except ValueError:
                mtype = MemoryType.FACT
            tags = [t for t in (meta.get("tags", "") or "").split(",") if t]
            items.append(
                MemoryItem(
                    text=doc,
                    type=mtype,
                    tags=tags,
                    source=meta.get("source") or None,
                    score=1.0 - dist,
                )
            )
        return items

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        return [m.text for m in self.recall(query, k=k)]

    # --- event logging -----------------------------------------------------

    def log_step(self, step: AgentStep) -> None:
        self.store.log_event(self.session_id, "step", step.model_dump(mode="json"))

    def log_plan(self, plan: Plan) -> None:
        self.store.log_event(self.session_id, "plan", plan.model_dump(mode="json"))
