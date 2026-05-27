"""Tool interface and structured result with normalization."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from jarvis.tools.limits import truncate

ActionClass = Literal["read", "write", "execute", "destructive"]


class ToolResult(BaseModel):
    """Canonical tool output."""

    ok: bool
    content: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    truncated: bool = False


class Tool(ABC):
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    action_class: ActionClass = "read"

    @abstractmethod
    def arguments_schema(self) -> dict[str, Any]:
        """JSON-schema-like dict describing accepted arguments."""

    @abstractmethod
    def _execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Subclass implementation."""

    # ----------------------------------------------------- public entry point
    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool with safety/normalization wrappers."""
        try:
            result = self._execute(arguments or {})
        except Exception as e:
            return ToolResult(
                ok=False,
                content="",
                error=f"{type(e).__name__}: {e}",
                metadata={"tool": self.name},
            )

        # Normalize: enforce content cap, attach tool name in metadata.
        content, truncated = truncate(result.content or "")
        meta = dict(result.metadata or {})
        meta.setdefault("tool", self.name)
        return ToolResult(
            ok=bool(result.ok),
            content=content,
            error=result.error,
            metadata=meta,
            truncated=truncated or result.truncated,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "action_class": self.action_class,
            "arguments_schema": self.arguments_schema(),
        }
