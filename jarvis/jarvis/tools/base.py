"""Tool interface and structured result."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

ActionClass = Literal["read", "write", "execute", "destructive"]


class ToolResult(BaseModel):
    """Structured tool output."""

    ok: bool
    content: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    action_class: ActionClass = "read"

    @abstractmethod
    def arguments_schema(self) -> dict[str, Any]:
        """JSON-schema-like dict describing accepted arguments."""

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with validated arguments."""

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "action_class": self.action_class,
            "arguments_schema": self.arguments_schema(),
        }
