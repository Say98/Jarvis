"""Structured data models for the agent loop. No string hacks."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A structured tool invocation request emitted by the LLM."""

    tool: str = Field(..., description="Name of the tool to call.")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the tool."
    )


class FinalAnswer(BaseModel):
    """Signals the agent has produced a final answer."""

    type: Literal["final_answer"] = "final_answer"
    content: str


class Thought(BaseModel):
    """The model's reasoning trace for a single step."""

    reasoning: str


class Action(BaseModel):
    """A decision: either call a tool or produce a final answer."""

    type: Literal["tool_call", "final_answer"]
    tool_call: ToolCall | None = None
    final_answer: str | None = None


class Observation(BaseModel):
    """The result of executing an action."""

    ok: bool
    content: str
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    """A complete agent loop iteration."""

    step: int
    thought: Thought
    action: Action
    observation: Observation | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None  # For tool messages: tool name


AgentEvent = Union[AgentStep, FinalAnswer]
