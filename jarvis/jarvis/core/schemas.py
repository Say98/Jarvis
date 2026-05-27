"""Structured data models for Jarvis v2. All control-flow data is typed."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

# --- chat ------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


# --- actions ---------------------------------------------------------------


class ToolCall(BaseModel):
    tool: str = Field(..., description="Name of the tool to call.")
    arguments: dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """All possible agent actions emitted by the Reasoner."""

    type: Literal[
        "tool_call",
        "step_complete",
        "final_answer",
        "replan",
        "abort",
    ]
    tool_call: ToolCall | None = None
    final_answer: str | None = None
    note: str | None = None
    reason: str | None = None


class Thought(BaseModel):
    reasoning: str


class ReasonerDecision(BaseModel):
    """Strict JSON schema the Reasoner LLM must emit."""

    thought: str
    action: Action


class FinalAnswer(BaseModel):
    type: Literal["final_answer"] = "final_answer"
    content: str


# --- observations ----------------------------------------------------------


class Observation(BaseModel):
    ok: bool
    content: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    truncated: bool = False


# --- plan ------------------------------------------------------------------


class PlanStep(BaseModel):
    id: int
    goal: str
    suggested_tool: str | None = None
    success_criteria: str | None = None
    expected_output: str | None = None
    depends_on: list[int] = Field(default_factory=list)
    status: Literal["pending", "in_progress", "done", "failed", "skipped"] = "pending"
    notes: str | None = None
    attempts: int = 0


class Plan(BaseModel):
    objective: str
    steps: list[PlanStep]
    revision: int = 0

    def _by_id(self, step_id: int) -> PlanStep | None:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def _deps_satisfied(self, step: PlanStep) -> bool:
        for dep_id in step.depends_on:
            dep = self._by_id(dep_id)
            if dep is None or dep.status not in ("done", "skipped"):
                return False
        return True

    def current(self) -> PlanStep | None:
        """Return the next ready step (deps satisfied), preferring in_progress."""
        for s in self.steps:
            if s.status == "in_progress":
                return s
        for s in self.steps:
            if s.status == "pending" and self._deps_satisfied(s):
                return s
        return None

    def mark(self, step_id: int, status: str, note: str | None = None) -> None:
        s = self._by_id(step_id)
        if s is None:
            return
        s.status = status  # type: ignore[assignment]
        if note:
            s.notes = note

    def is_complete(self) -> bool:
        return all(s.status in ("done", "skipped") for s in self.steps)

    def is_stalled(self) -> bool:
        """True if no step can progress (all blocked / failed)."""
        for s in self.steps:
            if s.status in ("pending", "in_progress") and self._deps_satisfied(s):
                return False
        return not self.is_complete()


class PlanDraft(BaseModel):
    """LLM-emitted plan before id assignment / validation."""

    objective: str
    steps: list[dict[str, Any]]


# --- scratchpad ------------------------------------------------------------


class ScratchpadEntry(BaseModel):
    step_index: int
    plan_step_id: int | None
    thought: str
    action: Action
    observation: Observation | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --- memory ----------------------------------------------------------------


class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    TASK_RESULT = "task_result"
    SKILL = "skill"


class MemoryItem(BaseModel):
    text: str
    type: MemoryType
    tags: list[str] = Field(default_factory=list)
    source: str | None = None
    score: float | None = None


class MemoryClassification(BaseModel):
    """LLM decision about whether to persist a snippet."""

    store: bool
    type: MemoryType | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)


# --- public events (UI / API / SQLite) -------------------------------------


class AgentStep(BaseModel):
    step: int
    plan_step_id: int | None
    thought: Thought
    action: Action
    observation: Observation | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


AgentEvent = Union[Plan, AgentStep, FinalAnswer]


# --- policy ----------------------------------------------------------------


class PolicyVerdict(BaseModel):
    """Deterministic decision emitted by the ExecutionPolicyEngine."""

    kind: Literal["continue", "replan", "abort"]
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
