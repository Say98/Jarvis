"""Central state object for one agent task. Mutated by the loop."""
from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from jarvis.core.schemas import (
    Observation,
    Plan,
    PlanStep,
    ScratchpadEntry,
)
from jarvis.logging_setup import get_logger

log = get_logger(__name__)


TaskPhase = Literal[
    "planning",
    "executing",
    "replanning",
    "finalizing",
    "aborted",
    "done",
]


class TaskState(BaseModel):
    """Single source of truth for an in-flight task.

    The agent loop mutates this object as it progresses; every other component
    is a pure function of it.
    """

    objective: str
    plan: Plan | None = None
    history: list[ScratchpadEntry] = Field(default_factory=list)

    phase: TaskPhase = "planning"
    step_index: int = 0  # monotonic counter of agent iterations
    failures: int = 0  # consecutive failed tool calls
    total_failures: int = 0
    replans: int = 0

    # Terminal flags ------------------------------------------------------
    aborted: bool = False
    abort_reason: str | None = None
    completed: bool = False
    final_answer: str | None = None

    # Per-tool failure counters. Reset by a successful call to that tool.
    tool_failures: dict[str, int] = Field(default_factory=dict)
    # Tools banned for the *current* plan step (e.g. they failed too often).
    banned_tools_for_step: dict[int, list[str]] = Field(default_factory=dict)
    # Tools banned for the entire task — survives replans.
    persistent_tool_bans: list[str] = Field(default_factory=list)
    # Failed-step archive: list of {id, goal, attempts, reason}.
    failed_step_history: list[dict[str, Any]] = Field(default_factory=list)

    # Free-form signals consumed by the PolicyEngine and prompts.
    signals: dict[str, Any] = Field(default_factory=dict)

    # ---- phase transitions -----------------------------------------------

    def transition(self, new_phase: TaskPhase) -> None:
        if new_phase == self.phase:
            return
        log.info("TaskState phase: %s → %s", self.phase, new_phase)
        self.phase = new_phase

    # ---- accessors --------------------------------------------------------

    def current_step(self) -> PlanStep | None:
        if self.plan is None:
            return None
        return self.plan.current()

    def banned_for(self, step_id: int) -> list[str]:
        per_step = list(self.banned_tools_for_step.get(step_id, []))
        for t in self.persistent_tool_bans:
            if t not in per_step:
                per_step.append(t)
        return per_step

    def is_terminal(self) -> bool:
        return self.aborted or self.completed

    # ---- mutators ---------------------------------------------------------

    def append_entry(self, entry: ScratchpadEntry) -> None:
        self.history.append(entry)

    def record_observation(self, tool: str, ok: bool) -> None:
        if ok:
            self.failures = 0
            self.tool_failures.pop(tool, None)
        else:
            self.failures += 1
            self.total_failures += 1
            self.tool_failures[tool] = self.tool_failures.get(tool, 0) + 1

    def ban_tool_for_step(self, step_id: int, tool: str) -> None:
        self.banned_tools_for_step.setdefault(step_id, [])
        if tool not in self.banned_tools_for_step[step_id]:
            self.banned_tools_for_step[step_id].append(tool)

    def ban_tool_persistent(self, tool: str) -> None:
        if tool not in self.persistent_tool_bans:
            self.persistent_tool_bans.append(tool)
            log.info("TaskState: persistent ban on tool '%s'", tool)

    def mark_step_failed(self, step_id: int, reason: str) -> None:
        """Force a plan step into the failed terminal state."""
        if self.plan is None:
            return
        step = self.plan._by_id(step_id)
        if step is None:
            return
        step.status = "failed"
        step.notes = (step.notes + " | " if step.notes else "") + f"failed: {reason}"
        self.failed_step_history.append(
            {
                "id": step_id,
                "goal": step.goal,
                "attempts": step.attempts,
                "reason": reason,
            }
        )
        log.warning(
            "TaskState: step %d marked FAILED after %d attempts (%s)",
            step_id,
            step.attempts,
            reason,
        )

    def mark_replan(self, new_plan: Plan) -> None:
        """Adopt a revised plan. Per-step bans reset; persistent bans survive."""
        self.plan = new_plan
        self.replans += 1
        self.failures = 0
        self.tool_failures.clear()
        self.banned_tools_for_step.clear()
        self.transition("executing")

    def mark_aborted(self, reason: str) -> None:
        self.aborted = True
        self.abort_reason = reason
        self.transition("aborted")

    def mark_completed(self, answer: str) -> None:
        self.completed = True
        self.final_answer = answer
        self.transition("done")

    # ---- introspection ----------------------------------------------------

    def recent_tool_calls(self, n: int = 5) -> list[str]:
        out: list[str] = []
        for entry in reversed(self.history):
            if entry.action.tool_call:
                out.append(entry.action.tool_call.tool)
                if len(out) >= n:
                    break
        return list(reversed(out))

    def repeating_failures(self, window: int = 3) -> tuple[str, int] | None:
        """If the last `window` actions all failed on the same tool, return it."""
        tail = self.history[-window:]
        if len(tail) < window:
            return None
        tools: list[str] = []
        for e in tail:
            if (
                e.observation is None
                or e.observation.ok
                or e.action.tool_call is None
            ):
                return None
            tools.append(e.action.tool_call.tool)
        c = Counter(tools)
        tool, cnt = c.most_common(1)[0]
        if cnt == window:
            return tool, cnt
        return None

    def completed_steps_summary(self) -> list[dict[str, Any]]:
        """Compact view of finished plan steps for replan context."""
        if self.plan is None:
            return []
        out: list[dict[str, Any]] = []
        for s in self.plan.steps:
            if s.status in ("done", "skipped"):
                out.append(
                    {
                        "id": s.id,
                        "status": s.status,
                        "goal": s.goal,
                        "notes": s.notes,
                        "attempts": s.attempts,
                    }
                )
        return out

    def snapshot_signals(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "step_index": self.step_index,
            "failures": self.failures,
            "total_failures": self.total_failures,
            "replans": self.replans,
            "tool_failures": dict(self.tool_failures),
            "persistent_bans": list(self.persistent_tool_bans),
            "failed_steps": list(self.failed_step_history),
        }


def make_state(objective: str) -> TaskState:
    return TaskState(objective=objective)


def render_history(history: list[ScratchpadEntry], max_chars: int = 2000) -> str:
    """JSON-friendly compact rendering of the scratchpad for prompt injection."""
    import json

    out: list[dict[str, Any]] = []
    for e in history:
        action_view: dict[str, Any] = {"type": e.action.type}
        if e.action.tool_call:
            action_view["tool"] = e.action.tool_call.tool
            action_view["arguments"] = e.action.tool_call.arguments
        if e.action.final_answer:
            action_view["final_answer"] = e.action.final_answer
        if e.action.note:
            action_view["note"] = e.action.note
        if e.action.reason:
            action_view["reason"] = e.action.reason

        obs_view: dict[str, Any] | None = None
        if e.observation is not None:
            content = e.observation.content or ""
            if len(content) > max_chars:
                content = (
                    content[:max_chars]
                    + f"\n[... truncated, full length={len(e.observation.content)} ...]"
                )
            obs_view = {
                "ok": e.observation.ok,
                "content": content,
                "error": e.observation.error,
            }
        out.append(
            {
                "step": e.step_index,
                "plan_step_id": e.plan_step_id,
                "thought": e.thought,
                "action": action_view,
                "observation": obs_view,
            }
        )
    return json.dumps(out, indent=2, default=str)


__all__ = ["TaskState", "TaskPhase", "make_state", "render_history"]
