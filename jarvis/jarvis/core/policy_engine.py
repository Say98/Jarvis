"""Deterministic execution policy engine.

This module enforces hard rules OUTSIDE the LLM. It runs:

  - BEFORE the Reasoner   (`before`) → may force replan or abort
  - AFTER each tool call  (`after`)  → updates task signals (e.g. ban a tool)

Rules are intentionally simple, transparent and testable. NO LLM in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

from jarvis.core.schemas import Observation, PolicyVerdict
from jarvis.core.task_state import TaskState
from jarvis.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class PolicyConfig:
    max_consecutive_failures: int = 3
    max_total_failures: int = 8
    max_replans: int = 2
    ban_tool_after_failures: int = 2
    same_tool_loop_window: int = 3
    # Per plan-step ceiling. If a single step exceeds this many attempts,
    # the step is marked failed → planner is invoked to recover.
    max_step_attempts: int = 5
    # When a tool fails this many times across the entire task (not just one
    # step), it is banned for the rest of the task.
    persistent_ban_threshold: int = 4


class ExecutionPolicyEngine:
    def __init__(self, cfg: PolicyConfig | None = None) -> None:
        self.cfg = cfg or PolicyConfig()

    # ------------------------------------------------------------ pre-step

    def before(self, state: TaskState) -> PolicyVerdict:
        """Inspect state BEFORE reasoning. Decide continue / replan / abort."""

        # Terminal kill switches.
        if state.aborted:
            return PolicyVerdict(kind="abort", reason=state.abort_reason or "aborted")

        if state.total_failures >= self.cfg.max_total_failures:
            return PolicyVerdict(
                kind="abort",
                reason=(
                    f"max_total_failures reached "
                    f"({state.total_failures} >= {self.cfg.max_total_failures})"
                ),
                metadata=state.snapshot_signals(),
            )

        if state.replans > self.cfg.max_replans:
            return PolicyVerdict(
                kind="abort",
                reason=f"max_replans exceeded ({state.replans})",
                metadata=state.snapshot_signals(),
            )

        # Per-step attempt ceiling — fail the step and force replan.
        current = state.current_step()
        if current is not None and current.attempts >= self.cfg.max_step_attempts:
            state.mark_step_failed(
                current.id,
                f"exceeded max_step_attempts ({current.attempts})",
            )
            return PolicyVerdict(
                kind="replan",
                reason=(
                    f"step {current.id} exceeded max_step_attempts "
                    f"({self.cfg.max_step_attempts})"
                ),
                metadata={"step_id": current.id},
            )

        # Stalled plan (cyclic deps, all blocked, or all remaining failed).
        if state.plan is not None and state.plan.is_stalled():
            return PolicyVerdict(
                kind="replan",
                reason="plan is stalled (no step can progress)",
            )

        # Repeated identical failures → ban + replan.
        loop = state.repeating_failures(window=self.cfg.same_tool_loop_window)
        if loop is not None:
            tool, _ = loop
            if current is not None:
                state.ban_tool_for_step(current.id, tool)
            # Promote to persistent ban if the same tool has failed enough
            # times across the whole task.
            if state.tool_failures.get(tool, 0) >= self.cfg.persistent_ban_threshold:
                state.ban_tool_persistent(tool)
            return PolicyVerdict(
                kind="replan",
                reason=f"detected failure loop on tool '{tool}'",
                metadata={"tool": tool},
            )

        # Soft consecutive-failure trigger.
        if state.failures >= self.cfg.max_consecutive_failures:
            return PolicyVerdict(
                kind="replan",
                reason=(
                    f"consecutive failures threshold reached "
                    f"({state.failures})"
                ),
                metadata=state.snapshot_signals(),
            )

        return PolicyVerdict(kind="continue")

    # ------------------------------------------------------------ post-step

    def after(self, state: TaskState, tool: str, obs: Observation) -> None:
        """Update state based on the observation."""
        state.record_observation(tool, obs.ok)

        current = state.current_step()

        if not obs.ok:
            fails = state.tool_failures.get(tool, 0)
            # Per-step ban.
            if (
                current is not None
                and fails >= self.cfg.ban_tool_after_failures
            ):
                state.ban_tool_for_step(current.id, tool)
                log.info(
                    "Policy: banned tool '%s' for step %d (failures=%d)",
                    tool,
                    current.id,
                    fails,
                )
            # Task-wide ban for chronic offenders.
            if fails >= self.cfg.persistent_ban_threshold:
                state.ban_tool_persistent(tool)

        # Always advance per-step attempt counter (success or failure).
        if current is not None:
            current.attempts += 1
