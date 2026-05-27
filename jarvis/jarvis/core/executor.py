"""Executor: safety + retry + guards around a single tool call."""
from __future__ import annotations

import time

from jarvis.core.schemas import Action, Observation, ToolCall
from jarvis.logging_setup import get_logger
from jarvis.safety.approver import Approver
from jarvis.tools.base import Tool
from jarvis.tools.registry import ToolRegistry

log = get_logger(__name__)


# Action classes for which automatic retry is allowed.
RETRYABLE_CLASSES = {"read", "execute"}


class Executor:
    def __init__(
        self,
        tools: ToolRegistry,
        approver: Approver,
        max_retries_per_call: int = 1,
        per_call_timeout_s: float | None = None,
    ) -> None:
        self.tools = tools
        self.approver = approver
        self.max_retries_per_call = max_retries_per_call
        self.per_call_timeout_s = per_call_timeout_s

    # ---------------------------------------------------------------- public

    def execute(self, action: Action) -> Observation:
        if action.type != "tool_call" or action.tool_call is None:
            return Observation(
                ok=False,
                error=f"Executor only handles tool_call actions, got {action.type}",
            )

        call: ToolCall = action.tool_call
        tool = self.tools.get(call.tool)
        if tool is None:
            return Observation(
                ok=False,
                error=f"Unknown tool '{call.tool}'. Available: {self.tools.names()}",
            )

        decision = self.approver.approve(tool, call)
        if not decision.allowed:
            return Observation(
                ok=False,
                error=f"Action denied by safety layer: {decision.reason}",
            )

        return self._run_with_retry(tool, call)

    # ---------------------------------------------------------------- guts

    def _run_once(self, tool: Tool, call: ToolCall) -> Observation:
        t0 = time.monotonic()
        try:
            result = tool.run(call.arguments)
        except Exception as e:  # last-resort guard; tools should not raise
            elapsed = time.monotonic() - t0
            log.exception("Tool %s raised an unhandled exception", tool.name)
            return Observation(
                ok=False,
                error=f"{type(e).__name__}: {e}",
                metadata={"elapsed_s": round(elapsed, 3), "unhandled": True},
            )
        elapsed = time.monotonic() - t0

        # Soft per-call wall-time guard (not a hard kill — tool internals
        # already enforce their own timeouts).
        if (
            self.per_call_timeout_s is not None
            and elapsed > self.per_call_timeout_s
            and result.ok
        ):
            log.warning(
                "Tool %s exceeded soft timeout: %.2fs > %.2fs",
                tool.name,
                elapsed,
                self.per_call_timeout_s,
            )

        return Observation(
            ok=result.ok,
            content=result.content,
            error=result.error,
            truncated=result.truncated,
            metadata={**result.metadata, "elapsed_s": round(elapsed, 3)},
        )

    def _run_with_retry(self, tool: Tool, call: ToolCall) -> Observation:
        attempts = 0
        last: Observation | None = None
        retryable = tool.action_class in RETRYABLE_CLASSES
        max_attempts = 1 + (self.max_retries_per_call if retryable else 0)

        while attempts < max_attempts:
            attempts += 1
            last = self._run_once(tool, call)
            log.info(
                "Executor %s attempt %d/%d ok=%s",
                tool.name,
                attempts,
                max_attempts,
                last.ok,
            )
            if last.ok:
                last.metadata["attempts"] = attempts
                return last
            # Don't retry permission / validation errors.
            if last.error and any(
                marker in last.error
                for marker in ("PermissionError", "not a string", "must be", "Refusing")
            ):
                break

        assert last is not None
        last.metadata["attempts"] = attempts
        return last
