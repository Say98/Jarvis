"""Executor: safety-checks and runs a single Action, returning an Observation."""
from __future__ import annotations

import time

from jarvis.core.schemas import Action, Observation, ToolCall
from jarvis.logging_setup import get_logger
from jarvis.safety.approver import Approver
from jarvis.tools.registry import ToolRegistry

log = get_logger(__name__)


class Executor:
    def __init__(
        self,
        tools: ToolRegistry,
        approver: Approver,
        max_consecutive_failures: int = 3,
    ) -> None:
        self.tools = tools
        self.approver = approver
        self.max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def reset_failures(self) -> None:
        self._consecutive_failures = 0

    def execute(self, action: Action) -> Observation:
        if action.type != "tool_call" or action.tool_call is None:
            return Observation(
                ok=False,
                error=f"Executor only handles tool_call actions, got {action.type}",
            )

        call: ToolCall = action.tool_call
        tool = self.tools.get(call.tool)
        if tool is None:
            self._consecutive_failures += 1
            return Observation(
                ok=False,
                error=(
                    f"Unknown tool '{call.tool}'. Available: {self.tools.names()}"
                ),
            )

        decision = self.approver.approve(tool, call)
        if not decision.allowed:
            self._consecutive_failures += 1
            return Observation(
                ok=False,
                error=f"Action denied by safety layer: {decision.reason}",
            )

        t0 = time.monotonic()
        result = tool.run(call.arguments)
        elapsed = time.monotonic() - t0

        obs = Observation(
            ok=result.ok,
            content=result.content,
            error=result.error,
            truncated=result.truncated,
            metadata={
                **result.metadata,
                "elapsed_s": round(elapsed, 3),
            },
        )

        if obs.ok:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

        log.info(
            "Executor ran %s (ok=%s, elapsed=%.2fs, fails=%d)",
            call.tool,
            obs.ok,
            elapsed,
            self._consecutive_failures,
        )
        return obs

    def should_abort(self) -> bool:
        return self._consecutive_failures >= self.max_consecutive_failures
