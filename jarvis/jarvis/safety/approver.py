"""Human-in-the-loop approval system."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

from rich.console import Console
from rich.panel import Panel

from jarvis.core.schemas import ToolCall
from jarvis.safety.policy import ApprovalDecision, PolicyEngine
from jarvis.tools.base import Tool


class Approver(ABC):
    """Decides whether a tool call should be executed."""

    @abstractmethod
    def approve(self, tool: Tool, call: ToolCall) -> ApprovalDecision: ...


class AutoApprover(Approver):
    """Approve everything. For non-interactive automation only."""

    def approve(self, tool: Tool, call: ToolCall) -> ApprovalDecision:
        return ApprovalDecision(allowed=True, reason="auto-approved")


class DenyApprover(Approver):
    def approve(self, tool: Tool, call: ToolCall) -> ApprovalDecision:
        return ApprovalDecision(allowed=False, reason="denied by policy")


class CLIApprover(Approver):
    """Interactive approver enforcing the configured PolicyEngine."""

    def __init__(
        self,
        policy: PolicyEngine,
        console: Console | None = None,
    ) -> None:
        self.policy = policy
        self.console = console or Console()

    def _prompt(self, prompt: str) -> bool:
        while True:
            self.console.print(f"[bold yellow]{prompt}[/bold yellow] [y/N]: ", end="")
            try:
                resp = input().strip().lower()
            except EOFError:
                return False
            if resp in ("y", "yes"):
                return True
            if resp in ("", "n", "no"):
                return False

    def approve(self, tool: Tool, call: ToolCall) -> ApprovalDecision:
        level = self.policy.policy_for(tool.action_class)

        panel = Panel(
            f"[bold]Tool:[/bold] {tool.name}  "
            f"[dim]({tool.action_class})[/dim]\n"
            f"[bold]Arguments:[/bold]\n{json.dumps(call.arguments, indent=2)}",
            title="Approval required",
            border_style="yellow",
        )

        if level == "auto":
            return ApprovalDecision(allowed=True, reason="policy=auto")
        if level == "deny":
            return ApprovalDecision(allowed=False, reason="policy=deny")

        self.console.print(panel)

        if level == "ask":
            ok = self._prompt(f"Allow {tool.name}?")
            return ApprovalDecision(allowed=ok, reason=f"policy=ask user={'yes' if ok else 'no'}")

        if level == "double_confirm":
            if not self._prompt(f"DESTRUCTIVE action via {tool.name}. Allow?"):
                return ApprovalDecision(allowed=False, reason="first confirm denied")
            if not self._prompt("Are you ABSOLUTELY sure? Type y to proceed"):
                return ApprovalDecision(allowed=False, reason="second confirm denied")
            return ApprovalDecision(allowed=True, reason="double confirmed")

        return ApprovalDecision(allowed=False, reason=f"unknown policy {level}")
