"""Shell execution tool."""
from __future__ import annotations

import shlex
import subprocess
import sys
from typing import Any

from jarvis.logging_setup import get_logger
from jarvis.tools.base import Tool, ToolResult

log = get_logger(__name__)


class ShellTool(Tool):
    name = "shell"
    description = (
        "Execute a shell command on the local machine and return stdout/stderr. "
        "Use sparingly; subject to safety approval."
    )
    action_class = "execute"

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def arguments_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The exact shell command to execute.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory.",
                },
            },
            "required": ["command"],
        }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        cmd = arguments.get("command")
        cwd = arguments.get("cwd")
        if not isinstance(cmd, str) or not cmd.strip():
            return ToolResult(ok=False, error="'command' must be a non-empty string.")

        log.info("ShellTool executing: %s", cmd)
        try:
            if sys.platform == "win32":
                # Use shell=True on Windows for built-ins like dir
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=cwd,
                )
            else:
                proc = subprocess.run(
                    shlex.split(cmd),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=cwd,
                )
        except subprocess.TimeoutExpired:
            return ToolResult(
                ok=False, error=f"Command timed out after {self.timeout}s."
            )
        except Exception as e:
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")

        out = proc.stdout or ""
        err = proc.stderr or ""
        body = out if not err else f"{out}\n--- stderr ---\n{err}"
        return ToolResult(
            ok=proc.returncode == 0,
            content=body.strip(),
            error=None if proc.returncode == 0 else f"exit_code={proc.returncode}",
            metadata={"returncode": proc.returncode, "command": cmd},
        )
