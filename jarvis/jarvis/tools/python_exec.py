"""Python code execution tool (subprocess-isolated)."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from jarvis.logging_setup import get_logger
from jarvis.tools.base import Tool, ToolResult

log = get_logger(__name__)


class PythonExecTool(Tool):
    name = "python_exec"
    description = (
        "Execute a snippet of Python 3 code in an isolated subprocess. "
        "Returns stdout/stderr. Subject to safety approval."
    )
    action_class = "execute"

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout

    def arguments_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        }

    def _execute(self, arguments: dict[str, Any]) -> ToolResult:
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return ToolResult(ok=False, error="'code' must be a non-empty string.")

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "snippet.py"
            script.write_text(code, encoding="utf-8")
            log.info("PythonExecTool running snippet (%d chars)", len(code))
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(script)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmp,
                )
            except subprocess.TimeoutExpired:
                return ToolResult(
                    ok=False, error=f"Execution timed out after {self.timeout}s."
                )

        out = proc.stdout or ""
        err = proc.stderr or ""
        body = out if not err else f"{out}\n--- stderr ---\n{err}"
        return ToolResult(
            ok=proc.returncode == 0,
            content=body.strip(),
            error=None if proc.returncode == 0 else f"exit_code={proc.returncode}",
            metadata={"returncode": proc.returncode},
        )
