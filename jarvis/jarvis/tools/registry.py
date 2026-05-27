"""Tool registry."""
from __future__ import annotations

from jarvis.config.settings import ToolsConfig
from jarvis.tools.base import Tool
from jarvis.tools.filesystem import (
    FileDeleteTool,
    FileListTool,
    FileReadTool,
    FileWriteTool,
)
from jarvis.tools.python_exec import PythonExecTool
from jarvis.tools.shell import ShellTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("Tool must define a non-empty name.")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())


def build_default_registry(cfg: ToolsConfig) -> ToolRegistry:
    reg = ToolRegistry()
    if cfg.shell.enabled:
        reg.register(ShellTool(timeout=cfg.shell.timeout))
    if cfg.filesystem.enabled:
        root = cfg.filesystem.root
        reg.register(FileReadTool(root=root))
        reg.register(FileWriteTool(root=root))
        reg.register(FileListTool(root=root))
        reg.register(FileDeleteTool(root=root))
    if cfg.python_exec.enabled:
        reg.register(PythonExecTool(timeout=cfg.python_exec.timeout))
    return reg
