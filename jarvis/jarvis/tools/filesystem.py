"""Filesystem tools: read, write, list, delete."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.logging_setup import get_logger
from jarvis.tools.base import Tool, ToolResult

log = get_logger(__name__)

MAX_READ_BYTES = 256 * 1024  # 256 KB


def _resolve_safe(root: Path, target: str) -> Path:
    """Resolve `target` relative to root and ensure it stays within root."""
    p = (root / target).resolve() if not Path(target).is_absolute() else Path(target).resolve()
    root_resolved = root.resolve()
    try:
        p.relative_to(root_resolved)
    except ValueError as e:
        raise PermissionError(
            f"Path '{p}' is outside of the allowed root '{root_resolved}'."
        ) from e
    return p


class FileReadTool(Tool):
    name = "file_read"
    description = "Read the contents of a UTF-8 text file under the configured root."
    action_class = "read"

    def __init__(self, root: str = ".") -> None:
        self.root = Path(root)

    def arguments_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative or absolute path."}
            },
            "required": ["path"],
        }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path")
        if not isinstance(path, str):
            return ToolResult(ok=False, error="'path' must be a string.")
        try:
            target = _resolve_safe(self.root, path)
            if not target.exists() or not target.is_file():
                return ToolResult(ok=False, error=f"File not found: {target}")
            data = target.read_bytes()
            truncated = len(data) > MAX_READ_BYTES
            text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
            if truncated:
                text += f"\n\n[... truncated at {MAX_READ_BYTES} bytes ...]"
            return ToolResult(
                ok=True,
                content=text,
                metadata={"path": str(target), "bytes": len(data), "truncated": truncated},
            )
        except Exception as e:
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")


class FileWriteTool(Tool):
    name = "file_write"
    description = (
        "Write text content to a file under the configured root. "
        "Creates parent directories. Overwrites existing files."
    )
    action_class = "write"

    def __init__(self, root: str = ".") -> None:
        self.root = Path(root)

    def arguments_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "append": {"type": "boolean", "default": False},
            },
            "required": ["path", "content"],
        }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path")
        content = arguments.get("content")
        append = bool(arguments.get("append", False))
        if not isinstance(path, str) or not isinstance(content, str):
            return ToolResult(ok=False, error="'path' and 'content' must be strings.")
        try:
            target = _resolve_safe(self.root, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with target.open(mode, encoding="utf-8") as fh:
                fh.write(content)
            return ToolResult(
                ok=True,
                content=f"Wrote {len(content)} chars to {target}",
                metadata={"path": str(target), "append": append},
            )
        except Exception as e:
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")


class FileListTool(Tool):
    name = "file_list"
    description = "List entries of a directory under the configured root."
    action_class = "read"

    def __init__(self, root: str = ".") -> None:
        self.root = Path(root)

    def arguments_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "recursive": {"type": "boolean", "default": False},
            },
        }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path", ".")
        recursive = bool(arguments.get("recursive", False))
        try:
            target = _resolve_safe(self.root, path)
            if not target.exists() or not target.is_dir():
                return ToolResult(ok=False, error=f"Directory not found: {target}")
            entries: list[str] = []
            iterator = target.rglob("*") if recursive else target.iterdir()
            for p in iterator:
                rel = p.relative_to(target)
                marker = "/" if p.is_dir() else ""
                entries.append(f"{rel}{marker}")
            entries.sort()
            return ToolResult(
                ok=True,
                content="\n".join(entries) if entries else "(empty)",
                metadata={"path": str(target), "count": len(entries)},
            )
        except Exception as e:
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")


class FileDeleteTool(Tool):
    name = "file_delete"
    description = "Delete a file under the configured root. Destructive."
    action_class = "destructive"

    def __init__(self, root: str = ".") -> None:
        self.root = Path(root)

    def arguments_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path")
        if not isinstance(path, str):
            return ToolResult(ok=False, error="'path' must be a string.")
        try:
            target = _resolve_safe(self.root, path)
            if not target.exists():
                return ToolResult(ok=False, error=f"Not found: {target}")
            if target.is_dir():
                return ToolResult(
                    ok=False,
                    error="Refusing to delete directory; use shell with care.",
                )
            target.unlink()
            return ToolResult(ok=True, content=f"Deleted {target}")
        except Exception as e:
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
