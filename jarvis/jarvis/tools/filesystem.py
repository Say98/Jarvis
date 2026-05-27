"""Filesystem tools: read, write, list, delete, patch (apply edits)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.logging_setup import get_logger
from jarvis.tools.base import Tool, ToolResult

log = get_logger(__name__)

MAX_READ_BYTES = 256 * 1024  # 256 KB


def _resolve_safe(root: Path, target: str) -> Path:
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
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    def _execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path")
        if not isinstance(path, str):
            return ToolResult(ok=False, error="'path' must be a string.")
        target = _resolve_safe(self.root, path)
        if not target.exists() or not target.is_file():
            return ToolResult(ok=False, error=f"File not found: {target}")
        data = target.read_bytes()
        truncated = len(data) > MAX_READ_BYTES
        text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
        return ToolResult(
            ok=True,
            content=text,
            metadata={"path": str(target), "bytes": len(data)},
            truncated=truncated,
        )


class FileWriteTool(Tool):
    name = "file_write"
    description = (
        "Write text content to a file under the configured root. "
        "Creates parent directories. Overwrites existing files unless append=true."
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

    def _execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path")
        content = arguments.get("content")
        append = bool(arguments.get("append", False))
        if not isinstance(path, str) or not isinstance(content, str):
            return ToolResult(ok=False, error="'path' and 'content' must be strings.")
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

    def _execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path", ".")
        recursive = bool(arguments.get("recursive", False))
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


class FileDeleteTool(Tool):
    name = "file_delete"
    description = "Delete a single file under the configured root. Destructive."
    action_class = "destructive"

    def __init__(self, root: str = ".") -> None:
        self.root = Path(root)

    def arguments_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    def _execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path")
        if not isinstance(path, str):
            return ToolResult(ok=False, error="'path' must be a string.")
        target = _resolve_safe(self.root, path)
        if not target.exists():
            return ToolResult(ok=False, error=f"Not found: {target}")
        if target.is_dir():
            return ToolResult(
                ok=False, error="Refusing to delete directory; use shell with care."
            )
        target.unlink()
        return ToolResult(ok=True, content=f"Deleted {target}")


class FilePatchTool(Tool):
    """Apply a search/replace edit to a file. Safer than full rewrite."""

    name = "file_patch"
    description = (
        "Apply a targeted edit to a UTF-8 text file by exact search/replace. "
        "Fails if 'old' is not found exactly once (no ambiguous edits)."
    )
    action_class = "write"

    def __init__(self, root: str = ".") -> None:
        self.root = Path(root)

    def arguments_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string", "description": "Exact text to replace."},
                "new": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old", "new"],
        }

    def _execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path")
        old = arguments.get("old")
        new = arguments.get("new")
        if not all(isinstance(x, str) for x in (path, old, new)):
            return ToolResult(ok=False, error="path/old/new must all be strings.")
        target = _resolve_safe(self.root, path)  # type: ignore[arg-type]
        if not target.exists() or not target.is_file():
            return ToolResult(ok=False, error=f"File not found: {target}")
        original = target.read_text(encoding="utf-8")
        count = original.count(old)  # type: ignore[arg-type]
        if count == 0:
            return ToolResult(ok=False, error="'old' string not found in file.")
        if count > 1:
            return ToolResult(
                ok=False,
                error=f"'old' string matches {count} times; refusing ambiguous edit.",
            )
        patched = original.replace(old, new, 1)  # type: ignore[arg-type]
        target.write_text(patched, encoding="utf-8")
        return ToolResult(
            ok=True,
            content=f"Patched {target} (delta={len(patched) - len(original)} chars)",
            metadata={"path": str(target)},
        )
