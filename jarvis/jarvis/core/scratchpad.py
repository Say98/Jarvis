"""Structured scratchpad of agent steps. Prompts are rebuilt from this."""
from __future__ import annotations

import json

from jarvis.core.schemas import Action, Observation, ScratchpadEntry

MAX_OBSERVATION_CHARS_IN_PROMPT = 2000


class Scratchpad:
    def __init__(self) -> None:
        self._entries: list[ScratchpadEntry] = []

    def add(self, entry: ScratchpadEntry) -> None:
        self._entries.append(entry)

    def all(self) -> list[ScratchpadEntry]:
        return list(self._entries)

    def last(self, n: int) -> list[ScratchpadEntry]:
        return self._entries[-n:] if n > 0 else []

    def clear(self) -> None:
        self._entries.clear()

    def render(self, last_n: int | None = None) -> str:
        """Render scratchpad as a clean JSON array for prompt injection."""
        entries = self._entries if last_n is None else self.last(last_n)
        out = []
        for e in entries:
            action_view: dict = {"type": e.action.type}
            if e.action.tool_call:
                action_view["tool"] = e.action.tool_call.tool
                action_view["arguments"] = e.action.tool_call.arguments
            if e.action.final_answer:
                action_view["final_answer"] = e.action.final_answer
            if e.action.note:
                action_view["note"] = e.action.note

            obs_view: dict | None = None
            if e.observation is not None:
                content = e.observation.content or ""
                if len(content) > MAX_OBSERVATION_CHARS_IN_PROMPT:
                    content = (
                        content[:MAX_OBSERVATION_CHARS_IN_PROMPT]
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
