"""Agent loop: THINK → DECIDE → ACT → OBSERVE → STORE → REPEAT."""
from __future__ import annotations

import json
import re
from typing import Iterator

from pydantic import ValidationError

from jarvis.core.prompts import build_system_prompt
from jarvis.core.schemas import (
    Action,
    AgentStep,
    ChatMessage,
    FinalAnswer,
    Observation,
    Thought,
    ToolCall,
)
from jarvis.llm.base import LLMProvider
from jarvis.logging_setup import get_logger
from jarvis.memory.manager import MemoryManager
from jarvis.safety.approver import Approver
from jarvis.tools.registry import ToolRegistry

log = get_logger(__name__)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> str:
    """Extract the first JSON object from a model response."""
    text = text.strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1)
    # Fallback: locate first { ... } balanced span.
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    raise ValueError("Unbalanced JSON in model output.")


def _parse_decision(raw: str) -> tuple[Thought, Action]:
    js = _extract_json(raw)
    try:
        data = json.loads(js)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from model: {e}") from e

    thought_text = data.get("thought", "")
    action_data = data.get("action")
    if not isinstance(action_data, dict):
        raise ValueError("Missing 'action' object in model output.")

    try:
        if action_data.get("type") == "tool_call":
            tc = ToolCall(**action_data["tool_call"])
            action = Action(type="tool_call", tool_call=tc)
        elif action_data.get("type") == "final_answer":
            action = Action(
                type="final_answer", final_answer=action_data.get("final_answer", "")
            )
        else:
            raise ValueError(f"Unknown action type: {action_data.get('type')!r}")
    except ValidationError as e:
        raise ValueError(f"Action validation failed: {e}") from e

    return Thought(reasoning=str(thought_text)), action


class Agent:
    """Tool-using agent. Stateless across `run` calls; uses MemoryManager for state."""

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        memory: MemoryManager,
        approver: Approver,
        max_steps: int = 12,
        retrieve_k: int = 5,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.approver = approver
        self.max_steps = max_steps
        self.retrieve_k = retrieve_k

    def _build_messages(self, user_input: str) -> list[ChatMessage]:
        snippets = self.memory.recall(user_input, k=self.retrieve_k)
        system = build_system_prompt(self.tools.all(), snippets)
        msgs: list[ChatMessage] = [ChatMessage(role="system", content=system)]
        msgs.extend(self.memory.conversation())
        msgs.append(ChatMessage(role="user", content=user_input))
        return msgs

    def _execute_tool(self, call: ToolCall) -> Observation:
        tool = self.tools.get(call.tool)
        if tool is None:
            return Observation(
                ok=False,
                content="",
                error=f"Unknown tool '{call.tool}'. Available: {self.tools.names()}",
            )

        decision = self.approver.approve(tool, call)
        if not decision.allowed:
            return Observation(
                ok=False,
                content="",
                error=f"Action denied by safety layer: {decision.reason}",
            )

        result = tool.run(call.arguments)
        return Observation(
            ok=result.ok,
            content=result.content,
            error=result.error,
            metadata=result.metadata,
        )

    def stream(self, user_input: str) -> Iterator[AgentStep | FinalAnswer]:
        """Run the agent loop, yielding each AgentStep and the FinalAnswer."""
        self.memory.add_message(ChatMessage(role="user", content=user_input))
        scratch: list[ChatMessage] = []

        for step_idx in range(1, self.max_steps + 1):
            base = self._build_messages(user_input)
            messages = base + scratch

            log.debug("Agent step %d: %d msgs", step_idx, len(messages))
            try:
                raw = self.llm.generate(messages)
            except Exception as e:
                err = f"LLM error: {type(e).__name__}: {e}"
                step = AgentStep(
                    step=step_idx,
                    thought=Thought(reasoning="LLM failure"),
                    action=Action(type="final_answer", final_answer=err),
                    observation=Observation(ok=False, content="", error=err),
                )
                self.memory.log_step(step)
                yield step
                yield FinalAnswer(content=err)
                return

            try:
                thought, action = _parse_decision(raw)
            except ValueError as e:
                log.warning("Parse failure: %s\nRaw:\n%s", e, raw)
                # Feed the error back to the model for self-correction.
                scratch.append(ChatMessage(role="assistant", content=raw))
                scratch.append(
                    ChatMessage(
                        role="user",
                        content=(
                            f"Your previous output could not be parsed: {e}. "
                            "Reply ONLY with a single valid JSON object matching the schema."
                        ),
                    )
                )
                continue

            if action.type == "final_answer":
                step = AgentStep(step=step_idx, thought=thought, action=action)
                self.memory.log_step(step)
                self.memory.add_message(
                    ChatMessage(role="assistant", content=action.final_answer or "")
                )
                self.memory.remember(
                    f"User asked: {user_input}\nAssistant answered: {action.final_answer}",
                    metadata={"kind": "qa"},
                )
                yield step
                yield FinalAnswer(content=action.final_answer or "")
                return

            assert action.tool_call is not None
            observation = self._execute_tool(action.tool_call)
            step = AgentStep(
                step=step_idx, thought=thought, action=action, observation=observation
            )
            self.memory.log_step(step)

            # Feed the observation back into the conversation for next iteration.
            scratch.append(ChatMessage(role="assistant", content=raw))
            scratch.append(
                ChatMessage(
                    role="user",
                    content=(
                        f"Observation from tool '{action.tool_call.tool}' "
                        f"(ok={observation.ok}):\n{observation.content or ''}"
                        + (f"\nerror: {observation.error}" if observation.error else "")
                        + "\n\nDecide your next action as JSON."
                    ),
                )
            )
            yield step

        # Exceeded max steps
        msg = f"Agent stopped: exceeded max_steps={self.max_steps}."
        self.memory.add_message(ChatMessage(role="assistant", content=msg))
        yield FinalAnswer(content=msg)

    def run(self, user_input: str) -> str:
        """Convenience: run to completion and return the final answer."""
        final = ""
        for event in self.stream(user_input):
            if isinstance(event, FinalAnswer):
                final = event.content
        return final
