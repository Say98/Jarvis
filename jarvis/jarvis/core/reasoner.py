"""Reasoner: decides the next Action based on TaskState."""
from __future__ import annotations

import json

from jarvis.core.context import ContextManager
from jarvis.core.prompts import (
    REASONER_SYSTEM,
    reasoner_user_prompt,
    render_tools,
)
from jarvis.core.schemas import ChatMessage, ReasonerDecision
from jarvis.core.task_state import TaskState, render_history
from jarvis.llm.base import LLMProvider
from jarvis.logging_setup import get_logger
from jarvis.tools.base import Tool

log = get_logger(__name__)


class Reasoner:
    def __init__(
        self,
        llm: LLMProvider,
        tools: list[Tool],
        context: ContextManager,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.context = context

    def decide(
        self,
        state: TaskState,
        memory_snippets: list[str],
    ) -> ReasonerDecision:
        step = state.current_step()
        if step is None:
            raise RuntimeError("Reasoner.decide called with no current step.")

        banned = state.banned_for(step.id)
        signals_json = json.dumps(state.snapshot_signals())
        completed_json = json.dumps(state.completed_steps_summary(), indent=2)

        msgs = [
            ChatMessage(
                role="system",
                content=REASONER_SYSTEM.format(tools_block=render_tools(self.tools)),
            ),
            ChatMessage(
                role="user",
                content=reasoner_user_prompt(
                    objective=state.objective,
                    current_step_json=step.model_dump_json(indent=2),
                    completed_steps_json=completed_json,
                    history_json=render_history(state.history),
                    signals_json=signals_json,
                    banned_tools=banned,
                    memory_snippets=memory_snippets,
                ),
            ),
        ]
        msgs = self.context.fit(msgs)
        decision = self.llm.generate_json(msgs, ReasonerDecision, temperature=0.0)

        # Defensive: if the model picked a banned tool, downgrade to replan.
        if (
            decision.action.type == "tool_call"
            and decision.action.tool_call is not None
            and decision.action.tool_call.tool in banned
        ):
            log.warning(
                "Reasoner attempted banned tool '%s' for step %d; converting to replan.",
                decision.action.tool_call.tool,
                step.id,
            )
            from jarvis.core.schemas import Action

            decision = ReasonerDecision(
                thought=decision.thought,
                action=Action(
                    type="replan",
                    reason=(
                        f"reasoner picked banned tool '{decision.action.tool_call.tool}'"
                    ),
                ),
            )

        log.debug(
            "Reasoner decision: step=%d action=%s",
            step.id,
            decision.action.type,
        )
        return decision
