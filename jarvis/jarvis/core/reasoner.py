"""Reasoner: decides the next Action for the current plan step."""
from __future__ import annotations

from jarvis.core.context import ContextManager
from jarvis.core.prompts import (
    REASONER_SYSTEM,
    reasoner_user_prompt,
    render_tools,
)
from jarvis.core.schemas import ChatMessage, PlanStep, ReasonerDecision
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
        objective: str,
        current_step: PlanStep,
        scratchpad_json: str,
        memory_snippets: list[str],
    ) -> ReasonerDecision:
        msgs = [
            ChatMessage(
                role="system",
                content=REASONER_SYSTEM.format(tools_block=render_tools(self.tools)),
            ),
            ChatMessage(
                role="user",
                content=reasoner_user_prompt(
                    objective=objective,
                    current_step_json=current_step.model_dump_json(indent=2),
                    scratchpad_json=scratchpad_json,
                    memory_snippets=memory_snippets,
                ),
            ),
        ]
        msgs = self.context.fit(msgs)
        decision = self.llm.generate_json(msgs, ReasonerDecision, temperature=0.0)
        log.debug(
            "Reasoner decision: step=%d action=%s",
            current_step.id,
            decision.action.type,
        )
        return decision
