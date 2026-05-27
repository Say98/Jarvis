"""Planner: turns an objective into a structured multi-step Plan."""
from __future__ import annotations

from jarvis.core.context import ContextManager
from jarvis.core.prompts import (
    PLANNER_SYSTEM,
    REPLAN_SYSTEM,
    planner_user_prompt,
    render_tools,
    replan_user_prompt,
)
from jarvis.core.schemas import ChatMessage, Plan, PlanDraft, PlanStep
from jarvis.llm.base import LLMProvider
from jarvis.logging_setup import get_logger
from jarvis.tools.base import Tool

log = get_logger(__name__)


class Planner:
    def __init__(
        self,
        llm: LLMProvider,
        tools: list[Tool],
        context: ContextManager,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.context = context

    def _draft_to_plan(self, draft: PlanDraft) -> Plan:
        steps: list[PlanStep] = []
        for idx, raw in enumerate(draft.steps, start=1):
            steps.append(
                PlanStep(
                    id=idx,
                    goal=str(raw.get("goal", "")).strip() or f"Step {idx}",
                    suggested_tool=raw.get("suggested_tool"),
                    success_criteria=raw.get("success_criteria"),
                )
            )
        if not steps:
            # Always at least one step.
            steps.append(PlanStep(id=1, goal=draft.objective))
        return Plan(objective=draft.objective, steps=steps)

    def plan(self, objective: str, memory_snippets: list[str]) -> Plan:
        msgs = [
            ChatMessage(
                role="system",
                content=PLANNER_SYSTEM.format(tools_block=render_tools(self.tools)),
            ),
            ChatMessage(
                role="user",
                content=planner_user_prompt(objective, memory_snippets),
            ),
        ]
        msgs = self.context.fit(msgs)
        draft = self.llm.generate_json(msgs, PlanDraft, temperature=0.0)
        plan = self._draft_to_plan(draft)
        log.info("Planner produced %d steps for objective: %s", len(plan.steps), objective)
        return plan

    def replan(self, plan: Plan, scratchpad_json: str) -> Plan:
        msgs = [
            ChatMessage(
                role="system",
                content=REPLAN_SYSTEM.format(tools_block=render_tools(self.tools)),
            ),
            ChatMessage(
                role="user",
                content=replan_user_prompt(
                    plan.objective,
                    plan.model_dump_json(indent=2),
                    scratchpad_json,
                ),
            ),
        ]
        msgs = self.context.fit(msgs)
        draft = self.llm.generate_json(msgs, PlanDraft, temperature=0.0)
        new_plan = self._draft_to_plan(draft)
        new_plan.revision = plan.revision + 1
        log.info(
            "Planner revised plan (rev=%d, %d steps)",
            new_plan.revision,
            len(new_plan.steps),
        )
        return new_plan
