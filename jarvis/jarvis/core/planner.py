"""Planner: turns an objective into a structured, dependency-aware Plan."""
from __future__ import annotations

import json
from typing import Any

from jarvis.core.context import ContextManager
from jarvis.core.prompts import (
    PLANNER_SYSTEM,
    REPLAN_SYSTEM,
    planner_user_prompt,
    render_tools,
    replan_user_prompt,
)
from jarvis.core.schemas import ChatMessage, Plan, PlanDraft, PlanStep
from jarvis.core.task_state import TaskState, render_history
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

    # ---------------------------------------------------------------- build

    def _draft_to_plan(self, draft: PlanDraft) -> Plan:
        steps: list[PlanStep] = []
        for idx, raw in enumerate(draft.steps, start=1):
            depends_on_raw = raw.get("depends_on") or []
            if not isinstance(depends_on_raw, list):
                depends_on_raw = []
            depends_on = [int(x) for x in depends_on_raw if isinstance(x, (int, str)) and str(x).isdigit()]

            steps.append(
                PlanStep(
                    id=idx,
                    goal=str(raw.get("goal", "")).strip() or f"Step {idx}",
                    suggested_tool=raw.get("suggested_tool"),
                    success_criteria=raw.get("success_criteria"),
                    expected_output=raw.get("expected_output"),
                    depends_on=[d for d in depends_on if 1 <= d < idx],
                )
            )
        if not steps:
            steps.append(PlanStep(id=1, goal=draft.objective))
        return Plan(objective=draft.objective, steps=steps)

    # ---------------------------------------------------------------- plan

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
        log.info(
            "Planner produced %d steps for objective: %s", len(plan.steps), objective
        )
        return plan

    # ---------------------------------------------------------------- replan

    def replan(self, state: TaskState) -> Plan:
        assert state.plan is not None, "Cannot replan without an existing plan."
        banned_json = json.dumps(
            {
                str(step_id): tools
                for step_id, tools in state.banned_tools_for_step.items()
            }
        )
        persistent_bans_json = json.dumps(list(state.persistent_tool_bans))
        signals_json = json.dumps(state.snapshot_signals())
        completed_json = json.dumps(state.completed_steps_summary(), indent=2)
        failed_json = json.dumps(list(state.failed_step_history), indent=2)

        msgs = [
            ChatMessage(
                role="system",
                content=REPLAN_SYSTEM.format(tools_block=render_tools(self.tools)),
            ),
            ChatMessage(
                role="user",
                content=replan_user_prompt(
                    objective=state.objective,
                    current_plan_json=state.plan.model_dump_json(indent=2),
                    completed_steps_json=completed_json,
                    failed_steps_json=failed_json,
                    history_json=render_history(state.history),
                    signals_json=signals_json,
                    banned_json=banned_json,
                    persistent_bans_json=persistent_bans_json,
                ),
            ),
        ]
        msgs = self.context.fit(msgs)
        draft = self.llm.generate_json(msgs, PlanDraft, temperature=0.0)
        new_plan = self._draft_to_plan(draft)
        new_plan.revision = state.plan.revision + 1
        log.info(
            "Planner revised plan (rev=%d, %d steps)",
            new_plan.revision,
            len(new_plan.steps),
        )
        return new_plan
