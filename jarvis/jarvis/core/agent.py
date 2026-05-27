"""Agent loop: orchestrates Planner → Reasoner → Executor with structured state."""
from __future__ import annotations

from typing import Iterator

from jarvis.core.executor import Executor
from jarvis.core.planner import Planner
from jarvis.core.reasoner import Reasoner
from jarvis.core.schemas import (
    Action,
    AgentEvent,
    AgentStep,
    ChatMessage,
    FinalAnswer,
    MemoryType,
    Observation,
    Plan,
    ScratchpadEntry,
    Thought,
)
from jarvis.core.scratchpad import Scratchpad
from jarvis.llm.base import LLMError
from jarvis.logging_setup import get_logger
from jarvis.memory.manager import MemoryManager

log = get_logger(__name__)


class Agent:
    """High-level loop:

      1. Planner.plan(objective)                       — once
      2. While plan not complete and budget left:
           a. Reasoner.decide(current_step, scratchpad)
           b. dispatch:
              - tool_call     → Executor.execute → append entry → continue
              - step_complete → mark step done   → continue
              - final_answer  → yield FinalAnswer, return
           c. on repeated failures → Planner.replan
      3. If budget exhausted → graceful FinalAnswer
    """

    def __init__(
        self,
        planner: Planner,
        reasoner: Reasoner,
        executor: Executor,
        memory: MemoryManager,
        max_steps: int = 12,
        retrieve_k: int = 5,
        replan_after_failures: int = 2,
    ) -> None:
        self.planner = planner
        self.reasoner = reasoner
        self.executor = executor
        self.memory = memory
        self.max_steps = max_steps
        self.retrieve_k = retrieve_k
        self.replan_after_failures = replan_after_failures

    # --------------------------------------------------------------- helpers

    def _recall(self, query: str) -> list[str]:
        return self.memory.recall_texts(query, k=self.retrieve_k)

    def _emit_step(
        self,
        step_idx: int,
        plan_step_id: int | None,
        thought: str,
        action: Action,
        observation: Observation | None,
    ) -> AgentStep:
        step = AgentStep(
            step=step_idx,
            plan_step_id=plan_step_id,
            thought=Thought(reasoning=thought),
            action=action,
            observation=observation,
        )
        self.memory.log_step(step)
        return step

    # --------------------------------------------------------------- main

    def stream(self, user_input: str) -> Iterator[AgentEvent]:
        self.memory.add_message(ChatMessage(role="user", content=user_input))
        memory_snippets = self._recall(user_input)

        # --- 1. PLAN ------------------------------------------------------
        try:
            plan: Plan = self.planner.plan(user_input, memory_snippets)
        except LLMError as e:
            err = f"Planner failed: {e}"
            log.error(err)
            yield FinalAnswer(content=err)
            return

        self.memory.log_plan(plan)
        yield plan

        scratchpad = Scratchpad()
        step_idx = 0

        # --- 2. LOOP ------------------------------------------------------
        while step_idx < self.max_steps:
            step_idx += 1

            current = plan.current()
            if current is None:
                # Plan complete but no final_answer yielded → synthesize one.
                final = self._synthesize_final(plan, scratchpad)
                yield self._emit_step(
                    step_idx, None, "All plan steps complete.", final, None
                )
                self.memory.add_message(
                    ChatMessage(role="assistant", content=final.final_answer or "")
                )
                self.memory.consider(
                    f"OBJECTIVE: {user_input}\nANSWER: {final.final_answer}",
                    source="agent",
                )
                yield FinalAnswer(content=final.final_answer or "")
                return

            plan.mark(current.id, "in_progress")

            try:
                decision = self.reasoner.decide(
                    objective=user_input,
                    current_step=current,
                    scratchpad_json=scratchpad.render(),
                    memory_snippets=memory_snippets,
                )
            except LLMError as e:
                err = f"Reasoner failed: {e}"
                log.error(err)
                yield FinalAnswer(content=err)
                return

            action = decision.action
            thought = decision.thought

            # --- dispatch -------------------------------------------------
            if action.type == "final_answer":
                step = self._emit_step(step_idx, current.id, thought, action, None)
                yield step
                self.memory.add_message(
                    ChatMessage(role="assistant", content=action.final_answer or "")
                )
                self.memory.consider(
                    f"OBJECTIVE: {user_input}\nANSWER: {action.final_answer}",
                    source="agent",
                )
                yield FinalAnswer(content=action.final_answer or "")
                return

            if action.type == "step_complete":
                plan.mark(current.id, "done", note=action.note)
                step = self._emit_step(step_idx, current.id, thought, action, None)
                scratchpad.add(
                    ScratchpadEntry(
                        step_index=step_idx,
                        plan_step_id=current.id,
                        thought=thought,
                        action=action,
                    )
                )
                yield step
                continue

            # tool_call
            observation = self.executor.execute(action)
            step = self._emit_step(
                step_idx, current.id, thought, action, observation
            )
            scratchpad.add(
                ScratchpadEntry(
                    step_index=step_idx,
                    plan_step_id=current.id,
                    thought=thought,
                    action=action,
                    observation=observation,
                )
            )
            yield step

            # --- replan / abort handling ----------------------------------
            if self.executor.should_abort():
                log.warning("Executor abort threshold reached; replanning.")
                try:
                    plan = self.planner.replan(plan, scratchpad.render())
                    self.memory.log_plan(plan)
                    self.executor.reset_failures()
                    yield plan
                except LLMError as e:
                    yield FinalAnswer(content=f"Replan failed: {e}")
                    return
                continue

            # Soft replan trigger if a single step keeps failing.
            if (
                not observation.ok
                and self.executor.consecutive_failures >= self.replan_after_failures
            ):
                plan.mark(current.id, "failed", note=observation.error)
                try:
                    plan = self.planner.replan(plan, scratchpad.render())
                    self.memory.log_plan(plan)
                    self.executor.reset_failures()
                    yield plan
                except LLMError as e:
                    yield FinalAnswer(content=f"Replan failed: {e}")
                    return

        # --- 3. BUDGET EXHAUSTED -----------------------------------------
        msg = f"Agent stopped: exceeded max_steps={self.max_steps}."
        self.memory.add_message(ChatMessage(role="assistant", content=msg))
        yield FinalAnswer(content=msg)

    # --------------------------------------------------------------- helpers

    def _synthesize_final(self, plan: Plan, scratchpad: Scratchpad) -> Action:
        """Build a fallback final answer summarizing scratchpad outputs."""
        last_obs = next(
            (
                e.observation.content
                for e in reversed(scratchpad.all())
                if e.observation and e.observation.ok
            ),
            None,
        )
        text = (
            f"Completed all plan steps for: {plan.objective}.\n"
            + (f"Last result:\n{last_obs}" if last_obs else "")
        )
        return Action(type="final_answer", final_answer=text)

    # --------------------------------------------------------------- convenience

    def run(self, user_input: str) -> str:
        final = ""
        for ev in self.stream(user_input):
            if isinstance(ev, FinalAnswer):
                final = ev.content
        return final
