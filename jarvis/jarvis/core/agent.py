"""Agent V3 loop: TaskState-centric, PolicyEngine-gated.

Responsibilities (intentionally narrow):
    - own the loop and the budget
    - dispatch decisions to the right component
    - emit AgentEvent(s) for UI/API consumers
    - drive TaskState lifecycle transitions

All *intelligence*  lives in Planner / Reasoner.
All *hard rules*    live in PolicyEngine.
All *side-effects*  live in Executor (with safety + retry).
"""
from __future__ import annotations

from typing import Iterator

from jarvis.core.executor import Executor
from jarvis.core.planner import Planner
from jarvis.core.policy_engine import ExecutionPolicyEngine
from jarvis.core.reasoner import Reasoner
from jarvis.core.schemas import (
    Action,
    AgentEvent,
    AgentStep,
    ChatMessage,
    FinalAnswer,
    Observation,
    Plan,
    ScratchpadEntry,
    Thought,
)
from jarvis.core.task_state import TaskState, make_state
from jarvis.llm.base import LLMError
from jarvis.logging_setup import get_logger
from jarvis.memory.manager import MemoryManager

log = get_logger(__name__)


class Agent:
    def __init__(
        self,
        planner: Planner,
        reasoner: Reasoner,
        executor: Executor,
        policy: ExecutionPolicyEngine,
        memory: MemoryManager,
        max_steps: int = 12,
        retrieve_k: int = 5,
    ) -> None:
        self.planner = planner
        self.reasoner = reasoner
        self.executor = executor
        self.policy = policy
        self.memory = memory
        self.max_steps = max_steps
        self.retrieve_k = retrieve_k

    # ---------------------------------------------------------------- public

    def stream(self, user_input: str) -> Iterator[AgentEvent]:
        self.memory.add_message(ChatMessage(role="user", content=user_input))
        memory_snippets = self.memory.recall_texts(user_input, k=self.retrieve_k)

        state = make_state(user_input)

        # ---- PLANNING phase --------------------------------------------
        try:
            state.plan = self.planner.plan(user_input, memory_snippets)
        except LLMError as e:
            yield from self._abort(state, f"Planner failed: {e}")
            return

        self.memory.log_plan(state.plan)
        yield state.plan
        state.transition("executing")

        # ---- EXECUTING phase -------------------------------------------
        while state.step_index < self.max_steps and not state.is_terminal():
            state.step_index += 1

            # 1) deterministic policy gate
            verdict = self.policy.before(state)
            if verdict.kind == "abort":
                yield from self._abort(state, verdict.reason)
                return
            if verdict.kind == "replan":
                new_plan = self._replan(state, verdict.reason)
                if new_plan is None:
                    yield from self._abort(state, "replan failed")
                    return
                yield new_plan
                continue

            # 2) plan complete? → finalize
            current = state.current_step()
            if current is None:
                yield from self._synthesize_final(state, user_input)
                return

            state.plan.mark(current.id, "in_progress")  # type: ignore[union-attr]

            # 3) reasoner picks one action
            try:
                decision = self.reasoner.decide(state, memory_snippets)
            except LLMError as e:
                yield from self._abort(state, f"Reasoner failed: {e}")
                return

            action = decision.action
            thought = decision.thought

            # 4) dispatch
            match action.type:
                case "final_answer":
                    yield self._emit(state, current.id, thought, action, None)
                    answer = action.final_answer or ""
                    self._persist_final(user_input, answer)
                    state.mark_completed(answer)
                    yield FinalAnswer(content=answer)
                    return

                case "abort":
                    yield self._emit(state, current.id, thought, action, None)
                    yield from self._abort(
                        state, action.reason or "reasoner abort"
                    )
                    return

                case "replan":
                    yield self._emit(state, current.id, thought, action, None)
                    state.append_entry(
                        self._entry(state, current.id, thought, action, None)
                    )
                    new_plan = self._replan(
                        state, action.reason or "reasoner replan"
                    )
                    if new_plan is None:
                        yield from self._abort(state, "replan failed")
                        return
                    yield new_plan
                    continue

                case "step_complete":
                    state.plan.mark(  # type: ignore[union-attr]
                        current.id, "done", note=action.note
                    )
                    yield self._emit(state, current.id, thought, action, None)
                    state.append_entry(
                        self._entry(state, current.id, thought, action, None)
                    )
                    continue

                case "tool_call":
                    observation = self.executor.execute(action)
                    assert action.tool_call is not None
                    self.policy.after(
                        state, action.tool_call.tool, observation
                    )
                    yield self._emit(
                        state, current.id, thought, action, observation
                    )
                    state.append_entry(
                        self._entry(
                            state, current.id, thought, action, observation
                        )
                    )
                    continue

                case _:
                    yield from self._abort(
                        state, f"unknown action type '{action.type}'"
                    )
                    return

        # ---- budget exhausted ------------------------------------------
        if not state.is_terminal():
            yield from self._abort(
                state, f"exceeded max_steps={self.max_steps}"
            )

    # ---------------------------------------------------------------- private

    def _replan(self, state: TaskState, reason: str) -> Plan | None:
        """Run the Planner and adopt the result. Returns None on LLM failure."""
        log.info("Replan triggered: %s", reason)
        state.transition("replanning")
        try:
            new_plan = self.planner.replan(state)
        except LLMError as e:
            log.error("Replan LLM failed: %s", e)
            return None
        state.mark_replan(new_plan)  # also transitions back to "executing"
        self.memory.log_plan(new_plan)
        return new_plan

    def _synthesize_final(
        self, state: TaskState, user_input: str
    ) -> Iterator[AgentEvent]:
        state.transition("finalizing")
        last_ok = next(
            (
                e.observation.content
                for e in reversed(state.history)
                if e.observation and e.observation.ok
            ),
            None,
        )
        text = (
            f"Completed all plan steps for: {state.objective}."
            + (f"\nLast result:\n{last_ok}" if last_ok else "")
        )
        action = Action(type="final_answer", final_answer=text)
        yield self._emit(state, None, "All plan steps complete.", action, None)
        self._persist_final(user_input, text)
        state.mark_completed(text)
        yield FinalAnswer(content=text)

    def _abort(self, state: TaskState, reason: str) -> Iterator[AgentEvent]:
        log.warning("Agent aborting: %s", reason)
        if not state.aborted:
            state.mark_aborted(reason)
        msg = f"Aborted: {reason}"
        self.memory.add_message(ChatMessage(role="assistant", content=msg))
        # Persist the abort context so future runs can learn from it.
        self.memory.consider(
            f"OBJECTIVE: {state.objective}\nABORTED: {reason}\n"
            f"SIGNALS: {state.snapshot_signals()}",
            source="agent.abort",
        )
        yield FinalAnswer(content=msg)

    def _persist_final(self, user_input: str, answer: str) -> None:
        self.memory.add_message(ChatMessage(role="assistant", content=answer))
        self.memory.consider(
            f"OBJECTIVE: {user_input}\nANSWER: {answer}", source="agent"
        )

    def _entry(
        self,
        state: TaskState,
        plan_step_id: int | None,
        thought: str,
        action: Action,
        observation: Observation | None,
    ) -> ScratchpadEntry:
        return ScratchpadEntry(
            step_index=state.step_index,
            plan_step_id=plan_step_id,
            thought=thought,
            action=action,
            observation=observation,
        )

    def _emit(
        self,
        state: TaskState,
        plan_step_id: int | None,
        thought: str,
        action: Action,
        observation: Observation | None,
    ) -> AgentStep:
        step = AgentStep(
            step=state.step_index,
            plan_step_id=plan_step_id,
            thought=Thought(reasoning=thought),
            action=action,
            observation=observation,
        )
        self.memory.log_step(step)
        return step

    # ---------------------------------------------------------------- convenience

    def run(self, user_input: str) -> str:
        final = ""
        for ev in self.stream(user_input):
            if isinstance(ev, FinalAnswer):
                final = ev.content
        return final
