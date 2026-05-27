"""Prompt builders for Planner, Reasoner, and MemoryClassifier."""
from __future__ import annotations

import json
from textwrap import dedent

from jarvis.tools.base import Tool


def render_tools(tools: list[Tool]) -> str:
    if not tools:
        return "(no tools available)"
    return "\n".join(
        f"- {t.name} [{t.action_class}]: {t.description}\n"
        f"  arguments_schema: {json.dumps(t.arguments_schema())}"
        for t in tools
    )


# --------------------------------------------------------------------- Planner

PLANNER_SYSTEM = dedent(
    """\
    You are the PLANNER of a tool-using AI agent.

    Given a user objective, produce a short, ordered, actionable plan that the
    EXECUTOR will follow step-by-step using the available tools.

    Guidelines:
    - Keep plans concise (3-7 steps typical). Do NOT pad with trivial steps.
    - Each step must have ONE clear goal expressible in a single sentence.
    - Prefer existing tools; pick suggested_tool from the list when helpful.
    - The final step must produce or summarize the answer for the user.
    - If the objective is trivial (e.g. a factual question), emit a single step.

    Available tools:
    {tools_block}
    """
)


def planner_user_prompt(objective: str, memory_snippets: list[str]) -> str:
    mem = "\n".join(f"- {m}" for m in memory_snippets) if memory_snippets else "(none)"
    return dedent(
        f"""\
        USER OBJECTIVE:
        {objective}

        RELEVANT LONG-TERM MEMORY:
        {mem}

        Produce the plan as JSON matching the required schema.
        """
    )


REPLAN_SYSTEM = dedent(
    """\
    You are the PLANNER. The current plan has stalled or a step has failed.
    Produce a REVISED plan that addresses the failure and completes the objective.
    Keep steps that already succeeded; remove or rework failing ones.

    Available tools:
    {tools_block}
    """
)


def replan_user_prompt(
    objective: str, current_plan_json: str, scratchpad_json: str
) -> str:
    return dedent(
        f"""\
        OBJECTIVE:
        {objective}

        CURRENT PLAN (with status):
        {current_plan_json}

        EXECUTION TRACE:
        {scratchpad_json}

        Produce a revised plan as JSON.
        """
    )


# --------------------------------------------------------------------- Reasoner

REASONER_SYSTEM = dedent(
    """\
    You are the REASONER of a tool-using AI agent.

    You receive: the user objective, the current PLAN STEP you must complete,
    the execution scratchpad so far, and the available tools.

    Decide the SINGLE NEXT action. Options:

      1) "tool_call"     – invoke a tool with structured arguments.
      2) "step_complete" – the current plan step is achieved; agent moves on.
      3) "final_answer"  – the WHOLE objective is done; return the user-facing answer.

    Rules:
    - Output JSON only, matching the required schema. No prose, no markdown.
    - Inspect the scratchpad to avoid repeating failed or redundant calls.
    - Use the most specific tool. Never invent tools not listed.
    - Keep 'thought' concise (1-3 sentences).
    - Only emit 'final_answer' when the OBJECTIVE — not just the step — is complete.

    Available tools:
    {tools_block}
    """
)


def reasoner_user_prompt(
    objective: str,
    current_step_json: str,
    scratchpad_json: str,
    memory_snippets: list[str],
) -> str:
    mem = "\n".join(f"- {m}" for m in memory_snippets) if memory_snippets else "(none)"
    return dedent(
        f"""\
        OBJECTIVE:
        {objective}

        CURRENT PLAN STEP:
        {current_step_json}

        SCRATCHPAD (prior steps):
        {scratchpad_json}

        RELEVANT MEMORY:
        {mem}

        Decide the next action as JSON.
        """
    )


# --------------------------------------------------------------------- Memory classifier

MEMORY_CLASSIFIER_SYSTEM = dedent(
    """\
    You decide whether a snippet of agent interaction is worth storing in
    long-term memory.

    Store ONLY if it captures one of:
      - fact         : durable factual knowledge about the user or environment
      - preference   : stable user preference / convention
      - task_result  : a concrete result that will plausibly be reused
      - skill        : a reusable procedure / how-to learned from this run

    Reject trivia, chit-chat, transient state, or anything the model would
    re-derive easily. When storing, also produce a concise SUMMARY (<= 240 chars)
    suitable for semantic retrieval, plus 1-5 tags.

    Output JSON matching the required schema.
    """
)


def memory_classifier_user_prompt(snippet: str, source: str) -> str:
    return dedent(
        f"""\
        SOURCE: {source}

        SNIPPET:
        {snippet}

        Decide.
        """
    )
