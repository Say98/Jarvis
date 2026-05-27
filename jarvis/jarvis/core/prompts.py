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
    You are the PLANNER of a tool-using autonomous AI agent.

    Given a user objective, produce a SHORT, ORDERED, EXECUTABLE plan.

    Each step must declare:
      - "goal"            : one-sentence objective for the step
      - "suggested_tool"  : a tool name from the available list (or null)
      - "success_criteria": observable outcome that proves the step succeeded
      - "expected_output" : what the next step needs from this one
      - "depends_on"      : list of step ids (1-based) this step requires

    Rules:
    - Keep plans concise (3-7 steps typical).
    - Use dependencies for real ordering only — they form a DAG.
    - Prefer existing tools; never invent unavailable ones.
    - The FINAL step must produce or summarize the answer for the user.

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
    You are the PLANNER. The current plan has stalled or failed.
    Produce a REVISED plan that recovers and completes the objective.

    Guidelines:
    - PRESERVE work already done: the COMPLETED STEPS list is authoritative;
      do not re-do them. Build the new plan as the remaining work only.
    - Avoid repeating known-failed tool/argument combinations (see HISTORY).
    - Respect the BANNED TOOLS list (per step AND persistent).
    - If a step in FAILED STEPS proved infeasible as written, change the
      approach — different tool, different decomposition, or skip it.
    - Each step must declare goal / suggested_tool / success_criteria /
      expected_output / depends_on (DAG, 1-based ids restarting at 1).

    Available tools:
    {tools_block}
    """
)


def replan_user_prompt(
    objective: str,
    current_plan_json: str,
    completed_steps_json: str,
    failed_steps_json: str,
    history_json: str,
    signals_json: str,
    banned_json: str,
    persistent_bans_json: str,
) -> str:
    return dedent(
        f"""\
        OBJECTIVE:
        {objective}

        CURRENT PLAN (with status / attempts):
        {current_plan_json}

        COMPLETED STEPS (do NOT redo these):
        {completed_steps_json}

        FAILED STEPS (do NOT retry the same approach):
        {failed_steps_json}

        EXECUTION HISTORY:
        {history_json}

        TASK SIGNALS:
        {signals_json}

        BANNED TOOLS PER STEP:
        {banned_json}

        PERSISTENT BANNED TOOLS (whole task):
        {persistent_bans_json}

        Produce a revised plan as JSON. Number the new steps starting at 1.
        """
    )


# --------------------------------------------------------------------- Reasoner

REASONER_SYSTEM = dedent(
    """\
    You are the REASONER of an autonomous AI agent.

    You receive: the user objective, the CURRENT PLAN STEP (including its
    expected_output and success_criteria), the execution history, current
    task signals, and the available tools.

    Decide the SINGLE NEXT action. Allowed action types:

      1) "tool_call"     – invoke a tool with structured arguments.
      2) "step_complete" – the CURRENT step's success_criteria is satisfied
                           by prior observations; agent moves on.
      3) "final_answer"  – the WHOLE objective is done; emit the user answer.
      4) "replan"        – the current plan is no longer viable; trigger replan.
      5) "abort"         – the task is impossible or unsafe; stop with reason.

    Hard rules:
    - Output JSON only, matching the required schema. No prose, no markdown.
    - NEVER invoke tools listed under BANNED_TOOLS for the current step.
    - The current step is DONE only when its expected_output / success_criteria
      is supported by the EXECUTION HISTORY. If yes → "step_complete".
    - Do not call tools redundantly; inspect history first.
    - Prefer "replan" over "abort"; reserve "abort" for truly impossible tasks.
    - Only emit "final_answer" when the WHOLE OBJECTIVE is complete.
    - Keep 'thought' concise (1-3 sentences).
    - Always set action.reason for replan or abort.

    Available tools:
    {tools_block}
    """
)


def reasoner_user_prompt(
    objective: str,
    current_step_json: str,
    completed_steps_json: str,
    history_json: str,
    signals_json: str,
    banned_tools: list[str],
    memory_snippets: list[str],
) -> str:
    mem = "\n".join(f"- {m}" for m in memory_snippets) if memory_snippets else "(none)"
    banned = ", ".join(banned_tools) if banned_tools else "(none)"
    return dedent(
        f"""\
        OBJECTIVE:
        {objective}

        CURRENT PLAN STEP (focus on its success_criteria + expected_output):
        {current_step_json}

        COMPLETED STEPS (already satisfied):
        {completed_steps_json}

        BANNED_TOOLS for this step: {banned}

        TASK SIGNALS:
        {signals_json}

        EXECUTION HISTORY:
        {history_json}

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
