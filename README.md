# Jarvis V3 – Production-grade Local AI Agent

Jarvis is a fully local, model-agnostic **autonomous AI agent**. Not a chatbot.

V3 introduces a `TaskState`-centric loop, a deterministic `ExecutionPolicyEngine`,
and an upgraded Reasoner that can explicitly choose to **replan** or **abort**.

## Architecture

```
                        ┌──────────────────────────┐
                        │       TaskState          │
                        │  objective, plan,        │
                        │  history, failures,      │
                        │  signals, banned_tools   │
                        └─────────────┬────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                ▼                     ▼                     ▼
          Planner               PolicyEngine            Reasoner
       (DAG of steps,       (deterministic rules:     (LLM, 5 actions:
        depends_on,          replan / abort /          tool_call,
        expected_output)     continue)                 step_complete,
                                                       replan, abort,
                                                       final_answer)
                                      │
                                      ▼
                                  Executor
                              (retry on read/execute,
                               size limits, timing)
                                      │
                                      ▼
                              MemoryManager
                          (typed + LLM-filtered)
```

## V3 changes vs V2

| Area | V3 upgrade |
|---|---|
| **State** | New [`TaskState`](jarvis/jarvis/core/task_state.py) — single source of truth: plan, history, failures, banned_tools_for_step, signals. |
| **Loop** | [`Agent`](jarvis/jarvis/core/agent.py) is a thin coordinator over `TaskState`; all intelligence in Planner/Reasoner, all hard rules in PolicyEngine. |
| **Reasoner** | Can emit `tool_call`, `step_complete`, `final_answer`, **`replan`**, **`abort`**. Banned-tool defense converts illegal picks into `replan`. |
| **PolicyEngine** | New deterministic [`ExecutionPolicyEngine`](jarvis/jarvis/core/policy_engine.py): `before()` returns `continue / replan / abort`; `after()` updates tool-failure counters and bans tools per step. |
| **Planner** | Steps carry `depends_on`, `expected_output`. `Plan.current()` respects deps; `Plan.is_stalled()` triggers replan. Replan receives banned-tools, signals, full history. |
| **Executor** | Per-call retry for `read`/`execute` classes (never for `write`/`destructive`); attempts + elapsed_s in metadata; permission/validation errors are not retried. |
| **Memory** | Typed (`fact`/`preference`/`task_result`/`skill`) with LLM-backed [`MemoryClassifier`](jarvis/jarvis/memory/classifier.py) gate. |
| **LLM** | `generate_json(schema)` with retry-on-validation; Ollama `format: json`. |
| **Context** | Token-budgeted message trimming. |

## Strict Reasoner output schema

```json
{
  "thought": "...",
  "action": {
    "type": "tool_call | step_complete | final_answer | replan | abort",
    "tool_call":   { "tool": "...", "arguments": { ... } },
    "final_answer": "...",
    "note":   "...",
    "reason": "..."
  }
}
```

Pydantic-validated; non-conforming output is retried up to N times with the
validation error fed back to the model.

## Loop pseudocode (exact)

```
state = TaskState(objective)
state.plan = Planner.plan(objective, memory)
while step_budget left:
    verdict = PolicyEngine.before(state)
    if verdict == "abort":  yield FinalAnswer(reason); return
    if verdict == "replan": state.plan = Planner.replan(state); continue
    step = state.current_step()
    if step is None: yield synthesized FinalAnswer; return
    decision = Reasoner.decide(state, memory)
    match decision.action.type:
        tool_call     → obs = Executor.execute(...); PolicyEngine.after(state, tool, obs)
        step_complete → plan.mark(step, "done")
        replan        → state.plan = Planner.replan(state)
        abort         → yield FinalAnswer(reason); return
        final_answer  → yield FinalAnswer; return
```

## Install & run

```bash
pip install -e .
ollama pull llama3
jarvis chat -v     # interactive REPL with full step trace + plan table
jarvis run "..."   # one-shot
jarvis serve       # FastAPI on :8000
jarvis tools       # list registered tools
python examples/basic_usage.py
```

## Configuration (key V3 fields)

`jarvis/config/default.yaml` (env override: `JARVIS_*` with `__` for nesting).

```yaml
agent:
  max_steps: 12
  retrieve_k: 5
  # ExecutionPolicyEngine thresholds
  max_consecutive_failures: 3
  max_total_failures: 8
  max_replans: 2
  ban_tool_after_failures: 2
  same_tool_loop_window: 3
  # Executor
  tool_retry: 1
  # Context window
  context_max_tokens: 6000
  context_keep_recent: 6
```

## Safety

| Action class | Policy          |
|--------------|-----------------|
| read         | auto            |
| write        | ask             |
| execute      | ask             |
| destructive  | double_confirm  |

The `Executor` is the single chokepoint: every tool call passes through the
`Approver` before running, then through the per-class retry policy.

## Project layout

```
jarvis/
├── core/
│   ├── agent.py            # V3 loop (TaskState + PolicyEngine)
│   ├── task_state.py       # NEW: central state
│   ├── policy_engine.py    # NEW: deterministic execution policy
│   ├── planner.py          # dep-aware plans
│   ├── reasoner.py         # tool_call/step_complete/final_answer/replan/abort
│   ├── executor.py         # retry + guards + size limits
│   ├── context.py          # token budgeting
│   ├── prompts.py
│   ├── schemas.py
│   └── orchestrator.py     # wiring
├── llm/                    # generate + generate_json + retry
├── memory/                 # short_term, long_term, store, classifier, manager
├── safety/                 # approver + safety policy
├── tools/                  # base + shell/fs/python_exec + file_patch + limits
├── api/server.py
├── config/
└── cli.py
```

## Extending

- **New tool**: subclass `Tool`, implement `_execute`; register in `tools/registry.py`. Wrapper handles error capture, size truncation, normalization.
- **New LLM provider**: subclass `LLMProvider`, implement `generate`; `generate_json` works out of the box.
- **New policy rule**: add a check in `ExecutionPolicyEngine.before/after`. It's deterministic Python — no LLM in the loop for safety rails.
- **New action type**: extend the `Action.type` Literal and handle it in `Agent.stream`.
