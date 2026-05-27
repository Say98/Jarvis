# Jarvis v2 – Local AI Agent

Jarvis is a fully local, model-agnostic AI **agent** (not a chatbot) with planning,
structured reasoning, real tool execution, typed memory, and a safety layer.

## What's new in v2

- **Planner / Reasoner / Executor split** – the old monolithic `Agent` is now a
  thin orchestrator over three single-purpose components.
- **`generate_json(schema=...)`** on the LLM provider – strict Pydantic-validated
  output with self-correcting retry. No regex parsing for control flow.
- **Ollama JSON mode** (`"format": "json"`) used for all structured calls.
- **Structured scratchpad** – previous steps are stored as typed entries and
  rebuilt into prompts as a clean JSON array; raw model text is never re-fed.
- **Typed long-term memory** – every entry has a `MemoryType` (`fact`,
  `preference`, `task_result`, `skill`); a **MemoryClassifier** decides whether
  a snippet is worth persisting before it touches Chroma.
- **Context manager** – token-budgeted trimming of system + recent + relevant
  messages.
- **Executor with safety + size limits + failure tracking + replan trigger**.
- **`file_patch` tool** – targeted, unambiguous search/replace edits.

## Architecture

```
Agent  (loop orchestrator)
 ├── Planner    → produces a Plan(objective, steps[])
 ├── Reasoner   → per-step ReasonerDecision(thought, action)
 ├── Executor   → safety-checks + runs tool, returns Observation
 └── MemoryManager
       ├── ShortTermMemory   (deque of ChatMessage)
       ├── LongTermMemory    (Chroma + sentence-transformers, typed)
       ├── EventStore        (SQLite: sessions, plans, steps)
       └── MemoryClassifier  (LLM, decides what to persist)

LLM layer
 ├── LLMProvider.generate()           — raw text
 └── LLMProvider.generate_json(schema)— strict Pydantic with retry
```

Loop:
```
plan = Planner.plan(objective)
while plan not done and step_budget left:
    step    = plan.current()
    decision= Reasoner.decide(step, scratchpad, memory)
    match decision.action.type:
        tool_call     → Executor.execute → Observation → scratchpad
        step_complete → mark step done
        final_answer  → return
    if too many failures → Planner.replan(plan, scratchpad)
```

## Strict LLM output schema (Reasoner)

```json
{
  "thought": "...",
  "action": {
    "type": "tool_call | step_complete | final_answer",
    "tool_call":   { "tool": "...", "arguments": { ... } },
    "final_answer": "...",
    "note": "..."
  }
}
```

The Planner emits a `PlanDraft` (objective + step dicts) validated to a `Plan`.
The MemoryClassifier emits a `MemoryClassification`.

## Install

```bash
pip install -e .
ollama pull llama3
```

## Run

```bash
jarvis chat -v           # interactive REPL with step-by-step trace
jarvis run "..."         # one-shot
jarvis serve             # FastAPI on :8000
jarvis tools             # list registered tools
```

## Configuration

`jarvis/config/default.yaml` (env override: `JARVIS_*` with `__` for nesting).

Key new keys:
```yaml
agent:
  max_steps: 12
  retrieve_k: 5
  max_consecutive_failures: 3
  replan_after_failures: 2
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

`Executor` is the single chokepoint: every tool call passes through the
`Approver` before running.

## Project layout

```
jarvis/
├── core/
│   ├── agent.py         # loop orchestrator
│   ├── planner.py       # NEW
│   ├── reasoner.py      # NEW
│   ├── executor.py      # NEW
│   ├── scratchpad.py    # NEW (structured state)
│   ├── context.py       # NEW (token budgeting)
│   ├── prompts.py       # split: planner/reasoner/memory
│   ├── schemas.py       # Plan, PlanStep, ReasonerDecision, MemoryItem, ...
│   └── orchestrator.py  # wiring
├── llm/
│   ├── base.py          # generate + generate_json + retry
│   ├── ollama_provider.py
│   └── retry.py         # NEW (JSON extraction + Pydantic validation)
├── memory/
│   ├── manager.py       # typed, filtered
│   ├── classifier.py    # NEW (LLM-based gate)
│   ├── long_term.py     # Chroma with metadata filtering
│   ├── short_term.py
│   └── store.py         # SQLite
├── tools/
│   ├── base.py          # wrapper: error capture, size limit, normalize
│   ├── limits.py        # NEW (truncate)
│   ├── shell.py
│   ├── filesystem.py    # + FilePatchTool
│   ├── python_exec.py
│   └── registry.py
├── safety/
├── api/server.py
├── config/
└── cli.py
```

## Extending

- **New tool**: subclass `Tool`, implement `_execute`; register in
  `tools/registry.py`. The base class handles error capture, size truncation,
  and metadata normalization automatically.
- **New LLM provider**: subclass `LLMProvider`, implement `generate`. The
  default `generate_json` works automatically using your `generate`.
- **New memory type**: extend the `MemoryType` enum.
