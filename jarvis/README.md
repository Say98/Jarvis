# Jarvis – Local AI Assistant

Jarvis is a fully local, model-agnostic AI agent framework. It is **not** a chatbot.
It is a tool-using agent with persistent memory and a human-in-the-loop safety layer.

## Features

- **Modular architecture**: strict separation of `core`, `llm`, `tools`, `memory`, `safety`, `api`.
- **Model-agnostic LLM interface**: Ollama provider included; vLLM/OpenAI pluggable.
- **Real agent loop**: `THINK → DECIDE → ACT → OBSERVE → STORE`.
- **Structured actions** via Pydantic – no string parsing hacks.
- **Working tools**: shell, filesystem, sandboxed Python execution.
- **Memory**:
  - Short-term conversational buffer
  - Long-term vector memory (ChromaDB + sentence-transformers)
  - Persistent SQLite event store
- **Safety layer**: per-action policies (`auto`, `ask`, `double_confirm`).
- **Interfaces**: CLI (Typer + Rich) and FastAPI HTTP server.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (e.g. `ollama serve` with `llama3` pulled).
- Optional GPU for the embedding model.

## Install

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Run

Pull a model in Ollama first:

```bash
ollama pull llama3
```

Then run Jarvis:

```bash
jarvis chat
# or
python -m jarvis chat
```

Run the HTTP API:

```bash
jarvis serve --host 127.0.0.1 --port 8000
```

## Configuration

Edit `jarvis/config/default.yaml` or set environment variables prefixed with `JARVIS_`
(e.g. `JARVIS_LLM__MODEL=qwen2.5`).

## Project layout

```
jarvis/
├── core/         # Agent loop, orchestrator, schemas, prompts
├── llm/          # LLM provider abstraction (Ollama)
├── tools/        # Tool interface + shell / fs / python tools
├── memory/       # Short-term, long-term (Chroma), SQLite store
├── safety/       # Approval policies + CLI approver
├── api/          # FastAPI server
├── config/       # Settings + default.yaml
├── cli.py        # Typer CLI entry point
└── logging_setup.py
```

## Safety

| Action class    | Policy          |
|-----------------|-----------------|
| read            | auto            |
| write           | ask             |
| execute         | ask             |
| destructive     | double confirm  |

All tool invocations pass through the safety layer before execution.

## Extending

- **Add a tool**: subclass `jarvis.tools.base.Tool`, register via `ToolRegistry.register`.
- **Add an LLM provider**: subclass `jarvis.llm.base.LLMProvider`, register in `llm/registry.py`.
