# JARVIS LOCAL AI ASSISTANT – FULL SPECIFICATION

## Overview
This project aims to build a fully local, model-agnostic AI assistant ("Jarvis") capable of orchestrating tasks, executing tools, and acting semi-autonomously with human-in-the-loop approval.

---

## Hardware Target
- GPU: NVIDIA 5060 Ti (16GB VRAM)
- RAM: 32GB
- Fully offline capability required

---

## Core Goals

### 1. System Automation
- Execute shell commands
- Manage filesystem
- Automate developer workflows
- Control local environment

### 2. Knowledge & Memory
- Index local files
- Provide contextual retrieval (RAG)
- Maintain persistent memory

### 3. AI Coding & Planning
- Code generation and refactoring
- Problem decomposition
- Multi-step task planning

---

## Autonomy Model
- Semi-autonomous agent
- Requires user approval for:
  - File writes
  - Command execution
  - Destructive actions
- Fully automatic for read-only operations

---

## Architecture

```
Interface (CLI / Web / Voice)
        ↓
Orchestrator (Core Logic)
        ↓
LLM Backend | Tools | Memory
```

---

## Tech Stack

### Backend
- Python
- FastAPI

### LLM Runtime
- Primary: Ollama
- Future: vLLM

### Models
- Llama 3
- Qwen
- Mixtral

### Memory
- Short-term: in-memory context
- Long-term: Chroma / FAISS
- Structured: SQLite

---

## Project Structure

```
jarvis/
│
├── core/
├── llm/
├── tools/
├── memory/
├── api/
├── ui/
└── config/
```

---

## Core Loop

```
while True:
    think
    decide action
    request approval (if needed)
    execute tool
    store memory
    update context
```

---

## Tooling System

Base interface:

```
class Tool:
    name: str
    def execute(self, input): pass
```

Initial tools:
- Shell execution
- File read/write
- Code execution

---

## LLM Abstraction

```
class LLMProvider:
    def generate(self, prompt, context): pass
```

Providers:
- Ollama
- vLLM
- OpenAI (optional fallback)

---

## Security Policy

| Action | Policy |
|--------|--------|
| Read file | Auto |
| Write file | Ask |
| Execute command | Ask |
| Delete data | Double confirm |

---

## Roadmap

### Phase 1
- Local chat
- Basic tools

### Phase 2
- Agent loop
- Tool decision

### Phase 3
- Planner
- Multi-step tasks

### Phase 4
- Voice interface

---

## Design Principles
- Model-agnostic
- Modular
- Extensible
- Safe by design

---

## PROMPT FOR CLAUDE OPUS 4.7

You are a senior AI systems architect and Python engineer.

Your task is to generate a complete production-ready codebase for a local AI assistant named "Jarvis" based on the following requirements:

- Fully modular architecture
- Model-agnostic LLM interface
- Tool-based agent system
- Persistent memory (short + long term)
- Human-in-the-loop safety layer
- Python (no frameworks abstraction unless justified)

Key requirements:
- Clean separation of concerns
- Production-level code quality
- Type hints everywhere
- Logging system
- Config-driven design

You must:
1. Generate full folder structure
2. Implement all modules
3. Provide minimal working MVP
4. Include extensibility patterns
5. Avoid mock implementations unless necessary

Focus on correctness, maintainability, and scalability.
