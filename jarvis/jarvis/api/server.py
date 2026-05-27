"""FastAPI HTTP server for Jarvis."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from jarvis.core.orchestrator import Orchestrator
from jarvis.core.schemas import AgentStep, FinalAnswer


class ChatRequest(BaseModel):
    message: str


class StepView(BaseModel):
    step: int
    thought: str
    action_type: str
    tool: str | None = None
    arguments: dict | None = None
    final_answer: str | None = None
    observation_ok: bool | None = None
    observation_content: str | None = None
    observation_error: str | None = None


class ChatResponse(BaseModel):
    final_answer: str
    steps: list[StepView]


def create_app(non_interactive: bool = True) -> FastAPI:
    app = FastAPI(title="Jarvis", version="0.1.0")
    orch = Orchestrator.from_config(non_interactive=non_interactive)

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "model": orch.settings.llm.model,
            "llm_reachable": orch.llm.health_check(),
            "tools": orch.tools.names(),
            "session": orch.memory.session_id,
        }

    @app.get("/tools")
    def tools() -> list[dict]:
        return [t.describe() for t in orch.tools.all()]

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        if not req.message.strip():
            raise HTTPException(400, "message must be non-empty")
        steps: list[StepView] = []
        final = ""
        for event in orch.agent.stream(req.message):
            if isinstance(event, AgentStep):
                steps.append(
                    StepView(
                        step=event.step,
                        thought=event.thought.reasoning,
                        action_type=event.action.type,
                        tool=event.action.tool_call.tool if event.action.tool_call else None,
                        arguments=(
                            event.action.tool_call.arguments
                            if event.action.tool_call
                            else None
                        ),
                        final_answer=event.action.final_answer,
                        observation_ok=event.observation.ok if event.observation else None,
                        observation_content=(
                            event.observation.content if event.observation else None
                        ),
                        observation_error=(
                            event.observation.error if event.observation else None
                        ),
                    )
                )
            elif isinstance(event, FinalAnswer):
                final = event.content
        return ChatResponse(final_answer=final, steps=steps)

    return app


# Allow `uvicorn jarvis.api.server:app`
app = create_app()
