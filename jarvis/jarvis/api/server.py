"""FastAPI HTTP server for Jarvis v2."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from jarvis.core.orchestrator import Orchestrator
from jarvis.core.schemas import AgentStep, FinalAnswer, Plan


class ChatRequest(BaseModel):
    message: str


class StepView(BaseModel):
    step: int
    plan_step_id: int | None
    thought: str
    action_type: str
    tool: str | None = None
    arguments: dict | None = None
    final_answer: str | None = None
    note: str | None = None
    observation_ok: bool | None = None
    observation_content: str | None = None
    observation_error: str | None = None


class PlanView(BaseModel):
    revision: int
    objective: str
    steps: list[dict]


class ChatResponse(BaseModel):
    final_answer: str
    plans: list[PlanView]
    steps: list[StepView]


def create_app(non_interactive: bool = True) -> FastAPI:
    app = FastAPI(title="Jarvis", version="2.0.0")
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
        plans: list[PlanView] = []
        final = ""
        for ev in orch.agent.stream(req.message):
            if isinstance(ev, Plan):
                plans.append(
                    PlanView(
                        revision=ev.revision,
                        objective=ev.objective,
                        steps=[s.model_dump() for s in ev.steps],
                    )
                )
            elif isinstance(ev, AgentStep):
                steps.append(
                    StepView(
                        step=ev.step,
                        plan_step_id=ev.plan_step_id,
                        thought=ev.thought.reasoning,
                        action_type=ev.action.type,
                        tool=ev.action.tool_call.tool if ev.action.tool_call else None,
                        arguments=(
                            ev.action.tool_call.arguments
                            if ev.action.tool_call
                            else None
                        ),
                        final_answer=ev.action.final_answer,
                        note=ev.action.note,
                        observation_ok=ev.observation.ok if ev.observation else None,
                        observation_content=(
                            ev.observation.content if ev.observation else None
                        ),
                        observation_error=(
                            ev.observation.error if ev.observation else None
                        ),
                    )
                )
            elif isinstance(ev, FinalAnswer):
                final = ev.content
        return ChatResponse(final_answer=final, plans=plans, steps=steps)

    return app


# Allow `uvicorn jarvis.api.server:app`
app = create_app()
