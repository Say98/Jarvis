"""Jarvis v2 CLI."""
from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from jarvis.core.orchestrator import Orchestrator
from jarvis.core.schemas import AgentStep, FinalAnswer, Plan

app = typer.Typer(add_completion=False, help="Jarvis v2 – local AI agent.")
console = Console()


def _render_plan(plan: Plan) -> None:
    table = Table(title=f"PLAN (rev {plan.revision}): {plan.objective}", show_lines=False)
    table.add_column("#", style="cyan", width=3)
    table.add_column("goal")
    table.add_column("tool", style="magenta")
    table.add_column("status", style="green")
    for s in plan.steps:
        table.add_row(str(s.id), s.goal, s.suggested_tool or "-", s.status)
    console.print(table)


def _render_step(step: AgentStep) -> None:
    title = f"step {step.step} · plan_step={step.plan_step_id} · {step.action.type}"
    body = f"[bold]thought:[/bold] {step.thought.reasoning}\n"
    if step.action.type == "tool_call" and step.action.tool_call:
        body += (
            f"[bold]tool:[/bold] {step.action.tool_call.tool}\n"
            f"[bold]args:[/bold] {json.dumps(step.action.tool_call.arguments)}\n"
        )
        if step.observation is not None:
            body += (
                f"[bold]ok:[/bold] {step.observation.ok}\n"
                f"[bold]output:[/bold]\n{step.observation.content or ''}"
            )
            if step.observation.error:
                body += f"\n[red]error:[/red] {step.observation.error}"
    elif step.action.type == "step_complete":
        body += f"[green]step complete[/green] {step.action.note or ''}"
    elif step.action.type == "final_answer":
        body += f"[bold]final:[/bold] {step.action.final_answer}"
    console.print(Panel(body, title=title, border_style="cyan"))


@app.command()
def chat(
    config: Optional[str] = typer.Option(None, help="Path to a YAML config file."),
    auto: bool = typer.Option(False, "--auto", help="Auto-approve all actions (UNSAFE)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Interactive REPL."""
    orch = Orchestrator.from_config(config_path=config, non_interactive=auto)
    console.print(
        Panel.fit(
            f"[bold]Jarvis v2[/bold] · model=[cyan]{orch.settings.llm.model}[/cyan] "
            f"· session=[dim]{orch.memory.session_id}[/dim]\n"
            f"Tools: {', '.join(orch.tools.names())}\n"
            f"Type [yellow]/quit[/yellow] to exit.",
            border_style="green",
        )
    )
    while True:
        try:
            console.print("\n[bold green]you[/bold green]> ", end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye.")
            return
        if not user_input:
            continue
        if user_input in ("/quit", "/exit", ":q"):
            console.print("bye.")
            return

        final = ""
        for ev in orch.agent.stream(user_input):
            if isinstance(ev, Plan):
                _render_plan(ev)
            elif isinstance(ev, AgentStep):
                if verbose:
                    _render_step(ev)
            elif isinstance(ev, FinalAnswer):
                final = ev.content

        console.print(
            Panel(Markdown(final or "(no answer)"), title="jarvis", border_style="magenta")
        )


@app.command()
def run(
    prompt: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None),
    auto: bool = typer.Option(True, "--auto/--ask"),
) -> None:
    """One-shot run."""
    orch = Orchestrator.from_config(config_path=config, non_interactive=auto)
    console.print(orch.agent.run(prompt))


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Start the FastAPI HTTP server."""
    import uvicorn

    from jarvis.api.server import create_app

    uvicorn.run(create_app(non_interactive=True), host=host, port=port)


@app.command()
def tools() -> None:
    orch = Orchestrator.from_config(non_interactive=True)
    for t in orch.tools.all():
        console.print(
            Panel(
                f"[bold]{t.name}[/bold] [dim]({t.action_class})[/dim]\n{t.description}\n\n"
                f"schema: {json.dumps(t.arguments_schema(), indent=2)}",
                border_style="blue",
            )
        )


if __name__ == "__main__":
    app()
