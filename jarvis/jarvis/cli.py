"""Jarvis CLI."""
from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from jarvis.core.orchestrator import Orchestrator
from jarvis.core.schemas import AgentStep, FinalAnswer

app = typer.Typer(add_completion=False, help="Jarvis – local AI agent.")
console = Console()


def _render_step(step: AgentStep) -> None:
    title = f"step {step.step} · {step.action.type}"
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
    elif step.action.type == "final_answer":
        body += f"[bold]final:[/bold] {step.action.final_answer}"
    console.print(Panel(body, title=title, border_style="cyan"))


@app.command()
def chat(
    config: Optional[str] = typer.Option(None, help="Path to a YAML config file."),
    auto: bool = typer.Option(False, "--auto", help="Auto-approve all actions (UNSAFE)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show every agent step."),
) -> None:
    """Interactive REPL with the Jarvis agent."""
    orch = Orchestrator.from_config(config_path=config, non_interactive=auto)
    console.print(
        Panel.fit(
            f"[bold]Jarvis[/bold] · model=[cyan]{orch.settings.llm.model}[/cyan] "
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
        for event in orch.agent.stream(user_input):
            if isinstance(event, AgentStep):
                if verbose:
                    _render_step(event)
            elif isinstance(event, FinalAnswer):
                final = event.content

        console.print(Panel(Markdown(final or "(no answer)"), title="jarvis", border_style="magenta"))


@app.command()
def run(
    prompt: str = typer.Argument(..., help="A single prompt to execute non-interactively."),
    config: Optional[str] = typer.Option(None),
    auto: bool = typer.Option(True, "--auto/--ask"),
) -> None:
    """Run a single prompt and print the final answer."""
    orch = Orchestrator.from_config(config_path=config, non_interactive=auto)
    answer = orch.agent.run(prompt)
    console.print(answer)


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    config: Optional[str] = typer.Option(None),
) -> None:
    """Start the FastAPI HTTP server."""
    import uvicorn

    from jarvis.api.server import create_app

    # Force re-creation honoring config / non_interactive=True for HTTP context
    server_app = create_app(non_interactive=True)
    uvicorn.run(server_app, host=host, port=port)


@app.command()
def tools() -> None:
    """List available tools."""
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
