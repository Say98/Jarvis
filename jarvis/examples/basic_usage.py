"""Example: run a single prompt through Jarvis programmatically."""
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.schemas import AgentStep, FinalAnswer


def main() -> None:
    orch = Orchestrator.from_config(non_interactive=True)

    prompt = (
        "List the files in the current directory, then tell me how many there are."
    )
    print(f"USER: {prompt}\n")

    for event in orch.agent.stream(prompt):
        if isinstance(event, AgentStep):
            print(f"[step {event.step}] {event.action.type}")
            if event.action.tool_call:
                print(f"  tool: {event.action.tool_call.tool}")
                print(f"  args: {event.action.tool_call.arguments}")
            if event.observation is not None:
                print(f"  ok={event.observation.ok}")
                print(f"  out={event.observation.content[:200]}")
        elif isinstance(event, FinalAnswer):
            print(f"\nJARVIS: {event.content}")


if __name__ == "__main__":
    main()
