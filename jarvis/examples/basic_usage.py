"""Example: run a single objective through Jarvis v2 and stream events."""
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.schemas import AgentStep, FinalAnswer, Plan


def main() -> None:
    orch = Orchestrator.from_config(non_interactive=True)

    objective = (
        "List the files in the current directory, count them, "
        "and write the count into a file named 'count.txt'."
    )
    print(f"OBJECTIVE: {objective}\n")

    for event in orch.agent.stream(objective):
        if isinstance(event, Plan):
            print(f"\n=== PLAN rev {event.revision} ===")
            for s in event.steps:
                print(f"  {s.id}. [{s.status}] {s.goal} (tool={s.suggested_tool})")
        elif isinstance(event, AgentStep):
            print(f"\n[step {event.step}/plan_step={event.plan_step_id}] {event.action.type}")
            print(f"  thought: {event.thought.reasoning}")
            if event.action.tool_call:
                print(f"  tool: {event.action.tool_call.tool}")
                print(f"  args: {event.action.tool_call.arguments}")
            if event.observation is not None:
                print(f"  ok={event.observation.ok}")
                print(f"  out={event.observation.content[:200]}")
                if event.observation.error:
                    print(f"  error={event.observation.error}")
        elif isinstance(event, FinalAnswer):
            print(f"\n=== FINAL ===\n{event.content}")


if __name__ == "__main__":
    main()
