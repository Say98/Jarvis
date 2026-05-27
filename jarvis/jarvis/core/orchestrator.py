"""Orchestrator: wires LLM, tools, memory, safety into an Agent."""
from __future__ import annotations

from pathlib import Path

from jarvis.config.settings import Settings, load_settings
from jarvis.core.agent import Agent
from jarvis.llm.registry import build_provider
from jarvis.logging_setup import get_logger, setup_logging
from jarvis.memory.manager import MemoryManager
from jarvis.safety.approver import Approver, AutoApprover, CLIApprover
from jarvis.safety.policy import PolicyEngine
from jarvis.tools.registry import ToolRegistry, build_default_registry

log = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        settings: Settings | None = None,
        approver: Approver | None = None,
        session_id: str | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        setup_logging(
            level=self.settings.app.log_level,
            log_file=Path(self.settings.app.data_dir) / "jarvis.log",
        )

        self.llm = build_provider(self.settings.llm)
        self.tools: ToolRegistry = build_default_registry(self.settings.tools)
        self.memory = MemoryManager(self.settings.memory, session_id=session_id)

        policy = PolicyEngine(self.settings.safety)
        self.approver: Approver = approver or CLIApprover(policy=policy)

        self.agent = Agent(
            llm=self.llm,
            tools=self.tools,
            memory=self.memory,
            approver=self.approver,
            max_steps=self.settings.agent.max_steps,
            retrieve_k=self.settings.agent.retrieve_k,
        )
        log.info(
            "Jarvis ready: model=%s tools=%s session=%s",
            self.settings.llm.model,
            self.tools.names(),
            self.memory.session_id,
        )

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        non_interactive: bool = False,
    ) -> "Orchestrator":
        settings = load_settings(config_path)
        approver = AutoApprover() if non_interactive else None
        return cls(settings=settings, approver=approver)
