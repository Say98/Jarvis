"""Orchestrator: wires Planner, Reasoner, Executor, MemoryManager."""
from __future__ import annotations

from pathlib import Path

from jarvis.config.settings import Settings, load_settings
from jarvis.core.agent import Agent
from jarvis.core.context import ContextManager
from jarvis.core.executor import Executor
from jarvis.core.planner import Planner
from jarvis.core.reasoner import Reasoner
from jarvis.llm.registry import build_provider
from jarvis.logging_setup import get_logger, setup_logging
from jarvis.memory.classifier import MemoryClassifier
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

        classifier = MemoryClassifier(self.llm)
        self.memory = MemoryManager(
            self.settings.memory,
            classifier=classifier,
            session_id=session_id,
        )

        policy = PolicyEngine(self.settings.safety)
        self.approver: Approver = approver or CLIApprover(policy=policy)

        context = ContextManager(
            max_tokens=self.settings.agent.context_max_tokens,
            keep_recent=self.settings.agent.context_keep_recent,
        )

        self.planner = Planner(self.llm, self.tools.all(), context)
        self.reasoner = Reasoner(self.llm, self.tools.all(), context)
        self.executor = Executor(
            self.tools,
            self.approver,
            max_consecutive_failures=self.settings.agent.max_consecutive_failures,
        )

        self.agent = Agent(
            planner=self.planner,
            reasoner=self.reasoner,
            executor=self.executor,
            memory=self.memory,
            max_steps=self.settings.agent.max_steps,
            retrieve_k=self.settings.agent.retrieve_k,
            replan_after_failures=self.settings.agent.replan_after_failures,
        )
        log.info(
            "Jarvis v2 ready: model=%s tools=%s session=%s",
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
