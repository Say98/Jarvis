"""Safety policy engine."""
from __future__ import annotations

from dataclasses import dataclass

from jarvis.config.settings import PolicyLevel, SafetyConfig
from jarvis.tools.base import ActionClass


@dataclass
class ApprovalDecision:
    allowed: bool
    reason: str = ""


class PolicyEngine:
    def __init__(self, cfg: SafetyConfig) -> None:
        self.cfg = cfg

    def policy_for(self, action_class: ActionClass) -> PolicyLevel:
        return self.cfg.policies.get(action_class, "ask")
