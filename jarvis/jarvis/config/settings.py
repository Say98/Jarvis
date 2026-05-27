"""Configuration system with YAML + environment override support."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseModel):
    name: str = "jarvis"
    log_level: str = "INFO"
    data_dir: str = ".jarvis_data"


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 2048
    request_timeout: int = 120


class MemoryConfig(BaseModel):
    short_term_max_messages: int = 30
    long_term_collection: str = "jarvis_longterm"
    embedding_model: str = "all-MiniLM-L6-v2"
    sqlite_path: str = ".jarvis_data/jarvis.db"
    chroma_path: str = ".jarvis_data/chroma"


class AgentConfig(BaseModel):
    max_steps: int = 12
    retrieve_k: int = 5


PolicyLevel = Literal["auto", "ask", "double_confirm", "deny"]


class SafetyConfig(BaseModel):
    policies: dict[str, PolicyLevel] = Field(
        default_factory=lambda: {
            "read": "auto",
            "write": "ask",
            "execute": "ask",
            "destructive": "double_confirm",
        }
    )


class ShellToolConfig(BaseModel):
    enabled: bool = True
    timeout: int = 30


class FilesystemToolConfig(BaseModel):
    enabled: bool = True
    root: str = "."


class PythonExecToolConfig(BaseModel):
    enabled: bool = True
    timeout: int = 15


class ToolsConfig(BaseModel):
    shell: ShellToolConfig = ShellToolConfig()
    filesystem: FilesystemToolConfig = FilesystemToolConfig()
    python_exec: PythonExecToolConfig = PythonExecToolConfig()


class Settings(BaseSettings):
    app: AppConfig = AppConfig()
    llm: LLMConfig = LLMConfig()
    memory: MemoryConfig = MemoryConfig()
    agent: AgentConfig = AgentConfig()
    safety: SafetyConfig = SafetyConfig()
    tools: ToolsConfig = ToolsConfig()

    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_nested_delimiter="__",
        extra="ignore",
    )


DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load settings from YAML; env vars override."""
    data: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                user_data = yaml.safe_load(fh) or {}
            data = _deep_merge(data, user_data)

    # Env vars override via BaseSettings constructor merging
    return Settings(**data)
