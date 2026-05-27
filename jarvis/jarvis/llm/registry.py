"""LLM provider factory / registry."""
from __future__ import annotations

from jarvis.config.settings import LLMConfig
from jarvis.llm.base import LLMProvider
from jarvis.llm.ollama_provider import OllamaProvider


def build_provider(cfg: LLMConfig) -> LLMProvider:
    provider = cfg.provider.lower()
    if provider == "ollama":
        return OllamaProvider(
            model=cfg.model,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            request_timeout=cfg.request_timeout,
        )
    raise ValueError(
        f"Unknown LLM provider '{cfg.provider}'. "
        f"Available: ollama. (vLLM/OpenAI can be plugged in here.)"
    )
