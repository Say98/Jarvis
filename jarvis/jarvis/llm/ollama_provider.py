"""Ollama LLM provider with JSON-mode support."""
from __future__ import annotations

import httpx

from jarvis.core.schemas import ChatMessage
from jarvis.llm.base import LLMError, LLMProvider
from jarvis.logging_setup import get_logger

log = get_logger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        request_timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout

    def health_check(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception as e:  # pragma: no cover
            log.warning("Ollama health check failed: %s", e)
            return False

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        json_mode: bool = False,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.role in ("system", "user", "assistant")
            ],
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            },
        }
        if stop:
            payload["options"]["stop"] = stop
        if json_mode:
            # Ollama supports `"format": "json"` to constrain output to valid JSON.
            payload["format"] = "json"

        log.debug(
            "Ollama request: model=%s messages=%d json_mode=%s",
            self.model,
            len(messages),
            json_mode,
        )
        try:
            with httpx.Client(timeout=self.request_timeout) as client:
                resp = client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama HTTP error: {e}") from e

        content = data.get("message", {}).get("content", "")
        if not content:
            raise LLMError(f"Ollama returned empty response: {data!r}")
        return content
