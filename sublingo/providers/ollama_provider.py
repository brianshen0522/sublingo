"""Ollama local LLM provider."""

from __future__ import annotations

import httpx

from sublingo.providers.base import BaseLLMProvider
from sublingo.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"


class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        timeout: float = 300.0,
    ):
        super().__init__(
            model=model or DEFAULT_MODEL,
            base_url=(base_url or DEFAULT_BASE_URL).rstrip("/"),
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return "ollama"

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        logger.debug("POST %s/api/chat model=%s", self.base_url, self.model)

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()

        data = resp.json()
        return data["message"]["content"]
