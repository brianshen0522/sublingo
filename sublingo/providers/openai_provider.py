"""OpenAI-compatible API provider."""

from __future__ import annotations

import httpx

from sublingo.providers.base import BaseLLMProvider
from sublingo.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        timeout: float = 120.0,
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
        return "openai"

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        logger.debug("POST %s/chat/completions model=%s", self.base_url, self.model)

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()

        data = resp.json()
        return data["choices"][0]["message"]["content"]
