"""vLLM provider (OpenAI-compatible API)."""

from __future__ import annotations

from sublingo.providers.openai_provider import OpenAIProvider

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "default"


class VLLMProvider(OpenAIProvider):
    """vLLM uses the OpenAI-compatible API format."""

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
            base_url=base_url or DEFAULT_BASE_URL,
            api_key=api_key or "EMPTY",
            temperature=temperature,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return "vllm"
