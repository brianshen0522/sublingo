"""Base LLM provider abstract class."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from sublingo.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT_BASE = """\
You are a professional subtitle translator. Translate the given subtitle lines \
accurately while preserving the original meaning, tone, and context. \
Keep translations concise and suitable for subtitle display.

Rules:
- Read ALL subtitle lines first to understand the full context before translating
- Use surrounding lines as context to resolve ambiguous words, pronouns, and implied meanings
- Maintain consistent terminology and tone across all lines in the batch
- Maintain the same number of entries in input and output
- Preserve line breaks within entries where appropriate for readability
- Do not add or remove subtitle entries
{keep_names_rule}\
- Respond ONLY with a JSON array of objects, each with "index" and "text" keys
"""

KEEP_NAMES_RULE = "- Do NOT translate personal names or place names — keep them in their original form\n"


def build_system_prompt(keep_names: bool = False) -> str:
    return SYSTEM_PROMPT_BASE.format(
        keep_names_rule=KEEP_NAMES_RULE if keep_names else "",
    )

USER_PROMPT_TEMPLATE = """\
Translate the following subtitle lines from {source_lang} to {target_lang}.

Respond with a JSON array where each element has "index" (the original index) and "text" (the translated text).

Subtitle lines:
{entries_json}
"""

MAX_RETRIES = 3


def format_entries_for_prompt(entries: list[dict[str, Any]]) -> str:
    return json.dumps(entries, ensure_ascii=False, indent=2)


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array from LLM response text, handling markdown fences."""
    # Try direct parse first
    text = text.strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting from markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding array in text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON array from response: {text[:200]}")


class BaseLLMProvider(ABC):
    """Abstract base class for LLM translation providers."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    @abstractmethod
    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Make the actual API call and return the raw text response."""

    def translate(
        self,
        texts: list[dict[str, Any]],
        source_lang: str,
        target_lang: str,
        temperature: float | None = None,
        keep_names: bool = False,
    ) -> list[dict[str, Any]]:
        """Translate a batch of subtitle entries.

        Args:
            texts: List of {"index": int, "text": str} dicts
            source_lang: Source language name or code
            target_lang: Target language name or code
            temperature: Override default temperature
            keep_names: If True, do not translate personal/place names

        Returns:
            List of {"index": int, "text": str} dicts with translations
        """
        system_prompt = build_system_prompt(keep_names=keep_names)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            entries_json=format_entries_for_prompt(texts),
        )

        logger.debug("System prompt:\n%s", system_prompt)
        logger.debug("User prompt:\n%s", user_prompt)

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = self._call_api(system_prompt, user_prompt)
                logger.debug("LLM raw response:\n%s", raw)
                result = extract_json_array(raw)
                if len(result) != len(texts):
                    logger.warning(
                        "Expected %d entries, got %d (attempt %d)",
                        len(texts), len(result), attempt,
                    )
                return result
            except (ValueError, json.JSONDecodeError, KeyError) as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d failed to parse response: %s",
                    attempt, MAX_RETRIES, e,
                )

        raise RuntimeError(
            f"Failed to get valid translation after {MAX_RETRIES} attempts: {last_error}"
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for display."""
