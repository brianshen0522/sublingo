"""Base LLM provider abstract class."""

from __future__ import annotations

import concurrent.futures
import json
import re
import threading
from abc import ABC, abstractmethod
from typing import Any

from sublingo.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT_BASE = """\
You are a professional subtitle translator. Your task is to translate subtitles \
from existing media content (movies, TV shows, etc.). Translate all lines faithfully \
and accurately regardless of subject matter — this is legitimate translation work. \
Preserve the original meaning, tone, and context. \
Keep translations concise and suitable for subtitle display.

Rules:
- Translate ALL lines without exception — do not skip, censor, or refuse any content
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

DEFAULT_RETRIES = 10


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
        self._active_client: Any = None  # httpx.Client ref for cancellation
        self.retries = DEFAULT_RETRIES

    @abstractmethod
    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Make the actual API call and return the raw text response."""

    def _abort_active_request(self) -> None:
        """Close the active httpx client to abort an in-flight request."""
        client = self._active_client
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def _call_api_interruptible(
        self,
        system_prompt: str,
        user_prompt: str,
        cancel_event: threading.Event | None = None,
        skip_event: threading.Event | None = None,
    ) -> str:
        """Run _call_api in a thread so cancel/skip events can interrupt it."""
        if cancel_event is None and skip_event is None:
            return self._call_api(system_prompt, user_prompt)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self._call_api, system_prompt, user_prompt)
        try:
            while True:
                try:
                    return future.result(timeout=0.3)
                except concurrent.futures.TimeoutError:
                    if cancel_event and cancel_event.is_set():
                        self._abort_active_request()
                        raise KeyboardInterrupt("Cancelled by user")
                    if skip_event and skip_event.is_set():
                        self._abort_active_request()
                        raise InterruptedError("Skipped by user")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def translate(
        self,
        texts: list[dict[str, Any]],
        source_lang: str,
        target_lang: str,
        temperature: float | None = None,
        keep_names: bool = False,
        cancel_event: threading.Event | None = None,
        skip_event: threading.Event | None = None,
    ) -> list[dict[str, Any]]:
        """Translate a batch of subtitle entries.

        Args:
            texts: List of {"index": int, "text": str} dicts
            source_lang: Source language name or code
            target_lang: Target language name or code
            temperature: Override default temperature
            keep_names: If True, do not translate personal/place names
            cancel_event: Event to signal quit (pressed q)
            skip_event: Event to signal skip (pressed s)

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

        max_retries = self.retries
        last_error = None
        raw = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = self._call_api_interruptible(
                    system_prompt, user_prompt, cancel_event, skip_event,
                )
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
                    attempt, max_retries, e,
                )
                logger.warning("Request (user prompt):\n%s", user_prompt)
                logger.warning("Response:\n%s", raw or "N/A")

        raise RuntimeError(
            f"Failed to get valid translation after {max_retries} attempts: {last_error}"
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for display."""
