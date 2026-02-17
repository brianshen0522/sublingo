"""Base LLM provider abstract class."""

from __future__ import annotations

import concurrent.futures
import json
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from sublingo.utils.logger import get_logger

logger = get_logger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt.txt"
USER_PROMPT_FILE = PROMPTS_DIR / "user_prompt.txt"
KEEP_NAMES_RULE_FILE = PROMPTS_DIR / "keep_names_rule.txt"


def _replace_placeholders(template: str, replacements: dict[str, str]) -> str:
    """Replace {placeholder} tokens in a template string.

    Uses str.replace() instead of str.format() so that literal braces
    (e.g. JSON examples like {"index": 1}) don't cause errors.
    """
    result = template
    for key, value in replacements.items():
        result = result.replace("{" + key + "}", value)
    return result


def build_prompts(
    source_lang: str,
    target_lang: str,
    entries_json: str,
    keep_names: bool = False,
    tvdb_context: str | None = None,
) -> tuple[str, str]:
    """Build both system and user prompts from template files.

    Returns (system_prompt, user_prompt).
    """
    keep_names_rule = KEEP_NAMES_RULE_FILE.read_text(encoding="utf-8") if keep_names else ""
    replacements = {
        "keep_names_rule": keep_names_rule,
        "tvdb_context": tvdb_context or "",
        "source_lang": source_lang,
        "target_lang": target_lang,
        "entries_json": entries_json,
    }
    system_template = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    user_template = USER_PROMPT_FILE.read_text(encoding="utf-8")
    return (
        _replace_placeholders(system_template, replacements),
        _replace_placeholders(user_template, replacements),
    )

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
        tvdb_context: str | None = None,
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
            tvdb_context: Optional TVDB context string for the system prompt

        Returns:
            List of {"index": int, "text": str} dicts with translations
        """
        system_prompt, user_prompt = build_prompts(
            source_lang=source_lang,
            target_lang=target_lang,
            entries_json=format_entries_for_prompt(texts),
            keep_names=keep_names,
            tvdb_context=tvdb_context,
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
                logger.warning("System prompt:\n%s", system_prompt)
                logger.warning("User prompt:\n%s", user_prompt)
                logger.warning("Response:\n%s", raw or "N/A")

        raise RuntimeError(
            f"Failed to get valid translation after {max_retries} attempts: {last_error}"
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for display."""
