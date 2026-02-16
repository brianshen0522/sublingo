"""Detect source language from subtitle entries using LLM."""

from __future__ import annotations

import json
import re
import threading

from sublingo.core.subtitle_parser import SubtitleEntry
from sublingo.providers.base import BaseLLMProvider
from sublingo.utils.logger import get_logger

logger = get_logger(__name__)

DETECTION_SYSTEM_PROMPT = """\
You are a language detection assistant. Given subtitle text, identify the language.
Respond with ONLY a JSON object: {"language": "<language name>", "code": "<ISO 639-1 code>"}
Do not repeat the input. Do not explain. Only output the JSON object.
"""

DETECTION_USER_PROMPT = """\
What language is this text written in? Reply with only a JSON object like {{"language": "English", "code": "en"}}.

Text:
{sample_text}
"""

MAX_RETRIES = 3


def detect_language(
    entries: list[SubtitleEntry],
    provider: BaseLLMProvider,
    sample_size: int = 5,
    cancel_event: threading.Event | None = None,
    skip_event: threading.Event | None = None,
) -> dict[str, str]:
    """Detect the language of subtitle entries.

    Returns dict with "language" (full name) and "code" (ISO 639-1).
    """
    sample = entries[:sample_size]
    sample_text = "\n".join(e.text for e in sample)

    user_prompt = DETECTION_USER_PROMPT.format(sample_text=sample_text)
    logger.debug("Detection system prompt:\n%s", DETECTION_SYSTEM_PROMPT)
    logger.debug("Detection user prompt:\n%s", user_prompt)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = provider._call_api_interruptible(
                DETECTION_SYSTEM_PROMPT, user_prompt, cancel_event, skip_event,
            )
            logger.debug("Detection raw response (attempt %d):\n%s", attempt, raw)

            raw = raw.strip()
            # Try direct parse
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                # Try extracting JSON object from response
                match = re.search(r"\{.*?\}", raw, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
                else:
                    raise ValueError(f"No JSON object found in response")

            # Validate it has the expected keys
            if "language" not in result and "code" not in result:
                raise ValueError(f"Response missing 'language'/'code' keys: {result}")

            logger.info("Detected language: %s (%s)", result.get("language"), result.get("code"))
            return result

        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            logger.warning(
                "Language detection attempt %d/%d failed: %s",
                attempt, MAX_RETRIES, e,
            )

    raise RuntimeError(
        f"Language detection failed after {MAX_RETRIES} attempts: {last_error}\n"
        f"Hint: try specifying the source language with --from instead of auto-detect"
    )
