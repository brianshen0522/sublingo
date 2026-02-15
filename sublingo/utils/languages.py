"""Language code to full name mapping."""

from __future__ import annotations

import json
from pathlib import Path

_LANGUAGES: dict[str, str] | None = None


def _load() -> dict[str, str]:
    global _LANGUAGES
    if _LANGUAGES is None:
        path = Path(__file__).parent / "languages.json"
        with open(path) as f:
            _LANGUAGES = json.load(f)
    return _LANGUAGES


def resolve_language(code_or_name: str) -> str:
    """Resolve a language code to its full name for use in prompts.

    If the input is a known code (e.g. "zh-TW"), returns the full name
    (e.g. "Traditional Chinese"). Otherwise returns the input as-is.
    """
    languages = _load()
    # Exact match
    if code_or_name in languages:
        return languages[code_or_name]
    # Case-insensitive match
    for code, name in languages.items():
        if code.lower() == code_or_name.lower():
            return name
    # Not a known code, return as-is (might already be a full name)
    return code_or_name


def list_languages() -> dict[str, str]:
    """Return all supported language codes and names."""
    return dict(_load())
