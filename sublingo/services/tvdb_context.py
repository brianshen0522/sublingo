"""Orchestration layer for building TVDB context for translation prompts."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from sublingo.services.tvdb_client import TVDBClient
from sublingo.utils.logger import get_logger

logger = get_logger(__name__)


def _google_translate(text: str, dest: str) -> str | None:
    """Translate text using googletrans. Returns translated text or None on failure."""
    try:
        from googletrans import Translator
        translator = Translator()
        # googletrans 4.x is async
        result = asyncio.run(translator.translate(text, dest=dest))
        return result.text
    except Exception as e:
        logger.debug("Google Translate failed: %s", e)
        return None


def _google_translate_dict(
    data: dict[str, str], dest: str,
) -> dict[str, str]:
    """Translate name/overview values in a TVDB translation dict via Google Translate."""
    result = {}
    for key in ("name", "overview"):
        val = data.get(key, "")
        if val:
            translated = _google_translate(val, dest)
            result[key] = translated if translated else val
        else:
            result[key] = val
    return result

_TVDB_LANGUAGES: dict[str, str] | None = None


def _load_tvdb_languages() -> dict[str, str]:
    global _TVDB_LANGUAGES
    if _TVDB_LANGUAGES is None:
        lang_file = Path(__file__).resolve().parent.parent / "utils" / "tvdb_languages.json"
        with open(lang_file, encoding="utf-8") as f:
            _TVDB_LANGUAGES = json.load(f)
    return _TVDB_LANGUAGES


# Patterns for extracting series info from filenames
_PATTERNS = [
    # Sonarr style: "South Park - S01E01 - Title.srt"
    re.compile(
        r"^(?P<series>.+?)\s*-\s*S(?P<season>\d+)E(?P<episode>\d+)",
        re.IGNORECASE,
    ),
    # Dot-separated: "South.Park.S01E01.720p.srt"
    re.compile(
        r"^(?P<series>.+?)\.S(?P<season>\d+)E(?P<episode>\d+)",
        re.IGNORECASE,
    ),
    # Numbered: "South Park 1x01.srt"
    re.compile(
        r"^(?P<series>.+?)\s+(?P<season>\d+)x(?P<episode>\d+)",
        re.IGNORECASE,
    ),
]


def parse_series_info(filename: str) -> dict[str, Any] | None:
    """Extract series name, season, and episode from a filename.

    Returns dict with keys: series, season, episode — or None if no match.
    """
    stem = Path(filename).stem
    for pattern in _PATTERNS:
        m = pattern.match(stem)
        if m:
            series = m.group("series").replace(".", " ").strip()
            return {
                "series": series,
                "season": int(m.group("season")),
                "episode": int(m.group("episode")),
            }
    return None


def resolve_tvdb_language(sublingo_code: str) -> str | None:
    """Map a sublingo language code to a TVDB 3-letter language code."""
    langs = _load_tvdb_languages()
    return langs.get(sublingo_code)


def _fetch_translations(
    client: TVDBClient,
    series_id: int,
    episode_id: int | None,
    tvdb_lang: str,
    sublingo_code: str,
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    """Fetch series and episode translations for a single language.

    Falls back to Google Translate (from English) when TVDB lacks a translation.
    """
    # googletrans uses lowercase hyphenated codes like "zh-tw"
    gtrans_dest = sublingo_code.lower()

    series_trans = client.get_series_translation(series_id, tvdb_lang)
    if series_trans is None and tvdb_lang != "eng":
        eng_series = client.get_series_translation(series_id, "eng")
        if eng_series:
            logger.debug("TVDB: no series translation for %s, falling back to Google Translate", tvdb_lang)
            series_trans = _google_translate_dict(eng_series, gtrans_dest)

    episode_trans = None
    if episode_id is not None:
        episode_trans = client.get_episode_translation(episode_id, tvdb_lang)
        if episode_trans is None and tvdb_lang != "eng":
            eng_episode = client.get_episode_translation(episode_id, "eng")
            if eng_episode:
                logger.debug("TVDB: no episode translation for %s, falling back to Google Translate", tvdb_lang)
                episode_trans = _google_translate_dict(eng_episode, gtrans_dest)

    return series_trans, episode_trans


def _append_translation_lines(
    lines: list[str],
    label: str,
    series_trans: dict[str, str] | None,
    episode_trans: dict[str, str] | None,
    season: int,
    episode: int,
) -> None:
    """Append formatted translation lines for one language."""
    if series_trans:
        if series_trans.get("name"):
            lines.append(f"Series title ({label}): {series_trans['name']}")
        if series_trans.get("overview"):
            lines.append(f"Series description ({label}): {series_trans['overview']}")
    if episode_trans:
        ep_label = f"S{season:02d}E{episode:02d}"
        if episode_trans.get("name"):
            lines.append(f"Episode title ({label}, {ep_label}): {episode_trans['name']}")
        if episode_trans.get("overview"):
            lines.append(f"Episode description ({label}, {ep_label}): {episode_trans['overview']}")


def build_tvdb_context(
    client: TVDBClient,
    filename: str,
    source_lang_code: str,
    target_lang_code: str,
) -> str | None:
    """Build a TVDB context string for the translation system prompt.

    Fetches series/episode descriptions in both the source and target languages
    so the LLM can map names and references between them.

    Returns a formatted context block, or None if info is unavailable.
    """
    info = parse_series_info(filename)
    if not info:
        logger.debug("TVDB: could not parse series info from %r", filename)
        return None

    source_tvdb = resolve_tvdb_language(source_lang_code)
    target_tvdb = resolve_tvdb_language(target_lang_code)

    if not source_tvdb and not target_tvdb:
        logger.debug(
            "TVDB: no TVDB language mapping for source %r or target %r",
            source_lang_code, target_lang_code,
        )
        return None

    series_name = info["series"]
    season = info["season"]
    episode_num = info["episode"]

    # Deduplicate if source and target resolve to the same TVDB code
    langs_to_fetch: list[tuple[str, str, str]] = []  # (label, tvdb_code, sublingo_code)
    if source_tvdb:
        langs_to_fetch.append((f"source: {source_lang_code}", source_tvdb, source_lang_code))
    if target_tvdb and target_tvdb != source_tvdb:
        langs_to_fetch.append((f"target: {target_lang_code}", target_tvdb, target_lang_code))

    logger.debug(
        "TVDB: looking up %r S%02dE%02d in languages %s",
        series_name, season, episode_num,
        [code for _, code, _ in langs_to_fetch],
    )

    # Search for series
    series_id = client.search_series(series_name)
    if series_id is None:
        logger.debug("TVDB: series %r not found", series_name)
        return None

    # Find episode ID (shared across languages)
    episode_id = client.get_episode_id(series_id, season, episode_num)

    # Fetch translations for each language
    lines = [
        "Context about this series/episode (use for accurate translation of names, places, and cultural references):",
    ]
    has_content = False

    for label, tvdb_lang, sublingo_code in langs_to_fetch:
        series_trans, episode_trans = _fetch_translations(
            client, series_id, episode_id, tvdb_lang, sublingo_code,
        )
        if series_trans or episode_trans:
            has_content = True
            _append_translation_lines(
                lines, label, series_trans, episode_trans, season, episode_num,
            )

    if not has_content:
        logger.debug("TVDB: no translations found for any requested language")
        return None

    context = "\n".join(lines)
    logger.debug("TVDB context:\n%s", context)
    return context
