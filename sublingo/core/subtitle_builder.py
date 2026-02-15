"""Write translated subtitles back to file."""

from __future__ import annotations

from pathlib import Path

import pysubs2

from sublingo.core.subtitle_parser import SubtitleEntry


def build_file(
    entries: list[SubtitleEntry],
    output_path: Path,
    original_path: Path | None = None,
    bilingual: bool = False,
    original_entries: list[SubtitleEntry] | None = None,
) -> None:
    """Write subtitle entries to a file.

    If bilingual=True and original_entries provided, each subtitle line
    will contain the original text followed by the translated text.
    """
    # If we have an original file, load it to preserve styles/metadata
    if original_path and original_path.is_file():
        subs = pysubs2.load(str(original_path))
        subs.events.clear()
    else:
        subs = pysubs2.SSAFile()

    for i, entry in enumerate(entries):
        text = entry.text
        if bilingual and original_entries and i < len(original_entries):
            text = f"{original_entries[i].text}\\N{entry.text}"

        event = pysubs2.SSAEvent(
            start=entry.start,
            end=entry.end,
            text=text,
            style=entry.style,
        )
        subs.events.append(event)

    # Determine output format from extension
    output_format = output_path.suffix.lstrip(".")
    subs.save(str(output_path), format_=output_format)
