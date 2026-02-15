"""Parse subtitle files into internal representation via pysubs2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pysubs2


@dataclass
class SubtitleEntry:
    index: int
    start: int  # milliseconds
    end: int  # milliseconds
    text: str
    style: str = "Default"


def parse_file(path: Path) -> list[SubtitleEntry]:
    """Parse a subtitle file into a list of SubtitleEntry."""
    subs = pysubs2.load(str(path))
    entries = []
    for i, event in enumerate(subs.events):
        if event.is_comment:
            continue
        entries.append(
            SubtitleEntry(
                index=i,
                start=event.start,
                end=event.end,
                text=event.plaintext,
                style=event.style,
            )
        )
    return entries


def parse_string(content: str, format: str = "srt") -> list[SubtitleEntry]:
    """Parse subtitle content from a string."""
    subs = pysubs2.SSAFile.from_string(content, format_=format)
    entries = []
    for i, event in enumerate(subs.events):
        if event.is_comment:
            continue
        entries.append(
            SubtitleEntry(
                index=i,
                start=event.start,
                end=event.end,
                text=event.plaintext,
                style=event.style,
            )
        )
    return entries
