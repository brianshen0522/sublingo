"""Group subtitle entries into batches for translation."""

from __future__ import annotations

from sublingo.core.subtitle_parser import SubtitleEntry


def create_batches(
    entries: list[SubtitleEntry],
    batch_size: int = 20,
) -> list[list[SubtitleEntry]]:
    """Split entries into batches of at most batch_size."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [
        entries[i : i + batch_size]
        for i in range(0, len(entries), batch_size)
    ]
