"""File path helpers."""

from __future__ import annotations

from pathlib import Path

SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa"}
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".webm", ".mov", ".flv", ".wmv"}


def is_subtitle_file(path: Path) -> bool:
    return path.suffix.lower() in SUBTITLE_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def generate_output_path(
    input_path: Path,
    target_lang: str,
    output_format: str | None = None,
    output_path: Path | None = None,
) -> Path:
    """Generate output file path based on input path and target language.

    Default pattern: input.en.srt -> input.ja.srt
    """
    if output_path is not None:
        return output_path

    suffix = output_format if output_format else input_path.suffix
    if not suffix.startswith("."):
        suffix = f".{suffix}"

    stem = input_path.stem
    # Strip existing language code if present (e.g., "movie.en" -> "movie")
    parts = stem.rsplit(".", 1)
    if len(parts) == 2 and len(parts[1]) <= 3 and parts[1].isalpha():
        stem = parts[0]

    return input_path.parent / f"{stem}.{target_lang}{suffix}"
