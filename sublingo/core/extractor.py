"""Extract subtitles from video files via ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from sublingo.utils.logger import get_logger

logger = get_logger(__name__)


def extract_subtitles(
    video_path: Path,
    output_path: Path | None = None,
    stream_index: int = 0,
) -> Path:
    """Extract subtitle stream from a video file using ffmpeg.

    Returns the path to the extracted subtitle file.
    """
    if output_path is None:
        output_path = video_path.with_suffix(".srt")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-map", f"0:s:{stream_index}",
        "-c:s", "srt",
        str(output_path),
    ]

    logger.info("Extracting subtitles: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg subtitle extraction failed:\n{result.stderr}"
        )

    logger.info("Extracted subtitles to %s", output_path)
    return output_path
