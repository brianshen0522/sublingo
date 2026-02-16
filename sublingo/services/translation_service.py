"""Orchestrate the full subtitle translation pipeline."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from sublingo.core.batching import create_batches
from sublingo.core.extractor import extract_subtitles
from sublingo.core.subtitle_builder import build_file
from sublingo.core.subtitle_parser import SubtitleEntry, parse_file
from sublingo.providers.base import BaseLLMProvider
from sublingo.providers.ollama_provider import OllamaProvider
from sublingo.providers.openai_provider import OpenAIProvider
from sublingo.providers.vllm_provider import VLLMProvider
from sublingo.services.language_detection import detect_language
from sublingo.utils.file_utils import generate_output_path, is_video_file
from sublingo.utils.languages import resolve_language
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from sublingo.utils.logger import get_logger

logger = get_logger(__name__)

# Global state for graceful shutdown
_cancel_event = threading.Event()
_skip_event = threading.Event()
_temp_files: list[Path] = []


class TranslationSkipped(Exception):
    """Raised when user presses 's' to skip current file."""


def _quit_listener() -> None:
    """Background thread that listens for key presses: 'q' to quit, 's' to skip."""
    try:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not _cancel_event.is_set():
                ch = sys.stdin.read(1)
                if ch.lower() == "q":
                    _cancel_event.set()
                    break
                elif ch.lower() == "s":
                    _skip_event.set()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except (ImportError, OSError, ValueError):
        # Not a terminal (e.g. piped input), skip listener
        pass


def _cleanup_temp_files() -> None:
    """Remove any extracted temporary subtitle files."""
    for path in _temp_files:
        if path.exists():
            path.unlink(missing_ok=True)
            logger.info("Cleaned up temp file: %s", path)
    _temp_files.clear()


PROVIDERS: dict[str, type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
    "vllm": VLLMProvider,
}


def get_provider(config: dict[str, Any]) -> BaseLLMProvider:
    provider_name = config.get("provider", "openai")
    provider_cls = PROVIDERS.get(provider_name)
    if provider_cls is None:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available: {', '.join(PROVIDERS)}"
        )
    kwargs: dict[str, Any] = {
        "model": config.get("model"),
        "base_url": config.get("base_url"),
        "api_key": config.get("api_key"),
        "temperature": config.get("temperature", 0.3),
    }
    if config.get("timeout") is not None:
        kwargs["timeout"] = float(config["timeout"])
    provider = provider_cls(**kwargs)
    if config.get("retries") is not None:
        provider.retries = int(config["retries"])
    return provider


def translate_file(
    input_path: Path,
    config: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """Translate a subtitle file end-to-end.

    Returns the output file path.
    """
    target_lang = config.get("target_language", "en")
    source_lang = config.get("source_language", "auto")
    batch_size = config.get("batch_size", 20)
    bilingual = config.get("bilingual", False)
    keep_names = config.get("keep_names", False)
    output_format = config.get("output_format")
    debug = config.get("debug", False)
    extracted_sub_path: Path | None = None

    # Reset events for this file
    _skip_event.clear()

    # If input is a video, extract subtitles first
    if is_video_file(input_path):
        logger.info("Extracting subtitles from video: %s", input_path)
        input_path = extract_subtitles(input_path)
        extracted_sub_path = input_path
        if not debug:
            _temp_files.append(extracted_sub_path)

    # Parse
    logger.info("Parsing subtitles: %s", input_path)
    entries = parse_file(input_path)
    logger.info("Found %d subtitle entries", len(entries))

    if not entries:
        raise ValueError("No subtitle entries found in file")

    # Get provider
    provider = get_provider(config)
    logger.info("Using provider: %s (model: %s)", provider.name, provider.model)

    # Resolve language codes to full names for prompts
    target_lang_full = resolve_language(target_lang)
    logger.info("Target language: %s -> %s", target_lang, target_lang_full)

    # Detect source language if auto
    try:
        if source_lang == "auto":
            detected = detect_language(
                entries, provider,
                cancel_event=_cancel_event, skip_event=_skip_event,
            )
            source_lang = detected.get("language", "Unknown")
            logger.info("Auto-detected source language: %s", source_lang)
        else:
            source_lang = resolve_language(source_lang)
            logger.info("Source language: %s", source_lang)
    except KeyboardInterrupt:
        _cleanup_temp_files()
        raise
    except InterruptedError:
        _cleanup_temp_files()
        raise TranslationSkipped()

    # Batch and translate
    batches = create_batches(entries, batch_size)
    logger.info("Processing %d batches (batch_size=%d)", len(batches), batch_size)

    translated_entries: list[SubtitleEntry] = []

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("entries"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    cancelled = False
    with progress:
        task = progress.add_task(
            f"Translating to {target_lang_full}",
            total=len(entries),
        )

        for i, batch in enumerate(batches, 1):
            if _cancel_event.is_set():
                cancelled = True
                break
            if _skip_event.is_set():
                break

            logger.debug("Translating batch %d/%d (%d entries)", i, len(batches), len(batch))

            try:
                texts = [{"index": e.index, "text": e.text} for e in batch]
                results = provider.translate(
                    texts, source_lang, target_lang_full,
                    keep_names=keep_names,
                    cancel_event=_cancel_event,
                    skip_event=_skip_event,
                )
            except KeyboardInterrupt:
                cancelled = True
                break
            except InterruptedError:
                break

            for j, result in enumerate(results):
                original = batch[j] if j < len(batch) else batch[-1]
                translated_entries.append(
                    SubtitleEntry(
                        index=original.index,
                        start=original.start,
                        end=original.end,
                        text=result.get("text", original.text),
                        style=original.style,
                    )
                )

            progress.advance(task, len(batch))

    if cancelled:
        logger.info("Translation cancelled by user")
        _cleanup_temp_files()
        raise KeyboardInterrupt("Translation cancelled by user (pressed q)")

    if _skip_event.is_set():
        logger.info("File skipped by user")
        _cleanup_temp_files()
        raise TranslationSkipped()

    # Build output
    out = output_path or generate_output_path(input_path, target_lang, output_format)
    logger.info("Writing output: %s", out)

    build_file(
        entries=translated_entries,
        output_path=out,
        original_path=input_path,
        bilingual=bilingual,
        original_entries=entries if bilingual else None,
    )

    # Clean up extracted subtitle file unless in debug mode
    if extracted_sub_path and not debug:
        extracted_sub_path.unlink(missing_ok=True)
        if extracted_sub_path in _temp_files:
            _temp_files.remove(extracted_sub_path)
        logger.info("Cleaned up extracted subtitle: %s", extracted_sub_path)

    logger.info("Translation complete: %s -> %s", input_path, out)
    return out
