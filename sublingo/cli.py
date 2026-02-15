"""Click CLI definitions."""

from __future__ import annotations

from pathlib import Path

import click

from sublingo import __version__
from sublingo.services.translation_service import PROVIDERS
from sublingo.utils.config import build_config
from sublingo.utils.file_utils import VIDEO_EXTENSIONS, SUBTITLE_EXTENSIONS
from sublingo.utils.logger import setup_logging


def _collect_files(input_path: Path, recursive: bool = False) -> list[Path]:
    """Collect translatable files from a path (file or directory)."""
    valid_extensions = SUBTITLE_EXTENSIONS | VIDEO_EXTENSIONS
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        pattern = "**/*" if recursive else "*"
        files = sorted(
            f for f in input_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in valid_extensions
        )
        return files
    return []


@click.group()
@click.version_option(version=__version__, prog_name="sublingo")
def cli() -> None:
    """SubLingo - Translate subtitles using LLMs."""


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--to", "target_lang", default=None, help="Target language (e.g., ja, en, es)")
@click.option("--from", "source_lang", default=None, help="Source language (default: auto-detect)")
@click.option("--provider", default=None, help="LLM provider (openai, ollama, vllm)")
@click.option("--model", default=None, help="Model name to use")
@click.option("--base-url", default=None, help="API base URL override")
@click.option("--api-key", default=None, help="API key")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output file path")
@click.option("--output-format", default=None, help="Output format (srt, vtt, ass)")
@click.option("--batch-size", type=int, default=None, help="Entries per translation batch")
@click.option("--temperature", type=float, default=None, help="LLM temperature")
@click.option("--timeout", type=float, default=None, help="LLM API timeout in seconds (default: 120)")
@click.option("--bilingual", is_flag=True, default=False, help="Include original text with translation")
@click.option("--keep-names", is_flag=True, default=False, help="Keep personal and place names untranslated")
@click.option("-r", "--recursive", is_flag=True, default=False, help="Recursively scan subdirectories")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output")
@click.option("--debug", is_flag=True, default=False, help="Debug mode: print raw LLM responses")
def translate(
    input_path: Path,
    target_lang: str,
    source_lang: str | None,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    output: Path | None,
    output_format: str | None,
    batch_size: int | None,
    temperature: float | None,
    timeout: float | None,
    bilingual: bool,
    keep_names: bool,
    recursive: bool,
    verbose: bool,
    debug: bool,
) -> None:
    """Translate a subtitle or video file, or all files in a directory."""
    setup_logging(verbose=verbose or debug, debug=debug)

    cli_args = {
        "target_language": target_lang,
        "source_language": source_lang or "auto",
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "output_format": output_format,
        "batch_size": batch_size,
        "temperature": temperature,
        "timeout": timeout,
        "bilingual": bilingual,
        "keep_names": keep_names,
        "debug": debug,
        "verbose": verbose,
    }
    # Remove None values so they don't override config
    cli_args = {k: v for k, v in cli_args.items() if v is not None}

    config = build_config(cli_args=cli_args)

    if not config.get("target_language"):
        raise click.ClickException(
            "Target language required. Use --to or set SUBLINGO_TARGET_LANGUAGE."
        )

    files = _collect_files(input_path, recursive=recursive)
    if not files:
        raise click.ClickException(
            f"No subtitle or video files found in: {input_path}"
        )

    # --output only valid for single file
    if output and len(files) > 1:
        raise click.ClickException(
            "-o/--output cannot be used when translating a directory."
        )

    from sublingo.services.translation_service import translate_file

    click.echo(f"Found {len(files)} file(s) to translate")

    for i, file in enumerate(files, 1):
        if len(files) > 1:
            click.echo(f"\n[{i}/{len(files)}] {file.name}")
        try:
            out = translate_file(file, config, output_path=output)
            click.echo(f"Translated: {out}")
        except Exception as e:
            click.echo(f"Error translating {file.name}: {e}", err=True)
            if len(files) == 1:
                raise click.ClickException(str(e))


@cli.command("providers")
def list_providers() -> None:
    """List available translation providers."""
    click.echo("Available providers:")
    for name in PROVIDERS:
        click.echo(f"  - {name}")


@cli.command("languages")
def list_languages() -> None:
    """List supported language codes."""
    from sublingo.utils.languages import list_languages

    langs = list_languages()
    click.echo("Supported language codes:")
    for code, name in sorted(langs.items()):
        click.echo(f"  {code:<8} {name}")


@cli.command()
def config() -> None:
    """Show current configuration."""
    cfg = build_config()
    for key, val in sorted(cfg.items()):
        if key == "api_key" and val:
            val = val[:4] + "..." + val[-4:] if len(val) > 8 else "***"
        click.echo(f"  {key}: {val}")
