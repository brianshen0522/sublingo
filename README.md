# SubLingo

CLI subtitle translation tool powered by LLMs. Translate `.srt`, `.vtt`, and `.ass` subtitle files using OpenAI, Ollama, vLLM, or any OpenAI-compatible API.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Translate a subtitle file to Traditional Chinese
sublingo translate movie.srt --to zh-TW

# Translate all video/subtitle files in a directory
sublingo translate /path/to/videos/

# Use Ollama locally
sublingo translate movie.srt --to es --provider ollama --model llama3.1

# Translate an MKV file directly (subtitles extracted via ffmpeg)
sublingo translate movie.mkv --to ja

# Bilingual output (original + translation)
sublingo translate movie.srt --to fr --bilingual

# Keep personal and place names untranslated
sublingo translate movie.srt --to zh-TW --keep-names

# Specify output path and format
sublingo translate movie.srt --to de -o movie.de.vtt --output-format vtt

# Debug mode: show full prompts and raw LLM responses
sublingo translate movie.srt --to ja --debug
```

## Configuration

Configure via `.env` file or environment variables. CLI arguments override everything.

Copy `.env.example` to `.env` and edit:

```bash
SUBLINGO_PROVIDER=openai
SUBLINGO_MODEL=gpt-4o-mini
SUBLINGO_BASE_URL=https://api.openai.com/v1
SUBLINGO_API_KEY=sk-...
SUBLINGO_TEMPERATURE=0.3
SUBLINGO_BATCH_SIZE=20
SUBLINGO_SOURCE_LANGUAGE=auto
SUBLINGO_TARGET_LANGUAGE=zh-TW
```

When `SUBLINGO_TARGET_LANGUAGE` is set, the `--to` flag becomes optional.

### Language Codes

Use standard language codes (e.g., `en`, `ja`, `zh-TW`, `zh-CN`, `es`, `fr`). Codes are automatically resolved to full names in prompts (e.g., `zh-TW` becomes "Traditional Chinese").

```bash
sublingo languages  # List all supported language codes
```

## Providers

| Provider | Description                | Default URL                  |
| -------- | -------------------------- | ---------------------------- |
| `openai` | OpenAI API (or compatible) | `https://api.openai.com/v1`  |
| `ollama` | Ollama local LLM           | `http://localhost:11434`     |
| `vllm`   | vLLM (OpenAI-compat mode)  | `http://localhost:8000/v1`   |

```bash
sublingo providers  # List available providers
```

## Commands

```bash
sublingo translate <file-or-dir>       # Translate subtitles
sublingo translate <file> --to <lang>  # Specify target language
sublingo providers                     # List providers
sublingo languages                     # List language codes
sublingo config                        # Show current config
```

## CLI Options

| Option             | Description                                  |
| ------------------ | -------------------------------------------- |
| `--to`             | Target language code (e.g., `ja`, `zh-TW`)   |
| `--from`           | Source language (default: auto-detect)        |
| `--provider`       | LLM provider (`openai`, `ollama`, `vllm`)    |
| `--model`          | Model name                                   |
| `--base-url`       | API base URL override                        |
| `--api-key`        | API key                                      |
| `-o`, `--output`   | Output file path                             |
| `--output-format`  | Output format (`srt`, `vtt`, `ass`)          |
| `--batch-size`     | Entries per translation batch (default: 20)  |
| `--temperature`    | LLM temperature (default: 0.3)               |
| `--timeout`        | API timeout in seconds (default: 120)        |
| `--bilingual`      | Include original text with translation       |
| `--keep-names`     | Keep personal/place names untranslated       |
| `-v`, `--verbose`  | Verbose output                               |
| `--debug`          | Show full prompts and raw LLM responses      |

## Supported Formats

- **SRT** (.srt) - SubRip
- **VTT** (.vtt) - WebVTT
- **ASS/SSA** (.ass, .ssa) - Advanced SubStation Alpha

Video files (.mkv, .mp4, .avi, .webm, .mov) are also supported — subtitles are extracted via ffmpeg automatically. Extracted subtitle files are cleaned up after translation unless `--debug` is used.

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

[MIT](LICENSE)
