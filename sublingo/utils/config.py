""".env loading with CLI override merging."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

DEFAULTS: dict[str, Any] = {
    "provider": "openai",
    "model": None,
    "base_url": None,
    "api_key": None,
    "temperature": 0.3,
    "batch_size": 20,
    "source_language": "auto",
    "target_language": None,
    "bilingual": False,
    "output_format": None,  # None means same as input
    "verbose": False,
}


def build_config(
    cli_args: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Merge defaults <- env vars <- CLI args."""
    load_dotenv()
    config = dict(DEFAULTS)

    # Layer 2: env vars (SUBLINGO_ prefix)
    env_map = {
        "SUBLINGO_PROVIDER": "provider",
        "SUBLINGO_MODEL": "model",
        "SUBLINGO_BASE_URL": "base_url",
        "SUBLINGO_API_KEY": "api_key",
        "SUBLINGO_TEMPERATURE": "temperature",
        "SUBLINGO_BATCH_SIZE": "batch_size",
        "SUBLINGO_SOURCE_LANGUAGE": "source_language",
        "SUBLINGO_TARGET_LANGUAGE": "target_language",
    }
    for env_key, cfg_key in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            if cfg_key == "temperature":
                config[cfg_key] = float(val)
            elif cfg_key == "batch_size":
                config[cfg_key] = int(val)
            else:
                config[cfg_key] = val

    # Also check provider-specific API key env vars
    if config["api_key"] is None:
        config["api_key"] = os.environ.get("OPENAI_API_KEY")

    # Layer 3: CLI args (override everything)
    if cli_args:
        for key, val in cli_args.items():
            if val is not None:
                config[key] = val

    return config
