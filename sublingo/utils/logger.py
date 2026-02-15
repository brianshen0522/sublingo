"""Structured logging setup."""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_FORMAT_VERBOSE = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"


def setup_logging(verbose: bool = False, debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if (verbose or debug) else logging.INFO
    fmt = LOG_FORMAT_VERBOSE if (verbose or debug) else LOG_FORMAT

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))

    logger = logging.getLogger("sublingo")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def get_logger(name: str = "sublingo") -> logging.Logger:
    return logging.getLogger(name)
