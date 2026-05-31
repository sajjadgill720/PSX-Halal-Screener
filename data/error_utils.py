"""Shared error logging utilities for the PSX Halal Screener."""

from __future__ import annotations

import logging
from pathlib import Path


LOGGER_NAME = "psx_halal_screener"


def get_logger() -> logging.Logger:
    """Return the shared project logger."""

    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    log_path = Path(__file__).resolve().parent / "errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_message(message: str, level: int = logging.ERROR) -> None:
    """Log a plain message to the shared error log."""

    get_logger().log(level, message)


def log_exception(message: str) -> None:
    """Log the current exception with a context message."""

    get_logger().exception(message)