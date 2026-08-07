"""Logging helpers for the DOCX pipeline.

Importing this module never changes Loguru's process-wide sinks.  Applications
that embed the library retain control; the CLI explicitly calls
``configure_logging`` for predictable command-line output.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def get_logger(**context: object):
    """Return the shared logger bound to this package and optional context."""

    return logger.bind(component="docx_pipeline", **context)


def configure_logging(level: str = "INFO", log_file: str | Path | None = None, serialize: bool = False) -> None:
    """Configure deterministic CLI sinks.

    This is intentionally opt-in because Loguru sinks are process-global.
    """

    logger.remove()
    sink_kwargs = {"level": level.upper(), "serialize": serialize, "enqueue": False}
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}", **sink_kwargs)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(path, format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}", **sink_kwargs)
