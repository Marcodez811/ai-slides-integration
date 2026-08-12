"""Small, content-safe Loguru observability for presentation generation runs."""

from __future__ import annotations

import sys
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger


def new_run_id() -> str:
    """Return a compact identifier for logs and failure reports only."""
    return uuid4().hex[:12]


def configure_run_logging(
    *, output_dir: str | Path, log_level: str = "INFO", run_id: str | None = None
) -> tuple[str, Path]:
    """Configure stderr and one persistent JSONL sink outside output staging."""
    level = str(log_level).upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("log_level must be DEBUG, INFO, WARNING, or ERROR")
    identifier = run_id or new_run_id()
    output = Path(output_dir).absolute()
    log_dir = output.parent / ".presentation-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{output.name}-{identifier}.jsonl"
    logger.remove()
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level} | {message}")
    logger.add(path, level="DEBUG", serialize=True, encoding="utf-8")
    logger.configure(extra={"run_id": identifier})
    logger.bind(run_id=identifier).info("run_logging_configured log_path={log_path}", log_path=str(path))
    return identifier, path


def safe_event(event: str, **fields: object) -> None:
    """Emit only metadata-shaped fields; discard arbitrary content containers."""
    safe: dict[str, object] = {key: _safe_value(value) for key, value in fields.items()}
    logger.bind(event=event, **safe).info(event)


def safe_error(event: str, **fields: object) -> None:
    safe: dict[str, object] = {key: _safe_value(value) for key, value in fields.items()}
    logger.bind(event=event, **safe).error(event)


def safe_debug(event: str, **fields: object) -> None:
    safe: dict[str, object] = {key: _safe_value(value) for key, value in fields.items()}
    logger.bind(event=event, **safe).debug(event)


class stage_timer(AbstractContextManager["stage_timer"]):
    """Log one content-free start/end/error event and elapsed milliseconds."""

    def __init__(self, stage: str, **fields: object) -> None:
        self.stage = stage
        self.fields = fields
        self.started = 0.0

    def __enter__(self) -> "stage_timer":
        self.started = time.perf_counter()
        safe_event("stage_start", stage=self.stage, **self.fields)
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, _traceback: object) -> bool:
        elapsed_ms = round((time.perf_counter() - self.started) * 1000, 2)
        if exc is None:
            safe_event("stage_end", stage=self.stage, elapsed_ms=elapsed_ms, **self.fields)
        else:
            safe_error(
                "stage_error",
                stage=self.stage,
                elapsed_ms=elapsed_ms,
                error_type=type(exc).__name__,
                **self.fields,
            )
        return False


def telemetry_logger(event: object) -> None:
    """Persist provider telemetry without request or response content."""
    values: dict[str, Any]
    if is_dataclass(event) and not isinstance(event, type):
        values = asdict(event)
    elif isinstance(event, dict):
        values = event
    else:
        values = {
            name: getattr(event, name)
            for name in (
                "provider", "model", "response_model", "request_id", "latency_ms",
                "input_tokens", "output_tokens", "total_tokens", "outcome", "error_type",
            )
            if hasattr(event, name)
        }
    safe_event("provider_call", **values)


def _safe_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        # IDs/stages/paths/types are safe. Avoid a generic data logger becoming
        # a source-text sink by clipping arbitrary accidental strings.
        return value[:512]
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value[:100]]
    if isinstance(value, dict):
        return {str(key)[:80]: _safe_value(item) for key, item in list(value.items())[:100]}
    return type(value).__name__
