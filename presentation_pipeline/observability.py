"""Small, content-safe Loguru observability for presentation generation runs."""

from __future__ import annotations

import sys
import time
from math import ceil
from contextvars import ContextVar, Token
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger


@dataclass(frozen=True, slots=True)
class _CallCorrelation:
    stage: str
    response_model: str
    estimated_input_tokens: int


_call_correlation: ContextVar[_CallCorrelation | None] = ContextVar(
    "presentation_llm_call_correlation", default=None
)


class RunMetrics:
    """Small in-memory, content-free counters for the active CLI run."""

    def __init__(self) -> None:
        self.provider_calls = 0
        self.provider_input_tokens = 0
        self.provider_output_tokens = 0
        self.provider_latency_ms = 0.0
        self.calls_by_response_model: dict[str, int] = {}
        self.calls_by_stage: dict[str, int] = {}
        self.estimated_to_actual_ratios: list[float] = []
        self.estimated_to_actual_ratios_by_stage: dict[str, list[float]] = {}
        self.estimated_to_actual_ratios_by_stage_and_model: dict[
            str, dict[str, list[float]]
        ] = {}
        self.document_count = 0
        self.window_count = 0
        self.chunk_digest_count = 0
        self.semantic_slide_count = 0
        self.physical_slide_count = 0

    def record_limiter_call(self, *, stage: str) -> None:
        self.calls_by_stage[stage] = self.calls_by_stage.get(stage, 0) + 1

    def record_provider(self, values: dict[str, Any]) -> None:
        self.provider_calls += 1
        model = values.get("response_model")
        if isinstance(model, str):
            self.calls_by_response_model[model] = self.calls_by_response_model.get(model, 0) + 1
        for key, destination in (("input_tokens", "provider_input_tokens"), ("output_tokens", "provider_output_tokens")):
            value = values.get(key)
            if isinstance(value, int) and value >= 0:
                setattr(self, destination, getattr(self, destination) + value)
        latency = values.get("latency_ms")
        if isinstance(latency, (int, float)) and latency >= 0:
            self.provider_latency_ms += float(latency)

    def record_estimate_ratio(
        self,
        ratio: float,
        *,
        stage: str | None = None,
        response_model: str | None = None,
    ) -> None:
        if ratio >= 0:
            self.estimated_to_actual_ratios.append(ratio)
            if stage is not None:
                self.estimated_to_actual_ratios_by_stage.setdefault(stage, []).append(ratio)
                if response_model is not None:
                    self.estimated_to_actual_ratios_by_stage_and_model.setdefault(
                        stage, {}
                    ).setdefault(response_model, []).append(ratio)

    def record_event(self, event: str, fields: dict[str, object]) -> None:
        if event == "document_digest_start":
            self.document_count += 1
            windows = fields.get("window_count")
            if isinstance(windows, int) and windows >= 0:
                self.window_count += windows
        elif event == "chunk_digest_complete":
            self.chunk_digest_count += 1
        elif event == "run_complete":
            for key in ("semantic_slide_count", "physical_slide_count"):
                value = fields.get(key)
                if isinstance(value, int) and value >= 0:
                    setattr(self, key, value)

    def summary(self) -> dict[str, object]:
        ratios = sorted(self.estimated_to_actual_ratios)
        stage_ratios = {
            stage: _ratio_summary(values)
            for stage, values in sorted(self.estimated_to_actual_ratios_by_stage.items())
        }
        stage_and_model_ratios = {
            stage: {
                model: _ratio_summary(values)
                for model, values in sorted(models.items())
            }
            for stage, models in sorted(
                self.estimated_to_actual_ratios_by_stage_and_model.items()
            )
        }
        return {
            "provider_call_count": self.provider_calls,
            "actual_provider_input_tokens": self.provider_input_tokens,
            "actual_provider_output_tokens": self.provider_output_tokens,
            "provider_latency_ms": round(self.provider_latency_ms, 2),
            "calls_by_response_model": dict(self.calls_by_response_model),
            "calls_by_stage": dict(self.calls_by_stage),
            "reduction_call_count": self.calls_by_stage.get("document_digest_reduction", 0),
            "compaction_call_count": self.calls_by_stage.get("document_digest_compaction", 0),
            "document_count": self.document_count,
            "window_count": self.window_count,
            "chunk_digest_count": self.chunk_digest_count,
            "semantic_slide_count": self.semantic_slide_count,
            "physical_slide_count": self.physical_slide_count,
            "average_estimate_to_actual_ratio": (sum(ratios) / len(ratios)) if ratios else None,
            "median_estimate_to_actual_ratio": (ratios[len(ratios) // 2] if ratios else None),
            "p90_estimate_to_actual_ratio": _nearest_rank_percentile(ratios, 0.90),
            "p95_estimate_to_actual_ratio": _nearest_rank_percentile(ratios, 0.95),
            "max_estimate_to_actual_ratio": (max(ratios) if ratios else None),
            "estimate_to_actual_ratio_by_stage": stage_ratios,
            "estimate_to_actual_ratio_by_stage_and_response_model": stage_and_model_ratios,
        }


_active_metrics: RunMetrics | None = None


def run_metrics() -> RunMetrics | None:
    return _active_metrics


def console_formatter(record: dict[str, Any]) -> str:
    """Render safe Loguru extras on stderr without serializing content blobs."""
    extra = record["extra"]
    event = extra.get("event")
    fields = []
    for key, value in extra.items():
        if key in {"event", "run_id"} or value is None:
            continue
        rendered = _console_value(value)
        if rendered is not None:
            # Loguru parses the returned formatter value as a format string.
            # Structured values naturally contain braces, so escape every
            # dynamic component before returning its template.
            fields.append(f"{_escape_loguru_braces(str(key))}={_escape_loguru_braces(rendered)}")
    suffix = (" | " + " ".join(fields)) if fields else ""
    rendered_event = "{message}" if event is None else _escape_loguru_braces(str(event))
    return "{time:HH:mm:ss} | {level} | " + rendered_event + suffix + "\n"


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
    global _active_metrics
    _active_metrics = RunMetrics()
    logger.remove()
    logger.add(sys.stderr, level=level, format=console_formatter)
    logger.add(path, level="DEBUG", serialize=True, encoding="utf-8")
    logger.configure(extra={"run_id": identifier})
    logger.bind(run_id=identifier).info("run_logging_configured log_path={log_path}", log_path=str(path))
    return identifier, path


def safe_event(event: str, **fields: object) -> None:
    """Emit only metadata-shaped fields; discard arbitrary content containers."""
    safe: dict[str, object] = {key: _safe_value(value) for key, value in fields.items()}
    if _active_metrics is not None:
        _active_metrics.record_event(event, safe)
    logger.bind(event=event, **safe).info(event)


def safe_error(event: str, **fields: object) -> None:
    safe: dict[str, object] = {key: _safe_value(value) for key, value in fields.items()}
    logger.bind(event=event, **safe).error(event)


def safe_debug(event: str, **fields: object) -> None:
    safe: dict[str, object] = {key: _safe_value(value) for key, value in fields.items()}
    if _active_metrics is not None:
        _active_metrics.record_event(event, safe)
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
    raw_values: dict[str, Any]
    if isinstance(event, dict):
        raw_values = event
    else:
        raw_values = {
            name: getattr(event, name)
            for name in _PROVIDER_TELEMETRY_FIELDS
            if hasattr(event, name)
        }
    # Providers may attach payloads or arbitrary debugging data to their event
    # objects. Keep this allow-list deliberately limited to operational metadata.
    values = {
        name: raw_values[name]
        for name in _PROVIDER_TELEMETRY_FIELDS
        if name in raw_values
    }
    if _active_metrics is not None:
        _active_metrics.record_provider(values)
    safe_event("provider_call", **values)
    correlation = _call_correlation.get()
    actual = values.get("input_tokens")
    if correlation is not None and isinstance(actual, int) and actual > 0:
        ratio = correlation.estimated_input_tokens / actual
        if _active_metrics is not None:
            _active_metrics.record_estimate_ratio(
                ratio,
                stage=correlation.stage,
                response_model=correlation.response_model,
            )
        safe_event(
            "estimated_vs_actual_tokens",
            stage=correlation.stage,
            response_model=correlation.response_model,
            estimated_input_tokens=correlation.estimated_input_tokens,
            actual_input_tokens=actual,
            estimate_to_actual_ratio=ratio,
        )


def record_limiter_call(*, stage: str) -> None:
    if _active_metrics is not None:
        _active_metrics.record_limiter_call(stage=stage)


def bind_limiter_call(
    *, stage: str, response_model: str, estimated_input_tokens: int
) -> Token[_CallCorrelation | None]:
    """Bind provider telemetry to the active async task, never a global queue."""
    return _call_correlation.set(
        _CallCorrelation(
            stage=stage,
            response_model=response_model,
            estimated_input_tokens=estimated_input_tokens,
        )
    )


def reset_limiter_call(token: Token[_CallCorrelation | None]) -> None:
    _call_correlation.reset(token)


def _console_value(value: object) -> str | None:
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
        return "[" + ",".join(str(item) for item in value[:20]) + "]"
    if isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(item, (str, int, float, bool))
        for key, item in value.items()
    ):
        return "{" + ",".join(f"{key}:{item}" for key, item in list(value.items())[:8]) + "}"
    return None


def _escape_loguru_braces(value: str) -> str:
    """Escape dynamic data embedded into a Loguru format-string template."""
    return value.replace("{", "{{").replace("}", "}}")


def _nearest_rank_percentile(values: list[float], percentile: float) -> float | None:
    """Return a deterministic nearest-rank percentile for pre-sorted values."""
    if not values:
        return None
    return values[ceil(percentile * len(values)) - 1]


def _ratio_summary(values: list[float]) -> dict[str, float | int | None]:
    """Stage-level ratio stats; p90/p95 use the nearest-rank convention."""
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        # Preserve the existing upper-middle median behavior for even samples.
        "median": sorted_values[len(sorted_values) // 2] if sorted_values else None,
        "p90": _nearest_rank_percentile(sorted_values, 0.90),
        "p95": _nearest_rank_percentile(sorted_values, 0.95),
    }


_PROVIDER_TELEMETRY_FIELDS = frozenset({
    "provider",
    "model",
    "response_model",
    "request_id",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "outcome",
    "error_type",
})


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
