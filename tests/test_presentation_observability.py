"""Persistent, content-free diagnostics around deck-generation failures."""

from __future__ import annotations

import asyncio
import json

import pytest

from presentation_pipeline.deck import DeckGenerationConfig, generate_deck
from presentation_pipeline.budgeting import CharacterTokenEstimator, GenerationLimiter, InputBudget
from presentation_pipeline.observability import (
    RunMetrics,
    configure_run_logging,
    console_formatter,
    run_metrics,
    safe_event,
    telemetry_logger,
)
from presentation_pipeline.planning import PresentationRequirements
from pydantic import BaseModel


def test_persistent_jsonl_has_run_id_and_provider_telemetry(tmp_path) -> None:
    run_id, path = configure_run_logging(output_dir=tmp_path / "deck", run_id="run123")
    telemetry_logger({
        "provider": "openai", "model": "test", "response_model": "Digest",
        "request_id": "request", "latency_ms": 1.5, "input_tokens": 3,
        "output_tokens": 4, "total_tokens": 7, "outcome": "success", "error_type": None,
        "input_data": "DO_NOT_LOG_PROVIDER_REQUEST", "response_text": "DO_NOT_LOG_PROVIDER_RESPONSE",
    })
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert run_id == "run123"
    telemetry = next(line for line in lines if line["record"]["extra"].get("event") == "provider_call")
    assert telemetry["record"]["extra"]["run_id"] == "run123"
    assert telemetry["record"]["extra"]["provider"] == "openai"
    serialized = json.dumps(lines)
    assert "input_data" not in serialized
    assert "DO_NOT_LOG_PROVIDER_REQUEST" not in serialized
    assert "DO_NOT_LOG_PROVIDER_RESPONSE" not in serialized


def test_console_formatter_escapes_structured_metadata_for_loguru() -> None:
    rendered = console_formatter({"extra": {"event": "digest_reduction_level", "doc_id": "doc-a", "level": 2, "group_sizes": [1, 1]}})
    assert "digest_reduction_level" in rendered
    assert "doc_id=doc-a" in rendered
    assert "level=2" in rendered
    assert "group_sizes=[1,1]" in rendered


def test_console_run_summary_renders_values_not_only_metric_keys() -> None:
    rendered = console_formatter({"extra": {
        "event": "run_summary", "documents": 2, "windows": 5, "llm_calls": 7,
        "actual_input_tokens": 120, "calls_by_stage": {"document_digest_reduction": 2},
    }})
    assert "documents=2" in rendered
    assert "windows=5" in rendered
    assert "actual_input_tokens=120" in rendered
    assert "calls_by_stage={{document_digest_reduction:2}}" in rendered


def test_console_sink_renders_dict_and_list_extras_without_loguru_format_errors(tmp_path, capsys) -> None:
    configure_run_logging(output_dir=tmp_path / "deck", run_id="structured")
    safe_event(
        "indexing_complete",
        evidence_kind_counts={"table": 2, "chart": 1},
        retained_evidence_kinds=["table", "chart"],
    )

    rendered = capsys.readouterr().err
    assert "evidence_kind_counts={table:2,chart:1}" in rendered
    assert "retained_evidence_kinds=[table,chart]" in rendered


def test_ratio_summary_includes_nearest_rank_percentiles_by_stage_and_model() -> None:
    metrics = RunMetrics()
    for ratio in range(1, 21):
        metrics.record_estimate_ratio(
            float(ratio),
            stage="document_digest_reduction",
            response_model="Digest",
        )
    metrics.record_estimate_ratio(0.5, stage="planning", response_model="Plan")

    summary = metrics.summary()
    assert summary["p90_estimate_to_actual_ratio"] == 18.0
    assert summary["p95_estimate_to_actual_ratio"] == 19.0
    by_stage = summary["estimate_to_actual_ratio_by_stage"]
    assert by_stage == {
        "document_digest_reduction": {"count": 20, "median": 11.0, "p90": 18.0, "p95": 19.0},
        "planning": {"count": 1, "median": 0.5, "p90": 0.5, "p95": 0.5},
    }
    assert summary["estimate_to_actual_ratio_by_stage_and_response_model"] == {
        "document_digest_reduction": {
            "Digest": {"count": 20, "median": 11.0, "p90": 18.0, "p95": 19.0}
        },
        "planning": {"Plan": {"count": 1, "median": 0.5, "p90": 0.5, "p95": 0.5}},
    }


def test_task_local_estimate_to_actual_correlation_handles_concurrent_calls(tmp_path) -> None:
    class Output(BaseModel):
        value: str

    class Generator:
        async def generate(self, *, input_data, **_kwargs):
            await asyncio.sleep(0.01 if input_data["id"] == 1 else 0)
            telemetry_logger({
                "provider": "test", "model": "test", "response_model": "Output",
                "request_id": str(input_data["id"]), "latency_ms": 1.0,
                "input_tokens": 10 + input_data["id"], "output_tokens": 1,
                "total_tokens": 11 + input_data["id"], "outcome": "success", "error_type": None,
            })
            return Output(value="ok")

    configure_run_logging(output_dir=tmp_path / "deck", run_id="correlation")
    limiter = GenerationLimiter(Generator(), token_counter=CharacterTokenEstimator(), concurrency=2)

    async def run() -> None:
        await asyncio.gather(*(
            limiter.invoke(
                system_prompt="prompt", input_data={"id": identifier}, response_model=Output,
                budget=InputBudget(max_input_tokens=100), stage=f"stage-{identifier}",
            )
            for identifier in (1, 2)
        ))

    asyncio.run(run())
    metrics = run_metrics()
    assert metrics is not None
    assert len(metrics.estimated_to_actual_ratios) == 2
    assert all(ratio > 0 for ratio in metrics.estimated_to_actual_ratios)
    assert metrics.summary()["calls_by_stage"] == {"stage-1": 1, "stage-2": 1}


def _failing_deck(tmp_path, monkeypatch, *, keep: bool, run_id: str):
    import presentation_pipeline.deck as deck

    async def fail(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(deck, "generate_plan", fail)
    output = tmp_path / "published"
    output.mkdir()
    sentinel = output / "prior.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(generate_deck(
            [tmp_path / "input.docx"],
            PresentationRequirements(goal="Goal", audience="Audience", target_slide_count=1),
            object(), output_dir=output, overwrite=True,
            config=DeckGenerationConfig(keep_failed_artifacts=keep, run_id=run_id, log_path="run.jsonl"),
        ))
    return output, sentinel


def test_normal_failure_report_removes_staging_and_preserves_prior_output(tmp_path, monkeypatch) -> None:
    output, sentinel = _failing_deck(tmp_path, monkeypatch, keep=False, run_id="normal")
    report = json.loads((tmp_path / ".presentation-failures" / "published-normal.json").read_text(encoding="utf-8"))
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert report["failed_stage"] == "planning"
    assert report["failed_artifacts_path"] is None
    assert not list(tmp_path.glob(".published.staging-*"))
    assert output.is_dir()


def test_debug_failure_preserves_staging_and_prior_output(tmp_path, monkeypatch) -> None:
    _output, sentinel = _failing_deck(tmp_path, monkeypatch, keep=True, run_id="debug")
    preserved = tmp_path / ".presentation-failures" / "published-debug"
    report = json.loads((tmp_path / ".presentation-failures" / "published-debug.json").read_text(encoding="utf-8"))
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert preserved.is_dir()
    assert report["failed_artifacts_path"] == str(preserved)


def test_failure_report_persists_safe_domain_metadata(tmp_path) -> None:
    from presentation_pipeline.deck import _failure_report
    from presentation_pipeline.understanding.reduction import ReductionNonReductionError

    error = ReductionNonReductionError(
        doc_id="doc-a", level=2, fragment_count=3, group_sizes=[1, 1, 1],
        estimates={"fragment_tokens": [10, 11, 12]}, limit=20, rounds=1,
        target_fragment_tokens=7, max_compaction_rounds=2,
        reason="max_compaction_rounds_at_level",
    )
    path, report = _failure_report(
        root=tmp_path / "output", run_id="metadata", failed_stage="document_digest_reduction",
        error=error, elapsed_seconds=1, input_paths=[], log_path=None, failed_artifacts_path=None,
    )
    assert path.name == "output-metadata.json"
    assert report["error_metadata"]["target_fragment_tokens"] == 7
    assert report["error_metadata"]["reason"] == "max_compaction_rounds_at_level"
