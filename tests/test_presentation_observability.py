"""Persistent, content-free diagnostics around deck-generation failures."""

from __future__ import annotations

import asyncio
import json

import pytest

from presentation_pipeline.deck import DeckGenerationConfig, generate_deck
from presentation_pipeline.observability import configure_run_logging, telemetry_logger
from presentation_pipeline.planning import PresentationRequirements


def test_persistent_jsonl_has_run_id_and_provider_telemetry(tmp_path) -> None:
    run_id, path = configure_run_logging(output_dir=tmp_path / "deck", run_id="run123")
    telemetry_logger({
        "provider": "openai", "model": "test", "response_model": "Digest",
        "request_id": "request", "latency_ms": 1.5, "input_tokens": 3,
        "output_tokens": 4, "total_tokens": 7, "outcome": "success", "error_type": None,
    })
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert run_id == "run123"
    telemetry = next(line for line in lines if line["record"]["extra"].get("event") == "provider_call")
    assert telemetry["record"]["extra"]["run_id"] == "run123"
    assert telemetry["record"]["extra"]["provider"] == "openai"
    assert "input_data" not in json.dumps(lines)


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
