"""Opt-in diagnostics and OpenAI evaluation for the local ``docs/`` corpus.

Nothing in this module runs during the ordinary test suite.  The deterministic
corpus audit and the paid OpenAI evaluation have separate opt-in switches so
extraction/indexing problems can be found before any model calls are made.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_DOCS_DIR = _ROOT / "docs"
_RUN_CORPUS_AUDIT = os.environ.get("RUN_DOCS_CORPUS_TESTS") == "1"
_RUN_OPENAI = os.environ.get("RUN_OPENAI_INTEGRATION_TESTS") == "1" and bool(
    os.environ.get("OPENAI_API_KEY")
)


def _document_paths() -> list[Path]:
    paths = sorted(_DOCS_DIR.glob("*.docx"))
    raw_limit = os.environ.get("OPENAI_DOCS_LIMIT")
    if raw_limit is None:
        return paths
    limit = int(raw_limit)
    if limit < 1:
        raise ValueError("OPENAI_DOCS_LIMIT must be at least 1")
    # A limited run is a cost-control smoke test, so prefer the smallest real
    # documents instead of whichever Unicode filename happens to sort first.
    return sorted(paths, key=lambda path: (path.stat().st_size, path.name))[:limit]


def _output_dir() -> Path:
    configured = Path(
        os.environ.get("OPENAI_DOCS_OUTPUT_DIR", "output/openai-docs-corpus")
    )
    path = configured if configured.is_absolute() else _ROOT / configured
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_report(filename: str, report: dict[str, object]) -> Path:
    path = _output_dir() / filename
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote corpus report: {path}")
    return path


@pytest.mark.skipif(
    not _RUN_CORPUS_AUDIT,
    reason="requires RUN_DOCS_CORPUS_TESTS=1",
)
def test_docs_corpus_extracts_indexes_and_preserves_provenance() -> None:
    """Audit every real DOCX without making an OpenAI request."""
    from loguru import logger

    from presentation_pipeline.corpus.batch import build_jobs, extract_batch
    from presentation_pipeline.indexing.builder import build_document_index
    from presentation_pipeline.budgeting import PlanningBudgets, Utf8ByteTokenEstimator
    from presentation_pipeline.scale import document_window_diagnostics
    from presentation_pipeline.validation import CorpusLookup

    paths = _document_paths()
    assert paths, f"no DOCX files found in {_DOCS_DIR}"

    started = perf_counter()
    logger.disable("docx_pipeline")
    try:
        batch = extract_batch(
            build_jobs(paths),
            max_workers=min(4, len(paths)),
            allow_degraded=True,
        )
    finally:
        logger.enable("docx_pipeline")
    indexes = []
    document_reports: list[dict[str, object]] = []
    index_failures: list[dict[str, str]] = []

    for artifact in batch.documents:
        extraction = artifact.extraction
        try:
            index = build_document_index(artifact)
            indexes.append(index)
            window_diagnostics = document_window_diagnostics(
                artifact, index, token_counter=Utf8ByteTokenEstimator(), budgets=PlanningBudgets()
            )
            evidence_counts = Counter(item.kind.value for item in index.evidence)
        except Exception as error:
            index_failures.append(
                {
                    "filename": artifact.filename,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            window_diagnostics = {}
            evidence_counts = Counter()

        document_reports.append(
            {
                "filename": artifact.filename,
                "health": artifact.health.value,
                "source_nodes": len(extraction.nodes),
                "normalized_blocks": len(extraction.views.blocks),
                "normalized_block_kinds": dict(
                    sorted(Counter(block.kind.value for block in extraction.views.blocks).items())
                ),
                "sections": len(extraction.views.sections),
                "caption_links": len(extraction.views.caption_links),
                "assets": len(extraction.assets),
                "evidence_kinds": dict(sorted(evidence_counts.items())),
                **window_diagnostics,
                "diagnostic_severities": dict(
                    sorted(
                        Counter(item.severity.value for item in extraction.diagnostics).items()
                    )
                ),
                "diagnostic_codes": dict(
                    sorted(Counter(item.code for item in extraction.diagnostics).items())
                ),
                "unsupported_nodes": extraction.coverage.unsupported_nodes,
                "silent_losses": extraction.coverage.silent_losses,
            }
        )

    provenance_error: dict[str, str] | None = None
    if not batch.failures and not index_failures:
        try:
            CorpusLookup.from_artifacts_indexes(batch.documents, indexes)
        except Exception as error:
            provenance_error = {
                "error_type": type(error).__name__,
                "message": str(error),
            }

    failure_reports = [
        {
            "filename": failure.input_path.name,
            "stage": failure.stage,
            "error_type": failure.error_type,
            "message": failure.message,
            "diagnostic_codes": [item.code for item in failure.diagnostics],
        }
        for failure in batch.failures
    ]
    report = {
        "success": not batch.failures
        and not index_failures
        and provenance_error is None
        and all(item["silent_losses"] == 0 for item in document_reports),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "document_count": len(paths),
        "documents": document_reports,
        "extraction_failures": failure_reports,
        "index_failures": index_failures,
        "provenance_error": provenance_error,
    }
    _write_report("deterministic-report.json", report)

    assert not batch.failures, failure_reports
    assert not index_failures, index_failures
    assert provenance_error is None, provenance_error
    assert all(item["silent_losses"] == 0 for item in document_reports), document_reports


@pytest.mark.skipif(
    not _RUN_OPENAI,
    reason="requires OPENAI_API_KEY and RUN_OPENAI_INTEGRATION_TESTS=1",
)
def test_openai_generates_validated_outline_from_docs_corpus() -> None:
    """Run the complete real-corpus path and retain inspectable artifacts."""
    from loguru import logger

    from presentation_pipeline import generate_outline, write_outline_json
    from presentation_pipeline.planning import PresentationRequirements
    from presentation_pipeline.providers import (
        OpenAIGenerationTelemetry,
        OpenAIStructuredGenerator,
    )

    paths = _document_paths()
    assert paths, f"no DOCX files found in {_DOCS_DIR}"

    model = os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")
    requirements = PresentationRequirements(
        goal=os.environ.get(
            "OPENAI_DOCS_GOAL",
            "Synthesize the supplied health-policy briefing documents into an actionable executive presentation.",
        ),
        audience=os.environ.get("OPENAI_DOCS_AUDIENCE", "Senior health-policy leadership"),
        target_slide_count=int(os.environ.get("OPENAI_DOCS_TARGET_SLIDES", "10")),
        tone=os.environ.get("OPENAI_DOCS_TONE", "concise, evidence-led, executive"),
        presentation_type=os.environ.get("OPENAI_DOCS_PRESENTATION_TYPE", "executive briefing"),
    )
    telemetry: list[OpenAIGenerationTelemetry] = []
    generator = OpenAIStructuredGenerator(
        model=model,
        telemetry_handler=telemetry.append,
    )
    started = perf_counter()

    try:
        logger.disable("docx_pipeline")
        try:
            outline = asyncio.run(
                generate_outline(
                    paths,
                    requirements,
                    generator,
                    extraction_workers=min(4, len(paths)),
                    llm_concurrency=int(
                        os.environ.get("OPENAI_DOCS_LLM_CONCURRENCY", "2")
                    ),
                )
            )
        finally:
            logger.enable("docx_pipeline")
    except Exception as error:
        _write_report(
            "openai-report.json",
            {
                "success": False,
                "model": model,
                "elapsed_seconds": round(perf_counter() - started, 3),
                "document_count": len(paths),
                "documents": [path.name for path in paths],
                "requirements": requirements.model_dump(mode="json"),
                "error_type": type(error).__name__,
                "message": str(error),
                "telemetry": [asdict(item) for item in telemetry],
            },
        )
        raise

    output_path = write_outline_json(
        outline,
        _output_dir() / "outline.json",
        overwrite=True,
    )
    response_models = Counter(item.response_model for item in telemetry)
    report = {
        "success": True,
        "model": model,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "document_count": len(paths),
        "documents": [path.name for path in paths],
        "requirements": requirements.model_dump(mode="json"),
        "outline_path": str(output_path),
        "slide_count": len(outline.all_slides()),
        "slide_evidence_references": sum(
            len(slide.evidence) for slide in outline.all_slides()
        ),
        "response_model_counts": dict(sorted(response_models.items())),
        "total_input_tokens": sum(item.input_tokens or 0 for item in telemetry),
        "total_output_tokens": sum(item.output_tokens or 0 for item in telemetry),
        "telemetry": [asdict(item) for item in telemetry],
    }
    _write_report("openai-report.json", report)

    assert len(outline.all_slides()) == requirements.target_slide_count
    assert response_models["ChunkDigest"] >= len(paths)
    assert response_models["EvidenceSelection"] == 1
    assert response_models["PresentationOutline"] in {1, 2}
    assert all(item.outcome == "success" for item in telemetry)
    assert type(outline).model_validate_json(output_path.read_text(encoding="utf-8")) == outline
