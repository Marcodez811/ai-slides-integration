"""Concurrent, failure-isolated DOCX extraction for a document corpus."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from docx_pipeline import Diagnostic, DiagnosticSeverity, ExtractionConfig, ExtractionResult
from docx_pipeline.api import extract_docx

from .models import ArtifactHealth, BatchExtractionResult, DocumentArtifact, DocumentFailure, DocumentJob


def build_jobs(
    input_paths: Iterable[str | Path],
    *,
    extraction_config: ExtractionConfig | None = None,
    asset_output_dir: str | Path | None = None,
) -> list[DocumentJob]:
    """Build independent jobs in input order without touching the filesystem.

    When an asset root is supplied, each job receives its own deterministic
    subdirectory named for its execution ID, avoiding concurrent collisions.
    """
    config = extraction_config or ExtractionConfig()
    root = Path(asset_output_dir) if asset_output_dir is not None else None
    jobs: list[DocumentJob] = []
    for input_path in input_paths:
        job = DocumentJob(input_path=Path(input_path), extraction_config=config)
        if root is not None:
            job = job.model_copy(update={"asset_output_dir": root / job.job_id})
        jobs.append(job)
    return jobs


def _extract_one(job: DocumentJob, *, allow_degraded: bool) -> DocumentArtifact | DocumentFailure:
    """Extract one job and convert unacceptable result states into failures."""
    result = extract_docx(
        job.input_path,
        asset_output_dir=job.asset_output_dir,
        config=job.extraction_config,
    )
    return _artifact_or_failure(job, result, allow_degraded=allow_degraded)


def extract_batch(
    jobs: Iterable[DocumentJob],
    *,
    max_workers: int = 4,
    allow_degraded: bool = False,
) -> BatchExtractionResult:
    """Extract jobs concurrently while retaining deterministic input-order output.

    A broken package, an extractor exception, and unacceptable output from one
    job become a :class:`DocumentFailure`; they never prevent other documents
    in the batch from completing.
    """
    _validate_max_workers(max_workers)
    ordered_jobs = list(jobs)
    _validate_unique_job_ids(ordered_jobs)
    if not ordered_jobs:
        return BatchExtractionResult()

    futures: list[tuple[DocumentJob, Future[DocumentArtifact | DocumentFailure]]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(ordered_jobs))) as executor:
        for job in ordered_jobs:
            futures.append((job, executor.submit(_extract_one, job, allow_degraded=allow_degraded)))

        documents: list[DocumentArtifact] = []
        failures: list[DocumentFailure] = []
        # Calling result in submission order makes externally visible ordering
        # deterministic even when workers complete in a different order.
        for job, future in futures:
            try:
                outcome = future.result()
            except Exception as error:
                outcome = _exception_failure(job, error)
            if isinstance(outcome, DocumentArtifact):
                documents.append(outcome)
            else:
                failures.append(outcome)
    return BatchExtractionResult(documents=documents, failures=failures)


def _artifact_or_failure(
    job: DocumentJob,
    result: ExtractionResult,
    *,
    allow_degraded: bool,
) -> DocumentArtifact | DocumentFailure:
    error_diagnostics = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.severity is DiagnosticSeverity.ERROR
    ]
    if error_diagnostics:
        return DocumentFailure(
            job_id=job.job_id,
            input_path=job.input_path,
            stage="extraction",
            error_type="ExtractionDiagnosticError",
            message=f"extraction produced {len(error_diagnostics)} error diagnostic(s)",
            diagnostics=error_diagnostics,
        )

    silent_losses = result.coverage.silent_losses
    if silent_losses and not allow_degraded:
        return DocumentFailure(
            job_id=job.job_id,
            input_path=job.input_path,
            stage="validation",
            error_type="SilentLossError",
            message=f"extraction reported {silent_losses} silent loss(es)",
            diagnostics=[
                Diagnostic(
                    code="batch.silent_loss",
                    severity=DiagnosticSeverity.ERROR,
                    message="strict batch extraction rejects artifacts with silent losses",
                    details={"silent_losses": silent_losses},
                )
            ],
        )

    return DocumentArtifact(
        job_id=job.job_id,
        doc_id=result.document.doc_id,
        filename=result.document.filename,
        extraction=result,
        health=ArtifactHealth.DEGRADED if silent_losses else ArtifactHealth.READY,
    )


def _exception_failure(job: DocumentJob, error: Exception) -> DocumentFailure:
    """Preserve an extractor exception as structured failure data."""
    error_type = type(error).__name__
    error_message = str(error) or repr(error)
    return DocumentFailure(
        job_id=job.job_id,
        input_path=job.input_path,
        stage="extraction",
        error_type=error_type,
        message=error_message,
        diagnostics=[
            Diagnostic(
                code="batch.extraction_exception",
                severity=DiagnosticSeverity.ERROR,
                message=f"{error_type}: {error_message}",
                details={"exception_type": error_type},
            )
        ],
    )


def _validate_max_workers(max_workers: int) -> None:
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be an integer greater than or equal to 1")


def _validate_unique_job_ids(jobs: list[DocumentJob]) -> None:
    job_ids = [job.job_id for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("job_id values must be unique within a batch")
