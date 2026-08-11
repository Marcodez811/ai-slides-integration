from __future__ import annotations

import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from docx_pipeline import (
    CoverageReport,
    Diagnostic,
    DiagnosticSeverity,
    DocumentMetadata,
    ExtractionResult,
)
from presentation_pipeline.common.ids import make_pipeline_id
from presentation_pipeline.corpus import batch
from presentation_pipeline.corpus.manifest import CorpusManifest, DocumentArtifactRef
from presentation_pipeline.corpus.models import (
    ArtifactHealth,
    BatchExtractionResult,
    DocumentFailure,
    DocumentJob,
)


def _result(name: str, *, silent_losses: int = 0, has_error: bool = False) -> ExtractionResult:
    diagnostics = []
    if has_error:
        diagnostics.append(
            Diagnostic(
                code="extractor.problem",
                severity=DiagnosticSeverity.ERROR,
                message="the source package contains an extraction error",
            )
        )
    return ExtractionResult(
        document=DocumentMetadata(
            doc_id=f"doc-{name}",
            filename=f"{name}.docx",
            sha256="a" * 64,
        ),
        diagnostics=diagnostics,
        coverage=CoverageReport(silent_losses=silent_losses),
    )


def test_extract_batch_keeps_successes_in_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = [DocumentJob(job_id=f"job-{name}", input_path=f"{name}.docx") for name in ("one", "two", "three")]

    def fake_extract(source: Path, **_: object) -> ExtractionResult:
        time.sleep({"one": 0.03, "two": 0.01, "three": 0.0}[source.stem])
        return _result(source.stem)

    monkeypatch.setattr(batch, "extract_docx", fake_extract)

    outcome = batch.extract_batch(jobs, max_workers=3)

    assert [document.job_id for document in outcome.documents] == [job.job_id for job in jobs]
    assert [document.doc_id for document in outcome.documents] == ["doc-one", "doc-two", "doc-three"]
    assert outcome.failures == []


def test_extract_batch_isolates_exceptions_and_preserves_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = [DocumentJob(job_id=f"job-{name}", input_path=f"{name}.docx") for name in ("good", "bad", "later")]

    def fake_extract(source: Path, **_: object) -> ExtractionResult:
        if source.stem == "bad":
            raise RuntimeError("broken document")
        return _result(source.stem)

    monkeypatch.setattr(batch, "extract_docx", fake_extract)

    outcome = batch.extract_batch(jobs, max_workers=2)

    assert [document.job_id for document in outcome.documents] == ["job-good", "job-later"]
    assert len(outcome.failures) == 1
    failure = outcome.failures[0]
    assert (failure.job_id, failure.stage, failure.error_type, failure.message) == (
        "job-bad",
        "extraction",
        "RuntimeError",
        "broken document",
    )
    assert failure.diagnostics[0].severity is DiagnosticSeverity.ERROR


def test_silent_loss_is_strict_by_default_and_can_be_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch, "extract_docx", lambda *_args, **_kwargs: _result("lossy", silent_losses=2))
    job = DocumentJob(job_id="job-lossy", input_path="lossy.docx")

    strict = batch.extract_batch([job])
    degraded = batch.extract_batch([job], allow_degraded=True)

    assert strict.documents == []
    assert strict.failures[0].error_type == "SilentLossError"
    assert strict.failures[0].diagnostics[0].code == "batch.silent_loss"
    assert degraded.failures == []
    assert degraded.documents[0].health is ArtifactHealth.DEGRADED


def test_error_diagnostics_are_failures_even_when_degraded_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch, "extract_docx", lambda *_args, **_kwargs: _result("error", has_error=True))

    outcome = batch.extract_batch([DocumentJob(job_id="job-error", input_path="error.docx")], allow_degraded=True)

    assert outcome.documents == []
    assert outcome.failures[0].error_type == "ExtractionDiagnosticError"
    assert outcome.failures[0].diagnostics[0].code == "extractor.problem"


def test_empty_batch_needs_no_workers() -> None:
    outcome = batch.extract_batch([], max_workers=1)

    assert outcome.documents == []
    assert outcome.failures == []
    assert outcome.batch_id.startswith("batch-")


@pytest.mark.parametrize("max_workers", [0, -1, True, 1.5, "2"])
def test_extract_batch_validates_worker_count(max_workers: object) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        batch.extract_batch([], max_workers=max_workers)  # type: ignore[arg-type]


def test_contracts_forbid_extra_fields_and_use_documents_as_canonical_result_field() -> None:
    with pytest.raises(ValidationError, match="unexpected"):
        DocumentJob(input_path="input.docx", unexpected=True)

    failure = DocumentFailure(
        job_id="job-failure",
        input_path="input.docx",
        stage="extraction",
        error_type="ValueError",
        message="invalid source",
    )
    result = BatchExtractionResult(failures=[failure])
    assert result.artifacts == []
    assert "documents" in result.model_dump()
    assert "artifacts" not in result.model_dump()


def test_manifest_is_a_strict_persistence_reference_index() -> None:
    reference = DocumentArtifactRef(
        doc_id="doc-one",
        filename="one.docx",
        artifact_uri="s3://corpus/doc-one.json",
        health=ArtifactHealth.READY,
    )
    manifest = CorpusManifest(documents=[reference])

    assert manifest.artifacts == [reference]
    assert manifest.model_dump()["documents"][0]["artifact_uri"] == "s3://corpus/doc-one.json"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DocumentArtifactRef.model_validate({**reference.model_dump(), "contents": {}})
    with pytest.raises(ValidationError, match="unique"):
        CorpusManifest(documents=[reference, reference])


def test_pipeline_ids_are_deterministic_and_length_safe() -> None:
    left = make_pipeline_id("evidence", "ab", "c")
    right = make_pipeline_id("evidence", "a", "bc")

    assert left == make_pipeline_id("evidence", "ab", "c")
    assert left != right
    assert left.startswith("evidence-")
    assert len(left.rsplit("-", 1)[1]) == 24
