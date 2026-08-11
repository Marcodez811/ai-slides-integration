"""In-memory contracts for extracting a document corpus."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from docx_pipeline import Diagnostic, ExtractionConfig, ExtractionResult

from ..common.ids import make_batch_id, make_job_id
from ..common.models import PipelineModel


class ArtifactHealth(str, Enum):
    """Whether a completed extraction is suitable for normal downstream use."""

    READY = "ready"
    DEGRADED = "degraded"


class DocumentJob(PipelineModel):
    """One isolated request to extract a DOCX source."""

    job_id: str = Field(default_factory=make_job_id, min_length=1)
    input_path: Path
    extraction_config: ExtractionConfig = Field(default_factory=ExtractionConfig)
    asset_output_dir: Path | None = None

    @field_validator("job_id")
    @classmethod
    def _job_id_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("job_id must not be blank")
        return value


class DocumentArtifact(PipelineModel):
    """A successful extraction retained only in the in-memory domain layer."""

    job_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    extraction: ExtractionResult
    health: ArtifactHealth = ArtifactHealth.READY

    @field_validator("job_id", "doc_id", "filename")
    @classmethod
    def _identifier_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifiers must not be blank")
        return value

    @model_validator(mode="after")
    def _matches_extraction_and_health(self) -> "DocumentArtifact":
        if self.doc_id != self.extraction.document.doc_id:
            raise ValueError("doc_id must match extraction.document.doc_id")
        if self.filename != self.extraction.document.filename:
            raise ValueError("filename must match extraction.document.filename")
        silent_losses = self.extraction.coverage.silent_losses
        if self.health is ArtifactHealth.READY and silent_losses:
            raise ValueError("a ready artifact cannot contain silent losses")
        if self.health is ArtifactHealth.DEGRADED and not silent_losses:
            raise ValueError("a degraded artifact requires at least one silent loss")
        return self


class DocumentFailure(PipelineModel):
    """An isolated job failure represented without re-raising batch exceptions."""

    job_id: str = Field(min_length=1)
    input_path: Path
    stage: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    diagnostics: list[Diagnostic] = Field(default_factory=list)

    @field_validator("job_id", "stage", "error_type", "message")
    @classmethod
    def _required_text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("failure text fields must not be blank")
        return value

class BatchExtractionResult(PipelineModel):
    """Ordered successes and isolated failures from one batch request."""

    batch_id: str = Field(default_factory=make_batch_id, min_length=1)
    documents: list[DocumentArtifact] = Field(default_factory=list)
    failures: list[DocumentFailure] = Field(default_factory=list)

    @property
    def artifacts(self) -> list[DocumentArtifact]:
        """Read-only compatibility alias for pre-release callers."""
        return self.documents

    @field_validator("batch_id")
    @classmethod
    def _batch_id_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("batch_id must not be blank")
        return value

    @model_validator(mode="after")
    def _jobs_are_reported_once(self) -> "BatchExtractionResult":
        artifact_job_ids = [artifact.job_id for artifact in self.documents]
        failure_job_ids = [failure.job_id for failure in self.failures]
        all_job_ids = artifact_job_ids + failure_job_ids
        if len(all_job_ids) != len(set(all_job_ids)):
            raise ValueError("each job may produce exactly one artifact or failure")
        return self
