"""Persistence-boundary references for extracted corpus artifacts."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from ..common.ids import make_corpus_id
from ..common.models import PipelineModel
from .models import ArtifactHealth


class DocumentArtifactRef(PipelineModel):
    """A serializable pointer to a stored document artifact, not the artifact itself."""

    doc_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    artifact_uri: str = Field(min_length=1)
    health: ArtifactHealth | None = None

    @field_validator("doc_id", "filename", "artifact_uri")
    @classmethod
    def _text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("artifact references must not contain blank identifiers or URIs")
        return value


class CorpusManifest(PipelineModel):
    """Serializable index of persisted artifacts for one corpus."""

    schema_version: str = "1.0.0"
    corpus_id: str = Field(default_factory=make_corpus_id, min_length=1)
    documents: list[DocumentArtifactRef] = Field(default_factory=list)

    @property
    def artifacts(self) -> list[DocumentArtifactRef]:
        """Read-only compatibility alias for pre-release callers."""
        return self.documents

    @field_validator("corpus_id")
    @classmethod
    def _corpus_id_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("corpus_id must not be blank")
        return value

    @model_validator(mode="after")
    def _references_are_unique(self) -> "CorpusManifest":
        document_ids = [artifact.doc_id for artifact in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("manifest doc_id values must be unique")
        return self
