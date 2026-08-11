"""Reusable in-memory results from presentation planning."""

from __future__ import annotations

from dataclasses import dataclass

from presentation_pipeline.corpus.models import DocumentArtifact
from presentation_pipeline.indexing.models import DocumentIndex
from presentation_pipeline.planning.models import (
    EvidenceSelection,
    PresentationOutline,
    PresentationRequirements,
)
from presentation_pipeline.understanding.models import DocumentDigest
from presentation_pipeline.retrieval.models import CandidateEvidenceSet


@dataclass(frozen=True, slots=True)
class PresentationPlanningResult:
    """Validated artifacts retained after a single planning pass."""

    requirements: PresentationRequirements
    artifacts: tuple[DocumentArtifact, ...]
    indexes: tuple[DocumentIndex, ...]
    digests: tuple[DocumentDigest, ...]
    selection: EvidenceSelection
    outline: PresentationOutline
    candidates: CandidateEvidenceSet | None = None

    def __post_init__(self) -> None:
        """Reject manually assembled plans whose document stages no longer align."""
        stages = (self.artifacts, self.indexes, self.digests)
        lengths = tuple(len(stage) for stage in stages)
        if not all(lengths) or len(set(lengths)) != 1:
            raise ValueError(
                "artifacts, indexes, and digests must have identical nonzero lengths"
            )
        artifact_ids = tuple(_document_id(artifact, "artifact") for artifact in self.artifacts)
        index_ids = tuple(_document_id(index, "index") for index in self.indexes)
        digest_ids = tuple(_document_id(digest, "digest") for digest in self.digests)
        if artifact_ids != index_ids or artifact_ids != digest_ids:
            raise ValueError(
                "artifacts, indexes, and digests must have matching doc_id order"
            )


def _document_id(value: object, label: str) -> str:
    doc_id = getattr(value, "doc_id", None)
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError(f"{label} must expose a non-empty doc_id")
    return doc_id
