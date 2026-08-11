"""Strict intermediate contracts for bounded evidence retrieval."""

from __future__ import annotations

from copy import deepcopy

from pydantic import Field, PrivateAttr, field_validator, model_validator

from presentation_pipeline.common.models import PipelineModel


def _nonempty(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


class CandidateEvidence(PipelineModel):
    """One source-backed item shortlisted before global selection."""

    doc_id: str
    evidence_id: str
    reason: str
    score: float | None = None
    _transport_content: dict[str, object] | None = PrivateAttr(default=None)

    _doc_id_is_nonempty = field_validator("doc_id")(_nonempty)
    _evidence_id_is_nonempty = field_validator("evidence_id")(_nonempty)
    _reason_is_nonempty = field_validator("reason")(_nonempty)

    def with_transport_content(self, content: dict[str, object]) -> "CandidateEvidence":
        """Attach deterministic source content without changing public identity/schema."""
        clone = self.model_copy(deep=True)
        clone._transport_content = deepcopy(content)
        return clone

    def transport_content(self) -> dict[str, object] | None:
        """Return a defensive copy of an ephemeral source slice, when present."""
        return deepcopy(self._transport_content)


class CandidateEvidenceSet(PipelineModel):
    """Deterministically ordered, globally bounded retrieval shortlist."""

    candidates: list[CandidateEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def _identities_are_unique(self) -> "CandidateEvidenceSet":
        identities = [(item.doc_id, item.evidence_id) for item in self.candidates]
        if len(identities) != len(set(identities)):
            raise ValueError("candidate evidence references must be unique")
        return self


class LocalCandidateSelection(PipelineModel):
    """A bounded provider response for exactly one evidence window."""

    candidates: list[CandidateEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _identities_are_unique(self) -> "LocalCandidateSelection":
        identities = [(item.doc_id, item.evidence_id) for item in self.candidates]
        if len(identities) != len(set(identities)):
            raise ValueError("local candidate evidence references must be unique")
        return self
