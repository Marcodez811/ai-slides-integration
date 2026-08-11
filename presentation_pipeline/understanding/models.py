"""Strict, provider-independent contracts for document understanding."""

from __future__ import annotations

from collections.abc import Sequence
from pydantic import ConfigDict, Field, field_validator

from presentation_pipeline.common.models import PipelineModel
from presentation_pipeline.common.references import EvidenceRef
from presentation_pipeline.generation import StructuredGenerator


class UnderstandingModel(PipelineModel):
    """Shared strict configuration for LLM-facing models."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _nonempty(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


class TopicDigest(UnderstandingModel):
    topic: str
    summary: str
    evidence: list[EvidenceRef] = Field(min_length=1)

    _topic_is_nonempty = field_validator("topic")(_nonempty)
    _summary_is_nonempty = field_validator("summary")(_nonempty)


class KeyFact(UnderstandingModel):
    claim: str
    evidence: list[EvidenceRef] = Field(min_length=1)

    _claim_is_nonempty = field_validator("claim")(_nonempty)


class DocumentDigest(UnderstandingModel):
    doc_id: str
    summary: str
    topics: list[TopicDigest] = Field(default_factory=list)
    key_facts: list[KeyFact] = Field(default_factory=list)

    _document_id_is_nonempty = field_validator("doc_id")(_nonempty)
    _summary_is_nonempty = field_validator("summary")(_nonempty)

    def evidence_refs(self) -> Sequence[EvidenceRef]:
        """Return references in source/model order without deduplication."""
        return [
            reference
            for topic in self.topics
            for reference in topic.evidence
        ] + [
            reference
            for fact in self.key_facts
            for reference in fact.evidence
        ]
