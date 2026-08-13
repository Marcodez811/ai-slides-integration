"""Strict, provider-independent contracts for document understanding."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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

    def scoped_to(self, evidence_ids: Iterable[str]) -> DocumentDigest:
        """Return a copy containing references limited to ``evidence_ids``.

        The document-level prose remains useful planning context, but references
        outside the caller's current evidence scope must not be exposed to a
        downstream model.  Rebuilding each nested value keeps this operation
        non-mutating even though their list fields are mutable containers.
        """
        if isinstance(evidence_ids, str):
            raise TypeError("evidence_ids must be an iterable of strings, not a string")

        allowed = set(evidence_ids)
        if any(not isinstance(evidence_id, str) for evidence_id in allowed):
            raise TypeError("evidence_ids must contain only strings")

        def scope_references(references: list[EvidenceRef]) -> list[EvidenceRef]:
            return [
                EvidenceRef(
                    doc_id=reference.doc_id,
                    evidence_ids=[
                        evidence_id
                        for evidence_id in reference.evidence_ids
                        if evidence_id in allowed
                    ],
                )
                for reference in references
                if any(evidence_id in allowed for evidence_id in reference.evidence_ids)
            ]

        topics = [
            TopicDigest(
                topic=topic.topic,
                summary=topic.summary,
                evidence=references,
            )
            for topic in self.topics
            if (references := scope_references(topic.evidence))
        ]
        key_facts = [
            KeyFact(
                claim=fact.claim,
                evidence=references,
            )
            for fact in self.key_facts
            if (references := scope_references(fact.evidence))
        ]
        return DocumentDigest(
            doc_id=self.doc_id,
            summary=self.summary,
            topics=topics,
            key_facts=key_facts,
        )


class DigestFragment(UnderstandingModel):
    """A bounded, internal semantic digest used during hierarchical reduction.

    ``fragment_id`` is a transport identifier only.  Evidence references always
    remain references to the original document index.
    """

    doc_id: str
    fragment_id: str
    summary: str
    topics: list[TopicDigest] = Field(default_factory=list)
    key_facts: list[KeyFact] = Field(default_factory=list)

    _document_id_is_nonempty = field_validator("doc_id")(_nonempty)
    _fragment_id_is_nonempty = field_validator("fragment_id")(_nonempty)
    _summary_is_nonempty = field_validator("summary")(_nonempty)

    def evidence_refs(self) -> Sequence[EvidenceRef]:
        return [
            reference
            for topic in self.topics
            for reference in topic.evidence
        ] + [
            reference
            for fact in self.key_facts
            for reference in fact.evidence
        ]


class ChunkDigest(UnderstandingModel):
    """The auditable digest of one bounded evidence window."""

    doc_id: str
    window_id: str
    summary: str
    topics: list[TopicDigest] = Field(default_factory=list)
    key_facts: list[KeyFact] = Field(default_factory=list)

    _document_id_is_nonempty = field_validator("doc_id")(_nonempty)
    _window_id_is_nonempty = field_validator("window_id")(_nonempty)
    _summary_is_nonempty = field_validator("summary")(_nonempty)

    def evidence_refs(self) -> Sequence[EvidenceRef]:
        return [
            reference
            for topic in self.topics
            for reference in topic.evidence
        ] + [
            reference
            for fact in self.key_facts
            for reference in fact.evidence
        ]
