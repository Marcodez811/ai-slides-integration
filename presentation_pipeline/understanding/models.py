"""Strict, provider-independent contracts for document understanding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from presentation_pipeline.common.models import PipelineModel


TStructured = TypeVar("TStructured", bound=BaseModel)


@runtime_checkable
class StructuredGenerator(Protocol):
    """Minimal adapter boundary for a structured-output LLM provider.

    Implementations receive immutable instructions plus compact provider-safe
    data and return data accepted by ``response_model``.
    """

    async def generate(
        self,
        *,
        system_prompt: str,
        input_data: dict[str, object],
        response_model: type[TStructured],
    ) -> TStructured | dict[str, object]: ...


class UnderstandingModel(PipelineModel):
    """Shared strict configuration for LLM-facing models."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _nonempty(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


class EvidenceRef(UnderstandingModel):
    """A claim's immutable reference to indexed evidence."""

    doc_id: str
    evidence_ids: list[str] = Field(min_length=1)

    _document_id_is_nonempty = field_validator("doc_id")(_nonempty)

    @field_validator("evidence_ids")
    @classmethod
    def _evidence_ids_are_unique_and_nonempty(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("evidence IDs must not be blank")
        if len(set(values)) != len(values):
            raise ValueError("evidence IDs must be unique within a reference")
        return values


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
