"""Shared, provenance-stable evidence reference contracts."""

from __future__ import annotations

from pydantic import ConfigDict, Field, field_validator

from .models import PipelineModel


def _nonempty(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


class EvidenceRef(PipelineModel):
    """An immutable reference to one or more items in a document index."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

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
