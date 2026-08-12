"""Strict intermediate contracts for bounded evidence retrieval."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence
import json

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
    _transport_contents: tuple[dict[str, object], ...] = PrivateAttr(default=())

    _doc_id_is_nonempty = field_validator("doc_id")(_nonempty)
    _evidence_id_is_nonempty = field_validator("evidence_id")(_nonempty)
    _reason_is_nonempty = field_validator("reason")(_nonempty)

    def with_transport_content(self, content: dict[str, object]) -> "CandidateEvidence":
        """Attach deterministic source content without changing public identity/schema."""
        clone = self.model_copy(deep=True)
        clone._transport_contents = (deepcopy(content),)
        return clone

    def with_transport_contents(self, contents: Sequence[dict[str, object]]) -> "CandidateEvidence":
        if any(not isinstance(content, dict) for content in contents):
            raise TypeError("transport contents must be dictionaries")
        clone = self.model_copy(deep=True)
        clone._transport_contents = tuple(
            deepcopy(content) for content in _dedupe_transport_contents(contents)
        )
        return clone

    def transport_contents(self) -> tuple[dict[str, object], ...]:
        return tuple(deepcopy(content) for content in self._transport_contents)

    def merged_transport_content(self) -> dict[str, object] | None:
        """Merge same-identity bounded slices in source order for downstream use."""
        if not self._transport_contents:
            return None
        if len(self._transport_contents) == 1:
            return deepcopy(self._transport_contents[0])
        contents = _dedupe_transport_contents(self._transport_contents)
        merged = deepcopy(contents[0])
        texts = [content.get("text") for content in contents]
        if all(isinstance(text, str) for text in texts):
            merged["text"] = "\n".join(text for text in texts if isinstance(text, str))
        content = merged.get("content")
        if isinstance(content, dict):
            all_items: list[object] = []
            for source in contents:
                source_content = source.get("content")
                if isinstance(source_content, dict) and isinstance(source_content.get("items"), list):
                    # Distinct slices may legitimately contain identical list
                    # entries. Slice-level deduplication already removed
                    # repeated transport fragments, so preserve item order and
                    # multiplicity here rather than losing source evidence.
                    all_items.extend(deepcopy(source_content["items"]))
            if all_items:
                content["items"] = all_items
        merged.pop("slice", None)
        return merged

    def transport_payload(self) -> dict[str, object] | None:
        """Return bounded provider transport without flattening several slices.

        Provider prompts retain each independently-windowed fragment rather
        than joining source text back into a new, unbounded evidence object.
        ``merged_transport_content`` is reserved for the later selected-slide
        context, where one canonical evidence identity is required.
        """
        contents = _dedupe_transport_contents(self._transport_contents)
        if not contents:
            return None
        if len(contents) == 1:
            return deepcopy(contents[0])
        first = deepcopy(contents[0])
        return {
            key: value
            for key, value in first.items()
            if key not in {"text", "content", "slice"}
        } | {"transport_slices": [deepcopy(content) for content in contents]}

    def transport_content(self) -> dict[str, object] | None:
        """Return a defensive copy of an ephemeral source slice, when present."""
        return deepcopy(self._transport_contents[0]) if self._transport_contents else None


def _canonical_value(value: object) -> str:
    """Stable identity for private transport deduplication only."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _dedupe_transport_contents(
    contents: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Retain distinct bounded slices once, ordered by their source index."""
    unique: list[tuple[int, int, dict[str, object]]] = []
    seen: set[tuple[str, object]] = set()
    for arrival, content in enumerate(contents):
        slice_data = content.get("slice")
        slice_id = slice_data.get("slice_id") if isinstance(slice_data, dict) else None
        marker: tuple[str, object] = (
            "slice" if isinstance(slice_id, str) and slice_id else "content",
            slice_id if isinstance(slice_id, str) and slice_id else _canonical_value(content),
        )
        if marker in seen:
            continue
        seen.add(marker)
        index = slice_data.get("index") if isinstance(slice_data, dict) else None
        unique.append((index if isinstance(index, int) and index >= 0 else arrival, arrival, content))
    return [content for _, _, content in sorted(unique, key=lambda item: (item[0], item[1]))]


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
