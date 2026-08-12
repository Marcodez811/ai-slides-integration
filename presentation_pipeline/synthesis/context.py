"""Deterministic, provider-safe evidence resolution for slide synthesis."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from presentation_pipeline.common.models import PipelineModel
from presentation_pipeline.indexing.compact import compact_evidence_item, is_provenance_key
from presentation_pipeline.indexing.models import DocumentIndex, EvidenceItem, EvidenceKind
from presentation_pipeline.planning.models import SlideOutline, SlidePurpose
from presentation_pipeline.results import PresentationPlanningResult


class SlideContextResolutionError(ValueError):
    """Raised when a validated plan cannot be resolved into safe slide contexts."""


class _SynthesisModel(PipelineModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ResolvedEvidence(_SynthesisModel):
    """Semantic evidence available to one slide-synthesis job."""

    doc_id: str
    evidence_id: str
    kind: EvidenceKind
    text: str | None = None
    structured_data: dict[str, object] = Field(default_factory=dict)
    section_ids: list[str] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    selection_reason: str | None = None

    @field_validator("doc_id", "evidence_id")
    @classmethod
    def _identity_is_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence identity values must not be blank")
        return value

    @field_validator("text", "selection_reason")
    @classmethod
    def _optional_text_is_nonempty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional semantic text must not be blank")
        return value

    @field_validator("section_ids", "asset_ids")
    @classmethod
    def _handles_are_valid(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values) or len(values) != len(set(values)):
            raise ValueError("semantic handles must be unique non-empty strings")
        return values

    @field_validator("structured_data")
    @classmethod
    def _structured_data_is_semantic(cls, value: dict[str, object]) -> dict[str, object]:
        _validate_semantic_value(value)
        return value


class SlideContext(_SynthesisModel):
    """Complete resolved input for one independent semantic slide job."""

    presentation_title: str
    presentation_objective: str
    presentation_narrative: str
    section_id: str
    section_title: str
    section_purpose: str
    slide_index: int = Field(ge=1)
    total_slides: int = Field(ge=1)
    slide: SlideOutline
    evidence: list[ResolvedEvidence] = Field(default_factory=list)

    @field_validator(
        "presentation_title",
        "presentation_objective",
        "presentation_narrative",
        "section_id",
        "section_title",
        "section_purpose",
    )
    @classmethod
    def _context_text_is_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("slide context text must not be blank")
        return value

    @field_validator("evidence")
    @classmethod
    def _evidence_is_unique(cls, values: list[ResolvedEvidence]) -> list[ResolvedEvidence]:
        identities = [(item.doc_id, item.evidence_id) for item in values]
        if len(identities) != len(set(identities)):
            raise ValueError("slide context evidence must be unique")
        return values

    @model_validator(mode="after")
    def _slide_index_is_within_presentation(self) -> "SlideContext":
        if self.slide_index > self.total_slides:
            raise ValueError("slide_index must not exceed total_slides")
        return self


def build_slide_contexts(plan: PresentationPlanningResult) -> list[SlideContext]:
    """Resolve a validated outline's ordered references into semantic evidence.

    Context construction is intentionally deterministic. It neither broadens the
    evidence selection nor exposes extraction mechanics to the synthesis stage.
    """
    index_by_doc = _index_indexes(plan.indexes)
    selected_reasons = _index_selection(plan)
    candidate_by_identity = _index_candidates(plan)
    _validate_selected_evidence(selected_reasons, index_by_doc)
    total_slides = len(plan.outline.all_slides())
    contexts: list[SlideContext] = []
    slide_index = 0

    for section in plan.outline.sections:
        for slide in section.slides:
            slide_index += 1
            evidence = _resolve_slide_evidence(
                slide,
                index_by_doc=index_by_doc,
                selected_reasons=selected_reasons,
                candidate_by_identity=candidate_by_identity,
            )
            if not evidence and slide.purpose not in {SlidePurpose.TITLE, SlidePurpose.SECTION}:
                raise SlideContextResolutionError(
                    f"slide {slide.slide_id!r} ({slide.purpose.value}) requires evidence"
                )
            try:
                contexts.append(
                    SlideContext(
                        presentation_title=plan.outline.title,
                        presentation_objective=plan.outline.objective,
                        presentation_narrative=plan.outline.narrative,
                        section_id=section.section_id,
                        section_title=section.title,
                        section_purpose=section.purpose,
                        slide_index=slide_index,
                        total_slides=total_slides,
                        slide=slide,
                        evidence=evidence,
                    )
                )
            except ValueError as error:
                raise SlideContextResolutionError(
                    f"slide {slide.slide_id!r} has malformed semantic context: {error}"
                ) from error
    return contexts


def _index_indexes(indexes: tuple[DocumentIndex, ...]) -> dict[str, dict[str, EvidenceItem]]:
    indexed: dict[str, dict[str, EvidenceItem]] = {}
    for index in indexes:
        if not isinstance(index.doc_id, str) or not index.doc_id.strip():
            raise SlideContextResolutionError("document index has a blank document ID")
        if index.doc_id in indexed:
            raise SlideContextResolutionError(f"duplicate document index {index.doc_id!r}")
        evidence_by_id: dict[str, EvidenceItem] = {}
        for item in index.evidence:
            if item.doc_id != index.doc_id:
                raise SlideContextResolutionError(
                    f"evidence {item.evidence_id!r} belongs to {item.doc_id!r}, not index {index.doc_id!r}"
                )
            if item.evidence_id in evidence_by_id:
                raise SlideContextResolutionError(
                    f"duplicate evidence {item.evidence_id!r} in document {index.doc_id!r}"
                )
            evidence_by_id[item.evidence_id] = item
        indexed[index.doc_id] = evidence_by_id
    return indexed


def _index_selection(plan: PresentationPlanningResult) -> dict[tuple[str, str], str]:
    selected: dict[tuple[str, str], str] = {}
    for item in plan.selection.selected:
        identity = (item.doc_id, item.evidence_id)
        if identity in selected:
            raise SlideContextResolutionError(f"duplicate selected evidence {identity!r}")
        selected[identity] = item.reason
    return selected


def _index_candidates(plan: PresentationPlanningResult) -> dict[tuple[str, str], object] | None:
    """Retain bounded retrieval transport slices without changing public models."""
    if plan.candidates is None:
        return None
    result: dict[tuple[str, str], object] = {}
    for candidate in plan.candidates.candidates:
        identity = (candidate.doc_id, candidate.evidence_id)
        if identity in result:
            raise SlideContextResolutionError(f"duplicate candidate evidence {identity!r}")
        result[identity] = candidate
    return result


def _validate_selected_evidence(
    selected_reasons: dict[tuple[str, str], str],
    index_by_doc: dict[str, dict[str, EvidenceItem]],
) -> None:
    for doc_id, evidence_id in selected_reasons:
        if doc_id not in index_by_doc or evidence_id not in index_by_doc[doc_id]:
            raise SlideContextResolutionError(
                f"selected evidence {(doc_id, evidence_id)!r} is not present in an index"
            )


def _resolve_slide_evidence(
    slide: SlideOutline,
    *,
    index_by_doc: dict[str, dict[str, EvidenceItem]],
    selected_reasons: dict[tuple[str, str], str],
    candidate_by_identity: dict[tuple[str, str], object] | None,
) -> list[ResolvedEvidence]:
    resolved: list[ResolvedEvidence] = []
    seen: set[tuple[str, str]] = set()
    for reference in slide.evidence:
        evidence_by_id = index_by_doc.get(reference.doc_id)
        if evidence_by_id is None:
            raise SlideContextResolutionError(
                f"slide {slide.slide_id!r} references unknown document {reference.doc_id!r}"
            )
        for evidence_id in reference.evidence_ids:
            identity = (reference.doc_id, evidence_id)
            if identity in seen:
                raise SlideContextResolutionError(
                    f"slide {slide.slide_id!r} repeats evidence {identity!r}"
                )
            seen.add(identity)
            item = evidence_by_id.get(evidence_id)
            if item is None:
                raise SlideContextResolutionError(
                    f"slide {slide.slide_id!r} references unknown evidence {identity!r}"
                )
            reason = selected_reasons.get(identity)
            if reason is None:
                raise SlideContextResolutionError(
                    f"slide {slide.slide_id!r} references unselected evidence {identity!r}"
                )
            candidate = candidate_by_identity.get(identity) if candidate_by_identity is not None else None
            transport = candidate.transport_content() if candidate is not None else None
            resolved.append(_resolved_evidence(item, reason, transport))
    return resolved


def _resolved_evidence(
    item: EvidenceItem, selection_reason: str, transport_content: dict[str, object] | None = None
) -> ResolvedEvidence:
    try:
        compact = transport_content if transport_content is not None else compact_evidence_item(item)
        if not isinstance(compact, dict):
            raise ValueError("candidate transport content must be a dictionary")
        return ResolvedEvidence(
            doc_id=item.doc_id,
            evidence_id=item.evidence_id,
            # Canonical identity and kind remain authoritative. Only semantic
            # content can come from a bounded retrieval transport slice.
            kind=item.kind,
            text=compact.get("text"),
            structured_data=compact.get("content", {}),
            section_ids=compact.get("section_ids", []),
            asset_ids=list(item.asset_ids),
            selection_reason=selection_reason,
        )
    except ValueError as error:
        raise SlideContextResolutionError(
            f"evidence {(item.doc_id, item.evidence_id)!r} has malformed semantic data: {error}"
        ) from error


def _validate_semantic_value(value: object) -> None:
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_semantic_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("structured semantic data keys must be strings")
            if is_provenance_key(key):
                raise ValueError(f"structured semantic data contains mechanical key {key!r}")
            _validate_semantic_value(item)
        return
    raise ValueError(f"structured semantic data has unsupported value {type(value).__name__}")
