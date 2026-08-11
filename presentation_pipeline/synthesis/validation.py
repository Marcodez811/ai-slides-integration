"""Deterministic validation for semantic slide content."""

from __future__ import annotations

from collections.abc import Sequence

from presentation_pipeline.indexing.models import EvidenceKind
from presentation_pipeline.planning.models import SlidePurpose

from .context import SlideContext
from .models import (
    BulletListContent,
    ChartContent,
    EquationContent,
    ImageContent,
    PresentationContent,
    SlideContent,
    TableContent,
    TextContent,
)


class SlideContentValidationError(ValueError):
    """Generated semantic content violates the context-bound contract."""


def _evidence_by_identity(context: SlideContext) -> dict[tuple[str, str], object]:
    resolved: dict[tuple[str, str], object] = {}
    for item in context.evidence:
        identity = (item.doc_id, item.evidence_id)
        if identity in resolved:
            raise SlideContentValidationError(f"duplicate context evidence {identity!r}")
        resolved[identity] = item
    return resolved


def _validate_refs(refs: object, allowed: dict[tuple[str, str], object]) -> None:
    for reference in refs:
        for evidence_id in reference.evidence_ids:
            if (reference.doc_id, evidence_id) not in allowed:
                raise SlideContentValidationError(
                    f"evidence {(reference.doc_id, evidence_id)!r} is not available to this slide"
                )


def _source_evidence(
    item: ChartContent | TableContent | ImageContent | EquationContent,
    allowed: dict[tuple[str, str], object],
) -> object:
    identity = (item.doc_id, item.evidence_id)
    try:
        return allowed[identity]
    except KeyError as error:
        raise SlideContentValidationError(f"source evidence {identity!r} is not available to this slide") from error


def validate_slide_content(content: SlideContent, context: SlideContext) -> None:
    """Ensure one model result is bound to exactly its deterministic context."""
    if content.slide_id != context.slide.slide_id:
        raise SlideContentValidationError(
            f"content slide ID {content.slide_id!r} does not match context {context.slide.slide_id!r}"
        )
    if context.slide.purpose in {SlidePurpose.CONTENT, SlidePurpose.SUMMARY} and not content.elements:
        raise SlideContentValidationError("content and summary slides require at least one semantic element")

    allowed = _evidence_by_identity(context)
    for element in content.elements:
        if isinstance(element, TextContent):
            _validate_refs(element.evidence, allowed)
        elif isinstance(element, BulletListContent):
            for bullet in element.items:
                _validate_refs(bullet.evidence, allowed)
        elif isinstance(element, ChartContent):
            evidence = _source_evidence(element, allowed)
            if evidence.kind not in {EvidenceKind.CHART, EvidenceKind.CHART_CANDIDATE}:
                raise SlideContentValidationError("chart content must reference chart or chart_candidate evidence")
        elif isinstance(element, TableContent):
            evidence = _source_evidence(element, allowed)
            if evidence.kind != EvidenceKind.TABLE:
                raise SlideContentValidationError("table content must reference table evidence")
        elif isinstance(element, ImageContent):
            evidence = _source_evidence(element, allowed)
            if evidence.kind != EvidenceKind.IMAGE:
                raise SlideContentValidationError("image content must reference image evidence")
            if not evidence.asset_ids:
                raise SlideContentValidationError("image content must reference evidence with an asset ID")
        elif isinstance(element, EquationContent):
            evidence = _source_evidence(element, allowed)
            if evidence.kind != EvidenceKind.EQUATION:
                raise SlideContentValidationError("equation content must reference equation evidence")


def validate_presentation_content(
    content: PresentationContent,
    contexts: Sequence[SlideContext],
) -> None:
    """Validate every content object and its exact authored presentation order."""
    expected_ids = [context.slide.slide_id for context in contexts]
    actual_ids = [slide.slide_id for slide in content.slides]
    if actual_ids != expected_ids:
        raise SlideContentValidationError(
            f"presentation slide IDs must exactly match context order: expected {expected_ids!r}, got {actual_ids!r}"
        )
    if len(actual_ids) != len(set(actual_ids)):
        raise SlideContentValidationError("presentation content slide IDs must be unique")
    for slide, context in zip(content.slides, contexts, strict=True):
        validate_slide_content(slide, context)
