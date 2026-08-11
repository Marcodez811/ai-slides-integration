"""Validation contracts that bind generated outlines to planning decisions."""

from __future__ import annotations

from presentation_pipeline.planning.models import EvidenceSelection, PresentationOutline


class OutlineEvidenceScopeValidationError(ValueError):
    """An outline cites corpus evidence omitted from the selected evidence set."""


def validate_outline_evidence_scope(
    outline: PresentationOutline, selection: EvidenceSelection
) -> None:
    """Require every outline reference to be included in the evidence selection."""
    allowed = {(item.doc_id, item.evidence_id) for item in selection.selected}
    for slide in outline.all_slides():
        for reference in slide.evidence:
            for evidence_id in reference.evidence_ids:
                identity = (reference.doc_id, evidence_id)
                if identity not in allowed:
                    raise OutlineEvidenceScopeValidationError(
                        f"slide {slide.slide_id!r} references unselected evidence {identity!r}"
                    )
