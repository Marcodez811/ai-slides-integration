"""LLM-assisted planning with externally enforced provenance validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from presentation_pipeline.generation import StructuredGenerator, invoke_structured
from presentation_pipeline.understanding.models import DocumentDigest

from .models import EvidenceSelection, PresentationOutline, PresentationRequirements
from .prompts import (
    EVIDENCE_SELECTION_PROMPT,
    OUTLINE_PROMPT,
    build_evidence_selection_input,
    build_outline_input,
    outline_system_prompt,
)

if TYPE_CHECKING:
    from presentation_pipeline.validation import CorpusLookup


async def select_evidence(
    digests: list[DocumentDigest],
    indexes: list[object],
    requirements: PresentationRequirements,
    generator: StructuredGenerator,
    *,
    lookup: "CorpusLookup | None" = None,
) -> EvidenceSelection:
    selection = await invoke_structured(
        generator,
        EVIDENCE_SELECTION_PROMPT,
        build_evidence_selection_input(digests, indexes, requirements),
        EvidenceSelection,
    )
    if lookup is not None:
        from presentation_pipeline.validation import validate_evidence_selection

        validate_evidence_selection(selection, lookup)
    return selection


async def generate_presentation_outline(
    digests: list[DocumentDigest],
    requirements: PresentationRequirements,
    selection: EvidenceSelection,
    indexes: list[object],
    generator: StructuredGenerator,
    *,
    lookup: "CorpusLookup | None" = None,
    retry_invalid_outline: bool = True,
) -> PresentationOutline:
    """Generate and validate once, with at most one full regeneration retry."""
    from presentation_pipeline.validation import (
        OutlineRequirementsValidationError,
        OutlineEvidenceScopeValidationError,
        ProvenanceValidationError,
        validate_outline_evidence_scope,
        validate_outline_requirements,
        validate_presentation_outline,
    )

    def validate(outline: PresentationOutline) -> None:
        if lookup is not None:
            validate_presentation_outline(outline, lookup)
        validate_outline_evidence_scope(outline, selection)
        validate_outline_requirements(outline, requirements)

    outline = await invoke_structured(
        generator,
        OUTLINE_PROMPT,
        build_outline_input(digests, requirements, selection, indexes),
        PresentationOutline,
    )
    try:
        validate(outline)
    except (
        ProvenanceValidationError,
        OutlineEvidenceScopeValidationError,
        OutlineRequirementsValidationError,
    ) as error:
        if not retry_invalid_outline:
            raise
        outline = await invoke_structured(
            generator,
            outline_system_prompt(str(error)),
            build_outline_input(digests, requirements, selection, indexes),
            PresentationOutline,
        )
        validate(outline)
    return outline
