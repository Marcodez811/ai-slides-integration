"""LLM-assisted planning with externally enforced provenance validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from presentation_pipeline.budgeting import (
    GenerationLimiter,
    InputBudget,
    PlanningBudgets,
    RetrievalBudget,
    TokenCounter,
    Utf8ByteTokenEstimator,
)
from presentation_pipeline.generation import StructuredGenerator
from presentation_pipeline.understanding.models import DocumentDigest
from presentation_pipeline.retrieval.models import CandidateEvidenceSet

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


def _bounded_context(
    generator: StructuredGenerator,
    *,
    token_counter: TokenCounter | None,
    budget: InputBudget | None,
    limiter: GenerationLimiter | None,
    default_budget: InputBudget,
) -> tuple[GenerationLimiter, InputBudget]:
    """Return a finite guard even for direct service callers."""
    effective_budget = budget if budget is not None else default_budget
    if not isinstance(effective_budget, InputBudget):
        raise TypeError("budget must be an InputBudget")
    if limiter is None:
        counter = token_counter if token_counter is not None else Utf8ByteTokenEstimator()
        return GenerationLimiter(generator, token_counter=counter, concurrency=1), effective_budget
    if not isinstance(limiter, GenerationLimiter):
        raise TypeError("limiter must be a GenerationLimiter")
    if token_counter is not None and token_counter is not limiter.token_counter:
        raise ValueError("token_counter must match the shared generation limiter")
    return limiter, effective_budget


async def select_evidence(
    digests: list[DocumentDigest],
    candidates: CandidateEvidenceSet,
    requirements: PresentationRequirements,
    generator: StructuredGenerator,
    *,
    indexes: list[object],
    token_counter: TokenCounter | None = None,
    budget: InputBudget | None = None,
    limiter: GenerationLimiter | None = None,
    lookup: "CorpusLookup | None" = None,
) -> EvidenceSelection:
    input_data = build_evidence_selection_input(digests, candidates, requirements, indexes)
    limiter, budget = _bounded_context(
        generator,
        token_counter=token_counter,
        budget=budget,
        limiter=limiter,
        default_budget=RetrievalBudget().selection_budget,
    )
    selection = await limiter.invoke(
        system_prompt=EVIDENCE_SELECTION_PROMPT,
        input_data=input_data,
        response_model=EvidenceSelection,
        budget=budget,
        stage="evidence_selection",
    )
    candidate_identities = {(item.doc_id, item.evidence_id) for item in candidates.candidates}
    selected_identities = {(item.doc_id, item.evidence_id) for item in selection.selected}
    unknown = sorted(selected_identities - candidate_identities)
    if unknown:
        raise ValueError(f"evidence selection contains non-candidate evidence: {unknown!r}")
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
    candidates: CandidateEvidenceSet | None = None,
    lookup: "CorpusLookup | None" = None,
    retry_invalid_outline: bool = True,
    token_counter: TokenCounter | None = None,
    budget: InputBudget | None = None,
    limiter: GenerationLimiter | None = None,
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

    input_data = build_outline_input(
        digests, requirements, selection, indexes, candidates=candidates
    )
    limiter, budget = _bounded_context(
        generator,
        token_counter=token_counter,
        budget=budget,
        limiter=limiter,
        default_budget=PlanningBudgets().outline_generation,
    )

    async def generate(system_prompt: str) -> PresentationOutline:
        return await limiter.invoke(
            system_prompt=system_prompt,
            input_data=input_data,
            response_model=PresentationOutline,
            budget=budget,
            stage="outline_generation",
        )

    outline = await generate(OUTLINE_PROMPT)
    try:
        validate(outline)
    except (
        ProvenanceValidationError,
        OutlineEvidenceScopeValidationError,
        OutlineRequirementsValidationError,
    ) as error:
        if not retry_invalid_outline:
            raise
        outline = await generate(outline_system_prompt(str(error)))
        validate(outline)
    return outline
