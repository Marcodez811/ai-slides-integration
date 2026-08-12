"""Top-level extraction, understanding, selection, and outline orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from presentation_pipeline.budgeting import GenerationLimiter
from presentation_pipeline.corpus.batch import build_jobs, extract_batch
from presentation_pipeline.corpus.models import BatchExtractionResult, DocumentFailure
from presentation_pipeline.indexing.builder import build_document_index
from presentation_pipeline.observability import safe_event, stage_timer
from presentation_pipeline.generation import StructuredGenerator
from presentation_pipeline.planning.models import PresentationOutline, PresentationRequirements
from presentation_pipeline.planning.service import generate_presentation_outline, select_evidence
from presentation_pipeline.retrieval import WindowedLLMEvidenceRetriever
from presentation_pipeline.results import PresentationPlanningResult
from presentation_pipeline.scale import PlanningScaleConfig
from presentation_pipeline.understanding.service import generate_digests
from presentation_pipeline.validation import (
    CorpusLookup,
    validate_document_digests,
    validate_evidence_selection,
)


class BatchExtractionError(RuntimeError):
    """Typed failure stopping planning when any requested source failed extraction."""

    def __init__(self, failures: Sequence[DocumentFailure]):
        self.failures = tuple(failures)
        super().__init__(f"batch extraction failed for {len(self.failures)} document(s)")


async def generate_plan(
    input_paths: Sequence[str | Path],
    requirements: PresentationRequirements,
    generator: StructuredGenerator,
    *,
    extraction_workers: int = 4,
    llm_concurrency: int = 4,
    scale_config: PlanningScaleConfig | None = None,
    asset_output_dir: str | Path | None = None,
) -> PresentationPlanningResult:
    """Return all validated, reusable intermediates from one planning pass.

    Extraction and indexing are deterministic. Every provider call across
    chunk digestion, reduction, candidate retrieval, selection, and outlining
    is bounded by one shared scale configuration and generation limiter.
    """
    if not input_paths:
        raise ValueError("at least one input document is required")
    # Preserve the exact legacy call shape when no renderer needs materialized
    # media; existing injected batch adapters then remain source-compatible.
    jobs = (
        build_jobs(input_paths, asset_output_dir=asset_output_dir)
        if asset_output_dir is not None
        else build_jobs(input_paths)
    )
    with stage_timer("extraction", document_count=len(jobs)):
        batch: BatchExtractionResult = extract_batch(jobs, max_workers=extraction_workers)
    if batch.failures:
        raise BatchExtractionError(batch.failures)
    artifacts = batch.documents
    for artifact in artifacts:
        safe_event(
            "extraction_complete",
            doc_id=artifact.doc_id,
            filename=getattr(artifact, "filename", None),
            node_count=len(getattr(artifact.extraction, "nodes", ())),
            asset_count=len(getattr(artifact.extraction, "assets", ())),
            diagnostic_count=len(getattr(artifact.extraction, "diagnostics", ())),
            coverage_health=getattr(getattr(artifact, "health", None), "value", getattr(artifact, "health", None)),
        )
    with stage_timer("indexing", document_count=len(artifacts)):
        indexes = [build_document_index(artifact) for artifact in artifacts]
    for index in indexes:
        kinds: dict[str, int] = {}
        for evidence in index.evidence:
            kind = str(getattr(evidence.kind, "value", evidence.kind))
            kinds[kind] = kinds.get(kind, 0) + 1
        safe_event(
            "indexing_complete", doc_id=index.doc_id, section_count=len(index.sections),
            evidence_count=len(index.evidence), evidence_kind_counts=kinds,
        )
    lookup = CorpusLookup.from_artifacts_indexes(artifacts, indexes)
    scale = scale_config or PlanningScaleConfig()
    limiter = GenerationLimiter(
        generator,
        token_counter=scale.token_counter,
        concurrency=llm_concurrency,
    )
    with stage_timer("document_digest_generation", document_count=len(artifacts)):
        digests = await generate_digests(
            artifacts, indexes, generator, token_counter=scale.token_counter,
            budgets=scale.budgets.understanding, limiter=limiter, concurrency=llm_concurrency,
        )
    validate_document_digests(digests, lookup)
    retriever = scale.retriever or WindowedLLMEvidenceRetriever(
        generator,
        token_counter=scale.token_counter,
        budget=scale.budgets.retrieval,
        artifacts=artifacts,
        concurrency=llm_concurrency,
    )
    with stage_timer("retrieval"):
        candidates = await retriever.retrieve(indexes, digests, requirements, limiter=limiter)
    safe_event(
        "retrieval_complete",
        candidate_count_before_reduction=len(candidates.candidates),
        candidate_count_after_reduction=len(candidates.candidates),
    )
    with stage_timer("evidence_selection", candidate_count=len(candidates.candidates)):
        selection = await select_evidence(
            digests,
            candidates,
            requirements,
            generator,
            indexes=indexes,
            token_counter=scale.token_counter,
            budget=scale.budgets.retrieval.selection_budget,
            limiter=limiter,
            lookup=lookup,
        )
    validate_evidence_selection(selection, lookup)
    safe_event(
        "evidence_selection_complete",
        candidate_count_before_reduction=len(candidates.candidates),
        candidate_count_after_reduction=len(candidates.candidates),
        selected_evidence_count=len(selection.selected),
    )
    with stage_timer("outline_generation", selected_evidence_count=len(selection.selected)):
        outline = await generate_presentation_outline(
            digests,
            requirements,
            selection,
            indexes,
            generator,
            lookup=lookup,
            candidates=candidates,
            token_counter=scale.token_counter,
            budget=scale.budgets.outline_generation,
            limiter=limiter,
        )
    safe_event(
        "outline_complete", section_count=len(outline.sections),
        semantic_slide_count=len(outline.all_slides()),
    )
    return PresentationPlanningResult(
        requirements=requirements,
        artifacts=tuple(artifacts),
        indexes=tuple(indexes),
        digests=tuple(digests),
        selection=selection,
        outline=outline,
        candidates=candidates,
    )


async def generate_outline(
    input_paths: Sequence[str | Path],
    requirements: PresentationRequirements,
    generator: StructuredGenerator,
    *,
    extraction_workers: int = 4,
    llm_concurrency: int = 4,
    scale_config: PlanningScaleConfig | None = None,
    asset_output_dir: str | Path | None = None,
) -> PresentationOutline:
    """Return the outline from a single complete presentation planning pass."""
    result = await generate_plan(
        input_paths,
        requirements,
        generator,
        extraction_workers=extraction_workers,
        llm_concurrency=llm_concurrency,
        scale_config=scale_config,
        asset_output_dir=asset_output_dir,
    )
    return result.outline
