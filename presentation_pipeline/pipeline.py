"""Top-level extraction, understanding, selection, and outline orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from presentation_pipeline.budgeting import GenerationLimiter
from presentation_pipeline.corpus.batch import build_jobs, extract_batch
from presentation_pipeline.corpus.models import BatchExtractionResult, DocumentFailure
from presentation_pipeline.indexing.builder import build_document_index
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
    batch: BatchExtractionResult = extract_batch(jobs, max_workers=extraction_workers)
    if batch.failures:
        raise BatchExtractionError(batch.failures)
    artifacts = batch.documents
    indexes = [build_document_index(artifact) for artifact in artifacts]
    lookup = CorpusLookup.from_artifacts_indexes(artifacts, indexes)
    scale = scale_config or PlanningScaleConfig()
    limiter = GenerationLimiter(
        generator,
        token_counter=scale.token_counter,
        concurrency=llm_concurrency,
    )
    digests = await generate_digests(
        artifacts,
        indexes,
        generator,
        token_counter=scale.token_counter,
        budgets=scale.budgets.understanding,
        limiter=limiter,
        concurrency=llm_concurrency,
    )
    validate_document_digests(digests, lookup)
    retriever = scale.retriever or WindowedLLMEvidenceRetriever(
        generator,
        token_counter=scale.token_counter,
        budget=scale.budgets.retrieval,
        artifacts=artifacts,
        concurrency=llm_concurrency,
    )
    candidates = await retriever.retrieve(indexes, digests, requirements, limiter=limiter)
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
