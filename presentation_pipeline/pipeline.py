"""Top-level extraction, understanding, selection, and outline orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from presentation_pipeline.corpus.batch import build_jobs, extract_batch
from presentation_pipeline.corpus.models import BatchExtractionResult, DocumentFailure
from presentation_pipeline.indexing.builder import build_document_index
from presentation_pipeline.planning.models import PresentationOutline, PresentationRequirements
from presentation_pipeline.planning.service import generate_presentation_outline, select_evidence
from presentation_pipeline.understanding.models import StructuredGenerator
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


async def generate_outline(
    input_paths: Sequence[str | Path],
    requirements: PresentationRequirements,
    generator: StructuredGenerator,
    *,
    extraction_workers: int = 4,
    llm_concurrency: int = 4,
) -> PresentationOutline:
    """Return a fully provenance-validated presentation outline.

    The extraction and indexing stages are deterministic; only the three
    structured generation stages delegate to the supplied provider adapter.
    """
    if not input_paths:
        raise ValueError("at least one input document is required")
    jobs = build_jobs(input_paths)
    batch: BatchExtractionResult = extract_batch(jobs, max_workers=extraction_workers)
    if batch.failures:
        raise BatchExtractionError(batch.failures)
    artifacts = batch.documents
    indexes = [build_document_index(artifact) for artifact in artifacts]
    lookup = CorpusLookup.from_artifacts_indexes(artifacts, indexes)
    digests = await generate_digests(
        artifacts, indexes, generator, concurrency=llm_concurrency
    )
    validate_document_digests(digests, lookup)
    selection = await select_evidence(digests, indexes, requirements, generator)
    validate_evidence_selection(selection, lookup)
    return await generate_presentation_outline(
        digests, requirements, selection, indexes, generator, lookup=lookup
    )
