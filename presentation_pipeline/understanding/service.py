"""Bounded, order-preserving hierarchical document digest generation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Sequence

from presentation_pipeline.budgeting import (
    GenerationLimiter,
    TokenCounter,
    UnderstandingBudgets,
    Utf8ByteTokenEstimator,
)
from presentation_pipeline.generation import StructuredGenerator
from presentation_pipeline.observability import safe_debug, safe_event

from .models import ChunkDigest, DigestFragment, DocumentDigest
from .prompts import (
    CHUNK_DIGEST_PROMPT,
    DIGEST_COMPACTION_PROMPT,
    DIGEST_REDUCTION_PROMPT,
    document_id,
)
from .reduction import reduce_chunk_digests, validate_digest_scope
from .windows import EvidenceWindow, build_evidence_windows


class ChunkDigestValidationError(ValueError):
    """A chunk response does not stay within its requested window scope."""


async def _generate_chunk_digest(
    window: EvidenceWindow,
    *,
    limiter: GenerationLimiter,
    budget: object,
) -> ChunkDigest:
    digest = await limiter.invoke(
        system_prompt=CHUNK_DIGEST_PROMPT,
        input_data=window.payload,
        response_model=ChunkDigest,
        budget=budget,
        stage="document_chunk_digest",
    )
    if digest.doc_id != window.doc_id:
        raise ChunkDigestValidationError("chunk digest document ID does not match its evidence window")
    if digest.window_id != window.window_id:
        raise ChunkDigestValidationError("chunk digest window ID does not match its evidence window")
    try:
        validate_digest_scope(digest, doc_id=window.doc_id, allowed_evidence_ids=set(window.evidence_ids))
    except ValueError as error:
        raise ChunkDigestValidationError("chunk digest cites evidence outside its window") from error
    digest_tokens = limiter.token_counter.count_payload(digest.model_dump(mode="json"))
    safe_debug(
        "chunk_digest_complete",
        doc_id=window.doc_id,
        window_id=window.window_id,
        topic_count=len(digest.topics),
        key_fact_count=len(digest.key_facts),
        window_input_tokens=window.estimated_tokens,
        digest_output_estimated_tokens=digest_tokens,
        compression_ratio=(digest_tokens / window.estimated_tokens) if window.estimated_tokens else 1.0,
    )
    return digest


async def generate_document_digest(
    artifact: object,
    index: object,
    generator: StructuredGenerator,
    *,
    token_counter: TokenCounter | None = None,
    budgets: UnderstandingBudgets | None = None,
    limiter: GenerationLimiter | None = None,
    concurrency: int = 4,
) -> DocumentDigest:
    """Generate a document digest through bounded windows and reduction.

    Direct callers receive finite conservative defaults; pipeline callers should
    pass their shared limiter to bound all generation stages together.
    """
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be an integer of at least 1")
    budgets = budgets if budgets is not None else UnderstandingBudgets()
    if not isinstance(budgets, UnderstandingBudgets):
        raise TypeError("budgets must be an UnderstandingBudgets")
    if limiter is None:
        token_counter = token_counter if token_counter is not None else Utf8ByteTokenEstimator()
        limiter = GenerationLimiter(generator, token_counter=token_counter, concurrency=concurrency)
    elif token_counter is not None and token_counter is not limiter.token_counter:
        raise ValueError("token_counter must match the shared generation limiter")
    counter = limiter.token_counter
    started = time.perf_counter()
    expected_doc_id = document_id(artifact)
    if getattr(index, "doc_id", None) != expected_doc_id:
        raise ValueError("artifact and index document IDs do not match")
    windows = build_evidence_windows(
        artifact, index, token_counter=counter, budget=budgets.window
    )
    safe_event(
        "document_digest_start",
        doc_id=expected_doc_id,
        filename=getattr(artifact, "filename", None),
        window_count=len(windows),
    )
    estimates = [window.estimated_tokens for window in windows]
    safe_event(
        "digest_windows",
        doc_id=expected_doc_id,
        window_count=len(windows),
        max_window_token_estimate=max(estimates, default=0),
        total_window_token_estimate=sum(estimates),
    )
    for window in windows:
        safe_debug(
            "digest_window",
            doc_id=expected_doc_id,
            window_id=window.window_id,
            ordinal=window.ordinal,
            evidence_count=len(window.evidence_ids),
            estimated_tokens=window.estimated_tokens,
        )
    chunks = list(
        await asyncio.gather(
            *(
                _generate_chunk_digest(window, limiter=limiter, budget=budgets.window)
                for window in windows
            )
        )
    )
    safe_event("chunk_digests_complete", doc_id=expected_doc_id, chunk_count=len(chunks))

    async def reduce_invoke(
        payload: dict[str, object], response_model: type[DigestFragment | DocumentDigest]
    ) -> DigestFragment | DocumentDigest:
        return await limiter.invoke(
            system_prompt=DIGEST_REDUCTION_PROMPT,
            input_data=payload,
            response_model=response_model,
            budget=budgets.reduction,
            stage="document_digest_reduction",
        )

    async def compact_invoke(
        payload: dict[str, object], response_model: type[DigestFragment | DocumentDigest]
    ) -> DigestFragment | DocumentDigest:
        return await limiter.invoke(
            system_prompt=DIGEST_COMPACTION_PROMPT,
            input_data=payload,
            response_model=response_model,
            budget=budgets.reduction,
            stage="document_digest_compaction",
        )

    digest = await reduce_chunk_digests(
        chunks,
        token_counter=counter,
        budget=budgets.reduction,
        invoke=reduce_invoke,
        compact_invoke=compact_invoke,
    )
    if digest.doc_id != expected_doc_id:
        raise ValueError("digest document ID does not match artifact")
    safe_event(
        "document_digest_complete",
        doc_id=expected_doc_id,
        filename=getattr(artifact, "filename", None),
        window_count=len(windows),
        chunk_count=len(chunks),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return digest


def _index_by_document(indexes: Iterable[object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for index in indexes:
        doc_id = getattr(index, "doc_id", None)
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("document index does not expose a non-empty document ID")
        if doc_id in result:
            raise ValueError(f"duplicate document index for {doc_id!r}")
        result[doc_id] = index
    return result


async def generate_digests(
    artifacts: Sequence[object],
    indexes: Sequence[object],
    generator: StructuredGenerator,
    *,
    token_counter: TokenCounter | None = None,
    budgets: UnderstandingBudgets | None = None,
    limiter: GenerationLimiter | None = None,
    concurrency: int = 4,
) -> list[DocumentDigest]:
    """Generate in parallel while retaining artifact input order.

    The optional limiter is deliberately shared with higher stages by the
    application layer; otherwise this function creates a finite local limiter.
    """
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be an integer of at least 1")
    budgets = budgets if budgets is not None else UnderstandingBudgets()
    if not isinstance(budgets, UnderstandingBudgets):
        raise TypeError("budgets must be an UnderstandingBudgets")
    if limiter is None:
        token_counter = token_counter if token_counter is not None else Utf8ByteTokenEstimator()
        limiter = GenerationLimiter(generator, token_counter=token_counter, concurrency=concurrency)
    elif token_counter is not None and token_counter is not limiter.token_counter:
        raise ValueError("token_counter must match the shared generation limiter")
    by_document = _index_by_document(indexes)
    artifact_ids = tuple(document_id(artifact) for artifact in artifacts)
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("document artifacts must have unique document IDs")
    if set(artifact_ids) != set(by_document):
        raise ValueError("document artifacts and indexes must have matching document IDs")

    async def one(artifact: object) -> DocumentDigest:
        doc_id = document_id(artifact)
        try:
            index = by_document[doc_id]
        except KeyError as error:
            raise ValueError(f"no index available for document {doc_id!r}") from error
        return await generate_document_digest(
            artifact,
            index,
            generator,
            budgets=budgets,
            limiter=limiter,
            concurrency=concurrency,
        )

    return list(await asyncio.gather(*(one(artifact) for artifact in artifacts)))
