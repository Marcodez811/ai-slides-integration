"""Concurrent, order-preserving document digest generation."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence

from presentation_pipeline.generation import StructuredGenerator, invoke_structured

from .models import DocumentDigest
from .prompts import DOCUMENT_DIGEST_PROMPT, build_document_digest_input, document_id


async def generate_document_digest(
    artifact: object,
    index: object,
    generator: StructuredGenerator,
) -> DocumentDigest:
    expected_doc_id = document_id(artifact)
    digest = await invoke_structured(
        generator,
        DOCUMENT_DIGEST_PROMPT,
        build_document_digest_input(artifact, index),
        DocumentDigest,
    )
    if digest.doc_id != expected_doc_id:
        raise ValueError(
            f"digest document ID {digest.doc_id!r} does not match artifact {expected_doc_id!r}"
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
    concurrency: int = 4,
) -> list[DocumentDigest]:
    """Generate in parallel while returning results in artifact input order."""
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be an integer of at least 1")
    by_document = _index_by_document(indexes)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(artifact: object) -> DocumentDigest:
        doc_id = document_id(artifact)
        try:
            index = by_document[doc_id]
        except KeyError as error:
            raise ValueError(f"no index available for document {doc_id!r}") from error
        async with semaphore:
            return await generate_document_digest(artifact, index, generator)

    return list(await asyncio.gather(*(one(artifact) for artifact in artifacts)))
