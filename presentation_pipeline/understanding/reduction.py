"""Deterministic hierarchical reduction of bounded chunk digests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from presentation_pipeline.budgeting import InputBudget, TokenCounter, estimate_request_tokens

from .models import ChunkDigest, DigestFragment, DocumentDigest
from .prompts import DIGEST_REDUCTION_PROMPT, build_digest_reduction_input


class DigestProvenanceError(ValueError):
    """A digest cites evidence outside the input scope for its operation."""


class ReductionNonReductionError(RuntimeError):
    """A configured reduction cannot make deterministic forward progress."""


def _limit(budget: InputBudget) -> int:
    if not isinstance(budget, InputBudget):
        raise TypeError("reduction budget must be an InputBudget")
    return budget.target_input_tokens_or_usable


def _fragment_input(fragment: DigestFragment) -> dict[str, object]:
    return fragment.model_dump(mode="json")


def fragment_from_chunk(chunk: ChunkDigest, ordinal: int) -> DigestFragment:
    return DigestFragment(
        doc_id=chunk.doc_id,
        fragment_id=f"fragment-window-{ordinal:04d}",
        summary=chunk.summary,
        topics=chunk.topics,
        key_facts=chunk.key_facts,
    )


def evidence_scope(items: Sequence[ChunkDigest | DigestFragment]) -> set[str]:
    return {
        evidence_id
        for item in items
        for reference in item.evidence_refs()
        for evidence_id in reference.evidence_ids
    }


def validate_digest_scope(
    digest: ChunkDigest | DigestFragment | DocumentDigest,
    *,
    doc_id: str,
    allowed_evidence_ids: set[str],
) -> None:
    """Verify that a model response stays within the supplied provenance scope."""
    if digest.doc_id != doc_id:
        raise DigestProvenanceError("digest document ID does not match its input scope")
    refs = digest.evidence_refs()
    for reference in refs:
        if reference.doc_id != doc_id:
            raise DigestProvenanceError("digest references evidence from another document")
        unknown = set(reference.evidence_ids) - allowed_evidence_ids
        if unknown:
            raise DigestProvenanceError("digest references evidence outside its input scope")


def _count(token_counter: TokenCounter, payload: dict[str, object]) -> int:
    return estimate_request_tokens(DIGEST_REDUCTION_PROMPT, payload, token_counter)


def pack_digest_fragments(
    fragments: Sequence[DigestFragment],
    *,
    doc_id: str,
    level: int,
    token_counter: TokenCounter,
    budget: InputBudget,
) -> list[list[DigestFragment]]:
    """Pack source-order fragments into groups using the real reduction payload."""
    if not fragments:
        raise ValueError("at least one fragment is required for reduction")
    limit = _limit(budget)
    groups: list[list[DigestFragment]] = []
    current: list[DigestFragment] = []
    for fragment in fragments:
        if fragment.doc_id != doc_id:
            raise DigestProvenanceError("cannot reduce fragments from different documents")
        candidate = [*current, fragment]
        payload = build_digest_reduction_input(
            doc_id=doc_id, level=level, fragments=[_fragment_input(value) for value in candidate]
        )
        if _count(token_counter, payload) <= limit:
            current.append(fragment)
            continue
        if not current:
            raise ReductionNonReductionError("one digest fragment exceeds the reduction input budget")
        groups.append(current)
        current = [fragment]
        payload = build_digest_reduction_input(
            doc_id=doc_id, level=level, fragments=[_fragment_input(fragment)]
        )
        if _count(token_counter, payload) > limit:
            raise ReductionNonReductionError("one digest fragment exceeds the reduction input budget")
    if current:
        groups.append(current)
    return groups


ReductionCall = Callable[
    [dict[str, object], type[DigestFragment | DocumentDigest]],
    Awaitable[DigestFragment | DocumentDigest],
]


async def reduce_chunk_digests(
    chunks: Sequence[ChunkDigest],
    *,
    token_counter: TokenCounter,
    budget: InputBudget,
    invoke: ReductionCall,
) -> DocumentDigest:
    """Recursively reduce chunks, with strict scope validation at every level."""
    if not chunks:
        raise ValueError("cannot generate a document digest from empty evidence")
    doc_id = chunks[0].doc_id
    for ordinal, chunk in enumerate(chunks):
        if chunk.doc_id != doc_id:
            raise DigestProvenanceError("cannot reduce chunks from different documents")
    current = [fragment_from_chunk(chunk, ordinal) for ordinal, chunk in enumerate(chunks)]
    level = 0
    while True:
        groups = pack_digest_fragments(current, doc_id=doc_id, level=level, token_counter=token_counter, budget=budget)
        # A final response must be checked against the same full request shape.
        if len(groups) == 1:
            payload = build_digest_reduction_input(
                doc_id=doc_id, level=level, fragments=[_fragment_input(value) for value in current]
            )
            result = await invoke(payload, DocumentDigest)
            if not isinstance(result, DocumentDigest):
                raise TypeError("final reduction did not return a DocumentDigest")
            validate_digest_scope(result, doc_id=doc_id, allowed_evidence_ids=evidence_scope(current))
            return result
        if len(groups) >= len(current):
            raise ReductionNonReductionError("reduction grouping did not reduce fragment count")

        async def reduce_group(ordinal: int, group: list[DigestFragment]) -> DigestFragment:
            payload = build_digest_reduction_input(
                doc_id=doc_id, level=level, fragments=[_fragment_input(value) for value in group]
            )
            result = await invoke(payload, DigestFragment)
            if not isinstance(result, DigestFragment):
                raise TypeError("intermediate reduction did not return a DigestFragment")
            # Model-visible fragment IDs must be transport IDs, never citations.
            result = result.model_copy(update={"fragment_id": f"fragment-l{level + 1}-{ordinal:04d}"})
            validate_digest_scope(result, doc_id=doc_id, allowed_evidence_ids=evidence_scope(group))
            return result

        current = list(await asyncio.gather(*(reduce_group(ordinal, group) for ordinal, group in enumerate(groups))))
        level += 1
