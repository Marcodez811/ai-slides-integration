"""Deterministic hierarchical reduction of bounded chunk digests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from presentation_pipeline.budgeting import InputBudget, TokenCounter, estimate_request_tokens
from presentation_pipeline.observability import safe_event

from .models import ChunkDigest, DigestFragment, DocumentDigest
from .prompts import (
    DIGEST_REDUCTION_PROMPT,
    build_digest_compaction_input,
    build_digest_reduction_input,
)


class DigestProvenanceError(ValueError):
    """A digest cites evidence outside the input scope for its operation."""


class ReductionNonReductionError(RuntimeError):
    """A configured reduction cannot make deterministic forward progress."""

    def __init__(
        self,
        *,
        doc_id: str,
        level: int,
        fragment_count: int,
        group_sizes: Sequence[int],
        estimates: dict[str, object],
        limit: int,
        rounds: int,
        reason: str,
    ) -> None:
        # Diagnostics deliberately retain identifiers and numeric estimates only;
        # fragment summaries and source evidence text must never reach errors.
        fragment_estimates = list(estimates.get("fragment_tokens", ()))
        self.metadata: dict[str, object] = {
            "doc_id": doc_id,
            "level": level,
            "fragment_count": fragment_count,
            "group_sizes": list(group_sizes),
            "estimates": estimates,
            "fragment_token_estimates": fragment_estimates,
            "limit": limit,
            "limit_tokens": limit,
            "rounds": rounds,
            "compaction_rounds": rounds,
            "reason": reason,
        }
        super().__init__(f"reduction cannot make forward progress ({reason})")


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
    value = estimate_request_tokens(DIGEST_REDUCTION_PROMPT, payload, token_counter)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token counter returned an invalid token count")
    return value


def _fragment_token_estimates(
    fragments: Sequence[DigestFragment], token_counter: TokenCounter
) -> list[int]:
    """Count canonical serialized fragment transport data, not source text."""
    estimates: list[int] = []
    for fragment in fragments:
        value = token_counter.count_payload(_fragment_input(fragment))
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("token counter returned an invalid token count")
        estimates.append(value)
    return estimates


def _diagnostic_estimates(
    fragments: Sequence[DigestFragment],
    groups: Sequence[Sequence[DigestFragment]],
    *,
    token_counter: TokenCounter,
    doc_id: str,
    level: int,
) -> dict[str, object]:
    fragment_tokens = _fragment_token_estimates(fragments, token_counter)
    group_tokens = [
        _count(
            token_counter,
            build_digest_reduction_input(
                doc_id=doc_id,
                level=level,
                fragments=[_fragment_input(fragment) for fragment in group],
            ),
        )
        for group in groups
    ]
    return {
        "fragment_tokens": fragment_tokens,
        "total_fragment_tokens": sum(fragment_tokens),
        "group_tokens": group_tokens,
    }


def _non_reduction_error(
    *,
    doc_id: str,
    level: int,
    fragments: Sequence[DigestFragment],
    groups: Sequence[Sequence[DigestFragment]],
    token_counter: TokenCounter,
    limit: int,
    rounds: int,
    reason: str,
    estimates: dict[str, object] | None = None,
) -> ReductionNonReductionError:
    return ReductionNonReductionError(
        doc_id=doc_id,
        level=level,
        fragment_count=len(fragments),
        group_sizes=[len(group) for group in groups],
        estimates=(
            estimates
            if estimates is not None
            else _diagnostic_estimates(
                fragments, groups, token_counter=token_counter, doc_id=doc_id, level=level
            )
        ),
        limit=limit,
        rounds=rounds,
        reason=reason,
    )


def pack_digest_fragments(
    fragments: Sequence[DigestFragment],
    *,
    doc_id: str,
    level: int,
    token_counter: TokenCounter,
    budget: InputBudget,
    rounds: int = 0,
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
            raise _non_reduction_error(
                doc_id=doc_id,
                level=level,
                fragments=fragments,
                groups=groups,
                token_counter=token_counter,
                limit=limit,
                rounds=rounds,
                reason="fragment_exceeds_limit",
            )
        groups.append(current)
        current = [fragment]
        payload = build_digest_reduction_input(
            doc_id=doc_id, level=level, fragments=[_fragment_input(fragment)]
        )
        if _count(token_counter, payload) > limit:
            raise _non_reduction_error(
                doc_id=doc_id,
                level=level,
                fragments=fragments,
                groups=[*groups, current],
                token_counter=token_counter,
                limit=limit,
                rounds=rounds,
                reason="fragment_exceeds_limit",
            )
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
    compact_invoke: ReductionCall | None = None,
    max_compaction_rounds: int = 2,
) -> DocumentDigest:
    """Recursively reduce chunks, with strict scope validation at every level."""
    if not chunks:
        raise ValueError("cannot generate a document digest from empty evidence")
    if compact_invoke is not None and not callable(compact_invoke):
        raise TypeError("compact_invoke must be callable")
    if (
        isinstance(max_compaction_rounds, bool)
        or not isinstance(max_compaction_rounds, int)
        or max_compaction_rounds < 0
    ):
        raise ValueError("max_compaction_rounds must be a non-negative integer")
    doc_id = chunks[0].doc_id
    for ordinal, chunk in enumerate(chunks):
        if chunk.doc_id != doc_id:
            raise DigestProvenanceError("cannot reduce chunks from different documents")
    current = [fragment_from_chunk(chunk, ordinal) for ordinal, chunk in enumerate(chunks)]
    level = 0
    compaction_rounds = 0
    while True:
        groups = pack_digest_fragments(
            current,
            doc_id=doc_id,
            level=level,
            token_counter=token_counter,
            budget=budget,
            rounds=compaction_rounds,
        )
        safe_event(
            "digest_reduction_level",
            doc_id=doc_id,
            level=level,
            fragment_count=len(current),
            group_count=len(groups),
            group_sizes=[len(group) for group in groups],
            fragment_token_estimates=_fragment_token_estimates(current, token_counter),
            input_limit_tokens=_limit(budget),
        )
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
            safe_event("compaction_required", doc_id=doc_id, level=level, fragment_count=len(current))
            if compact_invoke is None:
                raise _non_reduction_error(
                    doc_id=doc_id,
                    level=level,
                    fragments=current,
                    groups=groups,
                    token_counter=token_counter,
                    limit=_limit(budget),
                    rounds=compaction_rounds,
                    reason="singleton_group_deadlock",
                )
            if compaction_rounds >= max_compaction_rounds:
                raise _non_reduction_error(
                    doc_id=doc_id,
                    level=level,
                    fragments=current,
                    groups=groups,
                    token_counter=token_counter,
                    limit=_limit(budget),
                    rounds=compaction_rounds,
                    reason="max_compaction_rounds",
                )

            before = _fragment_token_estimates(current, token_counter)

            async def compact_fragment(ordinal: int, fragment: DigestFragment) -> DigestFragment:
                payload = build_digest_compaction_input(
                    doc_id=doc_id, level=level, fragment=_fragment_input(fragment)
                )
                result = await compact_invoke(payload, DigestFragment)
                if not isinstance(result, DigestFragment):
                    raise TypeError("digest compaction did not return a DigestFragment")
                validate_digest_scope(
                    result, doc_id=doc_id, allowed_evidence_ids=evidence_scope([fragment])
                )
                return result.model_copy(
                    update={
                        "fragment_id": (
                            f"fragment-c{compaction_rounds + 1}-l{level}-{ordinal:04d}"
                        )
                    }
                )

            compacted = list(
                await asyncio.gather(
                    *(compact_fragment(ordinal, fragment) for ordinal, fragment in enumerate(current))
                )
            )
            after = _fragment_token_estimates(compacted, token_counter)
            compaction_rounds += 1
            safe_event(
                "digest_compaction_complete",
                doc_id=doc_id,
                level=level,
                compaction_round=compaction_rounds,
                before_total_tokens=sum(before),
                after_total_tokens=sum(after),
                size_reduction_ratio=(sum(after) / sum(before)) if sum(before) else 1.0,
            )
            if sum(after) >= sum(before):
                estimates = _diagnostic_estimates(
                    compacted,
                    groups,
                    token_counter=token_counter,
                    doc_id=doc_id,
                    level=level,
                )
                estimates.update(
                    {
                        "before_fragment_tokens": before,
                        "before_total_fragment_tokens": sum(before),
                        "after_fragment_tokens": after,
                        "after_total_fragment_tokens": sum(after),
                    }
                )
                raise _non_reduction_error(
                    doc_id=doc_id,
                    level=level,
                    fragments=compacted,
                    groups=groups,
                    token_counter=token_counter,
                    limit=_limit(budget),
                    rounds=compaction_rounds,
                    reason="compaction_made_no_token_progress",
                    estimates=estimates,
                )
            current = compacted
            continue

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
