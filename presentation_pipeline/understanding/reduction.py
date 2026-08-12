"""Deterministic hierarchical reduction of already-bounded document digests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from presentation_pipeline.budgeting import InputBudget, TokenCounter, estimate_request_tokens
from presentation_pipeline.observability import safe_debug, safe_event

from .models import ChunkDigest, DigestFragment, DocumentDigest
from .prompts import DIGEST_REDUCTION_PROMPT, build_digest_reduction_input


class DigestProvenanceError(ValueError):
    """A digest cites evidence outside the input scope for its operation."""


class ReductionInvariantError(RuntimeError):
    """Bounded fragments could not make deterministic forward progress."""

    def __init__(
        self,
        *,
        doc_id: str,
        level: int,
        fragment_count: int,
        group_sizes: Sequence[int],
        fragment_token_estimates: Sequence[int] | None = None,
        limit_tokens: int | None = None,
        reason: str,
        # Legacy diagnostic arguments retained only so old failure-report tests
        # and persisted tooling can still construct the error. Production
        # reduction no longer uses compaction.
        estimates: dict[str, object] | None = None,
        limit: int | None = None,
        rounds: int | None = None,
        target_fragment_tokens: int | None = None,
        max_compaction_rounds: int | None = None,
    ) -> None:
        if fragment_token_estimates is None:
            raw = (estimates or {}).get("fragment_tokens", ())
            fragment_token_estimates = [value for value in raw if isinstance(value, int)] if isinstance(raw, (list, tuple)) else []
        if limit_tokens is None:
            limit_tokens = limit if isinstance(limit, int) else 0
        self.metadata: dict[str, object] = {
            "doc_id": doc_id,
            "level": level,
            "fragment_count": fragment_count,
            "group_sizes": list(group_sizes),
            "fragment_token_estimates": list(fragment_token_estimates),
            "limit_tokens": limit_tokens,
            "reason": reason,
        }
        if estimates is not None:
            self.metadata["estimates"] = estimates
        if rounds is not None:
            self.metadata["rounds"] = rounds
            self.metadata["compaction_rounds"] = rounds
            self.metadata["compaction_rounds_at_level"] = rounds
        if target_fragment_tokens is not None:
            self.metadata["target_fragment_tokens"] = target_fragment_tokens
        if max_compaction_rounds is not None:
            self.metadata["max_compaction_rounds_per_level"] = max_compaction_rounds
        super().__init__(f"reduction invariant violated ({reason})")


# Compatibility name used by failure reporting and older callers.  The
# production reducer no longer has a compaction/retry state machine.
ReductionNonReductionError = ReductionInvariantError


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
    for reference in digest.evidence_refs():
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
    estimates: list[int] = []
    for fragment in fragments:
        value = token_counter.count_payload(_fragment_input(fragment))
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("token counter returned an invalid token count")
        estimates.append(value)
    return estimates


def target_fragment_tokens_for_pairing(
    *,
    doc_id: str,
    level: int,
    token_counter: TokenCounter,
    budget: InputBudget,
) -> int:
    """Compatibility diagnostic for the old reducer tests/tools.

    Production output sizing is now controlled by DigestOutputContract.
    """
    wrapper_tokens = _count(
        token_counter,
        build_digest_reduction_input(doc_id=doc_id, level=level, fragments=[]),
    )
    return max(int(max(_limit(budget) - wrapper_tokens, 1) * 0.32), 1)


def pack_digest_fragments(
    fragments: Sequence[DigestFragment],
    *,
    doc_id: str,
    level: int,
    token_counter: TokenCounter,
    budget: InputBudget,
    rounds: int = 0,  # retained only for compatibility; intentionally unused
) -> list[list[DigestFragment]]:
    """Greedily pack source-ordered fragments using the real request shape."""
    del rounds
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
            doc_id=doc_id,
            level=level,
            fragments=[_fragment_input(value) for value in candidate],
        )
        if _count(token_counter, payload) <= limit:
            current.append(fragment)
            continue
        if not current:
            raise ReductionInvariantError(
                doc_id=doc_id,
                level=level,
                fragment_count=len(fragments),
                group_sizes=[],
                fragment_token_estimates=_fragment_token_estimates(fragments, token_counter),
                limit_tokens=limit,
                reason="bounded_fragment_exceeds_request_limit",
            )
        groups.append(current)
        current = [fragment]
        singleton = build_digest_reduction_input(
            doc_id=doc_id,
            level=level,
            fragments=[_fragment_input(fragment)],
        )
        if _count(token_counter, singleton) > limit:
            raise ReductionInvariantError(
                doc_id=doc_id,
                level=level,
                fragment_count=len(fragments),
                group_sizes=[*(len(group) for group in groups), 1],
                fragment_token_estimates=_fragment_token_estimates(fragments, token_counter),
                limit_tokens=limit,
                reason="bounded_fragment_exceeds_request_limit",
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
    max_compaction_rounds: int = 0,
) -> DocumentDigest:
    """Reduce bounded chunks without provider-based shrink/retry loops.

    ``compact_invoke`` and ``max_compaction_rounds`` remain accepted only so
    older callers do not crash at import/call time; they are intentionally
    ignored.  All production digest outputs are bounded before they enter this
    reducer.
    """
    del compact_invoke, max_compaction_rounds
    if not chunks:
        raise ValueError("cannot generate a document digest from empty evidence")
    doc_id = chunks[0].doc_id
    for chunk in chunks:
        if chunk.doc_id != doc_id:
            raise DigestProvenanceError("cannot reduce chunks from different documents")

    current = [fragment_from_chunk(chunk, ordinal) for ordinal, chunk in enumerate(chunks)]
    level = 0

    while True:
        groups = pack_digest_fragments(
            current,
            doc_id=doc_id,
            level=level,
            token_counter=token_counter,
            budget=budget,
        )
        estimates = _fragment_token_estimates(current, token_counter)
        adjacent = [
            _count(
                token_counter,
                build_digest_reduction_input(
                    doc_id=doc_id,
                    level=level,
                    fragments=[_fragment_input(current[i]), _fragment_input(current[i + 1])],
                ),
            )
            for i in range(len(current) - 1)
        ]
        safe_event(
            "digest_reduction_level",
            doc_id=doc_id,
            level=level,
            fragment_count=len(current),
            group_count=len(groups),
            group_sizes=[len(group) for group in groups],
            fragment_token_estimates=estimates,
            input_limit_tokens=_limit(budget),
            adjacent_pair_fit_count=sum(value <= _limit(budget) for value in adjacent),
        )
        safe_debug(
            "digest_reduction_pair_fit",
            doc_id=doc_id,
            level=level,
            adjacent_pair_tokens=[
                {"left": i, "right": i + 1, "tokens": value, "fits": value <= _limit(budget)}
                for i, value in enumerate(adjacent)
            ],
        )

        if len(groups) == 1:
            safe_event(
                "reduction_decision",
                doc_id=doc_id,
                level=level,
                decision="final_reduce",
                reason="all_fragments_fit",
                fragment_count=len(current),
            )
            payload = build_digest_reduction_input(
                doc_id=doc_id,
                level=level,
                fragments=[_fragment_input(value) for value in current],
            )
            result = await invoke(payload, DocumentDigest)
            if not isinstance(result, DocumentDigest):
                raise TypeError("final reduction did not return a DocumentDigest")
            validate_digest_scope(
                result,
                doc_id=doc_id,
                allowed_evidence_ids=evidence_scope(current),
            )
            return result

        if len(groups) >= len(current):
            safe_event(
                "reduction_invariant_failed",
                doc_id=doc_id,
                level=level,
                fragment_count=len(current),
                group_sizes=[len(group) for group in groups],
                reason="bounded_fragments_not_pairable",
            )
            raise ReductionInvariantError(
                doc_id=doc_id,
                level=level,
                fragment_count=len(current),
                group_sizes=[len(group) for group in groups],
                fragment_token_estimates=estimates,
                limit_tokens=_limit(budget),
                reason="bounded_fragments_not_pairable",
            )

        safe_event(
            "reduction_decision",
            doc_id=doc_id,
            level=level,
            decision="merge_groups",
            reason="fragment_count_will_decrease",
            fragment_count=len(current),
            group_count=len(groups),
        )

        async def reduce_group(ordinal: int, group: list[DigestFragment]) -> DigestFragment:
            if len(group) == 1:
                fragment = group[0]
                safe_event(
                    "reduction_singleton_carried_forward",
                    doc_id=doc_id,
                    level=level,
                    group_ordinal=ordinal,
                    estimated_tokens=_fragment_token_estimates([fragment], token_counter)[0],
                )
                return fragment.model_copy(
                    update={"fragment_id": f"fragment-l{level + 1}-{ordinal:04d}"}
                )

            payload = build_digest_reduction_input(
                doc_id=doc_id,
                level=level,
                fragments=[_fragment_input(value) for value in group],
            )
            result = await invoke(payload, DigestFragment)
            if not isinstance(result, DigestFragment):
                raise TypeError("intermediate reduction did not return a DigestFragment")
            result = result.model_copy(
                update={"fragment_id": f"fragment-l{level + 1}-{ordinal:04d}"}
            )
            validate_digest_scope(
                result,
                doc_id=doc_id,
                allowed_evidence_ids=evidence_scope(group),
            )
            safe_event(
                "digest_reduction_merge",
                doc_id=doc_id,
                level=level,
                group_ordinal=ordinal,
                input_group_tokens=_count(token_counter, payload),
                merged_fragment_tokens=_fragment_token_estimates([result], token_counter)[0],
            )
            return result

        next_level = list(
            await asyncio.gather(
                *(reduce_group(ordinal, group) for ordinal, group in enumerate(groups))
            )
        )
        if len(next_level) >= len(current):
            raise ReductionInvariantError(
                doc_id=doc_id,
                level=level,
                fragment_count=len(current),
                group_sizes=[len(group) for group in groups],
                fragment_token_estimates=estimates,
                limit_tokens=_limit(budget),
                reason="merge_did_not_reduce_fragment_count",
            )
        current = next_level
        level += 1
