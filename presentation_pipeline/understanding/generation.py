"""Bounded generation helpers for document-understanding digests."""

from __future__ import annotations

from typing import TypeVar

from presentation_pipeline.budgeting import GenerationLimiter, InputBudget
from presentation_pipeline.common.references import EvidenceRef
from presentation_pipeline.observability import safe_event

from .contracts import DigestOutputContract, bound_digest_to_contract
from .models import ChunkDigest, DigestFragment, DocumentDigest, KeyFact, TopicDigest
from .reduction import validate_digest_scope

DigestT = TypeVar("DigestT", ChunkDigest, DigestFragment, DocumentDigest)


async def generate_bounded_digest(
    *,
    limiter: GenerationLimiter,
    system_prompt: str,
    input_data: dict[str, object],
    response_model: type[DigestT],
    input_budget: InputBudget,
    output_contract: DigestOutputContract,
    stage: str,
    expected_doc_id: str,
    allowed_evidence_ids: set[str] | None = None,
    expected_window_id: str | None = None,
) -> DigestT:
    """Generate once, normalize deterministic provenance, then bound output.

    Document/window identities are transport metadata owned by the pipeline, not
    semantic choices for the model. Evidence IDs *are* model-selected, so they
    are intersected with the operation's allowed scope. Unsupported topics or
    facts are dropped rather than being assigned invented provenance.

    There is intentionally no provider-based compaction/repair loop here. A
    model may be verbose, but the reduction tree receives a predictably small,
    scope-safe projection.
    """
    result = await limiter.invoke(
        system_prompt=system_prompt,
        input_data=input_data,
        response_model=response_model,
        budget=input_budget,
        stage=stage,
    )
    normalized = _normalize_digest_scope(
        result,
        expected_doc_id=expected_doc_id,
        expected_window_id=expected_window_id,
        allowed_evidence_ids=allowed_evidence_ids,
        stage=stage,
    )
    if allowed_evidence_ids is not None:
        validate_digest_scope(
            normalized,
            doc_id=expected_doc_id,
            allowed_evidence_ids=allowed_evidence_ids,
        )
    return bound_digest_to_contract(
        normalized,
        contract=output_contract,
        token_counter=limiter.token_counter,
        stage=stage,
    )


def _normalize_digest_scope(
    result: DigestT,
    *,
    expected_doc_id: str,
    expected_window_id: str | None,
    allowed_evidence_ids: set[str] | None,
    stage: str,
) -> DigestT:
    """Project provider output onto deterministic pipeline provenance.

    The provider is allowed to select from supplied evidence IDs, but it does
    not own document/window identity. Unknown evidence IDs are removed. A claim
    with no remaining valid evidence is removed as unsupported.
    """

    normalized_topics: list[TopicDigest] = []
    normalized_facts: list[KeyFact] = []
    dropped_evidence_ids = 0
    canonicalized_reference_doc_ids = 0
    dropped_topics = 0
    dropped_facts = 0

    def normalize_refs(refs: list[EvidenceRef]) -> list[EvidenceRef]:
        nonlocal dropped_evidence_ids, canonicalized_reference_doc_ids
        normalized: list[EvidenceRef] = []
        for ref in refs:
            if ref.doc_id != expected_doc_id:
                canonicalized_reference_doc_ids += 1
            if allowed_evidence_ids is None:
                kept_ids = list(ref.evidence_ids)
            else:
                kept_ids = [
                    evidence_id
                    for evidence_id in ref.evidence_ids
                    if evidence_id in allowed_evidence_ids
                ]
                dropped_evidence_ids += len(ref.evidence_ids) - len(kept_ids)
            if kept_ids:
                normalized.append(
                    EvidenceRef(doc_id=expected_doc_id, evidence_ids=kept_ids)
                )
        return normalized

    for topic in result.topics:
        refs = normalize_refs(topic.evidence)
        if not refs:
            dropped_topics += 1
            continue
        normalized_topics.append(topic.model_copy(update={"evidence": refs}))

    for fact in result.key_facts:
        refs = normalize_refs(fact.evidence)
        if not refs:
            dropped_facts += 1
            continue
        normalized_facts.append(fact.model_copy(update={"evidence": refs}))

    updates: dict[str, object] = {
        "doc_id": expected_doc_id,
        "topics": normalized_topics,
        "key_facts": normalized_facts,
    }
    identity_changed = result.doc_id != expected_doc_id
    if isinstance(result, ChunkDigest) and expected_window_id is not None:
        identity_changed = identity_changed or result.window_id != expected_window_id
        updates["window_id"] = expected_window_id

    if (
        identity_changed
        or canonicalized_reference_doc_ids
        or dropped_evidence_ids
        or dropped_topics
        or dropped_facts
    ):
        safe_event(
            "digest_provenance_normalized",
            stage=stage,
            doc_id=expected_doc_id,
            top_level_identity_changed=identity_changed,
            canonicalized_reference_doc_ids=canonicalized_reference_doc_ids,
            dropped_unknown_evidence_ids=dropped_evidence_ids,
            dropped_unsupported_topics=dropped_topics,
            dropped_unsupported_key_facts=dropped_facts,
        )

    return result.model_copy(update=updates)  # type: ignore[return-value]
