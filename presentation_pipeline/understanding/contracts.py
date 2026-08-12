"""Deterministic output-size contracts for document digests.

The provider is asked to be concise, but correctness does not depend on it
obeying a soft prompt.  Generated digests are deterministically projected to a
small navigation summary before entering hierarchical reduction.  Canonical
source evidence remains in the document index, so low-priority digest detail
may be dropped without losing the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from presentation_pipeline.budgeting import InputBudget, TokenCounter
from presentation_pipeline.observability import safe_event

from .models import ChunkDigest, DigestFragment, DocumentDigest

DigestT = TypeVar("DigestT", ChunkDigest, DigestFragment, DocumentDigest)


@dataclass(frozen=True, slots=True)
class DigestOutputContract:
    """Small, provider-neutral contract for every digest node in the tree."""

    max_transport_tokens: int
    max_topics: int = 8
    max_key_facts: int = 12
    max_evidence_refs_per_item: int = 2
    max_evidence_ids_per_ref: int = 4

    def __post_init__(self) -> None:
        for name in (
            "max_transport_tokens",
            "max_topics",
            "max_key_facts",
            "max_evidence_refs_per_item",
            "max_evidence_ids_per_ref",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def as_provider_data(self) -> dict[str, object]:
        return {
            "max_transport_tokens_estimate": self.max_transport_tokens,
            "max_topics": self.max_topics,
            "max_key_facts": self.max_key_facts,
            "priority": "retain only the most presentation-relevant information",
        }


class DigestOutputContractError(ValueError):
    """A digest cannot be represented inside the configured bounded contract."""

    def __init__(
        self,
        *,
        doc_id: str,
        stage: str,
        estimated_transport_tokens: int,
        max_transport_tokens: int,
        topic_count: int,
        key_fact_count: int,
        reason: str,
    ) -> None:
        self.stage = stage
        self.metadata: dict[str, object] = {
            "doc_id": doc_id,
            "stage": stage,
            "estimated_transport_tokens": estimated_transport_tokens,
            "max_transport_tokens": max_transport_tokens,
            "topic_count": topic_count,
            "key_fact_count": key_fact_count,
            "reason": reason,
        }
        super().__init__(f"digest output contract violated ({reason})")


def output_contract_for_reduction_budget(budget: InputBudget) -> DigestOutputContract:
    """Return a deliberately conservative tree-node size.

    A node receives at most ~25% of the reduction request target.  Therefore two
    maximum-sized nodes leave substantial room for the reduction prompt, JSON
    wrapper, and metadata even with the conservative UTF-8-byte estimator.
    """
    if not isinstance(budget, InputBudget):
        raise TypeError("reduction budget must be an InputBudget")
    target = budget.target_input_tokens_or_usable
    return DigestOutputContract(
        max_transport_tokens=max(128, int(target * 0.25)),
        max_topics=8,
        max_key_facts=12,
    )


def estimate_digest_transport_tokens(digest: DigestT, token_counter: TokenCounter) -> int:
    value = token_counter.count_payload(digest.model_dump(mode="json"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token counter returned an invalid digest token estimate")
    return value


def _clip(value: object, maximum: int) -> object:
    if not isinstance(value, str):
        return value
    if len(value) <= maximum:
        return value
    # Keep the operation deterministic and Unicode-safe.  This is generated
    # summary prose, not source text; the canonical source remains untouched.
    return value[:maximum].rstrip() or value[:maximum]


def _bound_refs(item: dict[str, object], contract: DigestOutputContract) -> None:
    refs = item.get("evidence")
    if not isinstance(refs, list):
        return
    bounded: list[object] = []
    for ref in refs[: contract.max_evidence_refs_per_item]:
        if not isinstance(ref, dict):
            bounded.append(ref)
            continue
        clone = dict(ref)
        ids = clone.get("evidence_ids")
        if isinstance(ids, list):
            clone["evidence_ids"] = ids[: contract.max_evidence_ids_per_ref]
        bounded.append(clone)
    item["evidence"] = bounded


def _initial_projection(digest: DigestT, contract: DigestOutputContract) -> dict[str, object]:
    data = digest.model_dump(mode="python")
    data["summary"] = _clip(data.get("summary"), 700)

    topics = data.get("topics")
    if isinstance(topics, list):
        bounded_topics: list[object] = []
        for raw in topics[: contract.max_topics]:
            if not isinstance(raw, dict):
                bounded_topics.append(raw)
                continue
            item = dict(raw)
            item["topic"] = _clip(item.get("topic"), 100)
            item["summary"] = _clip(item.get("summary"), 280)
            _bound_refs(item, contract)
            bounded_topics.append(item)
        data["topics"] = bounded_topics

    facts = data.get("key_facts")
    if isinstance(facts, list):
        bounded_facts: list[object] = []
        for raw in facts[: contract.max_key_facts]:
            if not isinstance(raw, dict):
                bounded_facts.append(raw)
                continue
            item = dict(raw)
            item["claim"] = _clip(item.get("claim"), 280)
            _bound_refs(item, contract)
            bounded_facts.append(item)
        data["key_facts"] = bounded_facts
    return data


def _rebuild(digest: DigestT, data: dict[str, object]) -> DigestT:
    return type(digest).model_validate(data)  # type: ignore[return-value]


def bound_digest_to_contract(
    digest: DigestT,
    *,
    contract: DigestOutputContract,
    token_counter: TokenCounter,
    stage: str,
) -> DigestT:
    """Deterministically make a generated digest small enough for the tree.

    The routine first caps verbose fields and provenance fan-out, then removes
    lower-priority trailing facts/topics until the serialized digest fits.  It
    never edits canonical source evidence and never makes another provider call.
    """
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-blank string")

    before = estimate_digest_transport_tokens(digest, token_counter)
    data = _initial_projection(digest, contract)
    candidate = _rebuild(digest, data)
    after_projection = estimate_digest_transport_tokens(candidate, token_counter)

    # Prefer keeping a balanced navigation map rather than all facts or all
    # topics.  Remove one trailing item at a time until the hard transport bound
    # is met.  This is deterministic and terminates because the lists shrink.
    while after_projection > contract.max_transport_tokens:
        topics = data.get("topics")
        facts = data.get("key_facts")
        topic_count = len(topics) if isinstance(topics, list) else 0
        fact_count = len(facts) if isinstance(facts, list) else 0
        if fact_count > 2 and isinstance(facts, list):
            facts.pop()
        elif topic_count > 2 and isinstance(topics, list):
            topics.pop()
        elif fact_count > 0 and isinstance(facts, list):
            facts.pop()
        elif topic_count > 0 and isinstance(topics, list):
            topics.pop()
        else:
            break
        candidate = _rebuild(digest, data)
        after_projection = estimate_digest_transport_tokens(candidate, token_counter)

    if after_projection > contract.max_transport_tokens:
        # With topics/facts gone, only the mandatory summary/identity fields can
        # be responsible. Binary-search the generated summary to the largest
        # Unicode-safe prefix that satisfies the contract.
        original_summary = str(data.get("summary") or "")
        low, high, best = 1, max(len(original_summary), 1), 0
        while low <= high:
            middle = (low + high) // 2
            trial = dict(data)
            trial["summary"] = original_summary[:middle] or "-"
            rebuilt = _rebuild(digest, trial)
            tokens = estimate_digest_transport_tokens(rebuilt, token_counter)
            if tokens <= contract.max_transport_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best:
            data["summary"] = original_summary[:best]
            candidate = _rebuild(digest, data)
            after_projection = estimate_digest_transport_tokens(candidate, token_counter)

    if after_projection > contract.max_transport_tokens:
        raise DigestOutputContractError(
            doc_id=digest.doc_id,
            stage=stage,
            estimated_transport_tokens=after_projection,
            max_transport_tokens=contract.max_transport_tokens,
            topic_count=len(candidate.topics),
            key_fact_count=len(candidate.key_facts),
            reason="minimal_digest_exceeds_contract",
        )

    safe_event(
        "digest_contract_valid",
        doc_id=digest.doc_id,
        stage=stage,
        before_transport_tokens=before,
        estimated_output_tokens=after_projection,
        target_tokens=contract.max_transport_tokens,
        topic_count=len(candidate.topics),
        key_fact_count=len(candidate.key_facts),
        deterministically_bounded=after_projection < before,
    )
    return candidate
