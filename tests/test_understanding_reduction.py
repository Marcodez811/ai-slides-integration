import asyncio
from types import SimpleNamespace

import pytest

from presentation_pipeline.budgeting import CharacterTokenEstimator, InputBudget
from presentation_pipeline.common.references import EvidenceRef
from presentation_pipeline.understanding.models import ChunkDigest, DigestFragment, DocumentDigest, KeyFact
from presentation_pipeline.understanding.reduction import (
    DigestProvenanceError,
    ReductionInvariantError,
    ReductionNonReductionError,
    reduce_chunk_digests,
    target_fragment_tokens_for_pairing,
    validate_digest_scope,
)
from presentation_pipeline.understanding.service import generate_document_digest


def _chunk(ordinal: int) -> ChunkDigest:
    return ChunkDigest(
        doc_id="doc-a",
        window_id=f"window-{ordinal:04d}",
        summary=f"Summary {ordinal}",
        key_facts=[KeyFact(claim=f"Fact {ordinal}", evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=[f"e-{ordinal}"])])],
    )


def test_scope_validation_rejects_new_evidence() -> None:
    fragment = DigestFragment(
        doc_id="doc-a", fragment_id="f", summary="s",
        key_facts=[KeyFact(claim="x", evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=["missing"])])],
    )
    with pytest.raises(DigestProvenanceError, match="outside"):
        validate_digest_scope(fragment, doc_id="doc-a", allowed_evidence_ids={"known"})


def test_single_chunk_is_reduced_to_document_digest_with_original_references() -> None:
    async def invoke(payload, response_model):
        assert payload["fragments"][0]["key_facts"][0]["evidence"][0]["evidence_ids"] == ["e-0"]
        return DocumentDigest(
            doc_id="doc-a", summary="final",
            key_facts=[KeyFact(claim="Fact", evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=["e-0"])])],
        )

    digest = asyncio.run(
        reduce_chunk_digests(
            [_chunk(0)], token_counter=CharacterTokenEstimator(), budget=InputBudget(max_input_tokens=4_000), invoke=invoke
        )
    )
    assert digest.evidence_refs()[0].evidence_ids == ["e-0"]


def test_reduction_recurses_in_deterministic_groups() -> None:
    class Counter:
        def count_text(self, text):
            return 1

        def count_payload(self, payload):
            return 100 * len(payload.get("fragments", []))

    calls = []

    async def invoke(payload, response_model):
        calls.append((response_model, [fragment["fragment_id"] for fragment in payload["fragments"]]))
        refs = [evidence_id for fragment in payload["fragments"] for fact in fragment["key_facts"] for ref in fact["evidence"] for evidence_id in ref["evidence_ids"]]
        if response_model is DigestFragment:
            return DigestFragment(
                doc_id="doc-a", fragment_id="provider-id", summary="merged",
                key_facts=[KeyFact(claim="merged", evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=refs)])],
            )
        return DocumentDigest(
            doc_id="doc-a", summary="final",
            key_facts=[KeyFact(claim="final", evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=refs)])],
        )

    result = asyncio.run(
        reduce_chunk_digests(
            [_chunk(0), _chunk(1), _chunk(2)], token_counter=Counter(), budget=InputBudget(max_input_tokens=250), invoke=invoke
        )
    )
    assert [response for response, _ in calls] == [DigestFragment, DocumentDigest]
    assert calls[0][1] == ["fragment-window-0000", "fragment-window-0001"]
    assert calls[1][1] == ["fragment-l1-0000", "fragment-l1-0001"]
    assert {evidence_id for ref in result.evidence_refs() for evidence_id in ref.evidence_ids} == {"e-0", "e-1", "e-2"}


def test_singleton_reduction_deadlock_fails_without_compaction() -> None:
    class Counter:
        def count_text(self, text):
            return 0

        def count_payload(self, payload):
            if "summary" in payload:
                return len(payload["summary"])
            return sum(len(fragment["summary"]) for fragment in payload.get("fragments", [payload.get("fragment", {})]))

    async def invoke(payload, response_model):
        refs = [
            evidence_id
            for fragment in payload["fragments"]
            for fact in fragment["key_facts"]
            for ref in fact["evidence"]
            for evidence_id in ref["evidence_ids"]
        ]
        return DocumentDigest(
            doc_id="doc-a", summary="final",
            key_facts=[KeyFact(claim="final", evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=refs)])],
        )

    with pytest.raises(ReductionInvariantError) as raised:
        asyncio.run(reduce_chunk_digests(
            [_chunk(0), _chunk(1)], token_counter=Counter(), budget=InputBudget(max_input_tokens=12), invoke=invoke,
        ))
    assert raised.value.metadata["reason"] == "singleton_group_deadlock"


def test_reduction_invariant_error_is_content_free() -> None:
    secret = "do not disclose source prose"

    class Counter:
        def count_text(self, text): return 0
        def count_payload(self, payload):
            fragments = [payload] if "summary" in payload else payload.get("fragments", [payload.get("fragment", {})])
            return sum(len(fragment.get("summary", "")) for fragment in fragments)

    async def invoke(_payload, _response_model):
        raise AssertionError("normal reduction should not run")

    chunks = [_chunk(0).model_copy(update={"summary": secret}), _chunk(1).model_copy(update={"summary": secret})]
    with pytest.raises(ReductionNonReductionError) as raised:
        asyncio.run(reduce_chunk_digests(
            chunks, token_counter=Counter(), budget=InputBudget(max_input_tokens=len(secret) + 1),
            invoke=invoke,
        ))
    assert raised.value.metadata["reason"] == "singleton_group_deadlock"
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value.metadata)


def test_reduction_does_not_call_compatibility_compaction_callback() -> None:
    class Counter:
        def count_text(self, text): return 0
        def count_payload(self, payload):
            fragments = [payload] if "summary" in payload else payload.get("fragments", [payload.get("fragment", {})])
            return sum(len(item.get("summary", "")) for item in fragments)

    async def invoke(_payload, _response_model):
        raise AssertionError("normal reduction should not run")

    called = False
    async def compact(_payload, _response_model):
        nonlocal called
        called = True
        raise AssertionError("obsolete callback must remain inactive")
    with pytest.raises(ReductionInvariantError):
        asyncio.run(reduce_chunk_digests(
            [_chunk(0), _chunk(1)], token_counter=Counter(), budget=InputBudget(max_input_tokens=12),
            invoke=invoke, compact_invoke=compact,
        ))
    assert not called


def test_pairing_target_is_budget_derived() -> None:
    class Counter:
        def count_text(self, text): return 0
        def count_payload(self, payload):
            fragments = [payload] if "summary" in payload else payload.get("fragments", [payload.get("fragment", {})])
            return sum(len(item.get("summary", "")) for item in fragments)

    counter = Counter()
    assert target_fragment_tokens_for_pairing(doc_id="doc-a", level=0, token_counter=counter, budget=InputBudget(30)) < target_fragment_tokens_for_pairing(doc_id="doc-a", level=0, token_counter=counter, budget=InputBudget(60))


def test_reduction_preserves_scope_on_normal_merge() -> None:
    class Counter:
        def count_text(self, text): return 1
        def count_payload(self, payload):
            if "summary" in payload: return len(payload["summary"])
            return 5 + sum(len(item.get("summary", "")) for item in payload.get("fragments", [payload.get("fragment", {})]))

    counter = Counter()
    small = target_fragment_tokens_for_pairing(
        doc_id="doc-a", level=0, token_counter=counter, budget=InputBudget(max_input_tokens=30)
    )
    large = target_fragment_tokens_for_pairing(
        doc_id="doc-a", level=0, token_counter=counter, budget=InputBudget(max_input_tokens=60)
    )
    assert 0 < small < large
    async def invoke(payload, response_model):
        refs = [evidence_id for fragment in payload["fragments"] for fact in fragment["key_facts"] for ref in fact["evidence"] for evidence_id in ref["evidence_ids"]]
        return DocumentDigest(doc_id="doc-a", summary="final", key_facts=[KeyFact(claim="x", evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=refs)])])

    result = asyncio.run(reduce_chunk_digests([_chunk(0), _chunk(1)], token_counter=counter, budget=InputBudget(60), invoke=invoke))
    assert result.doc_id == "doc-a"


def test_reduction_carries_singleton_forward_after_bounded_merge() -> None:
    class Counter:
        def count_text(self, text): return 0
        def count_payload(self, payload):
            if "summary" in payload: return len(payload["summary"])
            return sum(len(item.get("summary", "")) for item in payload.get("fragments", [payload.get("fragment", {})]))

    async def invoke(payload, response_model):
        refs = [evidence_id for fragment in payload["fragments"] for fact in fragment["key_facts"] for ref in fact["evidence"] for evidence_id in ref["evidence_ids"]]
        if response_model is DigestFragment:
            return DigestFragment(doc_id="doc-a", fragment_id="merged", summary="x" * 10, key_facts=[KeyFact(claim="x", evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=refs)])])
        return DocumentDigest(doc_id="doc-a", summary="final", key_facts=[KeyFact(claim="x", evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=refs)])])

    result = asyncio.run(reduce_chunk_digests(
        [_chunk(i).model_copy(update={"summary": "x" * 10}) for i in range(3)],
        token_counter=Counter(), budget=InputBudget(max_input_tokens=21), invoke=invoke,
    ))
    assert result.doc_id == "doc-a"


def test_document_generation_uses_chunk_then_final_reduction() -> None:
    calls = []

    class Generator:
        async def generate(self, *, system_prompt, input_data, response_model):
            calls.append(response_model)
            if response_model is ChunkDigest:
                return ChunkDigest(
                    doc_id="doc-a", window_id=input_data["window"]["window_id"], summary="chunk",
                    key_facts=[KeyFact(claim="fact", evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=["e-1"])])],
                )
            assert response_model is DocumentDigest
            return DocumentDigest(
                doc_id="doc-a", summary="final",
                key_facts=[KeyFact(claim="fact", evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=["e-1"])])],
            )

    artifact = SimpleNamespace(doc_id="doc-a", filename="doc-a.docx")
    index = SimpleNamespace(
        doc_id="doc-a", evidence=[SimpleNamespace(evidence_id="e-1", kind="text", section_ids=[], text="content", structured_data={})], sections=[]
    )
    result = asyncio.run(
        generate_document_digest(
            artifact, index, Generator(), token_counter=CharacterTokenEstimator(), budgets=None
        )
    )
    assert result.summary == "final"
    assert calls == [ChunkDigest, DocumentDigest]


def test_empty_document_evidence_is_rejected_before_provider_call() -> None:
    artifact = SimpleNamespace(doc_id="doc-a", filename="doc-a.docx")
    index = SimpleNamespace(doc_id="doc-a", evidence=[], sections=[])
    with pytest.raises(ValueError, match="no evidence"):
        asyncio.run(generate_document_digest(artifact, index, object()))


@pytest.mark.parametrize(
    ("artifacts", "indexes", "message"),
    [
        (
            [SimpleNamespace(doc_id="doc-a"), SimpleNamespace(doc_id="doc-a")],
            [SimpleNamespace(doc_id="doc-a")],
            "unique",
        ),
        (
            [SimpleNamespace(doc_id="doc-a")],
            [SimpleNamespace(doc_id="doc-a"), SimpleNamespace(doc_id="doc-b")],
            "matching",
        ),
    ],
)
def test_generate_digests_rejects_misaligned_document_sets(
    artifacts: list[SimpleNamespace], indexes: list[SimpleNamespace], message: str
) -> None:
    from presentation_pipeline.understanding.service import generate_digests

    with pytest.raises(ValueError, match=message):
        asyncio.run(generate_digests(artifacts, indexes, object()))
