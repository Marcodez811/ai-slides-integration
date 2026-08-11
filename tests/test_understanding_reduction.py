import asyncio
from types import SimpleNamespace

import pytest

from presentation_pipeline.budgeting import CharacterTokenEstimator, InputBudget
from presentation_pipeline.common.references import EvidenceRef
from presentation_pipeline.understanding.models import ChunkDigest, DigestFragment, DocumentDigest, KeyFact
from presentation_pipeline.understanding.reduction import DigestProvenanceError, reduce_chunk_digests, validate_digest_scope
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
    assert [response for response, _ in calls] == [DigestFragment, DigestFragment, DocumentDigest]
    assert calls[0][1] == ["fragment-window-0000", "fragment-window-0001"]
    assert calls[1][1] == ["fragment-window-0002"]
    assert {evidence_id for ref in result.evidence_refs() for evidence_id in ref.evidence_ids} == {"e-0", "e-1", "e-2"}


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
