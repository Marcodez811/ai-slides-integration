import asyncio
from types import SimpleNamespace

import pytest

from presentation_pipeline.budgeting import (
    CharacterTokenEstimator,
    InputBudget,
    UnderstandingBudgets,
    Utf8ByteTokenEstimator,
    estimate_request_tokens,
)
from presentation_pipeline.common.references import EvidenceRef
from presentation_pipeline.understanding.contracts import (
    DigestOutputContract,
    bound_digest_to_contract,
    estimate_digest_transport_tokens,
    output_contract_for_reduction_budget,
)
from presentation_pipeline.understanding.generation import generate_bounded_digest
from presentation_pipeline.understanding.models import (
    ChunkDigest,
    DigestFragment,
    DocumentDigest,
    KeyFact,
    TopicDigest,
)
from presentation_pipeline.understanding.prompts import (
    DIGEST_REDUCTION_PROMPT,
    build_digest_reduction_input,
)
from presentation_pipeline.understanding.reduction import (
    DigestProvenanceError,
    ReductionInvariantError,
    reduce_chunk_digests,
    validate_digest_scope,
)
from presentation_pipeline.understanding.service import generate_document_digest


def _chunk(ordinal: int, *, summary: str | None = None) -> ChunkDigest:
    return ChunkDigest(
        doc_id="doc-a",
        window_id=f"window-{ordinal:04d}",
        summary=summary or f"Summary {ordinal}",
        key_facts=[
            KeyFact(
                claim=f"Fact {ordinal}",
                evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=[f"e-{ordinal}"])],
            )
        ],
    )


def test_scope_validation_rejects_new_evidence() -> None:
    fragment = DigestFragment(
        doc_id="doc-a",
        fragment_id="f",
        summary="s",
        key_facts=[
            KeyFact(
                claim="x",
                evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=["missing"])],
            )
        ],
    )
    with pytest.raises(DigestProvenanceError, match="outside"):
        validate_digest_scope(fragment, doc_id="doc-a", allowed_evidence_ids={"known"})


def test_single_chunk_is_reduced_to_document_digest_with_original_references() -> None:
    async def invoke(payload, response_model):
        assert payload["fragments"][0]["key_facts"][0]["evidence"][0]["evidence_ids"] == ["e-0"]
        return DocumentDigest(
            doc_id="doc-a",
            summary="final",
            key_facts=[
                KeyFact(
                    claim="Fact",
                    evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=["e-0"])],
                )
            ],
        )

    digest = asyncio.run(
        reduce_chunk_digests(
            [_chunk(0)],
            token_counter=CharacterTokenEstimator(),
            budget=InputBudget(max_input_tokens=4_000),
            invoke=invoke,
        )
    )
    assert digest.evidence_refs()[0].evidence_ids == ["e-0"]


def test_reduction_recurses_in_deterministic_groups_and_carries_singleton() -> None:
    class Counter:
        def count_text(self, text):
            return 1

        def count_payload(self, payload):
            return 100 * len(payload.get("fragments", []))

    calls = []

    async def invoke(payload, response_model):
        calls.append((response_model, [fragment["fragment_id"] for fragment in payload["fragments"]]))
        refs = [
            evidence_id
            for fragment in payload["fragments"]
            for fact in fragment["key_facts"]
            for ref in fact["evidence"]
            for evidence_id in ref["evidence_ids"]
        ]
        if response_model is DigestFragment:
            return DigestFragment(
                doc_id="doc-a",
                fragment_id="provider-id",
                summary="merged",
                key_facts=[
                    KeyFact(
                        claim="merged",
                        evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=refs)],
                    )
                ],
            )
        return DocumentDigest(
            doc_id="doc-a",
            summary="final",
            key_facts=[
                KeyFact(
                    claim="final",
                    evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=refs)],
                )
            ],
        )

    result = asyncio.run(
        reduce_chunk_digests(
            [_chunk(0), _chunk(1), _chunk(2)],
            token_counter=Counter(),
            budget=InputBudget(max_input_tokens=250),
            invoke=invoke,
        )
    )
    assert [response for response, _ in calls] == [DigestFragment, DocumentDigest]
    assert calls[0][1] == ["fragment-window-0000", "fragment-window-0001"]
    # The third singleton was carried forward, so the final request contains
    # the merged first group and the carried singleton.
    assert calls[1][1] == ["fragment-l1-0000", "fragment-l1-0001"]
    assert {
        evidence_id for ref in result.evidence_refs() for evidence_id in ref.evidence_ids
    } == {"e-0", "e-1", "e-2"}


def test_unbounded_direct_fragments_fail_as_application_invariant() -> None:
    class Counter:
        def count_text(self, text):
            return 0

        def count_payload(self, payload):
            if "summary" in payload:
                return len(payload["summary"])
            return sum(len(fragment.get("summary", "")) for fragment in payload.get("fragments", []))

    with pytest.raises(ReductionInvariantError) as raised:
        asyncio.run(
            reduce_chunk_digests(
                [_chunk(0, summary="x" * 10), _chunk(1, summary="x" * 10)],
                token_counter=Counter(),
                budget=InputBudget(max_input_tokens=11),
                invoke=lambda *_args, **_kwargs: None,  # type: ignore[arg-type]
            )
        )
    assert raised.value.metadata["reason"] == "bounded_fragments_not_pairable"


def _huge_digest() -> ChunkDigest:
    refs = [EvidenceRef(doc_id="doc-a", evidence_ids=[f"e-{i}-{j}" for j in range(8)]) for i in range(3)]
    return ChunkDigest(
        doc_id="doc-a",
        window_id="window-0000",
        summary="這是一段非常長的摘要。" * 600,
        topics=[
            TopicDigest(
                topic=("非常長的主題" * 30) + str(i),
                summary="非常長的主題摘要內容。" * 120,
                evidence=refs,
            )
            for i in range(20)
        ],
        key_facts=[
            KeyFact(
                claim=("非常重要但是很長的事實描述。" * 100) + str(i),
                evidence=refs,
            )
            for i in range(30)
        ],
    )


def test_generated_digest_is_deterministically_bounded_without_provider_retry() -> None:
    counter = Utf8ByteTokenEstimator()
    contract = DigestOutputContract(max_transport_tokens=7_000)
    bounded = bound_digest_to_contract(
        _huge_digest(), contract=contract, token_counter=counter, stage="test"
    )
    assert estimate_digest_transport_tokens(bounded, counter) <= contract.max_transport_tokens
    assert len(bounded.topics) <= contract.max_topics
    assert len(bounded.key_facts) <= contract.max_key_facts
    assert bounded.doc_id == "doc-a"
    assert bounded.window_id == "window-0000"
    assert all(ref.doc_id == "doc-a" for ref in bounded.evidence_refs())


def test_two_bounded_fragments_fit_real_default_reduction_request() -> None:
    counter = Utf8ByteTokenEstimator()
    budget = UnderstandingBudgets().reduction
    contract = output_contract_for_reduction_budget(budget)

    first_chunk = bound_digest_to_contract(
        _huge_digest(), contract=contract, token_counter=counter, stage="test"
    )
    second_chunk = first_chunk.model_copy(update={"window_id": "window-0001"})
    fragments = [
        DigestFragment(
            doc_id=chunk.doc_id,
            fragment_id=f"f-{i}",
            summary=chunk.summary,
            topics=chunk.topics,
            key_facts=chunk.key_facts,
        )
        for i, chunk in enumerate((first_chunk, second_chunk))
    ]
    payload = build_digest_reduction_input(
        doc_id="doc-a",
        level=0,
        fragments=[fragment.model_dump(mode="json") for fragment in fragments],
    )
    estimate = estimate_request_tokens(DIGEST_REDUCTION_PROMPT, payload, counter)
    assert estimate <= budget.target_input_tokens_or_usable


def test_document_generation_bounds_verbose_provider_output_without_extra_shrink_call() -> None:
    calls = []

    class Generator:
        async def generate(self, *, system_prompt, input_data, response_model):
            calls.append(response_model)
            if response_model is ChunkDigest:
                huge = _huge_digest()
                return huge.model_copy(update={"window_id": input_data["window"]["window_id"]})
            assert response_model is DocumentDigest
            refs = []
            for fragment in input_data["fragments"]:
                for field in ("topics", "key_facts"):
                    for item in fragment.get(field, []):
                        for ref in item.get("evidence", []):
                            refs.extend(ref.get("evidence_ids", []))
            refs = list(dict.fromkeys(refs)) or ["e-0-0"]
            evidence = [EvidenceRef(doc_id="doc-a", evidence_ids=refs[:4])]
            return DocumentDigest(
                doc_id="doc-a",
                summary="最後摘要" * 1000,
                topics=[TopicDigest(topic="主題" * 100, summary="摘要" * 1000, evidence=evidence) for _ in range(20)],
                key_facts=[KeyFact(claim="事實" * 1000, evidence=evidence) for _ in range(30)],
            )

    artifact = SimpleNamespace(doc_id="doc-a", filename="doc-a.docx")
    index = SimpleNamespace(
        doc_id="doc-a",
        evidence=[
            SimpleNamespace(
                evidence_id=f"e-{i}-{j}",
                kind="text",
                section_ids=[],
                text="content",
                structured_data={},
            )
            for i in range(3)
            for j in range(8)
        ],
        sections=[],
    )
    result = asyncio.run(
        generate_document_digest(
            artifact,
            index,
            Generator(),
            token_counter=CharacterTokenEstimator(),
            budgets=None,
        )
    )
    assert result.doc_id == "doc-a"
    # One chunk call + one final reduction call. No compaction/repair provider call.
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



def test_bounded_generation_canonicalizes_provider_identity_and_reference_doc_ids() -> None:
    class Generator:
        async def generate(self, *, system_prompt, input_data, response_model):
            del system_prompt, input_data, response_model
            return ChunkDigest(
                doc_id="provider-made-up-doc",
                window_id="provider-made-up-window",
                summary="summary",
                topics=[
                    TopicDigest(
                        topic="topic",
                        summary="summary",
                        evidence=[
                            EvidenceRef(
                                doc_id="provider-made-up-doc",
                                evidence_ids=["e-valid"],
                            )
                        ],
                    )
                ],
            )

    from presentation_pipeline.budgeting import GenerationLimiter

    limiter = GenerationLimiter(
        Generator(),
        token_counter=CharacterTokenEstimator(),
        concurrency=1,
    )
    result = asyncio.run(
        generate_bounded_digest(
            limiter=limiter,
            system_prompt="prompt",
            input_data={"document": {"doc_id": "doc-a"}},
            response_model=ChunkDigest,
            input_budget=InputBudget(max_input_tokens=4_000),
            output_contract=DigestOutputContract(max_transport_tokens=2_000),
            stage="test_chunk",
            expected_doc_id="doc-a",
            expected_window_id="window-0000",
            allowed_evidence_ids={"e-valid"},
        )
    )
    assert result.doc_id == "doc-a"
    assert result.window_id == "window-0000"
    assert result.evidence_refs()[0].doc_id == "doc-a"
    assert result.evidence_refs()[0].evidence_ids == ["e-valid"]


def test_bounded_generation_drops_out_of_scope_provenance_without_inventing_support() -> None:
    class Generator:
        async def generate(self, *, system_prompt, input_data, response_model):
            del system_prompt, input_data, response_model
            return ChunkDigest(
                doc_id="wrong-doc",
                window_id="wrong-window",
                summary="summary",
                topics=[
                    TopicDigest(
                        topic="partly supported",
                        summary="summary",
                        evidence=[
                            EvidenceRef(
                                doc_id="wrong-doc",
                                evidence_ids=["e-valid", "e-hallucinated"],
                            )
                        ],
                    ),
                    TopicDigest(
                        topic="unsupported",
                        summary="summary",
                        evidence=[
                            EvidenceRef(
                                doc_id="wrong-doc",
                                evidence_ids=["e-hallucinated-only"],
                            )
                        ],
                    ),
                ],
                key_facts=[
                    KeyFact(
                        claim="unsupported fact",
                        evidence=[
                            EvidenceRef(
                                doc_id="wrong-doc",
                                evidence_ids=["e-other"],
                            )
                        ],
                    )
                ],
            )

    from presentation_pipeline.budgeting import GenerationLimiter

    limiter = GenerationLimiter(
        Generator(),
        token_counter=CharacterTokenEstimator(),
        concurrency=1,
    )
    result = asyncio.run(
        generate_bounded_digest(
            limiter=limiter,
            system_prompt="prompt",
            input_data={"document": {"doc_id": "doc-a"}},
            response_model=ChunkDigest,
            input_budget=InputBudget(max_input_tokens=4_000),
            output_contract=DigestOutputContract(max_transport_tokens=2_000),
            stage="test_chunk",
            expected_doc_id="doc-a",
            expected_window_id="window-0000",
            allowed_evidence_ids={"e-valid"},
        )
    )
    assert len(result.topics) == 1
    assert result.topics[0].topic == "partly supported"
    assert result.topics[0].evidence[0].doc_id == "doc-a"
    assert result.topics[0].evidence[0].evidence_ids == ["e-valid"]
    assert result.key_facts == []
