import asyncio

import pytest

from presentation_pipeline.budgeting import CharacterTokenEstimator, GenerationLimiter, InputBudget, estimate_request_tokens
from presentation_pipeline.common.references import EvidenceRef
from presentation_pipeline.understanding.contracts import DigestOutputContract, DigestOutputContractError, digest_output_contract, estimate_digest_transport_tokens, validate_digest_contract
from presentation_pipeline.understanding.generation import generate_bounded_digest
from presentation_pipeline.understanding.models import ChunkDigest, DigestFragment, KeyFact, TopicDigest
from presentation_pipeline.understanding.prompts import DIGEST_REDUCTION_PROMPT, build_digest_reduction_input


def _chunk(summary="ok", evidence="e-1"):
    return ChunkDigest(doc_id="doc", window_id="w", summary=summary,
        key_facts=[KeyFact(claim="fact", evidence=[EvidenceRef(doc_id="doc", evidence_ids=[evidence])])])


class _Generator:
    def __init__(self, responses): self.responses, self.calls = iter(responses), []
    async def generate(self, **kwargs): self.calls.append(kwargs); return next(self.responses)


def _run(responses, **kwargs):
    generator = _Generator(responses)
    value = asyncio.run(generate_bounded_digest(
        limiter=GenerationLimiter(generator, token_counter=CharacterTokenEstimator(chars_per_token=1)),
        system_prompt="digest", input_data={"window": {"source": "must-not-repair"}},
        response_model=ChunkDigest, stage="document_chunk_digest", input_budget=InputBudget(10_000),
        output_contract=DigestOutputContract(300), allowed_evidence_ids={"e-1"},
        expected_doc_id="doc", expected_window_id="w", **kwargs))
    return value, generator


def test_bounded_digest_valid_initial_and_provider_contract() -> None:
    value, generator = _run([_chunk()])
    assert value.summary == "ok" and len(generator.calls) == 1
    contract = generator.calls[0]["input_data"]["output_contract"]
    assert contract == {"max_transport_tokens_estimate": 300, "max_topics": 8, "max_key_facts": 12, "priority": "retain only presentation-relevant information"}


def test_oversized_digest_repairs_once_without_source_payload() -> None:
    value, generator = _run([_chunk("x" * 500), _chunk("short")])
    assert value.summary == "short" and len(generator.calls) == 2
    assert set(generator.calls[1]["input_data"]) == {"digest", "output_contract"}
    assert "window" not in generator.calls[1]["input_data"]


def test_invalid_initial_and_repair_fail_after_exactly_two_calls() -> None:
    generator = _Generator([_chunk("x" * 500), _chunk("y" * 500)])
    with pytest.raises(DigestOutputContractError) as raised:
        asyncio.run(generate_bounded_digest(
            limiter=GenerationLimiter(generator, token_counter=CharacterTokenEstimator(chars_per_token=1)),
            system_prompt="digest", input_data={"window": {"source": "must-not-repair"}},
            response_model=ChunkDigest, stage="document_chunk_digest", input_budget=InputBudget(10_000),
            output_contract=DigestOutputContract(300), allowed_evidence_ids={"e-1"},
            expected_doc_id="doc", expected_window_id="w"))
    assert raised.value.metadata["reason"] == "repair_failed_contract"
    assert raised.value.metadata["original_reason"] == "transport_tokens_exceeded"
    assert len(generator.calls) == 2


@pytest.mark.parametrize("invalid", [_chunk(evidence="invented"), ChunkDigest(doc_id="wrong", window_id="w", summary="ok")])
def test_identity_and_provenance_are_repaired_but_original_scope_stays_strict(invalid) -> None:
    value, generator = _run([invalid, _chunk()])
    assert value.doc_id == "doc" and len(generator.calls) == 2


def test_repair_cannot_adopt_invented_provenance_from_invalid_output() -> None:
    generator = _Generator([_chunk(evidence="invented"), _chunk(evidence="invented")])
    with pytest.raises(DigestOutputContractError) as raised:
        asyncio.run(generate_bounded_digest(
            limiter=GenerationLimiter(generator, token_counter=CharacterTokenEstimator(chars_per_token=1)),
            system_prompt="digest", input_data={"window": {"source": "never resend"}}, response_model=ChunkDigest,
            stage="document_chunk_digest", input_budget=InputBudget(10_000), output_contract=DigestOutputContract(300),
            allowed_evidence_ids={"e-1"}, expected_doc_id="doc", expected_window_id="w"))
    assert raised.value.metadata["reason"] == "repair_failed_contract"
    assert raised.value.metadata["original_reason"] == "provenance_outside_scope"
    assert len(generator.calls) == 2


def test_complete_serialized_topic_fact_contract_enforcement() -> None:
    digest = _chunk()
    digest = digest.model_copy(update={"topics": [TopicDigest(topic=str(i), summary="s", evidence=[EvidenceRef(doc_id="doc", evidence_ids=["e-1"])]) for i in range(2)], "key_facts": digest.key_facts * 2})
    with pytest.raises(DigestOutputContractError, match="topic_count"):
        validate_digest_contract(digest, contract=DigestOutputContract(10_000, max_topics=1, max_key_facts=9), token_counter=CharacterTokenEstimator(), stage="x")
    with pytest.raises(DigestOutputContractError, match="key_fact_count"):
        validate_digest_contract(digest, contract=DigestOutputContract(10_000, max_topics=9, max_key_facts=1), token_counter=CharacterTokenEstimator(), stage="x")


def test_two_near_contract_fragments_fit_real_reduction_prompt() -> None:
    counter = CharacterTokenEstimator(chars_per_token=1)
    budget = InputBudget(4_000)
    contract = digest_output_contract(budget)
    fragments = []
    for index in range(2):
        summary = "x"
        while estimate_digest_transport_tokens(DigestFragment(doc_id="doc", fragment_id=str(index), summary=summary, key_facts=[]), counter) < contract.max_transport_tokens - 5:
            summary += "x"
        fragment = DigestFragment(doc_id="doc", fragment_id=str(index), summary=summary, key_facts=[])
        assert estimate_digest_transport_tokens(fragment, counter) >= contract.max_transport_tokens - 5
        fragments.append(fragment)
    payload = build_digest_reduction_input(doc_id="doc", level=0, fragments=[item.model_dump(mode="json") for item in fragments])
    production_prompt = DIGEST_REDUCTION_PROMPT + f"\nLIMIT:{contract.max_transport_tokens}/{contract.max_topics}/{contract.max_key_facts}"
    assert estimate_request_tokens(production_prompt, payload, counter) <= budget.target_input_tokens_or_usable


def test_intermediate_and_final_reduction_contracts_are_provider_visible() -> None:
    from types import SimpleNamespace
    from presentation_pipeline.understanding.service import generate_document_digest

    calls = []
    class Generator:
        async def generate(self, *, input_data, response_model, system_prompt):
            calls.append((response_model, input_data, system_prompt))
            if response_model is ChunkDigest:
                return ChunkDigest(doc_id="doc", window_id=input_data["window"]["window_id"], summary="chunk", key_facts=[KeyFact(claim="fact", evidence=[EvidenceRef(doc_id="doc", evidence_ids=["e-1"])])])
            if response_model is DigestFragment:
                return DigestFragment(doc_id="doc", fragment_id="provider", summary="merged", key_facts=[KeyFact(claim="fact", evidence=[EvidenceRef(doc_id="doc", evidence_ids=["e-1"])])])
            return __import__("presentation_pipeline.understanding.models", fromlist=["DocumentDigest"]).DocumentDigest(doc_id="doc", summary="final", key_facts=[KeyFact(claim="fact", evidence=[EvidenceRef(doc_id="doc", evidence_ids=["e-1"])])])
    artifact = SimpleNamespace(doc_id="doc", filename="doc.docx")
    index = SimpleNamespace(doc_id="doc", sections=[], evidence=[
        SimpleNamespace(evidence_id="e-1", kind="text", section_ids=[], text="first", structured_data={}),
        SimpleNamespace(evidence_id="e-2", kind="text", section_ids=[], text="second", structured_data={}),
    ])
    asyncio.run(generate_document_digest(artifact, index, Generator(), token_counter=CharacterTokenEstimator(chars_per_token=1)))
    chunk = next(call for call in calls if call[0] is ChunkDigest)
    assert "output_contract" in chunk[1]
    reduced = [call for call in calls if call[0] in {DigestFragment, __import__("presentation_pipeline.understanding.models", fromlist=["DocumentDigest"]).DocumentDigest}]
    assert reduced and all("LIMIT:" in call[2] for call in reduced)


def test_intermediate_fragment_repair_uses_compact_provider_limit() -> None:
    generator = _Generator([
        DigestFragment(doc_id="doc", fragment_id="bad", summary="x" * 500, key_facts=[]),
        DigestFragment(doc_id="doc", fragment_id="good", summary="short", key_facts=[]),
    ])
    value = asyncio.run(generate_bounded_digest(
        limiter=GenerationLimiter(generator, token_counter=CharacterTokenEstimator(chars_per_token=1)),
        system_prompt=DIGEST_REDUCTION_PROMPT, input_data={"fragments": []}, response_model=DigestFragment,
        stage="document_digest_reduction", input_budget=InputBudget(10_000), output_contract=DigestOutputContract(300),
        allowed_evidence_ids=set(), expected_doc_id="doc", include_contract_in_payload=False))
    assert value.fragment_id == "good" and len(generator.calls) == 2
    assert "LIMIT:300/8/12" in generator.calls[0]["system_prompt"]
