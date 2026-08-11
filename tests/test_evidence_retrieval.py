"""Offline contracts for bounded evidence candidate retrieval."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from presentation_pipeline.budgeting import InputBudget, RetrievalBudget, Utf8ByteTokenEstimator
from presentation_pipeline.planning.models import PresentationRequirements
from presentation_pipeline.retrieval import (
    CandidateEvidence,
    CandidateEvidenceSet,
    CandidateRetrievalError,
    LocalCandidateSelection,
    WindowedLLMEvidenceRetriever,
)
from presentation_pipeline.understanding.models import DocumentDigest


def _index(doc_id: str, count: int = 2) -> object:
    return SimpleNamespace(
        doc_id=doc_id, filename=f"{doc_id}.docx", sections=[],
        evidence=[SimpleNamespace(evidence_id=f"e-{doc_id}-{number}", kind="text", text=f"fact {number}", section_ids=[], structured_data={}) for number in range(count)],
    )


class _CandidateGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, *, input_data, response_model, **_kwargs):
        if response_model is LocalCandidateSelection:
            self.calls += 1
            evidence = input_data["evidence"]
            return {"candidates": [{"doc_id": input_data["window"]["doc_id"], "evidence_id": evidence[0]["evidence_id"], "reason": "relevant"}]}
        raise AssertionError(response_model)


class _InvalidCandidateGenerator:
    def __init__(self, candidates: list[dict[str, str]]) -> None:
        self._candidates = candidates

    async def generate(self, *, response_model, **_kwargs):
        assert response_model is LocalCandidateSelection
        return {"candidates": self._candidates}


def test_retrieval_deduplicates_in_source_order_and_rejects_out_of_scope() -> None:
    requirements = PresentationRequirements(goal="brief", audience="team", target_slide_count=1)
    budget = RetrievalBudget(
        request_budget=InputBudget(2_000), reduction_budget=InputBudget(2_000),
        selection_budget=InputBudget(2_000), max_candidates_per_window=2, max_global_candidates=2,
    )
    result = asyncio.run(WindowedLLMEvidenceRetriever(
        _CandidateGenerator(), token_counter=Utf8ByteTokenEstimator(), budget=budget, concurrency=1
    ).retrieve([_index("a")], [DocumentDigest(doc_id="a", summary="digest")], requirements))
    assert [item.evidence_id for item in result.candidates] == ["e-a-0"]

    with pytest.raises(CandidateRetrievalError, match="outside"):
        WindowedLLMEvidenceRetriever._validate_identities(
            [CandidateEvidence(doc_id="a", evidence_id="missing", reason="no")], {("a", "e-a-0")}, 2
        )


def test_global_selection_input_excludes_unrelated_document_digests() -> None:
    from presentation_pipeline.planning.models import EvidenceSelection, SelectedEvidence
    from presentation_pipeline.planning.prompts import (
        build_evidence_selection_input,
        build_outline_input,
    )

    candidate = CandidateEvidence(
        doc_id="a", evidence_id="e-a-0", reason="relevant"
    ).with_transport_content(
        {"evidence_id": "e-a-0", "kind": "text", "text": "bounded slice"}
    )
    candidates = CandidateEvidenceSet(candidates=[candidate])
    digests = [
        DocumentDigest(doc_id="a", summary="include"),
        DocumentDigest(doc_id="b", summary="exclude"),
    ]
    requirements = PresentationRequirements(
        goal="brief", audience="team", target_slide_count=1
    )
    payload = build_evidence_selection_input(
        digests,
        candidates,
        requirements,
        [_index("a"), _index("b")],
    )
    assert [item["doc_id"] for item in payload["documents"]] == ["a"]
    assert payload["candidate_evidence"][0]["text"] == "bounded slice"
    assert "evidence_catalogue" not in payload
    assert "transport" not in repr(candidate.model_dump(mode="json"))

    outline_payload = build_outline_input(
        digests,
        requirements,
        EvidenceSelection(
            selected=[
                SelectedEvidence(
                    doc_id="a", evidence_id="e-a-0", reason="selected"
                )
            ]
        ),
        [_index("a"), _index("b")],
        candidates,
    )
    assert [item["doc_id"] for item in outline_payload["documents"]] == ["a"]
    assert outline_payload["selected_evidence"][0]["text"] == "bounded slice"


def test_large_text_is_resliced_for_actual_discovery_overhead() -> None:
    generator = _CandidateGenerator()
    index = _index("a", count=1)
    index.evidence[0].text = "long source sentence. " * 300
    budget = RetrievalBudget(
        request_budget=InputBudget(1_800), reduction_budget=InputBudget(2_000),
        selection_budget=InputBudget(2_000), max_candidates_per_window=2,
        max_global_candidates=2,
    )
    result = asyncio.run(
        WindowedLLMEvidenceRetriever(
            generator,
            token_counter=Utf8ByteTokenEstimator(),
            budget=budget,
            concurrency=1,
        ).retrieve(
            [index],
            [DocumentDigest(doc_id="a", summary="digest context")],
            PresentationRequirements(goal="brief", audience="team", target_slide_count=1),
        )
    )
    assert generator.calls > 1
    assert [candidate.evidence_id for candidate in result.candidates] == ["e-a-0"]


@pytest.mark.parametrize(
    ("candidates", "match"),
    [
        ([{"doc_id": "a", "evidence_id": "outside", "reason": "bad"}], "outside"),
        ([
            {"doc_id": "a", "evidence_id": "e-a-0", "reason": "one"},
            {"doc_id": "a", "evidence_id": "e-a-1", "reason": "two"},
        ], "exceeds"),
    ],
)
def test_local_candidate_responses_reject_out_of_window_and_over_cap(candidates, match) -> None:
    budget = RetrievalBudget(
        request_budget=InputBudget(2_000), reduction_budget=InputBudget(2_000), selection_budget=InputBudget(2_000),
        max_candidates_per_window=1, max_global_candidates=2,
    )
    retriever = WindowedLLMEvidenceRetriever(
        _InvalidCandidateGenerator(candidates), token_counter=Utf8ByteTokenEstimator(), budget=budget, concurrency=1
    )
    with pytest.raises(CandidateRetrievalError, match=match):
        asyncio.run(retriever.retrieve([_index("a")], [DocumentDigest(doc_id="a", summary="digest")], PresentationRequirements(goal="brief", audience="team", target_slide_count=1)))
