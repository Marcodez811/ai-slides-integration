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
from presentation_pipeline.planning.models import EvidenceRef
from presentation_pipeline.understanding.models import DocumentDigest, KeyFact, TopicDigest


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


def test_candidate_dedup_keeps_distinct_transport_slices_in_source_order() -> None:
    from presentation_pipeline.retrieval.windowed import _dedupe_candidates
    first = CandidateEvidence(doc_id="a", evidence_id="e-a-0", reason="first").with_transport_content(
        {"evidence_id": "e-a-0", "kind": "list", "text": "first", "content": {"items": [{"text": "first"}]}, "slice": {"slice_id": "slice-0001", "index": 1, "count": 2}}
    )
    second = CandidateEvidence(doc_id="a", evidence_id="e-a-0", reason="second").with_transport_content(
        {"evidence_id": "e-a-0", "kind": "list", "text": "zero", "content": {"items": [{"text": "zero"}]}, "slice": {"slice_id": "slice-0000", "index": 0, "count": 2}}
    )
    result = _dedupe_candidates([first, second, first])
    assert len(result) == 1
    assert [content["slice"]["slice_id"] for content in result[0].transport_contents()] == ["slice-0000", "slice-0001"]
    assert result[0].merged_transport_content()["content"]["items"] == [{"text": "zero"}, {"text": "first"}]

    requirements = PresentationRequirements(goal="brief", audience="team", target_slide_count=1)
    payload = __import__("presentation_pipeline.planning.prompts", fromlist=["build_evidence_selection_input"]).build_evidence_selection_input(
        [DocumentDigest(doc_id="a", summary="digest")], CandidateEvidenceSet(candidates=result), requirements, [_index("a")]
    )
    compact = payload["candidate_evidence"][0]
    assert "text" not in compact and "content" not in compact
    assert [slice_["text"] for slice_ in compact["transport_slices"]] == ["zero", "first"]

    repeated = CandidateEvidence(doc_id="a", evidence_id="e-a-1", reason="repeat").with_transport_contents([
        {"evidence_id": "e-a-1", "kind": "list", "content": {"items": [{"text": "same"}]}, "slice": {"slice_id": "slice-0000", "index": 0}},
        {"evidence_id": "e-a-1", "kind": "list", "content": {"items": [{"text": "same"}]}, "slice": {"slice_id": "slice-0001", "index": 1}},
    ])
    assert repeated.merged_transport_content()["content"]["items"] == [{"text": "same"}, {"text": "same"}]


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


def test_transport_trimming_fails_loudly_for_first_nonfitting_slice_and_keeps_later_candidate() -> None:
    requirements = PresentationRequirements(goal="brief", audience="team", target_slide_count=1)
    budget = RetrievalBudget(
        request_budget=InputBudget(2_000), reduction_budget=InputBudget(1_000),
        selection_budget=InputBudget(1_000), max_candidates_per_window=2, max_global_candidates=2,
    )
    retriever = WindowedLLMEvidenceRetriever(
        _CandidateGenerator(), token_counter=Utf8ByteTokenEstimator(), budget=budget, concurrency=1
    )
    giant = CandidateEvidence(doc_id="a", evidence_id="e-a-0", reason="giant").with_transport_contents([
        {"evidence_id": "e-a-0", "kind": "text", "text": "x" * 5_000, "slice": {"slice_id": "slice-0000", "index": 0}},
        {"evidence_id": "e-a-0", "kind": "text", "text": "small", "slice": {"slice_id": "slice-0001", "index": 1}},
    ])
    with pytest.raises(CandidateRetrievalError, match="one candidate transport slice"):
        retriever._fit_candidate_transport(giant, [_index("a")], requirements)

    first = CandidateEvidence(doc_id="a", evidence_id="e-a-0", reason="first").with_transport_contents([
        {"evidence_id": "e-a-0", "kind": "text", "text": "one", "slice": {"slice_id": "slice-0000", "index": 0}},
        {"evidence_id": "e-a-0", "kind": "text", "text": "x" * 5_000, "slice": {"slice_id": "slice-0001", "index": 1}},
    ])
    later = CandidateEvidence(doc_id="a", evidence_id="e-a-1", reason="later").with_transport_content(
        {"evidence_id": "e-a-1", "kind": "text", "text": "later"}
    )
    retained = retriever._fit_selection_transport(
        [first, later], [_index("a")], [DocumentDigest(doc_id="a", summary="digest")], requirements
    )
    assert [candidate.evidence_id for candidate in retained] == ["e-a-0", "e-a-1"]
    assert [item["text"] for item in retained[0].transport_contents()] == ["one"]


def test_candidate_discovery_digest_context_strips_global_evidence_ids() -> None:
    from presentation_pipeline.retrieval.windowed import build_local_candidate_input
    from presentation_pipeline.understanding.windows import EvidenceWindow

    digest = DocumentDigest(
        doc_id="a",
        summary="whole-document context",
        topics=[
            TopicDigest(
                topic="global topic",
                summary="topic summary",
                evidence=[EvidenceRef(doc_id="a", evidence_ids=["outside-topic-id"])],
            )
        ],
        key_facts=[
            KeyFact(
                claim="global fact",
                evidence=[EvidenceRef(doc_id="a", evidence_ids=["outside-fact-id"])],
            )
        ],
    )
    window = EvidenceWindow(
        doc_id="a", window_id="window-0000", ordinal=0,
        evidence_ids=("e-a-0",),
        payload={"evidence": [{"evidence_id": "e-a-0", "kind": "text", "text": "local"}]},
        estimated_tokens=1,
    )
    payload = build_local_candidate_input(
        PresentationRequirements(goal="brief", audience="team", target_slide_count=1),
        digest, window,
    )
    rendered = repr(payload["document_digest"])
    assert "whole-document context" in rendered
    assert "global topic" in rendered
    assert "global fact" in rendered
    assert "outside-topic-id" not in rendered
    assert "outside-fact-id" not in rendered
    assert payload["evidence"][0]["evidence_id"] == "e-a-0"


def test_local_candidate_response_filters_out_of_scope_but_keeps_valid() -> None:
    candidates = [
        {"doc_id": "a", "evidence_id": "outside", "reason": "bad"},
        {"doc_id": "a", "evidence_id": "e-a-0", "reason": "good"},
    ]
    budget = RetrievalBudget(
        request_budget=InputBudget(2_000), reduction_budget=InputBudget(2_000),
        selection_budget=InputBudget(2_000), max_candidates_per_window=2,
        max_global_candidates=2,
    )
    retriever = WindowedLLMEvidenceRetriever(
        _InvalidCandidateGenerator(candidates), token_counter=Utf8ByteTokenEstimator(),
        budget=budget, concurrency=1,
    )
    result = asyncio.run(
        retriever.retrieve(
            [_index("a")], [DocumentDigest(doc_id="a", summary="digest")],
            PresentationRequirements(goal="brief", audience="team", target_slide_count=1),
        )
    )
    assert [candidate.evidence_id for candidate in result.candidates] == ["e-a-0"]


def test_local_candidate_response_caps_excess_without_failing() -> None:
    candidates = [
        {"doc_id": "a", "evidence_id": "e-a-0", "reason": "one"},
        {"doc_id": "a", "evidence_id": "e-a-1", "reason": "two"},
    ]
    budget = RetrievalBudget(
        request_budget=InputBudget(2_000), reduction_budget=InputBudget(2_000),
        selection_budget=InputBudget(2_000), max_candidates_per_window=1,
        max_global_candidates=2,
    )
    retriever = WindowedLLMEvidenceRetriever(
        _InvalidCandidateGenerator(candidates), token_counter=Utf8ByteTokenEstimator(),
        budget=budget, concurrency=1,
    )
    result = asyncio.run(
        retriever.retrieve(
            [_index("a")], [DocumentDigest(doc_id="a", summary="digest")],
            PresentationRequirements(goal="brief", audience="team", target_slide_count=1),
        )
    )
    assert [candidate.evidence_id for candidate in result.candidates] == ["e-a-0"]


def test_all_invalid_local_candidates_fail_only_after_discovery_finishes() -> None:
    budget = RetrievalBudget(
        request_budget=InputBudget(2_000), reduction_budget=InputBudget(2_000),
        selection_budget=InputBudget(2_000), max_candidates_per_window=2,
        max_global_candidates=2,
    )
    retriever = WindowedLLMEvidenceRetriever(
        _InvalidCandidateGenerator([
            {"doc_id": "a", "evidence_id": "outside", "reason": "bad"}
        ]),
        token_counter=Utf8ByteTokenEstimator(), budget=budget, concurrency=1,
    )
    with pytest.raises(CandidateRetrievalError, match="returned no evidence"):
        asyncio.run(
            retriever.retrieve(
                [_index("a")], [DocumentDigest(doc_id="a", summary="digest")],
                PresentationRequirements(goal="brief", audience="team", target_slide_count=1),
            )
        )
