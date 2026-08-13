"""Offline scale regression: all planning requests remain bounded."""

from __future__ import annotations

import asyncio

from docx import Document
import pytest

from presentation_pipeline import PlanningScaleConfig, generate_plan
from presentation_pipeline.budgeting import (
    InputBudget,
    PlanningBudgets,
    RetrievalBudget,
    UnderstandingBudgets,
    Utf8ByteTokenEstimator,
    estimate_request_tokens,
)
from presentation_pipeline.planning import PresentationRequirements
from presentation_pipeline.common.references import EvidenceRef
from presentation_pipeline.planning.models import EvidenceSelection, PresentationOutline
from presentation_pipeline.planning.prompts import EVIDENCE_SELECTION_PROMPT
from presentation_pipeline.retrieval.models import (
    CandidateReductionSelection,
    LocalCandidateSelection,
)
from presentation_pipeline.understanding.models import (
    ChunkDigest,
    DigestFragment,
    DocumentDigest,
    KeyFact,
)


class _BoundedLargeCorpusGenerator:
    """A deterministic provider double that records every bounded request."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], type[object]]] = []
        self.active = 0
        self.maximum_active = 0

    async def generate(self, *, system_prompt, input_data, response_model):
        self.calls.append((system_prompt, input_data, response_model))
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        if response_model is ChunkDigest:
            item = input_data["evidence"][0]
            return {
                "doc_id": input_data["document"]["doc_id"],
                "window_id": input_data["window"]["window_id"],
                "summary": "bounded chunk",
                "key_facts": [{"claim": "source fact", "evidence": [{"doc_id": input_data["document"]["doc_id"], "evidence_ids": [item["evidence_id"]]}]}],
            }
        if response_model is DocumentDigest:
            fragment = input_data["fragments"][0]
            return {"doc_id": input_data["document"]["doc_id"], "summary": "bounded digest", "topics": [], "key_facts": fragment["key_facts"]}
        if response_model is DigestFragment:
            fragment = input_data["fragments"][0]
            return {"doc_id": input_data["document"]["doc_id"], "fragment_id": "reduced", "summary": "reduced", "topics": [], "key_facts": fragment["key_facts"]}
        if response_model is CandidateReductionSelection:
            return {
                "candidates": [
                    {
                        "doc_id": item["doc_id"],
                        "evidence_id": item["evidence_id"],
                    }
                    for item in input_data["candidate_descriptors"]
                ]
            }
        if response_model is LocalCandidateSelection:
            return {"candidates": [
                {"doc_id": input_data["window"]["doc_id"], "evidence_id": item["evidence_id"], "reason": "local"}
                for item in input_data["evidence"][:2]
            ]}
        if response_model is EvidenceSelection:
            item = input_data["candidate_evidence"][0]
            return {"selected": [{"doc_id": item["doc_id"], "evidence_id": item["evidence_id"], "reason": "selected"}]}
        if response_model is PresentationOutline:
            return {"title": "Bounded", "objective": "Brief", "narrative": "Source-backed", "sections": [{"section_id": "s", "title": "S", "purpose": "brief", "slides": [{"slide_id": "slide-1", "title": "Title", "purpose": "title", "message": "Bounded"}]}]}
        raise AssertionError(response_model)


def _scale() -> PlanningScaleConfig:
    return PlanningScaleConfig(
        token_counter=Utf8ByteTokenEstimator(),
        budgets=PlanningBudgets(
            understanding=UnderstandingBudgets(window=InputBudget(1_500), reduction=InputBudget(1_200)),
            retrieval=RetrievalBudget(
                request_budget=InputBudget(1_800), reduction_budget=InputBudget(1_800),
                selection_budget=InputBudget(1_800), max_candidates_per_window=2,
                max_global_candidates=2,
            ),
            # Outline transport includes the selected bounded evidence slice;
            # keep this test-stage cap above the retrieval-selection cap's
            # possible single-candidate payload while remaining finite.
            outline_generation=InputBudget(2_000),
        ),
    )


def test_large_synthetic_docx_uses_only_bounded_requests_and_reduces_candidates(tmp_path) -> None:
    source = tmp_path / "large.docx"
    document = Document()
    document.add_heading("Large corpus", level=1)
    table = document.add_table(rows=1, cols=3)
    for column, heading in zip(table.rows[0].cells, ("Policy", "Value", "Notes")):
        column.text = heading
    for number in range(80):
        cells = table.add_row().cells
        cells[0].text = f"Policy {number}"
        cells[1].text = f"Value {number}"
        cells[2].text = "oversized table evidence " * 12
    for number in range(24):
        document.add_paragraph(f"Finding {number}: " + ("evidence text " * 16))
    document.save(source)
    scale = _scale()
    generator = _BoundedLargeCorpusGenerator()
    result = asyncio.run(generate_plan(
        [source], PresentationRequirements(goal="Brief", audience="Leaders", target_slide_count=1),
        generator, extraction_workers=1, llm_concurrency=2, scale_config=scale,
    ))

    assert result.candidates is not None
    assert len(result.candidates.candidates) <= 2
    assert result.outline.all_slides()[0].slide_id == "slide-1"
    assert generator.maximum_active <= 2
    reduction_calls = [
        payload
        for _, payload, model in generator.calls
        if model is CandidateReductionSelection
    ]
    discovery_calls = [payload for _, payload, model in generator.calls if model is LocalCandidateSelection and "evidence" in payload]
    assert len(discovery_calls) > 1
    assert len(reduction_calls) >= 2  # at least two levels/groups of reduction
    assert all("candidate_descriptors" in payload for payload in reduction_calls)
    assert all(
        set(descriptor)
        == {"doc_id", "evidence_id", "kind", "reason", "score", "hit_count", "slice_count"}
        for payload in reduction_calls
        for descriptor in payload["candidate_descriptors"]
    )
    assert all("evidence_catalogue" not in payload for _, payload, _ in generator.calls)

    counter = scale.token_counter
    for prompt, payload, model in generator.calls:
        if model is ChunkDigest:
            budget = scale.budgets.understanding.window
        elif model in (DocumentDigest, DigestFragment):
            budget = scale.budgets.understanding.reduction
        elif model is CandidateReductionSelection:
            budget = scale.budgets.retrieval.reduction_budget
        elif model is LocalCandidateSelection:
            budget = scale.budgets.retrieval.request_budget
        elif model is EvidenceSelection:
            budget = scale.budgets.retrieval.selection_budget
        else:
            budget = scale.budgets.outline_generation
        assert estimate_request_tokens(prompt, payload, counter) <= budget.usable_input_tokens


class _AsymmetricHydrationGenerator:
    """Keep one identity per document through reduction and record planning inputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], type[object]]] = []

    async def generate(self, *, system_prompt, input_data, response_model):
        self.calls.append((system_prompt, input_data, response_model))
        if response_model is LocalCandidateSelection:
            return {
                "candidates": [
                    {
                        "doc_id": input_data["window"]["doc_id"],
                        "evidence_id": item["evidence_id"],
                        "reason": "local",
                    }
                    for item in input_data["evidence"]
                ]
            }
        if response_model is CandidateReductionSelection:
            descriptors = input_data["candidate_descriptors"]
            target = input_data["target_count"]
            chosen: list[dict[str, str]] = []
            represented: set[str] = set()
            for item in descriptors:
                if item["doc_id"] in represented:
                    continue
                chosen.append({"doc_id": item["doc_id"], "evidence_id": item["evidence_id"]})
                represented.add(item["doc_id"])
                if len(chosen) == target:
                    return {"candidates": chosen}
            for item in descriptors:
                identity = {"doc_id": item["doc_id"], "evidence_id": item["evidence_id"]}
                if identity in chosen:
                    continue
                chosen.append(identity)
                if len(chosen) == target:
                    break
            return {"candidates": chosen}
        if response_model is EvidenceSelection:
            return {
                "selected": [
                    {
                        "doc_id": item["doc_id"],
                        "evidence_id": item["evidence_id"],
                        "reason": "selected",
                    }
                    for item in input_data["candidate_evidence"]
                ]
            }
        if response_model is PresentationOutline:
            return {
                "title": "Bounded",
                "objective": "Brief",
                "narrative": "Source-backed",
                "sections": [
                    {
                        "section_id": "s",
                        "title": "S",
                        "purpose": "brief",
                        "slides": [
                            {
                                "slide_id": "slide-1",
                                "title": "Title",
                                "purpose": "title",
                                "message": "Bounded",
                            }
                        ],
                    }
                ],
            }
        raise AssertionError(response_model)


def _evidence_ids_in_documents(documents: list[dict[str, object]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for document in documents:
        doc_id = document["doc_id"]
        ids: set[str] = set()
        for field in ("topics", "key_facts"):
            for item in document.get(field, []):
                for reference in item.get("evidence", []):
                    ids.update(reference["evidence_ids"])
        result[doc_id] = ids
    return result


def test_asymmetric_two_document_hydration_keeps_bounded_transport_and_scoped_digests(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Digest scope avoids a below-cap reduction caused only by stale references."""
    sources = []
    for document_number in range(2):
        source = tmp_path / f"source-{document_number}.docx"
        document = Document()
        evidence_count = 2 if document_number == 0 else 6
        for evidence_number in range(evidence_count):
            document.add_paragraph(
                f"Document {document_number} evidence {evidence_number}: "
                + ("bounded source text " * 24)
            )
        document.save(source)
        sources.append(source)

    async def oversized_digests(_artifacts, indexes, *_args, **_kwargs):
        digests = []
        for index in indexes:
            evidence_ids = [item.evidence_id for item in index.evidence]
            digests.append(
                DocumentDigest(
                    doc_id=index.doc_id,
                    summary="digest",
                    key_facts=[
                        KeyFact(
                            claim="irrelevant digest detail " * 10 + str(number),
                            evidence=[EvidenceRef(
                                doc_id=index.doc_id,
                                evidence_ids=[evidence_ids[number % len(evidence_ids)]],
                            )],
                        )
                        for number in range(12)
                    ],
                )
            )
        return digests

    captured_events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr("presentation_pipeline.pipeline.generate_digests", oversized_digests)
    monkeypatch.setattr(
        "presentation_pipeline.retrieval.windowed.safe_event",
        lambda event, **fields: captured_events.append((event, fields)),
    )
    scale = PlanningScaleConfig(
        token_counter=Utf8ByteTokenEstimator(),
        budgets=PlanningBudgets(
            understanding=UnderstandingBudgets(window=InputBudget(10_000), reduction=InputBudget(10_000)),
            retrieval=RetrievalBudget(
                request_budget=InputBudget(10_000),
                reduction_budget=InputBudget(10_000),
                selection_budget=InputBudget(6_500),
                max_candidates_per_window=4,
                max_global_candidates=2,
            ),
            outline_generation=InputBudget(7_500),
        ),
    )
    generator = _AsymmetricHydrationGenerator()
    requirements = PresentationRequirements(goal="Brief", audience="Leaders", target_slide_count=1)

    result = asyncio.run(
        generate_plan(
            sources,
            requirements,
            generator,
            extraction_workers=1,
            llm_concurrency=2,
            scale_config=scale,
        )
    )

    assert result.candidates is not None
    final_candidates = result.candidates.candidates
    assert len(final_candidates) == 2
    assert len({item.doc_id for item in final_candidates}) == 2
    assert all(item.transport_contents() for item in final_candidates)

    reduction_calls = [
        payload for _, payload, model in generator.calls
        if model is CandidateReductionSelection
    ]
    # The asymmetric 2+6 local candidates become four, then two. A third call
    # would mean hydration leaked an unscoped digest and forced an unnecessary
    # post-cap pass.
    assert len(reduction_calls) == 2

    hydration = next(fields for event, fields in captured_events if event == "candidate_selection_hydration")
    assert hydration["input_candidate_count"] == 2
    assert hydration["hydrated_candidate_count"] == 2
    assert hydration["dropped_candidate_count"] == 0
    cap = next(fields for event, fields in captured_events if event == "candidate_over_limit_cap")
    assert cap["final_candidate_count"] == len(final_candidates)

    selection_call = next(
        (prompt, payload) for prompt, payload, model in generator.calls
        if model is EvidenceSelection
    )
    final_ids_by_document: dict[str, set[str]] = {}
    for candidate in final_candidates:
        final_ids_by_document.setdefault(candidate.doc_id, set()).add(candidate.evidence_id)
    assert _evidence_ids_in_documents(selection_call[1]["documents"]) == final_ids_by_document
    assert all(
        "text" in item or "transport_slices" in item
        for item in selection_call[1]["candidate_evidence"]
    )
    assert estimate_request_tokens(*selection_call, scale.token_counter) <= scale.budgets.retrieval.selection_budget.usable_input_tokens

    # The same payload with the intentionally oversized, unscoped digests is
    # over budget; digest scoping is what lets the already-capped set return.
    unscoped_selection = {
        "requirements": requirements.model_dump(mode="json"),
        "documents": [digest.model_dump(mode="json") for digest in result.digests],
        "candidate_evidence": selection_call[1]["candidate_evidence"],
    }
    assert estimate_request_tokens(
        EVIDENCE_SELECTION_PROMPT, unscoped_selection, scale.token_counter
    ) > scale.budgets.retrieval.selection_budget.usable_input_tokens

    outline_call = next(
        (prompt, payload) for prompt, payload, model in generator.calls
        if model is PresentationOutline
    )
    selected_ids_by_document: dict[str, set[str]] = {}
    for selected in result.selection.selected:
        selected_ids_by_document.setdefault(selected.doc_id, set()).add(selected.evidence_id)
    assert _evidence_ids_in_documents(outline_call[1]["documents"]) == selected_ids_by_document
    assert all(
        "text" in item or "transport_slices" in item
        for item in outline_call[1]["selected_evidence"]
    )
    assert estimate_request_tokens(*outline_call, scale.token_counter) <= scale.budgets.outline_generation.usable_input_tokens
