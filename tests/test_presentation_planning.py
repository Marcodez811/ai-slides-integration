"""Contracts for provider-neutral, provenance-safe presentation planning."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from presentation_pipeline.pipeline import generate_outline, generate_plan
from presentation_pipeline.results import PresentationPlanningResult
from presentation_pipeline.planning.models import (
    EvidenceSelection,
    OutlineSection,
    PresentationOutline,
    PresentationRequirements,
    SelectedEvidence,
    SlideOutline,
    SlidePurpose,
)
from presentation_pipeline.planning.service import generate_presentation_outline
from presentation_pipeline.understanding.models import DocumentDigest, EvidenceRef, KeyFact
from presentation_pipeline.understanding.models import ChunkDigest
from presentation_pipeline.retrieval.models import LocalCandidateSelection, CandidateEvidence
from presentation_pipeline.understanding.prompts import build_document_digest_input
from presentation_pipeline.understanding.service import generate_digests
from presentation_pipeline.validation import (
    CorpusLookup,
    ProvenanceValidationError,
    validate_document_digests,
    validate_presentation_outline,
)


def _artifact(doc_id: str = "doc-a") -> SimpleNamespace:
    return SimpleNamespace(
        doc_id=doc_id,
        filename=f"{doc_id}.docx",
        extraction=SimpleNamespace(
            document=SimpleNamespace(doc_id=doc_id, filename=f"{doc_id}.docx"),
            views=SimpleNamespace(blocks=[SimpleNamespace(block_id=f"block-{doc_id}")]),
            nodes=[SimpleNamespace(node_id=f"node-{doc_id}")],
        ),
    )


def _index(doc_id: str = "doc-a", *, block_id: str | None = None, node_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        doc_id=doc_id,
        sections=[],
        evidence=[
            SimpleNamespace(
                evidence_id=f"evidence-{doc_id}",
                kind="text",
                block_ids=[block_id or f"block-{doc_id}"],
                source_node_ids=[node_id or f"node-{doc_id}"],
                text=f"Text for {doc_id}",
                asset_ids=[],
                section_ids=[],
                structured_data={"payload": {"content": f"Text for {doc_id}"}},
            )
        ],
    )


class _DigestGenerator:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0
        self.seen_inputs: list[dict[str, object]] = []

    async def generate(self, *, system_prompt, input_data, response_model):
        self.seen_inputs.append(input_data)
        if response_model in (DocumentDigest, ChunkDigest):
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            doc_id = input_data["document"]["doc_id"]
            await asyncio.sleep(0.02 if doc_id == "doc-a" else 0.001)
            self.active -= 1
            result = DocumentDigest(
                doc_id=doc_id,
                summary=f"Summary {doc_id}",
                key_facts=[
                    KeyFact(
                        claim=f"Fact {doc_id}",
                        evidence=[EvidenceRef(doc_id=doc_id, evidence_ids=[f"evidence-{doc_id}"])],
                    )
                ],
            )
            if response_model is ChunkDigest:
                return ChunkDigest(window_id=input_data["window"]["window_id"], **result.model_dump())
            return result
        raise AssertionError(response_model)


def test_digest_input_is_compact_and_does_not_mutate_extraction() -> None:
    artifact = _artifact()
    artifact.extraction.nodes[0].source = {"xml_path": "/very/private/xml"}
    artifact.extraction.views.blocks[0].source_node_ids = ["node-doc-a"]
    before = deepcopy(artifact)

    input_data = build_document_digest_input(artifact, _index())

    assert input_data["document"] == {"doc_id": "doc-a", "filename": "doc-a.docx"}
    assert "source_node_ids" not in repr(input_data)
    assert "xml_path" not in repr(input_data)
    assert artifact == before


def test_digest_input_keeps_structured_chart_content_without_provenance_internals() -> None:
    index = _index()
    item = index.evidence[0]
    item.kind = "chart_candidate"
    item.text = None
    item.structured_data = {
        "categories": ["Q1", "Q2"],
        "series": [{"name": "Revenue", "values": [100, 118]}],
        "source_node_ids": ["node-secret"],
        "table_node_id": "node-table",
        "relationship_id": "rId5",
    }

    input_data = build_document_digest_input(_artifact(), index)

    evidence = input_data["evidence"][0]
    assert evidence["content"] == {
        "categories": ["Q1", "Q2"],
        "series": [{"name": "Revenue", "values": [100, 118]}],
    }
    assert "source_node" not in repr(input_data)
    assert "relationship" not in repr(input_data)


def test_generate_digests_respects_concurrency_and_input_order() -> None:
    generator = _DigestGenerator()
    digests = asyncio.run(
        generate_digests(
            [_artifact("doc-a"), _artifact("doc-b")],
            [_index("doc-a"), _index("doc-b")],
            generator,
            concurrency=2,
        )
    )

    assert generator.maximum_active == 2
    assert [digest.doc_id for digest in digests] == ["doc-a", "doc-b"]
    with pytest.raises(ValueError, match="at least 1"):
        asyncio.run(generate_digests([], [], generator, concurrency=0))
    with pytest.raises(ValueError, match="integer"):
        asyncio.run(generate_digests([], [], generator, concurrency=True))


def test_unknown_document_and_evidence_references_are_rejected() -> None:
    lookup = CorpusLookup.from_artifacts_indexes([_artifact()], [_index()])
    unknown_document = DocumentDigest(
        doc_id="missing",
        summary="summary",
        key_facts=[KeyFact(claim="claim", evidence=[EvidenceRef(doc_id="missing", evidence_ids=["evidence-doc-a"])])],
    )
    unknown_evidence = DocumentDigest(
        doc_id="doc-a",
        summary="summary",
        key_facts=[KeyFact(claim="claim", evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=["missing"])])],
    )

    with pytest.raises(ProvenanceValidationError, match="unknown document"):
        validate_document_digests([unknown_document], lookup)
    with pytest.raises(ProvenanceValidationError, match="unknown evidence"):
        validate_document_digests([unknown_evidence], lookup)


def test_lookup_rejects_evidence_with_dangling_deep_provenance() -> None:
    with pytest.raises(ProvenanceValidationError, match="unknown block IDs"):
        CorpusLookup.from_artifacts_indexes([_artifact()], [_index(block_id="missing-block")])
    with pytest.raises(ProvenanceValidationError, match="unknown source node IDs"):
        CorpusLookup.from_artifacts_indexes([_artifact()], [_index(node_id="missing-node")])

    index = _index()
    index.evidence[0].asset_ids = ["asset-missing"]
    with pytest.raises(ProvenanceValidationError, match="unknown asset IDs"):
        CorpusLookup.from_artifacts_indexes([_artifact()], [index])


def test_digest_cannot_reference_another_documents_evidence() -> None:
    lookup = CorpusLookup.from_artifacts_indexes(
        [_artifact("doc-a"), _artifact("doc-b")],
        [_index("doc-a"), _index("doc-b")],
    )
    digest = DocumentDigest(
        doc_id="doc-a",
        summary="summary",
        key_facts=[
            KeyFact(
                claim="borrowed claim",
                evidence=[EvidenceRef(doc_id="doc-b", evidence_ids=["evidence-doc-b"])],
            )
        ],
    )

    with pytest.raises(ProvenanceValidationError, match="another document"):
        validate_document_digests([digest], lookup)


def test_content_and_summary_slides_require_evidence() -> None:
    lookup = CorpusLookup.from_artifacts_indexes([_artifact()], [_index()])
    outline = PresentationOutline(
        title="Title",
        objective="Objective",
        narrative="Narrative",
        sections=[
            OutlineSection(
                section_id="section-1",
                title="Section",
                purpose="Explain",
                slides=[
                    SlideOutline(
                        slide_id="slide-1",
                        title="Content",
                        purpose=SlidePurpose.CONTENT,
                        message="Message",
                    )
                ],
            )
        ],
    )
    with pytest.raises(ProvenanceValidationError, match="must include evidence"):
        validate_presentation_outline(outline, lookup)


class _PipelineGenerator:
    async def generate(self, *, system_prompt, input_data, response_model):
        if response_model in (DocumentDigest, ChunkDigest):
            doc_id = input_data["document"]["doc_id"]
            if response_model is ChunkDigest:
                return ChunkDigest(
                    doc_id=doc_id, window_id=input_data["window"]["window_id"], summary="A summary",
                    key_facts=[KeyFact(claim="A fact", evidence=[EvidenceRef(doc_id=doc_id, evidence_ids=[f"evidence-{doc_id}"])])],
                )
            return DocumentDigest(
                doc_id="doc-a",
                summary="A summary",
                key_facts=[KeyFact(claim="A fact", evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=["evidence-doc-a"])])],
            )
        if response_model is LocalCandidateSelection:
            evidence = input_data["evidence"][0]
            return LocalCandidateSelection(candidates=[CandidateEvidence(doc_id=input_data["window"]["doc_id"], evidence_id=evidence["evidence_id"], reason="central fact")])
        if response_model is EvidenceSelection:
            assert input_data["candidate_evidence"]
            return EvidenceSelection(
                selected=[SelectedEvidence(doc_id="doc-a", evidence_id="evidence-doc-a", reason="central fact")],
                strategy="Use the key fact",
            )
        if response_model is PresentationOutline:
            assert input_data["selected_evidence"][0]["evidence_id"] == "evidence-doc-a"
            return PresentationOutline(
                title="Deck",
                objective="Explain",
                narrative="Fact then close",
                sections=[
                    OutlineSection(
                        section_id="section-1",
                        title="Body",
                        purpose="Explain",
                        slides=[
                            SlideOutline(
                                slide_id="slide-1",
                                title="Fact",
                                purpose=SlidePurpose.CONTENT,
                                message="A fact",
                                evidence=[EvidenceRef(doc_id="doc-a", evidence_ids=["evidence-doc-a"])],
                            )
                        ],
                    )
                ],
            )
        raise AssertionError(response_model)


def test_end_to_end_orchestration_uses_deterministic_stages(monkeypatch) -> None:
    import presentation_pipeline.pipeline as pipeline

    artifact, index = _artifact(), _index()
    monkeypatch.setattr(pipeline, "build_jobs", lambda paths: list(paths))
    monkeypatch.setattr(
        pipeline,
        "extract_batch",
        lambda jobs, max_workers: SimpleNamespace(documents=[artifact], failures=[]),
    )
    monkeypatch.setattr(pipeline, "build_document_index", lambda item: index)

    outline = asyncio.run(
        generate_outline(
            ["ignored.docx"],
            PresentationRequirements(goal="Explain", audience="Team", target_slide_count=1),
            _PipelineGenerator(),
        )
    )
    assert outline.all_slides()[0].slide_id == "slide-1"

    with pytest.raises(ValueError, match="at least one input"):
        asyncio.run(
            generate_outline(
                [],
                PresentationRequirements(goal="Explain", audience="Team", target_slide_count=1),
                _PipelineGenerator(),
            )
        )


def test_generate_plan_preserves_order_and_reusable_intermediates(monkeypatch) -> None:
    import presentation_pipeline.pipeline as pipeline

    artifact_a, artifact_b = _artifact("doc-a"), _artifact("doc-b")
    index_a, index_b = _index("doc-a"), _index("doc-b")
    requirements = PresentationRequirements(goal="Explain", audience="Team", target_slide_count=1)
    outline = PresentationOutline(
        title="Deck",
        objective="Explain",
        narrative="Narrative",
        sections=[OutlineSection(section_id="s", title="S", purpose="P", slides=[SlideOutline(slide_id="slide-1", title="T", purpose=SlidePurpose.TITLE, message="M")])],
    )
    digests = [DocumentDigest(doc_id="doc-a", summary="A"), DocumentDigest(doc_id="doc-b", summary="B")]
    selection = EvidenceSelection(selected=[SelectedEvidence(doc_id="doc-a", evidence_id="evidence-doc-a", reason="A")])
    calls = {"extract": 0, "index": 0}
    monkeypatch.setattr(pipeline, "build_jobs", lambda paths: list(paths))
    def extract(jobs, max_workers):
        calls["extract"] += 1
        return SimpleNamespace(documents=[artifact_a, artifact_b], failures=[])
    def index(artifact):
        calls["index"] += 1
        return {"doc-a": index_a, "doc-b": index_b}[artifact.doc_id]
    async def generate_digests(*args, **kwargs):
        return digests
    async def retrieve(*args, **kwargs):
        from presentation_pipeline.retrieval.models import CandidateEvidenceSet, CandidateEvidence
        return CandidateEvidenceSet(candidates=[CandidateEvidence(doc_id="doc-a", evidence_id="evidence-doc-a", reason="A")])
    async def select(*args, **kwargs):
        return selection
    async def generate_outline_stage(*args, **kwargs):
        return outline
    monkeypatch.setattr(pipeline, "extract_batch", extract)
    monkeypatch.setattr(pipeline, "build_document_index", index)
    monkeypatch.setattr(pipeline, "generate_digests", generate_digests)
    monkeypatch.setattr(pipeline.WindowedLLMEvidenceRetriever, "retrieve", retrieve)
    monkeypatch.setattr(pipeline, "select_evidence", select)
    monkeypatch.setattr(pipeline, "generate_presentation_outline", generate_outline_stage)

    result = asyncio.run(generate_plan(["a.docx", "b.docx"], requirements, _PipelineGenerator()))

    assert isinstance(result, PresentationPlanningResult)
    assert result.requirements is requirements
    assert result.artifacts == (artifact_a, artifact_b)
    assert result.indexes == (index_a, index_b)
    assert result.digests == tuple(digests)
    assert result.selection is selection
    assert result.outline is outline
    assert calls == {"extract": 1, "index": 2}


def test_generate_outline_delegates_to_generate_plan_without_repeating_work(monkeypatch) -> None:
    import presentation_pipeline.pipeline as pipeline

    requirements = PresentationRequirements(goal="Explain", audience="Team", target_slide_count=1)
    artifact, index = _artifact(), _index()
    digest = DocumentDigest(doc_id="doc-a", summary="A")
    outline = PresentationOutline(
        title="Deck", objective="Explain", narrative="Narrative",
        sections=[OutlineSection(section_id="s", title="S", purpose="P", slides=[SlideOutline(slide_id="slide-1", title="T", purpose=SlidePurpose.TITLE, message="M")])],
    )
    result = PresentationPlanningResult(
        requirements=requirements,
        artifacts=(artifact,), indexes=(index,), digests=(digest,),
        selection=EvidenceSelection(selected=[SelectedEvidence(doc_id="doc-a", evidence_id="evidence-doc-a", reason="A")]),
        outline=outline,
    )
    calls = 0
    async def fake_generate_plan(*args, **kwargs):
        nonlocal calls
        calls += 1
        return result
    monkeypatch.setattr(pipeline, "generate_plan", fake_generate_plan)

    returned = asyncio.run(generate_outline(["source.docx"], requirements, _PipelineGenerator()))

    assert returned is outline
    assert calls == 1


@pytest.mark.parametrize(
    ("artifacts", "indexes", "digests", "match"),
    [
        ((), (), (), "identical nonzero lengths"),
        ((_artifact(),), (), (DocumentDigest(doc_id="doc-a", summary="A"),), "identical nonzero lengths"),
        ((_artifact(),), (_index("doc-b"),), (DocumentDigest(doc_id="doc-a", summary="A"),), "matching doc_id order"),
    ],
)
def test_planning_result_rejects_misaligned_document_stages(
    artifacts, indexes, digests, match: str
) -> None:
    requirements = PresentationRequirements(goal="Explain", audience="Team", target_slide_count=1)
    outline = PresentationOutline(
        title="Deck", objective="Explain", narrative="Narrative",
        sections=[OutlineSection(section_id="s", title="S", purpose="P", slides=[SlideOutline(slide_id="slide-1", title="T", purpose=SlidePurpose.TITLE, message="M")])],
    )
    selection = EvidenceSelection(
        selected=[SelectedEvidence(doc_id="doc-a", evidence_id="evidence-doc-a", reason="A")]
    )
    with pytest.raises(ValueError, match=match):
        PresentationPlanningResult(requirements, artifacts, indexes, digests, selection, outline)
