from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from presentation_pipeline.indexing.models import DocumentIndex, EvidenceItem, EvidenceKind
from presentation_pipeline.planning.models import (
    EvidenceSelection,
    OutlineSection,
    PresentationOutline,
    PresentationRequirements,
    SlideOutline,
    SlidePurpose,
)
from presentation_pipeline.results import PresentationPlanningResult
from presentation_pipeline.common.references import EvidenceRef
from presentation_pipeline.synthesis.context import (
    ResolvedEvidence,
    SlideContext,
    SlideContextResolutionError,
    build_slide_contexts,
)


def _item(evidence_id: str = "evidence-1") -> EvidenceItem:
    return EvidenceItem(
        doc_id="doc-1",
        evidence_id=evidence_id,
        kind=EvidenceKind.CHART,
        block_ids=["block-1"],
        source_node_ids=["node-1"],
        section_ids=["section-a"],
        asset_ids=["asset-1"],
        text="Revenue grew.",
        structured_data={
            "series": [{"label": "Revenue", "values": [10, 12]}],
            "source_node_id": "node-1",
            "nested": {"xml_path": "/word/document.xml", "label": "safe"},
        },
    )


def _plan(*, slides: list[SlideOutline], indexes: tuple[DocumentIndex, ...] | None = None, selection: EvidenceSelection | None = None) -> PresentationPlanningResult:
    plan_indexes = indexes or (
        DocumentIndex(
            doc_id="doc-1",
            filename="source.docx",
            extraction_schema_version="1",
            extractor_version="1",
            evidence=[_item()],
        ),
    )
    return PresentationPlanningResult(
        requirements=PresentationRequirements(goal="Explain", audience="Leaders", target_slide_count=len(slides)),
        artifacts=tuple(SimpleNamespace(doc_id=index.doc_id) for index in plan_indexes),  # type: ignore[arg-type]
        indexes=plan_indexes,
        digests=tuple(SimpleNamespace(doc_id=index.doc_id) for index in plan_indexes),  # type: ignore[arg-type]
        selection=selection or EvidenceSelection(selected=[{"doc_id": "doc-1", "evidence_id": "evidence-1", "reason": "Core finding"}]),
        outline=PresentationOutline(
            title="Growth",
            objective="Explain growth",
            narrative="Start with the result.",
            sections=[OutlineSection(section_id="section-1", title="Findings", purpose="Explain", slides=slides)],
        ),
    )


def _content_slide(*, references: list[EvidenceRef] | None = None) -> SlideOutline:
    return SlideOutline(
        slide_id="slide-1",
        title="Revenue",
        purpose=SlidePurpose.CONTENT,
        message="Revenue grew.",
        evidence=references if references is not None else [EvidenceRef(doc_id="doc-1", evidence_ids=["evidence-1"])],
    )


def test_build_slide_contexts_preserves_outline_order_and_only_semantic_evidence() -> None:
    title = SlideOutline(slide_id="slide-title", title="Growth", purpose=SlidePurpose.TITLE, message="Growth story")
    context = build_slide_contexts(_plan(slides=[title, _content_slide()]))

    assert [item.slide_index for item in context] == [1, 2]
    assert [item.total_slides for item in context] == [2, 2]
    assert context[1].presentation_title == "Growth"
    assert context[1].presentation_objective == "Explain growth"
    assert context[1].presentation_narrative == "Start with the result."
    assert (context[1].section_id, context[1].section_title, context[1].section_purpose) == (
        "section-1",
        "Findings",
        "Explain",
    )
    assert context[0].evidence == []
    evidence = context[1].evidence[0]
    assert evidence.selection_reason == "Core finding"
    assert evidence.section_ids == ["section-a"]
    assert evidence.asset_ids == ["asset-1"]
    assert evidence.structured_data == {"series": [{"label": "Revenue", "values": [10, 12]}], "nested": {"label": "safe"}}
    dumped = evidence.model_dump()
    assert "block_ids" not in dumped and "source_node_ids" not in dumped
    assert "xml_path" not in str(dumped) and "source_node_id" not in str(dumped)


@pytest.mark.parametrize("purpose", [SlidePurpose.CONTENT, SlidePurpose.SUMMARY])
def test_evidence_required_for_content_and_summary(purpose: SlidePurpose) -> None:
    slide = SlideOutline(slide_id="slide-1", title="Finding", purpose=purpose, message="A finding")
    with pytest.raises(SlideContextResolutionError, match="requires evidence"):
        build_slide_contexts(_plan(slides=[slide]))


def test_section_slide_is_permitted_without_evidence() -> None:
    slide = SlideOutline(slide_id="slide-1", title="Findings", purpose=SlidePurpose.SECTION, message="The findings")
    assert build_slide_contexts(_plan(slides=[slide]))[0].evidence == []


@pytest.mark.parametrize(
    ("references", "match"),
    [
        ([EvidenceRef(doc_id="missing", evidence_ids=["evidence-1"])], "unknown document"),
        ([EvidenceRef(doc_id="doc-1", evidence_ids=["missing"])], "unknown evidence"),
        ([EvidenceRef(doc_id="doc-1", evidence_ids=["evidence-1"]), EvidenceRef(doc_id="doc-1", evidence_ids=["evidence-1"])], "repeats evidence"),
    ],
)
def test_rejects_unknown_and_repeated_outline_evidence(references: list[EvidenceRef], match: str) -> None:
    with pytest.raises(SlideContextResolutionError, match=match):
        build_slide_contexts(_plan(slides=[_content_slide(references=references)]))


def test_rejects_unselected_outline_evidence() -> None:
    index = DocumentIndex(doc_id="doc-1", filename="source.docx", extraction_schema_version="1", extractor_version="1", evidence=[_item("selected"), _item("unselected")])
    selection = EvidenceSelection(selected=[{"doc_id": "doc-1", "evidence_id": "selected", "reason": "Core"}])
    slide = _content_slide(references=[EvidenceRef(doc_id="doc-1", evidence_ids=["unselected"])])
    with pytest.raises(SlideContextResolutionError, match="unselected evidence"):
        build_slide_contexts(_plan(slides=[slide], indexes=(index,), selection=selection))


def test_rejects_selected_evidence_missing_from_indexes_even_when_unused() -> None:
    selection = EvidenceSelection(selected=[{"doc_id": "doc-1", "evidence_id": "missing", "reason": "Core"}])
    title = SlideOutline(slide_id="slide-1", title="Growth", purpose=SlidePurpose.TITLE, message="Growth story")
    with pytest.raises(SlideContextResolutionError, match="selected evidence"):
        build_slide_contexts(_plan(slides=[title], selection=selection))


def test_rejects_duplicate_indexes_and_evidence() -> None:
    index = DocumentIndex(doc_id="doc-1", filename="source.docx", extraction_schema_version="1", extractor_version="1", evidence=[_item(), _item()])
    with pytest.raises(SlideContextResolutionError, match="duplicate evidence"):
        build_slide_contexts(_plan(slides=[_content_slide()], indexes=(index,)))
    index_without_evidence = index.model_copy(update={"evidence": []})
    with pytest.raises(SlideContextResolutionError, match="duplicate document index"):
        build_slide_contexts(_plan(slides=[_content_slide()], indexes=(index_without_evidence, index_without_evidence)))


def test_resolved_evidence_rejects_mechanical_or_malformed_semantic_data() -> None:
    with pytest.raises(ValidationError, match="mechanical key"):
        ResolvedEvidence(doc_id="doc", evidence_id="evidence", kind=EvidenceKind.TEXT, structured_data={"block_id": "block"})
    with pytest.raises(ValidationError, match="unsupported value"):
        ResolvedEvidence(doc_id="doc", evidence_id="evidence", kind=EvidenceKind.TEXT, structured_data={"values": (1, 2)})


def test_slide_context_rejects_slide_index_after_presentation_end() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        SlideContext(
            presentation_title="Growth",
            presentation_objective="Explain growth",
            presentation_narrative="Start with the result.",
            section_id="section-1",
            section_title="Findings",
            section_purpose="Explain",
            slide_index=2,
            total_slides=1,
            slide=_content_slide(),
        )


def test_multiple_outline_references_preserve_order_and_semantic_payloads() -> None:
    list_item = _item("list-evidence").model_copy(
        update={
            "kind": EvidenceKind.LIST,
            "asset_ids": ["list-asset"],
            "structured_data": {"items": [{"text": "First"}], "block_id": "hidden"},
        }
    )
    chart_item = _item("chart-evidence").model_copy(
        update={
            "kind": EvidenceKind.CHART_CANDIDATE,
            "asset_ids": ["chart-asset"],
            "structured_data": {"series": [{"name": "Revenue", "values": [10, 12]}], "source_node_ids": ["hidden"]},
        }
    )
    image_item = _item("image-evidence").model_copy(
        update={
            "kind": EvidenceKind.IMAGE,
            "asset_ids": ["image-asset"],
            "structured_data": {"caption": "Product photo", "relationship_id": "hidden"},
        }
    )
    index = DocumentIndex(
        doc_id="doc-1",
        filename="source.docx",
        extraction_schema_version="1",
        extractor_version="1",
        evidence=[list_item, chart_item, image_item],
    )
    selection = EvidenceSelection(
        selected=[
            {"doc_id": "doc-1", "evidence_id": "list-evidence", "reason": "List reason"},
            {"doc_id": "doc-1", "evidence_id": "chart-evidence", "reason": "Chart reason"},
            {"doc_id": "doc-1", "evidence_id": "image-evidence", "reason": "Image reason"},
        ]
    )
    slide = _content_slide(
        references=[
            EvidenceRef(doc_id="doc-1", evidence_ids=["list-evidence", "chart-evidence"]),
            EvidenceRef(doc_id="doc-1", evidence_ids=["image-evidence"]),
        ]
    )

    evidence = build_slide_contexts(_plan(slides=[slide], indexes=(index,), selection=selection))[0].evidence

    assert [(item.evidence_id, item.kind, item.selection_reason) for item in evidence] == [
        ("list-evidence", EvidenceKind.LIST, "List reason"),
        ("chart-evidence", EvidenceKind.CHART_CANDIDATE, "Chart reason"),
        ("image-evidence", EvidenceKind.IMAGE, "Image reason"),
    ]
    assert [item.asset_ids for item in evidence] == [["list-asset"], ["chart-asset"], ["image-asset"]]
    assert [item.structured_data for item in evidence] == [
        {"items": [{"text": "First"}]},
        {"series": [{"name": "Revenue", "values": [10, 12]}]},
        {"caption": "Product photo"},
    ]


def test_compaction_rejects_non_string_structured_data_keys() -> None:
    malformed = _item().model_copy(update={"structured_data": {1: "invalid"}})
    with pytest.raises(SlideContextResolutionError, match="non-string key"):
        build_slide_contexts(_plan(slides=[_content_slide()], indexes=(DocumentIndex(doc_id="doc-1", filename="source.docx", extraction_schema_version="1", extractor_version="1", evidence=[malformed]),)))
