from __future__ import annotations

from copy import deepcopy

import pytest

from docx_pipeline import (
    Asset,
    DocumentMetadata,
    ExtractionResult,
    NormalizedBlock,
    NormalizedViews,
    SectionView,
    SourceLocator,
    SourceNode,
)
from presentation_pipeline.corpus.models import DocumentArtifact
from presentation_pipeline.indexing import EvidenceKind, build_document_index


def _node(sequence: int, kind: str, parent_id: str | None = None, payload: dict | None = None) -> SourceNode:
    return SourceNode(
        node_id=f"node-{sequence}",
        sequence=sequence,
        parent_id=parent_id,
        sibling_index=sequence,
        kind=kind,
        source=SourceLocator(part_name="word/document.xml", xml_path=f"/w:document/w:body/*[{sequence + 1}]"),
        payload=payload or {},
    )


def _artifact() -> DocumentArtifact:
    nodes = [
        _node(0, "paragraph", payload={"style_id": "Heading 1"}),
        _node(1, "text_run", "node-0", {"text": "Quarterly results"}),
        _node(2, "paragraph"),
        _node(3, "text_run", "node-2", {"text": "Revenue grew."}),
        _node(4, "paragraph", payload={"numbering": {"num_id": "7", "ilvl": 0}}),
        _node(5, "text_run", "node-4", {"text": "North region"}),
        _node(6, "paragraph", payload={"numbering": {"num_id": "7", "ilvl": 1}}),
        _node(7, "text_run", "node-6", {"text": "Enterprise"}),
        _node(8, "table", payload={"grid_column_count": 2}),
        _node(9, "table_row", "node-8", {"row_index": 0}),
        _node(10, "table_cell", "node-9", {"column_index": 0}),
        _node(11, "paragraph", "node-10"),
        _node(12, "text_run", "node-11", {"text": "Month"}),
        _node(13, "table_cell", "node-9", {"column_index": 1}),
        _node(14, "paragraph", "node-13"),
        _node(15, "text_run", "node-14", {"text": "Sales"}),
        _node(16, "table_row", "node-8", {"row_index": 1}),
        _node(17, "table_cell", "node-16", {"column_index": 0}),
        _node(18, "paragraph", "node-17"),
        _node(19, "text_run", "node-18", {"text": "Jan"}),
        _node(20, "table_cell", "node-16", {"column_index": 1}),
        _node(21, "paragraph", "node-20"),
        _node(22, "text_run", "node-21", {"text": "1200"}),
        _node(23, "table_row", "node-8", {"row_index": 2}),
        _node(24, "table_cell", "node-23", {"column_index": 0}),
        _node(25, "paragraph", "node-24"),
        _node(26, "text_run", "node-25", {"text": "Feb"}),
        _node(27, "table_cell", "node-23", {"column_index": 1}),
        _node(28, "paragraph", "node-27"),
        _node(29, "text_run", "node-28", {"text": "1300"}),
        _node(30, "image", payload={"asset_id": "asset-figure", "placement": "inline", "alt_text": "Sales chart"}),
        _node(31, "equation", payload={"latex": "y = mx + b"}),
        _node(32, "chart", payload={"title": "Pipeline chart", "series_count": 2}),
    ]
    blocks = [
        NormalizedBlock(block_id="block-title", kind="title", source_node_ids=["node-0", "node-1"], payload={"content": "Quarterly results"}),
        NormalizedBlock(block_id="block-text", kind="text", source_node_ids=["node-2", "node-3"], payload={"content": "Revenue grew."}),
        NormalizedBlock(
            block_id="block-list",
            kind="list",
            source_node_ids=["node-4", "node-6"],
            payload={
                "items": [
                    {
                        "source_node_id": "node-4",
                        "level": 0,
                        "num_id": "7",
                        "ordered": False,
                        "marker": "•",
                        "children": [
                            {
                                "source_node_id": "node-6",
                                "level": 1,
                                "num_id": "7",
                                "ordered": False,
                                "marker": "•",
                                "children": [],
                            }
                        ],
                    }
                ]
            },
        ),
        NormalizedBlock(
            block_id="block-table",
            kind="table",
            source_node_ids=["node-8"],
            payload={"text": "Month Sales\nJan 1200\nFeb 1300"},
        ),
        NormalizedBlock(
            block_id="block-image",
            kind="image",
            source_node_ids=["node-30"],
            payload={"asset_id": "asset-figure", "placement": "inline", "width_emu": 914400, "alt_text": "Sales chart"},
        ),
        NormalizedBlock(block_id="block-equation", kind="equation", source_node_ids=["node-31"], payload={"latex": "y = mx + b"}),
        NormalizedBlock(block_id="block-chart", kind="chart", source_node_ids=["node-32"], payload={"title": "Pipeline chart", "series_count": 2}),
        NormalizedBlock(block_id="block-ignored", kind="unsupported", source_node_ids=["node-3"], payload={}),
    ]
    result = ExtractionResult(
        schema_version="2.0",
        extractor_version="2.4",
        document=DocumentMetadata(doc_id="doc-index", filename="quarterly.docx", sha256="d" * 64),
        nodes=nodes,
        assets=[Asset(asset_id="asset-figure", sha256="f" * 64, original_name="sales.png", mime_type="image/png")],
        views=NormalizedViews(
            blocks=blocks,
            sections=[
                SectionView(
                    section_id="section-parent",
                    source_heading_node_id="node-0",
                    title="Quarterly results",
                    block_ids=["block-title", "block-text", "block-list", "block-table", "block-image", "block-equation", "block-chart", "block-ignored"],
                    level=1,
                ),
                SectionView(
                    section_id="section-child",
                    source_heading_node_id="node-0",
                    title="Detail",
                    block_ids=["block-table", "block-image"],
                    parent_id="section-parent",
                    level=2,
                ),
            ],
        ),
    )
    # model_construct intentionally avoids coupling these indexing tests to
    # unrelated corpus-health policy while still using the production field.
    return DocumentArtifact.model_construct(
        job_id="job-index", doc_id="doc-index", filename="quarterly.docx", health=None, extraction=result
    )


def test_document_index_is_deterministic_and_does_not_mutate_input():
    artifact = _artifact()
    before = deepcopy(artifact.extraction.model_dump(mode="json"))

    first = build_document_index(artifact)
    second = build_document_index(artifact)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert artifact.extraction.model_dump(mode="json") == before
    assert first.doc_id == "doc-index"
    assert first.filename == "quarterly.docx"
    assert first.extraction_schema_version == "2.0"
    assert first.extractor_version == "2.4"


def test_index_maps_supported_blocks_and_derives_chart_candidate():
    index = build_document_index(_artifact())
    evidence_by_kind = {item.kind: item for item in index.evidence if item.kind is not EvidenceKind.CHART_CANDIDATE}

    assert [item.kind for item in index.evidence] == [
        EvidenceKind.TITLE,
        EvidenceKind.TEXT,
        EvidenceKind.LIST,
        EvidenceKind.TABLE,
        EvidenceKind.CHART_CANDIDATE,
        EvidenceKind.IMAGE,
        EvidenceKind.EQUATION,
        EvidenceKind.CHART,
    ]
    assert set(evidence_by_kind) == {
        EvidenceKind.TITLE,
        EvidenceKind.TEXT,
        EvidenceKind.LIST,
        EvidenceKind.TABLE,
        EvidenceKind.IMAGE,
        EvidenceKind.EQUATION,
        EvidenceKind.CHART,
    }
    assert evidence_by_kind[EvidenceKind.TEXT].text == "Revenue grew."
    assert evidence_by_kind[EvidenceKind.LIST].text == "North region\nEnterprise"
    assert evidence_by_kind[EvidenceKind.LIST].structured_data["items"][0]["children"][0]["source_node_id"] == "node-6"
    assert evidence_by_kind[EvidenceKind.TABLE].text == "Month Sales\nJan 1200\nFeb 1300"
    assert evidence_by_kind[EvidenceKind.IMAGE].asset_ids == ["asset-figure"]
    assert evidence_by_kind[EvidenceKind.IMAGE].structured_data["asset"]["mime_type"] == "image/png"
    assert evidence_by_kind[EvidenceKind.EQUATION].text == "y = mx + b"
    assert evidence_by_kind[EvidenceKind.CHART].structured_data["payload"]["series_count"] == 2

    candidate = next(item for item in index.evidence if item.kind is EvidenceKind.CHART_CANDIDATE)
    assert candidate.block_ids == ["block-table"]
    assert candidate.structured_data["categories"] == ["Jan", "Feb"]
    assert candidate.structured_data["series"] == [{"name": "Sales", "column": 1, "values": [1200.0, 1300.0]}]


def test_sections_preserve_view_membership_and_all_ids_resolve():
    index = build_document_index(_artifact())

    assert [section.section_id for section in index.sections] == ["section-parent", "section-child"]
    assert index.sections[0].block_ids == ["block-title", "block-text", "block-list", "block-table", "block-image", "block-equation", "block-chart", "block-ignored"]
    table_evidence = evidence_by_block(index, "block-table")
    assert all(item.section_ids == ["section-parent", "section-child"] for item in table_evidence)
    assert evidence_by_block(index, "block-ignored") == []

    artifact = _artifact()
    block_ids = {block.block_id for block in artifact.extraction.views.blocks}
    node_ids = {node.node_id for node in artifact.extraction.nodes}
    asset_ids = {asset.asset_id for asset in artifact.extraction.assets}
    evidence_ids = {item.evidence_id for item in index.evidence}
    assert all(block_id in block_ids for item in index.evidence for block_id in item.block_ids)
    assert all(source_id in node_ids for item in index.evidence for source_id in item.source_node_ids)
    assert all(asset_id in asset_ids for item in index.evidence for asset_id in item.asset_ids)
    assert all(evidence_id in evidence_ids for section in index.sections for evidence_id in section.evidence_ids)


def test_unresolved_image_asset_is_allowed_but_dangling_non_null_asset_fails_fast():
    artifact = _artifact()
    image = next(block for block in artifact.extraction.views.blocks if block.block_id == "block-image")
    image.payload["asset_id"] = None
    index = build_document_index(artifact)
    image_evidence = next(item for item in index.evidence if item.kind is EvidenceKind.IMAGE)
    assert image_evidence.asset_ids == []

    dangling = _artifact()
    dangling_image = next(block for block in dangling.extraction.views.blocks if block.block_id == "block-image")
    dangling_image.payload["asset_id"] = "asset-missing"
    with pytest.raises(ValueError, match="unknown asset 'asset-missing'"):
        build_document_index(dangling)


def test_malformed_table_source_mapping_fails_fast():
    artifact = _artifact()
    table = next(block for block in artifact.extraction.views.blocks if block.block_id == "block-table")
    table.source_node_ids[0] = "node-2"

    with pytest.raises(ValueError, match="must start with a structural table node"):
        build_document_index(artifact)


def evidence_by_block(index, block_id: str):
    return [item for item in index.evidence if item.block_ids == [block_id]]
