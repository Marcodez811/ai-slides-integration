from docx_pipeline import Asset, SourceLocator, SourceNode
from docx_pipeline.enrichment.image_classifier import (
    ImageClassification,
    InMemoryClassificationCache,
    classification_cache_key,
)
from docx_pipeline.enrichment.table_profiler import profile_table


def _node(sequence, kind, parent_id=None, payload=None):
    return SourceNode(
        node_id=f"node-{sequence}", sequence=sequence, parent_id=parent_id,
        sibling_index=sequence, kind=kind,
        source=SourceLocator(part_name="word/document.xml", xml_path=f"/x[{sequence}]"), payload=payload or {},
    )


def test_image_classification_cache_uses_hash_and_versions_only():
    asset = Asset(asset_id="asset-a", sha256="a" * 64, original_name="one.png")
    result = ImageClassification(
        asset_id=asset.asset_id, asset_sha256=asset.sha256, classifier_version="vision-1",
        prompt_version="prompt-2", relevance="content", decorative=False,
    )
    cache = InMemoryClassificationCache()
    cache.put(result)
    assert cache.get(asset, "vision-1", "prompt-2") == result
    assert cache.get(asset, "vision-2", "prompt-2") is None
    assert result.cache_key == classification_cache_key(asset.sha256, "vision-1", "prompt-2")


def test_structural_table_profile_infers_header_and_chart_without_mutation():
    nodes = [
        _node(0, "table", payload={"grid_column_count": 2}),
        _node(1, "table_row", "node-0", {"row_index": 0}),
        _node(2, "table_cell", "node-1", {"row_index": 0, "column_index": 0}),
        _node(3, "paragraph", "node-2"), _node(4, "text_run", "node-3", {"text": "Month"}),
        _node(5, "table_cell", "node-1", {"row_index": 0, "column_index": 1}),
        _node(6, "paragraph", "node-5"), _node(7, "text_run", "node-6", {"text": "Sales"}),
        _node(8, "table_row", "node-0", {"row_index": 1}),
        _node(9, "table_cell", "node-8", {"row_index": 1, "column_index": 0}),
        _node(10, "paragraph", "node-9"), _node(11, "text_run", "node-10", {"text": "Jan"}),
        _node(12, "table_cell", "node-8", {"row_index": 1, "column_index": 1}),
        _node(13, "paragraph", "node-12"), _node(14, "text_run", "node-13", {"text": "1,200"}),
        _node(15, "table_row", "node-0", {"row_index": 2}),
        _node(16, "table_cell", "node-15", {"row_index": 2, "column_index": 0}),
        _node(17, "paragraph", "node-16"), _node(18, "text_run", "node-17", {"text": "Feb"}),
        _node(19, "table_cell", "node-15", {"row_index": 2, "column_index": 1}),
        _node(20, "paragraph", "node-19"), _node(21, "text_run", "node-20", {"text": "1300"}),
    ]
    original = nodes[2].payload.copy()
    profile = profile_table(nodes[0], nodes)
    assert profile.rectangular and profile.header_rows == (0,)
    assert [cell.original for cell in profile.cells][:2] == ["Month", "Sales"]
    assert profile.chart_candidate is not None
    assert profile.chart_candidate.categories == ("Jan", "Feb")
    assert profile.chart_candidate.series[0].values == (1200.0, 1300.0)
    assert nodes[2].payload == original
