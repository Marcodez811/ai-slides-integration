import json

from docx_pipeline import SourceLocator, SourceNode
from docx_pipeline.config import ExtractionConfig
from docx_pipeline.normalize import build_normalized_views
from docx_pipeline.normalize.captions import link_captions
from docx_pipeline.normalize.lists import build_list_views
from docx_pipeline.normalize.sections import detect_sections


def _node(sequence, kind, parent_id=None, payload=None):
    return SourceNode(
        node_id=f"node-{sequence}", sequence=sequence, parent_id=parent_id,
        sibling_index=sequence, kind=kind,
        source=SourceLocator(part_name="word/document.xml", xml_path=f"/x[{sequence}]"),
        payload=payload or {},
    )


def test_numbered_heading_is_not_a_list_and_nested_items_are_retained():
    nodes = [
        _node(0, "paragraph", payload={"style_id": "Heading 1", "numbering": {"num_id": "1", "ilvl": 0}}),
        _node(1, "paragraph", payload={"numbering": {"num_id": "2", "ilvl": 0, "num_fmt": "decimal"}}),
        _node(2, "paragraph", payload={"numbering": {"num_id": "2", "ilvl": 1, "num_fmt": "decimal"}}),
        _node(3, "paragraph", payload={"numbering": {"num_id": "2", "ilvl": 0, "num_fmt": "decimal"}}),
    ]
    views = build_list_views(nodes)
    assert len(views) == 1
    assert views[0].source_node_ids == ("node-1", "node-2", "node-3")
    assert [item.source_node_id for item in views[0].items] == ["node-1", "node-3"]
    assert views[0].items[0].children[0].source_node_id == "node-2"


def test_normalized_list_payload_recursively_uses_json_arrays_without_mutating_view():
    nodes = [
        _node(0, "paragraph", payload={"numbering": {"num_id": "7", "ilvl": 0, "num_fmt": "bullet", "marker": "•"}}),
        _node(1, "paragraph", payload={"numbering": {"num_id": "7", "ilvl": 1, "num_fmt": "bullet", "marker": "◦"}}),
        _node(2, "paragraph", payload={"numbering": {"num_id": "7", "ilvl": 2, "num_fmt": "decimal", "marker": "%3."}}),
        _node(3, "paragraph", payload={"numbering": {"num_id": "7", "ilvl": 0, "num_fmt": "bullet", "marker": "•"}}),
    ]

    list_view = build_list_views(nodes)[0]
    first = build_normalized_views(nodes, config=ExtractionConfig())
    second = build_normalized_views(nodes, config=ExtractionConfig())
    list_block = next(block for block in first.blocks if block.kind == "list")

    assert list_view.items[0].children.__class__ is tuple
    assert list_block.payload == {
        "num_id": "7",
        "items": [
            {
                "source_node_id": "node-0",
                "level": 0,
                "num_id": "7",
                "ordered": False,
                "marker": "•",
                "children": [
                    {
                        "source_node_id": "node-1",
                        "level": 1,
                        "num_id": "7",
                        "ordered": False,
                        "marker": "◦",
                        "children": [
                            {
                                "source_node_id": "node-2",
                                "level": 2,
                                "num_id": "7",
                                "ordered": True,
                                "marker": "%3.",
                                "children": [],
                            }
                        ],
                    }
                ],
            },
            {
                "source_node_id": "node-3",
                "level": 0,
                "num_id": "7",
                "ordered": False,
                "marker": "•",
                "children": [],
            },
        ],
    }
    assert isinstance(list_block.payload["items"], list)
    assert isinstance(list_block.payload["items"][0]["children"], list)
    assert isinstance(list_block.payload["items"][0]["children"][0]["children"], list)
    assert json.loads(json.dumps(list_block.payload)) == list_block.payload
    assert first.model_dump(mode="python") == second.model_dump(mode="python")
    assert list_block.source_node_ids == ["node-0", "node-1", "node-2", "node-3"]


def test_sections_use_levels_one_through_nine_and_do_not_reparent_source():
    nodes = [
        _node(0, "paragraph", payload={"outline_level": 0}),
        _node(1, "text_run", "node-0", {"text": "One"}),
        _node(2, "paragraph", payload={"style_id": "Heading 3"}),
        _node(3, "text_run", "node-2", {"text": "Three"}),
        _node(4, "paragraph", payload={"outline_level": 8}),
        _node(5, "text_run", "node-4", {"text": "Nine"}),
    ]
    sections = detect_sections(nodes)
    assert [(section.title, section.level) for section in sections] == [("One", 1), ("Three", 3), ("Nine", 9)]
    assert sections[1].parent_id == sections[0].section_id
    assert nodes[2].parent_id is None


def test_caption_is_retained_and_linked_to_adjacent_image():
    nodes = [
        _node(0, "paragraph"),
        _node(1, "image", "node-0", {"asset_id": "asset-a"}),
        _node(2, "paragraph", payload={"style_name": "Caption", "text": "Figure 1: example"}),
    ]
    links = link_captions(nodes)
    assert len(links) == 1
    assert links[0].caption_node_id == "node-2"
    assert links[0].target_node_id == "node-1"
    assert links[0].confidence >= 0.9
