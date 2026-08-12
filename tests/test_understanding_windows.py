from types import SimpleNamespace

import pytest

from presentation_pipeline.budgeting import CharacterTokenEstimator, InputBudget
from presentation_pipeline.indexing.compact import compact_evidence_item
from presentation_pipeline.understanding.windows import OversizedEvidenceError, build_evidence_windows


def _artifact() -> SimpleNamespace:
    return SimpleNamespace(doc_id="doc-a", filename="doc-a.docx")


def _index(*items: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        doc_id="doc-a",
        evidence=list(items),
        sections=[SimpleNamespace(section_id="section-a", title="A", evidence_ids=[item.evidence_id for item in items])],
    )


def _item(evidence_id: str, text: str | None, *, kind: str = "text") -> SimpleNamespace:
    return SimpleNamespace(
        evidence_id=evidence_id,
        kind=kind,
        section_ids=["section-a"],
        text=text,
        structured_data={} if text is not None else {"rows": [["a", "b"]] * 500},
    )


def _list_item(evidence_id: str, items: list[dict[str, object]]) -> SimpleNamespace:
    return SimpleNamespace(
        evidence_id=evidence_id,
        kind="list",
        section_ids=["section-a"],
        text="\n".join(str(node.get("text", "")) for node in items),
        structured_data={
            "payload": {"items": items, "list_style": "numbered"},
            "items": items,
        },
    )


def _window_items(windows: list[object]) -> list[dict[str, object]]:
    return [item for window in windows for item in window.payload["evidence"]]


def _texts(nodes: list[dict[str, object]]) -> list[str]:
    result: list[str] = []
    for node in nodes:
        if isinstance(node.get("text"), str):
            result.append(node["text"])
        children = node.get("children")
        if isinstance(children, list):
            result.extend(_texts([child for child in children if isinstance(child, dict)]))
    return result


def test_windows_are_stable_section_aware_and_preserve_canonical_ids() -> None:
    counter = CharacterTokenEstimator(chars_per_token=1)
    index = _index(_item("one", "first " * 90), _item("two", "second " * 90))
    windows = build_evidence_windows(
        _artifact(), index, token_counter=counter, budget=InputBudget(max_input_tokens=1_800)
    )

    assert [window.window_id for window in windows] == [f"window-{i:04d}" for i in range(len(windows))]
    assert all(window.estimated_tokens <= 1_800 for window in windows)
    assert [item["evidence_id"] for window in windows for item in window.payload["evidence"]] == ["one", "two"]
    assert all(window.payload["sections"] == [{"section_id": "section-a", "title": "A"}] for window in windows)


def test_oversized_text_is_sliced_without_changing_evidence_id() -> None:
    windows = build_evidence_windows(
        _artifact(),
        _index(_item("long", ("A sentence. " * 400))),
        token_counter=CharacterTokenEstimator(chars_per_token=1),
        budget=InputBudget(max_input_tokens=900),
    )

    items = [item for window in windows for item in window.payload["evidence"]]
    assert len(items) > 1
    assert {item["evidence_id"] for item in items} == {"long"}
    assert all("slice" in item and item["slice"]["slice_id"] for item in items)


def test_section_boundary_is_preferred_only_when_next_section_would_overflow() -> None:
    first, second = _item("one", "first " * 100), _item("two", "second " * 100)
    first.section_ids, second.section_ids = ["section-one"], ["section-two"]
    index = SimpleNamespace(
        doc_id="doc-a", evidence=[first, second],
        sections=[SimpleNamespace(section_id="section-one", title="One"), SimpleNamespace(section_id="section-two", title="Two")],
    )
    windows = build_evidence_windows(
        _artifact(), index, token_counter=CharacterTokenEstimator(chars_per_token=1),
        budget=InputBudget(max_input_tokens=1_500),
    )
    assert [window.evidence_ids for window in windows] == [("one",), ("two",)]


def test_small_neighboring_sections_share_a_window() -> None:
    first, second = _item("one", "first"), _item("two", "second")
    first.section_ids, second.section_ids = ["section-one"], ["section-two"]
    index = SimpleNamespace(
        doc_id="doc-a", evidence=[first, second],
        sections=[SimpleNamespace(section_id="section-one", title="One"), SimpleNamespace(section_id="section-two", title="Two")],
    )
    windows = build_evidence_windows(
        _artifact(), index, token_counter=CharacterTokenEstimator(), budget=InputBudget(max_input_tokens=4_000)
    )
    assert [window.evidence_ids for window in windows] == [("one", "two")]


def test_sliced_text_reconstructs_the_source_exactly() -> None:
    source = "First sentence.  Second sentence!\n\nThird paragraph? Tail"
    windows = build_evidence_windows(
        _artifact(), _index(_item("long", source * 40)),
        token_counter=CharacterTokenEstimator(chars_per_token=1),
        budget=InputBudget(max_input_tokens=900),
    )
    slices = [
        item["text"]
        for window in windows
        for item in window.payload["evidence"]
    ]
    assert "".join(slices) == source * 40


def test_oversized_malformed_list_fails_with_safe_metadata() -> None:
    with pytest.raises(OversizedEvidenceError, match="unsupported oversized structured"):
        build_evidence_windows(
            _artifact(),
            _index(_item("chart", None, kind="chart")),
            token_counter=CharacterTokenEstimator(chars_per_token=1),
            budget=InputBudget(max_input_tokens=900),
        )

    malformed = _item("malformed-list", "SECRET" * 400, kind="list")
    with pytest.raises(OversizedEvidenceError, match="no sliceable") as captured:
        build_evidence_windows(
            _artifact(), _index(malformed), token_counter=CharacterTokenEstimator(chars_per_token=1),
            budget=InputBudget(max_input_tokens=900),
        )
    assert captured.value.metadata == {
        "evidence_id": "malformed-list", "kind": "list", "reason": "malformed_list_items",
        "doc_id": "doc-a", "estimated_tokens": captured.value.metadata["estimated_tokens"], "limit_tokens": 900,
    }
    assert "SECRET" not in repr(captured.value.metadata)


def test_compact_list_transport_keeps_items_once_without_mutating_canonical() -> None:
    source_items = [{"text": "Parent", "source_node_id": "node-1", "children": []}]
    item = _list_item("list", source_items)
    compact = compact_evidence_item(item)
    assert compact["content"]["items"] == [{"text": "Parent", "children": []}]
    assert compact["content"]["payload"] == {"list_style": "numbered"}
    assert "items" not in compact["content"]["payload"]
    assert item.structured_data["payload"]["items"] == source_items


def test_oversized_flat_list_slices_without_loss_or_full_text_leakage() -> None:
    source = [{"text": f"item-{number}-" + "x" * 40, "children": []} for number in range(12)]
    windows = build_evidence_windows(
        _artifact(), _index(_list_item("list", source)),
        token_counter=CharacterTokenEstimator(chars_per_token=1), budget=InputBudget(max_input_tokens=1_000),
    )
    slices = _window_items(windows)
    assert len(slices) > 1
    assert {item["evidence_id"] for item in slices} == {"list"}
    assert [item["slice"]["index"] for item in slices] == list(range(len(slices)))
    assert {item["slice"]["count"] for item in slices} == {len(slices)}
    assert _texts([node for item in slices for node in item["content"]["items"]]) == _texts(source)
    assert all(item["text"] == "\n".join(_texts(item["content"]["items"])) for item in slices)
    assert all("items" not in item["content"].get("payload", {}) for item in slices)
    assert all(window.estimated_tokens <= 1_000 for window in windows)


def test_oversized_nested_list_keeps_fitting_parent_subtrees_together() -> None:
    source = [
        {"text": f"Parent {letter}", "children": [
            {"text": f"{letter}-one", "children": []},
            {"text": f"{letter}-two", "children": []},
        ]}
        for letter in ("A", "B", "C", "D")
    ]
    windows = build_evidence_windows(
        _artifact(), _index(_list_item("nested", source)),
        token_counter=CharacterTokenEstimator(chars_per_token=1), budget=InputBudget(max_input_tokens=950),
    )
    slices = _window_items(windows)
    assert len(slices) == 4
    assert [[node["text"] for node in item["content"]["items"]] for item in slices] == [
        ["Parent A"], ["Parent B"], ["Parent C"], ["Parent D"]
    ]
    assert all(len(item["content"]["items"][0]["children"]) == 2 for item in slices)


def test_oversized_nested_subtree_recurses_without_losing_children() -> None:
    source = [{"text": "Parent" + "x" * 10, "children": [
        {"text": f"child-{number}" + "x" * 5, "children": []}
        for number in range(10)
    ]}]
    windows = build_evidence_windows(
        _artifact(), _index(_list_item("nested", source)),
        token_counter=CharacterTokenEstimator(chars_per_token=1), budget=InputBudget(max_input_tokens=950),
    )
    slices = _window_items(windows)
    children = [
        child["text"]
        for item in slices
        for root in item["content"]["items"]
        for child in root.get("children", [])
    ]
    assert children == [child["text"] for child in source[0]["children"]]
    assert all(item["content"]["items"][0]["text"] == "Parent" + "x" * 10 for item in slices)


def test_oversized_list_leaf_text_slices_without_truncation() -> None:
    source_text = "leaf text " * 500
    windows = build_evidence_windows(
        _artifact(), _index(_list_item("leaf", [{"text": source_text, "children": []}])),
        token_counter=CharacterTokenEstimator(chars_per_token=1), budget=InputBudget(max_input_tokens=900),
    )
    slices = _window_items(windows)
    assert "".join(item["content"]["items"][0]["text"] for item in slices) == source_text
    assert {item["slice"]["unit"] for item in slices} == {"list_item_text"}


def test_malformed_nested_list_descendant_fails_without_silent_loss() -> None:
    malformed = [{"text": "Parent", "children": [{"text": "good", "children": []}, "bad-child"]}]
    with pytest.raises(OversizedEvidenceError, match="malformed descendant") as captured:
        build_evidence_windows(
            _artifact(), _index(_list_item("bad-tree", malformed)),
            token_counter=CharacterTokenEstimator(chars_per_token=1), budget=InputBudget(max_input_tokens=900),
        )
    assert captured.value.metadata == {
        "kind": "list", "reason": "malformed_list_tree", "doc_id": "doc-a",
        "evidence_id": "bad-tree", "estimated_tokens": captured.value.metadata["estimated_tokens"], "limit_tokens": 900,
    }


def test_oversized_parent_text_and_descendants_preserve_all_source_content() -> None:
    parent_text = "PARENT-COVERAGE " * 200
    children = [{"text": f"CHILD-{number}-" + "x" * 50, "children": []} for number in range(6)]
    windows = build_evidence_windows(
        _artifact(), _index(_list_item("parent-and-children", [{"text": parent_text, "children": children}])),
        token_counter=CharacterTokenEstimator(chars_per_token=1), budget=InputBudget(max_input_tokens=900),
    )
    fragments = _window_items(windows)
    flattened = "\n".join(fragment["text"] for fragment in fragments)
    # Parent text is either repeated as context with children or carried as
    # leaf text slices; joining its text fragments must retain every character.
    leaf_parts = [
        node["text"]
        for fragment in fragments
        for node in fragment["content"]["items"]
        if fragment["slice"]["unit"] == "list_item_text"
        and isinstance(node.get("text"), str)
    ]
    assert "".join(leaf_parts)[:len(parent_text)] == parent_text
    assert "".join(leaf_parts)[len(parent_text):] == "".join(child["text"] for child in children)
    assert "PARENT-COVERAGE" in flattened


def test_oversized_list_diagnostics_are_content_free(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "presentation_pipeline.understanding.windows.safe_event",
        lambda name, **fields: events.append((name, fields)),
    )
    build_evidence_windows(
        _artifact(), _index(_list_item("safe-list", [{"text": "SECRET-LIST-TEXT " * 100, "children": []}])),
        token_counter=CharacterTokenEstimator(chars_per_token=1), budget=InputBudget(max_input_tokens=900),
    )
    detected = next(fields for name, fields in events if name == "oversized_evidence_detected")
    sliced = next(fields for name, fields in events if name == "oversized_evidence_sliced")
    assert detected == {"doc_id": "doc-a", "evidence_id": "safe-list", "kind": "list", "estimated_tokens": detected["estimated_tokens"], "input_limit_tokens": 900}
    assert sliced["slice_count"] > 1 and sliced["max_slice_tokens"] <= 900
    assert "SECRET-LIST-TEXT" not in repr(events)


def test_oversized_table_is_row_sliced_without_changing_values_or_identity() -> None:
    rows = ["Year\tRevenue\n", *[f"20{number:02d}\t{number * 1234}.50\n" for number in range(80)]]
    text = "".join(rows)
    table = _item("table", text, kind="table")
    table.structured_data = {
        "payload": {"grid_column_count": 2, "text": text}
    }
    windows = build_evidence_windows(
        _artifact(), _index(table),
        token_counter=CharacterTokenEstimator(chars_per_token=1),
        budget=InputBudget(max_input_tokens=900),
    )
    slices = [
        item
        for window in windows
        for item in window.payload["evidence"]
    ]
    assert len(slices) > 1
    assert {item["evidence_id"] for item in slices} == {"table"}
    assert "".join(item["text"] for item in slices) == text
    assert "".join(item["content"]["payload"]["text"] for item in slices) == text
    assert all(item["content"]["payload"]["grid_column_count"] == 2 for item in slices)
    assert all(item["slice"]["unit"] == "table_rows" for item in slices)
