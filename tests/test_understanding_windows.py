from types import SimpleNamespace

import pytest

from presentation_pipeline.budgeting import CharacterTokenEstimator, InputBudget
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


def test_oversized_unsupported_structured_evidence_fails_loudly() -> None:
    with pytest.raises(OversizedEvidenceError, match="unsupported oversized structured"):
        build_evidence_windows(
            _artifact(),
            _index(_item("list", None, kind="list")),
            token_counter=CharacterTokenEstimator(chars_per_token=1),
            budget=InputBudget(max_input_tokens=900),
        )


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
