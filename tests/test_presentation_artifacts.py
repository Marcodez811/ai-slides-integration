"""Contracts for deterministic local outline JSON artifacts."""

from __future__ import annotations

import json

import pytest

from presentation_pipeline.artifacts import write_outline_json, write_presentation_content_json
from presentation_pipeline.planning.models import (
    OutlineSection,
    PresentationOutline,
    SlideOutline,
    SlidePurpose,
)
from presentation_pipeline.synthesis.models import PresentationContent, SlideContent


def _outline() -> PresentationOutline:
    return PresentationOutline(
        title="A deck",
        objective="Explain the result",
        narrative="Start with the result",
        sections=[
            OutlineSection(
                section_id="section-1",
                title="Opening",
                purpose="Introduce the topic",
                slides=[
                    SlideOutline(
                        slide_id="slide-1",
                        title="Welcome",
                        purpose=SlidePurpose.TITLE,
                        message="The result",
                    )
                ],
            )
        ],
    )


def test_write_outline_json_creates_parent_dirs_and_round_trips(tmp_path) -> None:
    outline = _outline()
    output_path = write_outline_json(outline, tmp_path / "nested" / "outline.json")

    assert output_path == tmp_path / "nested" / "outline.json"
    assert output_path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(output_path.read_text(encoding="utf-8")) == outline.model_dump(mode="json")
    assert PresentationOutline.model_validate_json(output_path.read_text(encoding="utf-8")) == outline


def test_write_outline_json_is_pretty_deterministic_and_protects_existing_file(tmp_path) -> None:
    output_path = tmp_path / "outline.json"
    first = _outline()
    write_outline_json(first, output_path)
    initial_contents = output_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_outline_json(first, output_path)
    assert output_path.read_text(encoding="utf-8") == initial_contents

    replacement = first.model_copy(update={"title": "Replacement deck"})
    write_outline_json(replacement, output_path, overwrite=True)
    assert json.loads(output_path.read_text(encoding="utf-8"))["title"] == "Replacement deck"


def test_write_outline_json_requires_outline_model(tmp_path) -> None:
    with pytest.raises(TypeError, match="PresentationOutline"):
        write_outline_json({"title": "not validated"}, tmp_path / "outline.json")  # type: ignore[arg-type]


def test_write_presentation_content_json_round_trips_and_protects_existing_file(tmp_path) -> None:
    content = PresentationContent(slides=[SlideContent(slide_id="slide-1")])
    output_path = write_presentation_content_json(content, tmp_path / "content.json")

    assert PresentationContent.model_validate_json(output_path.read_text(encoding="utf-8")) == content
    with pytest.raises(FileExistsError):
        write_presentation_content_json(content, output_path)
    with pytest.raises(TypeError, match="PresentationContent"):
        write_presentation_content_json(_outline(), tmp_path / "wrong.json")  # type: ignore[arg-type]
