"""Strict, layout-free semantic content contracts for presentation slides."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from presentation_pipeline.common.models import PipelineModel
from presentation_pipeline.common.references import EvidenceRef


def _nonempty(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


def _optional_nonempty(value: str | None) -> str | None:
    return _nonempty(value) if value is not None else None


class TextContent(PipelineModel):
    """A source-backed text statement; this is content, never layout."""

    kind: Literal["text"]
    text: str
    evidence: list[EvidenceRef] = Field(min_length=1)

    _text_is_nonempty = field_validator("text")(_nonempty)


class BulletItem(PipelineModel):
    text: str
    evidence: list[EvidenceRef] = Field(min_length=1)

    _text_is_nonempty = field_validator("text")(_nonempty)


class BulletListContent(PipelineModel):
    kind: Literal["list"]
    items: list[BulletItem] = Field(min_length=1)


class ChartContent(PipelineModel):
    kind: Literal["chart"]
    doc_id: str
    evidence_id: str
    chart_type: Literal["bar", "line", "area", "pie"]
    title: str | None = None

    _doc_id_is_nonempty = field_validator("doc_id")(_nonempty)
    _evidence_id_is_nonempty = field_validator("evidence_id")(_nonempty)
    _title_is_nonempty = field_validator("title")(_optional_nonempty)


class TableContent(PipelineModel):
    kind: Literal["table"]
    doc_id: str
    evidence_id: str
    title: str | None = None

    _doc_id_is_nonempty = field_validator("doc_id")(_nonempty)
    _evidence_id_is_nonempty = field_validator("evidence_id")(_nonempty)
    _title_is_nonempty = field_validator("title")(_optional_nonempty)


class ImageContent(PipelineModel):
    kind: Literal["image"]
    doc_id: str
    evidence_id: str
    caption: str | None = None

    _doc_id_is_nonempty = field_validator("doc_id")(_nonempty)
    _evidence_id_is_nonempty = field_validator("evidence_id")(_nonempty)
    _caption_is_nonempty = field_validator("caption")(_optional_nonempty)


class EquationContent(PipelineModel):
    kind: Literal["equation"]
    doc_id: str
    evidence_id: str

    _doc_id_is_nonempty = field_validator("doc_id")(_nonempty)
    _evidence_id_is_nonempty = field_validator("evidence_id")(_nonempty)


SlideElement = Annotated[
    TextContent | BulletListContent | ChartContent | TableContent | ImageContent | EquationContent,
    Field(discriminator="kind"),
]


class SlideContent(PipelineModel):
    slide_id: str
    elements: list[SlideElement] = Field(default_factory=list)

    _slide_id_is_nonempty = field_validator("slide_id")(_nonempty)


class PresentationContent(PipelineModel):
    """Ordered semantic content for a complete presentation, without rendering data."""

    schema_version: str = "1.0.0"
    slides: list[SlideContent] = Field(min_length=1)

    _schema_version_is_nonempty = field_validator("schema_version")(_nonempty)

    @model_validator(mode="after")
    def _slide_ids_are_unique(self) -> "PresentationContent":
        slide_ids = [slide.slide_id for slide in self.slides]
        if len(slide_ids) != len(set(slide_ids)):
            raise ValueError("slide IDs must be unique in presentation content")
        return self
