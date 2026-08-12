"""Typed, renderer-facing contracts for editable physical PowerPoint slides."""

from __future__ import annotations

from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from presentation_pipeline.common.models import PipelineModel


class RenderingModel(PipelineModel):
    """Base model for the render boundary; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True)


class LayoutArchetype(str, Enum):
    TITLE = "title"
    SECTION_DIVIDER = "section_divider"
    HEADLINE_BODY = "headline_body"
    KEY_POINTS = "key_points"
    TWO_COLUMN = "two_column"
    VISUAL_TEXT = "visual_text"
    TABLE_FOCUS = "table_focus"
    CHART_FOCUS = "chart_focus"


class Box(RenderingModel):
    """A rectangular region in inches on a 16:9 slide."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)

    @field_validator("x", "y", "width", "height")
    @classmethod
    def _is_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("box values must be finite")
        return value

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def intersects(self, other: "Box") -> bool:
        return self.x < other.right and self.right > other.x and self.y < other.bottom and self.bottom > other.y


class SourceAttribution(RenderingModel):
    """Stable provenance shown in a slide footer when a payload has a source."""

    doc_id: str = Field(min_length=1)
    evidence_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    filename: str | None = None
    label: str | None = None
    locator: str | None = None

    @field_validator("doc_id", "evidence_id", "filename", "label", "locator")
    @classmethod
    def _non_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("source attribution values must not be blank")
        return value

    @field_validator("evidence_ids")
    @classmethod
    def _evidence_ids_are_not_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("evidence IDs must not be blank")
        return values

    @model_validator(mode="after")
    def _normalise_evidence_ids(self) -> "SourceAttribution":
        values = list(dict.fromkeys([*self.evidence_ids, *([self.evidence_id] if self.evidence_id else [])]))
        if not values:
            raise ValueError("source attribution requires at least one evidence ID")
        self.evidence_ids = values
        self.evidence_id = values[0]
        return self

    @property
    def footer_text(self) -> str:
        reference = self.filename or self.label or self.doc_id
        suffix = f" · {self.locator}" if self.locator else ""
        return f"Source: {reference} ({', '.join(self.evidence_ids)}){suffix}"


class TableCell(RenderingModel):
    """Exact source-table cell content, retaining simple merge metadata."""

    text: str
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)


class TablePayload(RenderingModel):
    """An editable table payload.  ``rows`` is rectangular before spans apply."""

    rows: list[list[TableCell]] = Field(min_length=1)
    header_rows: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rectangular(self) -> "TablePayload":
        if not self.rows or not self.rows[0] or any(len(row) != len(self.rows[0]) for row in self.rows):
            raise ValueError("table rows must be a non-empty rectangular matrix")
        if any(index < 0 or index >= len(self.rows) for index in self.header_rows):
            raise ValueError("header row indexes must identify table rows")
        return self

    @property
    def column_count(self) -> int:
        return len(self.rows[0])

    def plain_rows(self) -> list[list[str]]:
        return [[cell.text for cell in row] for row in self.rows]


class ChartSeries(RenderingModel):
    name: str = Field(min_length=1)
    values: list[float] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _name_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("series name must not be blank")
        return value

    @field_validator("values")
    @classmethod
    def _values_are_finite(cls, values: list[float]) -> list[float]:
        if any(not isfinite(value) for value in values):
            raise ValueError("chart values must be finite")
        return values


ResolvedElementKind = Literal["text", "list", "image", "table", "chart", "equation"]
PositionedElementKind = Literal["title", "subtitle", "text", "list", "image", "table", "chart", "equation", "caption", "footer"]


class ResolvedElement(RenderingModel):
    """Resolved rendering data, including optional source provenance.

    ``rows``/``header_rows`` remain as a compatibility input. New callers should
    use ``table`` so table cell structure is explicit.
    """

    element_index: int = Field(ge=-1)
    kind: ResolvedElementKind
    text: str | None = None
    items: list[str] | None = None
    path: str | None = None
    table: TablePayload | None = None
    rows: list[list[str]] | None = None
    header_rows: list[int] = Field(default_factory=list)
    categories: list[str] | None = None
    series: list[ChartSeries] | None = None
    chart_type: Literal["bar", "line", "area", "pie"] | None = None
    title: str | None = None
    caption: str | None = None
    source_attribution: SourceAttribution | None = None
    attribution: list[SourceAttribution] = Field(default_factory=list)

    @field_validator("text", "path", "title", "caption")
    @classmethod
    def _optional_text_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional text values must not be blank")
        return value

    @field_validator("items", "categories")
    @classmethod
    def _string_items_are_not_blank(cls, values: list[str] | None) -> list[str] | None:
        if values is not None and any(not value.strip() for value in values):
            raise ValueError("string collection values must not be blank")
        return values

    @field_validator("rows")
    @classmethod
    def _legacy_rows_are_rectangular(cls, rows: list[list[str]] | None) -> list[list[str]] | None:
        if rows is None:
            return rows
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("table rows must be a non-empty rectangular matrix")
        return rows

    @model_validator(mode="after")
    def _payload_matches_kind(self) -> "ResolvedElement":
        sources = list(self.attribution)
        if self.source_attribution is not None:
            sources.insert(0, self.source_attribution)
        deduplicated: list[SourceAttribution] = []
        seen_sources: set[tuple[str, tuple[str, ...]]] = set()
        for source in sources:
            identity = (source.doc_id, tuple(source.evidence_ids))
            if identity not in seen_sources:
                seen_sources.add(identity)
                deduplicated.append(source)
        self.attribution = deduplicated
        self.source_attribution = deduplicated[0] if deduplicated else None
        if self.table is None and self.rows is not None:
            self.table = TablePayload(rows=[[TableCell(text=value) for value in row] for row in self.rows], header_rows=self.header_rows)
        if self.table is not None and self.header_rows and not self.table.header_rows:
            self.table = self.table.model_copy(update={"header_rows": self.header_rows})
        requirements: dict[str, bool] = {
            "text": self.text is not None,
            "list": self.items is not None and bool(self.items),
            "image": self.path is not None,
            "table": self.table is not None,
            "chart": self.categories is not None and self.series is not None and self.chart_type is not None,
            "equation": self.text is not None,
        }
        if not requirements[self.kind]:
            raise ValueError(f"resolved {self.kind} element has incomplete payload")
        if self.kind == "chart":
            assert self.categories is not None and self.series is not None
            if not self.categories or any(len(item.values) != len(self.categories) for item in self.series):
                raise ValueError("every chart series must match the category count")
        return self

    @property
    def image_path(self) -> Path | None:
        return Path(self.path) if self.path is not None else None

    @property
    def table_payload(self) -> TablePayload | None:
        return self.table


class RenderInputResult(RenderingModel):
    """One semantic slide's resolved render payloads and non-blocking warnings."""

    slide_id: str = Field(min_length=1)
    elements: list[ResolvedElement] = Field(default_factory=list)
    diagnostics: list["RenderDiagnostic"] = Field(default_factory=list)
    source_attributions: list[SourceAttribution] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_sources(self) -> "RenderInputResult":
        seen = set()
        ordered: list[SourceAttribution] = []
        for attribution in [*self.source_attributions, *(source for item in self.elements for source in item.attribution)]:
            assert attribution is not None
            identity = (attribution.doc_id, attribution.evidence_id)
            if identity not in seen:
                seen.add(identity)
                ordered.append(attribution)
        self.source_attributions = ordered
        return self


class PositionedElement(RenderingModel):
    """A resolved element with an assigned physical rectangle and typography."""

    element_index: int | None = None
    kind: PositionedElementKind
    role: str = Field(min_length=1)
    box: Box
    font_size_pt: float | None = Field(default=None, ge=10, le=72)
    payload: ResolvedElement | None = None
    text: str | None = None
    items: list[str] | None = None

    @field_validator("role", "text")
    @classmethod
    def _text_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("text values must not be blank")
        return value

    @model_validator(mode="after")
    def _positioned_payload_is_consistent(self) -> "PositionedElement":
        if self.kind in {"title", "subtitle", "text", "equation", "caption", "footer"} and self.text is None and self.payload is None:
            raise ValueError("textual positioned elements require text or a resolved payload")
        if self.kind == "list" and self.items is None and self.payload is None:
            raise ValueError("list positioned elements require items or a resolved payload")
        if self.kind in {"image", "table", "chart"} and self.payload is None:
            raise ValueError(f"{self.kind} positioned elements require a resolved payload")
        if self.payload is not None and self.element_index is not None and self.payload.element_index != self.element_index:
            raise ValueError("positioned element index must match its payload")
        return self


class ExecutivePolicyTheme(RenderingModel):
    """The fixed executive-policy visual system used by all slides."""

    name: str = "Executive Policy"
    slide_width_inches: float = Field(default=13.333, gt=0)
    slide_height_inches: float = Field(default=7.5, gt=0)
    title_font: str = "Aptos Display"
    body_font: str = "Aptos"
    east_asian_font: str = "Microsoft JhengHei"
    background_color: str = "F7F8FA"
    primary_color: str = "153B5B"
    accent_color: str = "2D7D9A"
    text_color: str = "1C2733"
    muted_color: str = "5C6B78"
    minimum_body_font_pt: float = Field(default=18, ge=10, le=30)
    maximum_body_font_pt: float = Field(default=22, ge=10, le=40)

    @field_validator("name", "title_font", "body_font", "east_asian_font")
    @classmethod
    def _theme_text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("theme text must not be blank")
        return value

    @field_validator("background_color", "primary_color", "accent_color", "text_color", "muted_color")
    @classmethod
    def _hex_colour_is_valid(cls, value: str) -> str:
        if len(value) != 6 or any(character not in "0123456789ABCDEFabcdef" for character in value):
            raise ValueError("theme colors must be six-digit RGB hex values")
        return value.upper()

    @model_validator(mode="after")
    def _body_font_limits_are_ordered(self) -> "ExecutivePolicyTheme":
        if self.minimum_body_font_pt > self.maximum_body_font_pt:
            raise ValueError("minimum body font must not exceed maximum body font")
        return self


class PhysicalSlideLayout(RenderingModel):
    slide_id: str = Field(min_length=1)
    physical_slide_id: str | None = None
    semantic_slide_id: str | None = None
    continuation_index: int = Field(default=0, ge=0)
    is_continuation: bool = False
    page_number: int = Field(default=1, ge=1)
    page_count: int = Field(default=1, ge=1)
    archetype: LayoutArchetype
    elements: list[PositionedElement] = Field(min_length=1)
    source_attributions: list[SourceAttribution] = Field(default_factory=list)
    theme: ExecutivePolicyTheme = Field(default_factory=ExecutivePolicyTheme)

    @field_validator("slide_id", "physical_slide_id", "semantic_slide_id")
    @classmethod
    def _slide_id_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("slide ID must not be blank")
        return value

    @model_validator(mode="after")
    def _element_indices_are_unique(self) -> "PhysicalSlideLayout":
        indices = [element.element_index for element in self.elements if element.element_index is not None and element.element_index >= 0]
        # A cell/table page can legitimately keep one semantic index only once.
        if len(indices) != len(set(indices)):
            raise ValueError("positioned element indexes must be unique per slide")
        if self.page_number > self.page_count:
            raise ValueError("page_number cannot exceed page_count")
        if self.semantic_slide_id is None:
            self.semantic_slide_id = self.slide_id
        if self.physical_slide_id is None:
            self.physical_slide_id = self.slide_id
        if self.physical_slide_id != self.slide_id:
            raise ValueError("physical_slide_id must match slide_id")
        self.continuation_index = self.page_number - 1
        self.is_continuation = self.continuation_index > 0
        return self


class PresentationLayout(RenderingModel):
    slides: list[PhysicalSlideLayout] = Field(min_length=1)
    theme: ExecutivePolicyTheme = Field(default_factory=ExecutivePolicyTheme)

    @model_validator(mode="after")
    def _slide_ids_are_unique_and_themed(self) -> "PresentationLayout":
        slide_ids = [slide.slide_id for slide in self.slides]
        if len(slide_ids) != len(set(slide_ids)):
            raise ValueError("presentation layout slide IDs must be unique")
        return self


class RenderDiagnostic(RenderingModel):
    severity: Literal["info", "warning", "error"]
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    slide_id: str | None = None
    element_index: int | None = None
    physical_slide_id: str | None = None
    source_attribution: SourceAttribution | None = None

    @field_validator("code", "message", "slide_id", "physical_slide_id")
    @classmethod
    def _diagnostic_text_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("diagnostic text must not be blank")
        return value


class RenderReport(RenderingModel):
    output_path: str = Field(min_length=1)
    slides_rendered: int = Field(ge=0)
    semantic_slides_rendered: int = Field(default=0, ge=0)
    physical_slide_ids: list[str] = Field(default_factory=list)
    source_footer_count: int = Field(default=0, ge=0)
    diagnostics: list[RenderDiagnostic] = Field(default_factory=list)
    verified: bool = False
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    source_assets: list[str] = Field(default_factory=list)
    generated_assets: list[str] = Field(default_factory=list)
    independent_verifier_mode: Literal["office", "ooxml_zip"] | None = None
    independent_verified: bool | None = None
