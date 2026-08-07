"""Lossless, ordered Source IR records."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IRModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceLocator(IRModel):
    """Exact package/XML provenance for one source occurrence."""

    part_name: str
    xml_path: str
    relationship_id: str | None = None
    body_index: int | None = Field(default=None, ge=0)
    char_range: tuple[int, int] | None = None

    @field_validator("part_name", "xml_path")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("source locator values must not be empty")
        return value

    @field_validator("char_range")
    @classmethod
    def _valid_character_range(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is not None and (value[0] < 0 or value[1] < value[0]):
            raise ValueError("char_range must be a non-negative increasing range")
        return value


class SourceNodeKind(str, Enum):
    DOCUMENT_BODY = "document_body"
    HEADER = "header"
    FOOTER = "footer"
    FOOTNOTE = "footnote"
    ENDNOTE = "endnote"
    COMMENT = "comment"
    PART = "part"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    TEXT_RUN = "text_run"
    HYPERLINK = "hyperlink"
    IMAGE = "image"
    EQUATION = "equation"
    LINE_BREAK = "line_break"
    TAB = "tab"
    FIELD = "field"
    BOOKMARK = "bookmark"
    COMMENT_REFERENCE = "comment_reference"
    PAGE_BREAK = "page_break"
    SECTION_BREAK = "section_break"
    CHART = "chart"
    CONTENT_CONTROL = "content_control"
    TEXT_BOX = "text_box"
    UNSUPPORTED = "unsupported"


class ExtractionStatus(str, Enum):
    EXTRACTED = "extracted"
    IGNORED = "ignored"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class SourceNode(IRModel):
    """One mechanically discovered source occurrence in global pre-order."""

    node_id: str
    sequence: int = Field(ge=0)
    parent_id: str | None = None
    sibling_index: int = Field(ge=0)
    kind: SourceNodeKind
    source: SourceLocator
    payload: dict[str, Any] = Field(default_factory=dict)
    status: ExtractionStatus = ExtractionStatus.EXTRACTED
    diagnostic_ids: list[str] = Field(default_factory=list)


class Asset(IRModel):
    """A deduplicated binary asset; occurrences remain separate image nodes."""

    asset_id: str
    sha256: str
    original_name: str | None = None
    mime_type: str | None = None
    path: str | None = None
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    size_bytes: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_parts: list[str] = Field(default_factory=list)
    derivatives: dict[str, str] = Field(default_factory=dict)

    @property
    def byte_size(self) -> int | None:
        """Compatibility spelling for callers that use ``byte_size``."""
        return self.size_bytes


class DocumentMetadata(IRModel):
    doc_id: str
    filename: str
    file_type: Literal["docx"] = "docx"
    sha256: str
    sanitized_sha256: str | None = None


class PartMetadata(IRModel):
    part_name: str
    content_type: str | None = None
    kind: str | None = None
    relationship_count: int = Field(default=0, ge=0)
