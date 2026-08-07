"""Disposable normalized/semantic views derived from Source IR IDs."""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from .source import IRModel


class NormalizedBlockKind(str, Enum):
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    EQUATION = "equation"
    CHART = "chart"
    UNSUPPORTED = "unsupported"
    TITLE = "title"
    TEXT = "text"
    CAPTION = "caption"
    INDEX = "index"
    HEADER = "header"
    FOOTER = "footer"


class NormalizedBlock(IRModel):
    block_id: str
    kind: NormalizedBlockKind
    source_node_ids: list[str] = Field(min_length=1)
    payload: dict[str, object] = Field(default_factory=dict)


class SectionView(IRModel):
    section_id: str
    source_heading_node_id: str | None = None
    title: str | None = None
    block_ids: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    level: int | None = Field(default=None, ge=1, le=9)
    confidence: float | None = Field(default=None, ge=0, le=1)
    detector_name: str | None = None
    detector_version: str | None = None


class CaptionLink(IRModel):
    caption_node_id: str
    target_node_id: str
    confidence: float = Field(ge=0, le=1)
    detector_name: str
    detector_version: str | None = None


class NormalizedViews(IRModel):
    blocks: list[NormalizedBlock] = Field(default_factory=list)
    sections: list[SectionView] = Field(default_factory=list)
    caption_links: list[CaptionLink] = Field(default_factory=list)
