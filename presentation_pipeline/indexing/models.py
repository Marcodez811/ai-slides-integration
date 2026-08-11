"""Stable, provenance-preserving models for presentation evidence indexing."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from presentation_pipeline.common.models import PipelineModel


class EvidenceKind(str, Enum):
    """Evidence types understood by the presentation planning stages."""

    TEXT = "text"
    TITLE = "title"
    CAPTION = "caption"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    EQUATION = "equation"
    CHART = "chart"
    CHART_CANDIDATE = "chart_candidate"


class EvidenceItem(PipelineModel):
    """One indexed, source-traceable item available to slide planning."""

    doc_id: str
    evidence_id: str
    kind: EvidenceKind
    block_ids: list[str] = Field(min_length=1)
    source_node_ids: list[str] = Field(min_length=1)
    section_ids: list[str] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    text: str | None = None
    structured_data: dict[str, Any] = Field(default_factory=dict)


class IndexedSection(PipelineModel):
    """A faithful copy of a normalized section plus its indexed evidence."""

    section_id: str
    source_heading_node_id: str | None = None
    title: str | None = None
    block_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    level: int | None = Field(default=None, ge=1, le=9)
    confidence: float | None = Field(default=None, ge=0, le=1)
    detector_name: str | None = None
    detector_version: str | None = None


class DocumentIndex(PipelineModel):
    """Deterministic, ordered evidence derived from a document artifact."""

    schema_version: str = "1.0.0"
    doc_id: str
    filename: str
    extraction_schema_version: str
    extractor_version: str
    indexer_version: str = "1.0.0"
    evidence: list[EvidenceItem] = Field(default_factory=list)
    sections: list[IndexedSection] = Field(default_factory=list)
