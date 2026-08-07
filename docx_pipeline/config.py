"""Configuration shared by mechanical DOCX extraction and derived views."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ExtractionConfig(BaseModel):
    """Versioned, serializable extraction policy.

    The ordering is part of the Source IR contract: it is deliberately data
    rather than an implementation detail of a package reader.
    """

    model_config = ConfigDict(extra="forbid")

    include_headers_footers: bool = True
    include_notes: bool = True
    include_comments: bool = True
    include_other_linked_parts: bool = False
    extract_linked_media: bool = False
    anchor_reading_order: str = "xml"
    part_order: list[str] = Field(
        default_factory=lambda: [
            "body",
            "headers",
            "footers",
            "footnotes",
            "endnotes",
            "comments",
            "other",
        ]
    )
    normalizer_version: str = "1.0.0"
