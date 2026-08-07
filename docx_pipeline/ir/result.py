"""Top-level versioned extraction artifact."""

from __future__ import annotations

from pydantic import Field, model_validator

from ..config import ExtractionConfig
from .diagnostics import CoverageReport, Diagnostic, FeatureInventory
from .normalized import NormalizedViews
from .source import Asset, DocumentMetadata, IRModel, PartMetadata, SourceNode


class ExtractionResult(IRModel):
    schema_version: str = "1.0.0"
    extractor_version: str = "0.1.0"
    document: DocumentMetadata
    parts: list[PartMetadata] = Field(default_factory=list)
    nodes: list[SourceNode] = Field(default_factory=list)
    assets: list[Asset] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    inventory: FeatureInventory = Field(default_factory=FeatureInventory)
    coverage: CoverageReport = Field(default_factory=CoverageReport)
    views: NormalizedViews = Field(default_factory=NormalizedViews)
    config: ExtractionConfig = Field(default_factory=ExtractionConfig)

    @model_validator(mode="after")
    def _validate_references(self) -> "ExtractionResult":
        node_ids = {node.node_id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("SourceNode node_id values must be unique")
        sequences = [node.sequence for node in self.nodes]
        if len(sequences) != len(set(sequences)) or sequences != sorted(sequences):
            raise ValueError("SourceNode sequence values must be unique and strictly increasing")
        for node in self.nodes:
            if node.parent_id is not None and node.parent_id not in node_ids:
                raise ValueError(f"unknown parent_id for node {node.node_id}")
        for block in self.views.blocks:
            unknown = set(block.source_node_ids) - node_ids
            if unknown:
                raise ValueError(f"normalized block references unknown nodes: {sorted(unknown)}")
        return self

    def canonical_json(self) -> str:
        """Return the stable JSON representation used by golden tests."""
        from ..ids import canonical_json

        return canonical_json(self)

    def to_canonical_json(self) -> str:
        """Alias retained for serializer adapters."""
        return self.canonical_json()
