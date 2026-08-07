"""Public Source IR and normalized-view model surface."""

from .diagnostics import (
    CoverageMetric,
    CoverageReport,
    Diagnostic,
    DiagnosticSeverity,
    FeatureInventory,
    InventoryOccurrence,
)
from .normalized import CaptionLink, NormalizedBlock, NormalizedBlockKind, NormalizedViews, SectionView
from .result import ExtractionResult
from .source import (
    Asset,
    DocumentMetadata,
    ExtractionStatus,
    PartMetadata,
    SourceLocator,
    SourceNode,
    SourceNodeKind,
)

__all__ = [
    "Asset", "CaptionLink", "CoverageMetric", "CoverageReport", "Diagnostic",
    "DiagnosticSeverity", "DocumentMetadata", "ExtractionResult", "ExtractionStatus",
    "FeatureInventory", "InventoryOccurrence", "NormalizedBlock", "NormalizedBlockKind",
    "NormalizedViews", "PartMetadata", "SectionView", "SourceLocator", "SourceNode",
    "SourceNodeKind",
]
