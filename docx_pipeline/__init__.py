"""Auditable, provenance-preserving DOCX extraction."""

from .config import ExtractionConfig
from .ids import canonical_json, make_asset_id, make_doc_id, make_node_id, make_view_id
from .inventory import canonical_xml_paths, inventory_package
from .ir import (
    Asset,
    CaptionLink,
    CoverageMetric,
    CoverageReport,
    Diagnostic,
    DiagnosticSeverity,
    DocumentMetadata,
    ExtractionResult,
    ExtractionStatus,
    FeatureInventory,
    InventoryOccurrence,
    NormalizedBlock,
    NormalizedBlockKind,
    NormalizedViews,
    PartMetadata,
    SectionView,
    SourceLocator,
    SourceNode,
    SourceNodeKind,
)
from .package import DocxPackage, PackageDiagnostic, PackagePreflight
from .relationships import Relationship, resolve_relationship_target

__all__ = [
    "Asset", "CaptionLink", "CoverageMetric", "CoverageReport", "Diagnostic",
    "DiagnosticSeverity", "DocumentMetadata", "DocxPackage", "ExtractionConfig",
    "ExtractionResult", "ExtractionStatus", "FeatureInventory", "InventoryOccurrence",
    "NormalizedBlock", "NormalizedBlockKind", "NormalizedViews", "PackageDiagnostic",
    "PackagePreflight", "PartMetadata", "Relationship", "SectionView", "SourceLocator",
    "SourceNode", "SourceNodeKind", "canonical_json", "canonical_xml_paths", "inventory_package", "make_asset_id",
    "make_doc_id", "make_node_id", "make_view_id", "resolve_relationship_target",
]
