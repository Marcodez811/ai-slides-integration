"""Deterministic presentation evidence indexing."""

from .builder import INDEXER_VERSION, build_document_index
from .models import DocumentIndex, EvidenceItem, EvidenceKind, IndexedSection

__all__ = [
    "DocumentIndex",
    "EvidenceItem",
    "EvidenceKind",
    "INDEXER_VERSION",
    "IndexedSection",
    "build_document_index",
]
