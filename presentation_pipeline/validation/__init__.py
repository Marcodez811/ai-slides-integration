"""Deterministic provenance checks for generated planning artifacts."""

from .provenance import (
    CorpusLookup,
    ProvenanceValidationError,
    validate_document_digests,
    validate_evidence_selection,
    validate_presentation_outline,
)
from .requirements import OutlineRequirementsValidationError, validate_outline_requirements

validate_digests = validate_document_digests
validate_outline = validate_presentation_outline

__all__ = [
    "CorpusLookup",
    "ProvenanceValidationError",
    "OutlineRequirementsValidationError",
    "validate_document_digests",
    "validate_digests",
    "validate_evidence_selection",
    "validate_outline",
    "validate_presentation_outline",
    "validate_outline_requirements",
]
