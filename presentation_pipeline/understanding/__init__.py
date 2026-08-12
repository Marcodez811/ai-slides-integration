"""Document-level structured understanding."""

from presentation_pipeline.common.references import EvidenceRef
from presentation_pipeline.generation import StructuredGenerator

from .contracts import DigestOutputContract, DigestOutputContractError
from .models import ChunkDigest, DigestFragment, DocumentDigest, KeyFact, TopicDigest
from .reduction import DigestProvenanceError, ReductionInvariantError, ReductionNonReductionError, validate_digest_scope
from .service import ChunkDigestValidationError, generate_digests, generate_document_digest
from .windows import EvidenceWindow, EvidenceWindowingError, OversizedEvidenceError, build_evidence_windows

__all__ = [
    "ChunkDigest",
    "ChunkDigestValidationError",
    "DigestFragment",
    "DigestOutputContract",
    "DigestOutputContractError",
    "DigestProvenanceError",
    "DocumentDigest",
    "EvidenceWindow",
    "EvidenceWindowingError",
    "EvidenceRef",
    "KeyFact",
    "OversizedEvidenceError",
    "ReductionInvariantError",
    "ReductionNonReductionError",
    "StructuredGenerator",
    "TopicDigest",
    "build_evidence_windows",
    "generate_digests",
    "generate_document_digest",
    "validate_digest_scope",
]
