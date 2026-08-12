"""Document-level structured understanding."""

from presentation_pipeline.common.references import EvidenceRef
from presentation_pipeline.generation import StructuredGenerator

from .models import ChunkDigest, DigestFragment, DocumentDigest, KeyFact, TopicDigest
from .contracts import DigestOutputContract, DigestOutputContractError, digest_output_contract, validate_digest_contract
from .generation import generate_bounded_digest
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
    "ReductionNonReductionError",
    "ReductionInvariantError",
    "StructuredGenerator",
    "TopicDigest",
    "build_evidence_windows",
    "digest_output_contract",
    "generate_bounded_digest",
    "generate_digests",
    "generate_document_digest",
    "validate_digest_scope",
    "validate_digest_contract",
]
