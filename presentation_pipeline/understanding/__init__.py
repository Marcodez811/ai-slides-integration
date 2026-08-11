"""Document-level structured understanding."""

from presentation_pipeline.common.references import EvidenceRef
from presentation_pipeline.generation import StructuredGenerator

from .models import DocumentDigest, KeyFact, TopicDigest
from .service import generate_digests, generate_document_digest

__all__ = [
    "DocumentDigest",
    "EvidenceRef",
    "KeyFact",
    "StructuredGenerator",
    "TopicDigest",
    "generate_digests",
    "generate_document_digest",
]
