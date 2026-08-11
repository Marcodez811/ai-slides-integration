"""Document-level structured understanding."""

from .models import DocumentDigest, EvidenceRef, KeyFact, StructuredGenerator, TopicDigest
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
