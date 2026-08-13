"""Bounded candidate retrieval for corpus-scale presentation planning."""

from .models import (
    CandidateDescriptor,
    CandidateEvidence,
    CandidateEvidenceSet,
    CandidateHit,
    CandidateIdentity,
    CandidateReductionSelection,
    LocalCandidateSelection,
)
from .protocol import EvidenceRetriever
from .windowed import CandidateRetrievalError, WindowedLLMEvidenceRetriever

__all__ = [
    "CandidateDescriptor",
    "CandidateEvidence",
    "CandidateEvidenceSet",
    "CandidateHit",
    "CandidateIdentity",
    "CandidateReductionSelection",
    "CandidateRetrievalError",
    "EvidenceRetriever",
    "LocalCandidateSelection",
    "WindowedLLMEvidenceRetriever",
]
