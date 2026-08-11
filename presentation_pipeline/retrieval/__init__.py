"""Bounded candidate retrieval for corpus-scale presentation planning."""

from .models import CandidateEvidence, CandidateEvidenceSet, LocalCandidateSelection
from .protocol import EvidenceRetriever
from .windowed import CandidateRetrievalError, WindowedLLMEvidenceRetriever

__all__ = [
    "CandidateEvidence",
    "CandidateEvidenceSet",
    "CandidateRetrievalError",
    "EvidenceRetriever",
    "LocalCandidateSelection",
    "WindowedLLMEvidenceRetriever",
]
