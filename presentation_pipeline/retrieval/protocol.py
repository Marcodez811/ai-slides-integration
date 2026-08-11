"""Provider-neutral boundary for bounded evidence retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from presentation_pipeline.budgeting import GenerationLimiter
from presentation_pipeline.planning.models import PresentationRequirements
from presentation_pipeline.retrieval.models import CandidateEvidenceSet
from presentation_pipeline.understanding.models import DocumentDigest


class EvidenceRetriever(Protocol):
    async def retrieve(
        self,
        indexes: Sequence[object],
        digests: Sequence[DocumentDigest],
        requirements: PresentationRequirements,
        *,
        limiter: GenerationLimiter | None = None,
    ) -> CandidateEvidenceSet: ...
