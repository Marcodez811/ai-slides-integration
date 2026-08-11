"""Provider-neutral presentation planning and semantic synthesis pipeline."""

from .artifacts import write_outline_json, write_presentation_content_json
from .pipeline import BatchExtractionError, generate_outline, generate_plan
from .results import PresentationPlanningResult
from .scale import PlanningScaleConfig, document_window_diagnostics
from .retrieval import CandidateEvidence, CandidateEvidenceSet, WindowedLLMEvidenceRetriever

__all__ = [
    "BatchExtractionError",
    "PresentationPlanningResult",
    "PlanningScaleConfig",
    "document_window_diagnostics",
    "CandidateEvidence",
    "CandidateEvidenceSet",
    "WindowedLLMEvidenceRetriever",
    "generate_outline",
    "generate_plan",
    "write_outline_json",
    "write_presentation_content_json",
]
