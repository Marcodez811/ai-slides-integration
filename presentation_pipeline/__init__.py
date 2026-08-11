"""Provider-neutral presentation planning and semantic synthesis pipeline."""

from .artifacts import write_outline_json, write_presentation_content_json
from .pipeline import BatchExtractionError, generate_outline, generate_plan
from .results import PresentationPlanningResult

__all__ = [
    "BatchExtractionError",
    "PresentationPlanningResult",
    "generate_outline",
    "generate_plan",
    "write_outline_json",
    "write_presentation_content_json",
]
