"""Evidence selection and validated presentation outline generation."""

from .models import (
    ContentForm,
    EvidenceSelection,
    OutlineSection,
    PresentationOutline,
    PresentationRequirements,
    SelectedEvidence,
    SlideOutline,
    SlidePurpose,
)
from .service import generate_presentation_outline, select_evidence

__all__ = [
    "ContentForm",
    "EvidenceSelection",
    "OutlineSection",
    "PresentationOutline",
    "PresentationRequirements",
    "SelectedEvidence",
    "SlideOutline",
    "SlidePurpose",
    "generate_presentation_outline",
    "select_evidence",
]
