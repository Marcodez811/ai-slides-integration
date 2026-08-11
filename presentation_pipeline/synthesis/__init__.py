"""Deterministic slide contexts and validated semantic slide content."""

from .context import ResolvedEvidence, SlideContext, build_slide_contexts
from .models import (
    BulletItem,
    BulletListContent,
    ChartContent,
    EquationContent,
    ImageContent,
    PresentationContent,
    SlideContent,
    TableContent,
    TextContent,
)
from .service import build_presentation_content, generate_slide_content, generate_slide_contents
from .validation import SlideContentValidationError, validate_presentation_content, validate_slide_content

__all__ = [
    "BulletItem", "BulletListContent", "ChartContent", "EquationContent", "ImageContent",
    "PresentationContent", "ResolvedEvidence", "SlideContent", "SlideContentValidationError",
    "SlideContext", "TableContent", "TextContent", "build_presentation_content", "build_slide_contexts", "generate_slide_content",
    "generate_slide_contents", "validate_presentation_content", "validate_slide_content",
]
