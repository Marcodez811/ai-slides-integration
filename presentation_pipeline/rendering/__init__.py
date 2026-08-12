"""Deterministic layout and editable PPTX rendering contracts."""

from .layout import (
    LayoutValidationError,
    build_presentation_layout,
    build_slide_layout,
    select_layout,
    text_capacity,
    validate_layout,
)
from .inputs import resolve_render_inputs
from .models import (
    Box,
    ChartSeries,
    ExecutivePolicyTheme,
    LayoutArchetype,
    PhysicalSlideLayout,
    PositionedElement,
    PresentationLayout,
    RenderDiagnostic,
    RenderReport,
    RenderInputResult,
    ResolvedElement,
    SourceAttribution,
    TableCell,
    TablePayload,
)
from .renderer import PresentationRenderError, render_presentation

__all__ = [
    "Box", "ChartSeries", "ExecutivePolicyTheme", "LayoutArchetype", "LayoutValidationError",
    "PhysicalSlideLayout", "PositionedElement", "PresentationLayout", "PresentationRenderError",
    "RenderDiagnostic", "RenderInputResult", "RenderReport", "ResolvedElement", "SourceAttribution",
    "TableCell", "TablePayload", "build_presentation_layout", "build_slide_layout",
    "render_presentation", "resolve_render_inputs", "select_layout", "text_capacity", "validate_layout",
]
