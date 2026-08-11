"""Deterministic checks that generated outlines meet user requirements."""

from __future__ import annotations

from presentation_pipeline.planning.models import PresentationOutline, PresentationRequirements


class OutlineRequirementsValidationError(ValueError):
    """Generated outline does not meet a deterministic presentation requirement."""


def validate_outline_requirements(
    outline: PresentationOutline, requirements: PresentationRequirements
) -> None:
    """Ensure the outline has exactly the requested number of slides."""
    actual = len(outline.all_slides())
    expected = requirements.target_slide_count
    if actual != expected:
        raise OutlineRequirementsValidationError(
            f"outline has {actual} slides; expected exactly {expected}"
        )
