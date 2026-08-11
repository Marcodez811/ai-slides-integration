"""Concurrent, context-bound semantic slide-content generation."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from presentation_pipeline.generation import StructuredGenerator, invoke_structured

from .context import SlideContext
from .models import PresentationContent, SlideContent
from .prompts import SLIDE_CONTENT_PROMPT, build_slide_content_input, slide_content_repair_prompt
from .validation import (
    SlideContentValidationError,
    validate_presentation_content,
    validate_slide_content,
)


async def generate_slide_content(context: SlideContext, generator: StructuredGenerator) -> SlideContent:
    """Generate one slide, retrying exactly once only after deterministic invalidity."""
    input_data = build_slide_content_input(context)
    content = await invoke_structured(generator, SLIDE_CONTENT_PROMPT, input_data, SlideContent)
    try:
        validate_slide_content(content, context)
    except SlideContentValidationError as error:
        content = await invoke_structured(
            generator,
            slide_content_repair_prompt(str(error)),
            input_data,
            SlideContent,
        )
        validate_slide_content(content, context)
    return content


async def generate_slide_contents(
    contexts: Sequence[SlideContext],
    generator: StructuredGenerator,
    *,
    concurrency: int = 4,
) -> list[SlideContent]:
    """Bound concurrent requests while retaining input order and propagating failures."""
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be an integer of at least 1")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(context: SlideContext) -> SlideContent:
        async with semaphore:
            return await generate_slide_content(context, generator)

    return list(await asyncio.gather(*(one(context) for context in contexts)))


def build_presentation_content(
    slides: Sequence[SlideContent],
    contexts: Sequence[SlideContext],
) -> PresentationContent:
    """Construct the persistence-ready aggregate only after full validation."""
    content = PresentationContent(slides=list(slides))
    validate_presentation_content(content, contexts)
    return content
