"""Concurrent, context-bound semantic slide-content generation."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from presentation_pipeline.budgeting import GenerationLimiter, TokenCounter, Utf8ByteTokenEstimator
from presentation_pipeline.generation import StructuredGenerator
from presentation_pipeline.observability import safe_debug

from .budgeting import SynthesisBudgets
from .context import SlideContext
from .models import PresentationContent, SlideContent
from .prompts import SLIDE_CONTENT_PROMPT, build_slide_content_input, slide_content_repair_prompt
from .validation import (
    SlideContentValidationError,
    validate_presentation_content,
    validate_slide_content,
)


async def generate_slide_content(
    context: SlideContext,
    generator: StructuredGenerator,
    *,
    limiter: GenerationLimiter | None = None,
    budget: SynthesisBudgets | None = None,
    token_counter: TokenCounter | None = None,
    concurrency: int = 4,
) -> SlideContent:
    """Generate one slide, retrying exactly once only after deterministic invalidity."""
    input_data = build_slide_content_input(context)
    budgets = budget if budget is not None else SynthesisBudgets()
    if not isinstance(budgets, SynthesisBudgets):
        raise TypeError("budget must be a SynthesisBudgets")
    if limiter is None:
        counter = token_counter if token_counter is not None else Utf8ByteTokenEstimator()
        limiter = GenerationLimiter(generator, token_counter=counter, concurrency=concurrency)
    elif token_counter is not None and token_counter is not limiter.token_counter:
        raise ValueError("token_counter must match the shared generation limiter")
    content = await limiter.invoke(
        system_prompt=SLIDE_CONTENT_PROMPT,
        input_data=input_data,
        response_model=SlideContent,
        budget=budgets.slide_content,
        stage="slide_content",
    )
    repaired = False
    try:
        validate_slide_content(content, context)
    except SlideContentValidationError as error:
        repaired = True
        content = await limiter.invoke(
            system_prompt=slide_content_repair_prompt(str(error)),
            input_data=input_data,
            response_model=SlideContent,
            budget=budgets.slide_content,
            stage="slide_content_repair",
        )
        validate_slide_content(content, context)
    kinds: dict[str, int] = {}
    for element in content.elements:
        kinds[element.kind] = kinds.get(element.kind, 0) + 1
    safe_debug(
        "slide_content_complete",
        slide_id=context.slide.slide_id,
        purpose=context.slide.purpose.value,
        element_count=len(content.elements),
        element_kind_counts=kinds,
        validation_retry=repaired,
    )
    return content


async def generate_slide_contents(
    contexts: Sequence[SlideContext],
    generator: StructuredGenerator,
    *,
    concurrency: int = 4,
    limiter: GenerationLimiter | None = None,
    budget: SynthesisBudgets | None = None,
    token_counter: TokenCounter | None = None,
) -> list[SlideContent]:
    """Bound concurrent requests while retaining input order and propagating failures."""
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be an integer of at least 1")
    budgets = budget if budget is not None else SynthesisBudgets()
    if not isinstance(budgets, SynthesisBudgets):
        raise TypeError("budget must be a SynthesisBudgets")
    if limiter is None:
        counter = token_counter if token_counter is not None else Utf8ByteTokenEstimator()
        limiter = GenerationLimiter(generator, token_counter=counter, concurrency=concurrency)
    elif token_counter is not None and token_counter is not limiter.token_counter:
        raise ValueError("token_counter must match the shared generation limiter")

    async def one(context: SlideContext) -> SlideContent:
        return await generate_slide_content(
            context, generator, limiter=limiter, budget=budgets
        )

    return list(await asyncio.gather(*(one(context) for context in contexts)))


def build_presentation_content(
    slides: Sequence[SlideContent],
    contexts: Sequence[SlideContext],
) -> PresentationContent:
    """Construct the persistence-ready aggregate only after full validation."""
    content = PresentationContent(slides=list(slides))
    validate_presentation_content(content, contexts)
    return content
