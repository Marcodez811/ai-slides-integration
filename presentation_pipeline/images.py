"""Optional, explicitly decorative image generation for rendered decks."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """A local decorative image produced for a single slide."""

    slide_id: str
    path: Path
    prompt: str
    model: str


class ImageGenerator(Protocol):
    """Generate one deliberately non-evidentiary illustration for a slide."""

    async def generate_illustration(
        self,
        *,
        slide_id: str,
        title: str,
        message: str,
        audience: str,
        tone: str | None,
        output_dir: str | Path,
    ) -> GeneratedImage: ...


class OpenAIImageGenerator:
    """Direct OpenAI Image API adapter for optional deck illustrations."""

    def __init__(self, *, model: str = "gpt-image-2", client: object | None = None) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
        self.model = model
        self.client = client

    async def generate_illustration(
        self,
        *,
        slide_id: str,
        title: str,
        message: str,
        audience: str,
        tone: str | None,
        output_dir: str | Path,
    ) -> GeneratedImage:
        prompt = illustration_prompt(
            title=title,
            message=message,
            audience=audience,
            tone=tone,
        )
        response = await self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size="2048x1152",
            quality="medium",
            output_format="png",
        )
        encoded = response.data[0].b64_json
        if not isinstance(encoded, str) or not encoded:
            raise RuntimeError("OpenAI image response contained no image data")
        output_path = Path(output_dir) / f"{slide_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(encoded))
        return GeneratedImage(slide_id=slide_id, path=output_path, prompt=prompt, model=self.model)


def illustration_prompt(*, title: str, message: str, audience: str, tone: str | None) -> str:
    """Build a bounded prompt that cannot turn an illustration into evidence."""
    style = tone or "restrained executive-policy briefing"
    return (
        "Create one polished, landscape editorial illustration for a presentation slide. "
        f"Theme: {title}. Concept to evoke: {message}. Audience: {audience}. Style: {style}. "
        "Use an abstract or non-identifying scene with a calm contemporary public-service aesthetic. "
        "This is a decorative illustration only: include no words, numbers, data charts, diagrams, "
        "logos, emblems, screenshots, document pages, named people, or factual claims."
    )
