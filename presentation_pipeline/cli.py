"""Command line entry point for an auditable DOCX-to-PPTX generation run."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from .deck import DeckGenerationConfig, generate_deck
from .images import OpenAIImageGenerator
from .observability import configure_run_logging, telemetry_logger
from .planning import PresentationRequirements
from .providers import OpenAIStructuredGenerator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    generate = subcommands.add_parser("generate", help="Generate a PPTX and inspectable JSON artifacts.")
    generate.add_argument("documents", nargs="+", type=Path)
    generate.add_argument("--goal", required=True)
    generate.add_argument("--audience", required=True)
    generate.add_argument("--slides", required=True, type=int)
    generate.add_argument("--tone")
    generate.add_argument("--presentation-type")
    generate.add_argument("--must-include", action="append", default=[])
    generate.add_argument("--must-avoid", action="append", default=[])
    generate.add_argument("--output-dir", required=True, type=Path)
    generate.add_argument("--generate-images", action="store_true")
    generate.add_argument("--max-generated-images", type=int, default=3)
    generate.add_argument("--image-concurrency", type=int, default=1)
    generate.add_argument("--overwrite", action="store_true")
    generate.add_argument("--extraction-workers", type=int, default=4)
    generate.add_argument("--llm-concurrency", type=int, default=4)
    generate.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    generate.add_argument("--debug", action="store_true")
    generate.add_argument("--keep-failed-artifacts", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> Path:
    log_level = "DEBUG" if args.debug else args.log_level
    run_id, log_path = configure_run_logging(output_dir=args.output_dir, log_level=log_level)
    requirements = PresentationRequirements(
        goal=args.goal,
        audience=args.audience,
        target_slide_count=args.slides,
        tone=args.tone,
        presentation_type=args.presentation_type,
        must_include=args.must_include,
        must_avoid=args.must_avoid,
    )
    generator = OpenAIStructuredGenerator(
        model=os.environ.get("OPENAI_MODEL", "gpt-5.6-terra"),
        telemetry_handler=telemetry_logger,
    )
    image_generator = (
        OpenAIImageGenerator(model=os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2"))
        if args.generate_images
        else None
    )
    result = await generate_deck(
        args.documents,
        requirements,
        generator,
        output_dir=args.output_dir,
        image_generator=image_generator,
        config=DeckGenerationConfig(
            generate_images=args.generate_images,
            max_generated_images=args.max_generated_images,
            image_concurrency=args.image_concurrency,
            keep_failed_artifacts=args.keep_failed_artifacts or args.debug,
            run_id=run_id,
            log_path=str(log_path),
        ),
        extraction_workers=args.extraction_workers,
        llm_concurrency=args.llm_concurrency,
        overwrite=args.overwrite,
    )
    return result.pptx_path


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        print(f"Wrote presentation: {asyncio.run(_run(args))}")
