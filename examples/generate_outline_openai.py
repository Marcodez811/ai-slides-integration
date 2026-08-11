#!/usr/bin/env python3
"""Generate a provenance-validated outline JSON artifact from DOCX files.

Usage:
    OPENAI_API_KEY=... uv run python examples/generate_outline_openai.py \\
      briefing.docx --goal "Inform a leadership team" --audience "Executives" \\
      --slides 3 --output output/outline.json
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from presentation_pipeline import generate_outline, write_outline_json
from presentation_pipeline.planning import PresentationRequirements
from presentation_pipeline.providers import OpenAIStructuredGenerator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="+", type=Path, help="One or more DOCX source files.")
    parser.add_argument("--goal", required=True, help="What this presentation must accomplish.")
    parser.add_argument("--audience", required=True, help="Who will consume the presentation.")
    parser.add_argument("--slides", required=True, type=int, help="Requested number of slides (1-100).")
    parser.add_argument("--tone", help="Optional presentation tone, such as executive or educational.")
    parser.add_argument("--presentation-type", help="Optional format, such as briefing or proposal.")
    parser.add_argument("--must-include", action="append", default=[], help="Requirement to include; repeatable.")
    parser.add_argument("--must-avoid", action="append", default=[], help="Requirement to avoid; repeatable.")
    parser.add_argument("--output", type=Path, default=Path("outline.json"), help="Explicit JSON output path.")
    parser.add_argument("--overwrite", action="store_true", help="Permit replacement of an existing output file.")
    return parser


async def _run(args: argparse.Namespace) -> Path:
    requirements = PresentationRequirements(
        goal=args.goal,
        audience=args.audience,
        target_slide_count=args.slides,
        tone=args.tone,
        presentation_type=args.presentation_type,
        must_include=args.must_include,
        must_avoid=args.must_avoid,
    )
    # Keep the model configurable without needing to put a secret or provider
    # configuration in source control.
    model = os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")
    generator = OpenAIStructuredGenerator(model=model)
    outline = await generate_outline(args.documents, requirements, generator)
    return write_outline_json(outline, args.output, overwrite=args.overwrite)


def main() -> None:
    args = _parser().parse_args()
    output_path = asyncio.run(_run(args))
    print(f"Wrote outline JSON: {output_path}")


if __name__ == "__main__":
    main()
