"""Command-line entry points for extraction and artifact validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .api import extract_docx
from .ir import ExtractionResult
from .logging import configure_logging, get_logger


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docx-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="extract a DOCX to canonical JSON")
    extract.add_argument("input", type=Path)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--assets-dir", type=Path)
    _add_logging_arguments(extract)

    validate = subparsers.add_parser("validate", help="validate a saved extraction artifact")
    validate.add_argument("input", type=Path)
    _add_logging_arguments(validate)
    return parser


def _add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", default="INFO", choices=("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"))
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--log-json", action="store_true", help="write structured JSON log records")


def _run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level, args.log_file, args.log_json)
    log = get_logger(command=args.command)
    if args.command == "extract":
        log.info("CLI extraction started")
        result = extract_docx(args.input, asset_output_dir=args.assets_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.canonical_json(), encoding="utf-8")
        log.info("CLI extraction wrote artifact with {} nodes and {} assets", len(result.nodes), len(result.assets))
        return 1 if _has_validation_failure(result) else 0

    log.info("CLI validation started")
    result = ExtractionResult.model_validate_json(args.input.read_text(encoding="utf-8"))
    log.info("CLI validation completed with {} silent losses", result.coverage.silent_losses)
    return 1 if _has_validation_failure(result) else 0


def _has_validation_failure(result: ExtractionResult) -> bool:
    return bool(
        result.coverage.silent_losses
        or any(diagnostic.severity.value == "error" for diagnostic in result.diagnostics)
    )


def main() -> None:
    raise SystemExit(_run())
