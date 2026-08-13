"""End-to-end assembly of validated semantic content into a reviewable deck."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from docx_pipeline.enrichment.table_profiler import profile_table

from presentation_pipeline.artifacts import write_outline_json, write_presentation_content_json
from presentation_pipeline.generation import StructuredGenerator
from presentation_pipeline.images import ImageGenerator
from presentation_pipeline.pipeline import generate_plan
from presentation_pipeline.observability import new_run_id, run_metrics, safe_event, stage_timer
from presentation_pipeline.planning.models import PresentationRequirements, SlidePurpose
from presentation_pipeline.results import PresentationPlanningResult
from presentation_pipeline.synthesis import (
    PresentationContent,
    TableContent,
    build_presentation_content,
    build_slide_contexts,
    generate_slide_contents,
)


@dataclass(frozen=True, slots=True)
class DeckGenerationConfig:
    """Deliberately narrow configuration for the first reviewable renderer."""

    generate_images: bool = False
    max_generated_images: int = 3
    image_concurrency: int = 1
    keep_failed_artifacts: bool = False
    run_id: str | None = None
    log_path: str | None = None

    def __post_init__(self) -> None:
        if self.max_generated_images < 0:
            raise ValueError("max_generated_images must not be negative")
        if self.image_concurrency < 1:
            raise ValueError("image_concurrency must be at least 1")
        if self.run_id is not None and not self.run_id.strip():
            raise ValueError("run_id must not be blank")


@dataclass(frozen=True, slots=True)
class DeckGenerationResult:
    """Inspectable products of one completed DOCX-to-PPTX run."""

    planning: PresentationPlanningResult
    content: PresentationContent
    pptx_path: Path
    outline_path: Path
    content_path: Path
    layout_path: Path
    report_path: Path
    diagnostics: tuple[dict[str, object], ...] = field(default_factory=tuple)


class DeckPromotionError(RuntimeError):
    """Raised when a staged deck cannot safely become the requested output."""


async def generate_deck(
    input_paths: Sequence[str | Path],
    requirements: PresentationRequirements,
    generator: StructuredGenerator,
    *,
    output_dir: str | Path,
    image_generator: ImageGenerator | None = None,
    config: DeckGenerationConfig | None = None,
    extraction_workers: int = 4,
    llm_concurrency: int = 4,
    overwrite: bool = False,
) -> DeckGenerationResult:
    """Create a real PPTX plus JSON audit artifacts from DOCX sources.

    Source media is materialized below the requested output directory, so the
    renderer can consume the same asset identities validated by synthesis.
    """
    from presentation_pipeline.rendering import (
        build_presentation_layout,
        render_presentation,
        validate_layout,
    )

    settings = config or DeckGenerationConfig()
    run_id = settings.run_id or new_run_id()
    started = time.perf_counter()
    root = Path(output_dir).absolute()
    staging_root = _create_staging_root(root, overwrite=overwrite)
    artifacts = _artifact_paths(staging_root)
    published_artifacts = _artifact_paths(root)
    failed_stage = "planning"
    try:
        with stage_timer("planning", document_count=len(input_paths)):
            plan = await generate_plan(
                input_paths,
                requirements,
                generator,
                extraction_workers=extraction_workers,
                llm_concurrency=llm_concurrency,
                asset_output_dir=staging_root / "assets" / "source",
            )
        failed_stage = "slide_context_resolution"
        with stage_timer(failed_stage, semantic_slide_count=len(plan.outline.all_slides())):
            contexts = build_slide_contexts(plan)
        failed_stage = "slide_content_generation"
        with stage_timer(failed_stage, slide_count=len(contexts)):
            slides = await generate_slide_contents(contexts, generator, concurrency=llm_concurrency)
        content = build_presentation_content(slides, contexts)
        diagnostics: list[dict[str, object]] = []
        failed_stage = "render_input_resolution"
        with stage_timer(failed_stage, slide_count=len(content.slides)):
            resolved_by_slide = _resolve_render_elements(plan, contexts, content, staging_root, diagnostics)
        image_outcomes: list[dict[str, object]] = []
        if settings.generate_images and image_generator is not None:
            failed_stage = "decorative_image_generation"
            with stage_timer(failed_stage):
                image_outcomes = await _add_generated_images(
                    contexts, content, resolved_by_slide, image_generator,
                    staging_root / "assets" / "generated", settings, diagnostics,
                    audience=plan.requirements.audience, tone=plan.requirements.tone,
                )

        failed_stage = "layout_generation"
        with stage_timer(failed_stage):
            layout = build_presentation_layout(contexts, content.slides, resolved_by_slide)
        archetypes: dict[str, int] = {}
        for physical_slide in layout.slides:
            key = physical_slide.archetype.value
            archetypes[key] = archetypes.get(key, 0) + 1
        safe_event(
            "layout_complete",
            semantic_slide_count=len(content.slides),
            physical_slide_count=len(layout.slides),
            continuation_slide_count=sum(slide.is_continuation for slide in layout.slides),
            layout_archetype_counts=archetypes,
        )
        failed_stage = "layout_validation"
        with stage_timer(failed_stage, physical_slide_count=len(layout.slides)):
            validation = [diagnostic for physical_slide in layout.slides for diagnostic in validate_layout(physical_slide)]
        diagnostics.extend(_as_json(item) for item in validation)
        blocking = [item for item in validation if _diagnostic_severity(item) == "error"]
        if blocking:
            raise ValueError(f"layout validation failed with {len(blocking)} error(s)")

        failed_stage = "pptx_render"
        with stage_timer(failed_stage):
            render_report = render_presentation(layout, artifacts["pptx"])
        safe_event(
            "pptx_render_complete",
            semantic_slide_count=len(content.slides),
            physical_slide_count=len(layout.slides),
            continuation_slide_count=sum(slide.is_continuation for slide in layout.slides),
            render_warning_count=sum(_diagnostic_severity(item) == "warning" for item in render_report.diagnostics),
            render_error_count=sum(_diagnostic_severity(item) == "error" for item in render_report.diagnostics),
            source_image_count=len(render_report.source_assets),
            generated_image_count=len(render_report.generated_assets),
        )
        # Do not trust the renderer's library-level reopen alone.  The package
        # validator is the promotion gate and has no python-pptx dependency.
        from presentation_pipeline.rendering.verification import verify_pptx

        failed_stage = "pptx_verification"
        with stage_timer(failed_stage):
            verification = verify_pptx(
                artifacts["pptx"], expected_slide_count=len(getattr(layout, "slides", ())),
                reported_path=published_artifacts["pptx"],
            )
        diagnostics.extend(_as_json(item) for item in verification.diagnostics)
        safe_event(
            "pptx_verification_complete",
            verification_mode=verification.verifier_mode,
            verified=verification.verified,
            expected_slide_count=len(layout.slides),
            actual_slide_count=verification.slide_count,
        )
        if not verification.verified:
            raise DeckPromotionError("PPTX package verification failed; staged output was not promoted")
        render_report = render_report.model_copy(update={
            "independent_verifier_mode": verification.verifier_mode,
            "independent_verified": verification.verified,
        })

        failed_stage = "artifact_write"
        with stage_timer(failed_stage):
            write_outline_json(plan.outline, artifacts["outline"], overwrite=False)
            write_presentation_content_json(content, artifacts["content"], overwrite=False)
            _write_json(artifacts["layout"], layout, overwrite=False, staging_root=staging_root, published_root=root)
        full_report = {
            "pptx_path": str(published_artifacts["pptx"]),
            "semantic_slide_count": len(content.slides),
            "physical_slide_count": len(getattr(layout, "slides", ())),
            "renderer": _replace_staging_paths(_as_json(render_report), staging_root, root),
            "verification": verification.as_dict(),
            "image_generation": {
                "enabled": settings.generate_images,
                "max_generated_images": settings.max_generated_images,
                "concurrency": settings.image_concurrency,
                "outcomes": image_outcomes,
            },
            "diagnostics": diagnostics,
        }
        _write_json(artifacts["report"], full_report, overwrite=False, staging_root=staging_root, published_root=root)
        failed_stage = "promotion"
        with stage_timer(failed_stage):
            _promote_staging_root(staging_root, root, overwrite=overwrite)
    except BaseException as error:
        failed_stage = _failure_stage(failed_stage, error)
        failed_root = _preserve_failure(
            staging_root, root, run_id=run_id, keep=settings.keep_failed_artifacts
        )
        report = _failure_report(
            root=root,
            run_id=run_id,
            failed_stage=failed_stage,
            error=error,
            elapsed_seconds=time.perf_counter() - started,
            input_paths=input_paths,
            log_path=settings.log_path,
            failed_artifacts_path=failed_root,
        )
        _write_json(report[0], report[1], overwrite=True)
        safe_event("deck_failed", run_id=run_id, failed_stage=failed_stage, error_type=type(error).__name__)
        raise
    metrics = run_metrics()
    if metrics is not None:
        metrics.semantic_slide_count = len(content.slides)
        metrics.physical_slide_count = len(layout.slides)
    safe_event(
        "run_complete",
        run_id=run_id,
        document_count=len(input_paths),
        semantic_slide_count=len(content.slides),
        physical_slide_count=len(layout.slides),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        metrics=metrics.summary() if metrics is not None else {},
    )
    if metrics is not None:
        summary = metrics.summary()
        safe_event(
            "run_summary",
            documents=summary["document_count"],
            windows=summary["window_count"],
            chunk_digests=summary["chunk_digest_count"],
            llm_calls=summary["provider_call_count"],
            reduction_calls=summary["reduction_call_count"],
            compaction_calls=summary["compaction_call_count"],
            actual_input_tokens=summary["actual_provider_input_tokens"],
            actual_output_tokens=summary["actual_provider_output_tokens"],
            provider_latency_ms=summary["provider_latency_ms"],
            semantic_slides=summary["semantic_slide_count"],
            physical_slides=summary["physical_slide_count"],
            average_estimate_to_actual_ratio=summary["average_estimate_to_actual_ratio"],
            median_estimate_to_actual_ratio=summary["median_estimate_to_actual_ratio"],
            p90_estimate_to_actual_ratio=summary["p90_estimate_to_actual_ratio"],
            p95_estimate_to_actual_ratio=summary["p95_estimate_to_actual_ratio"],
            max_estimate_to_actual_ratio=summary["max_estimate_to_actual_ratio"],
            estimate_to_actual_ratio_by_stage=summary["estimate_to_actual_ratio_by_stage"],
            estimate_to_actual_ratio_by_stage_and_response_model=summary[
                "estimate_to_actual_ratio_by_stage_and_response_model"
            ],
            calls_by_response_model=summary["calls_by_response_model"],
            calls_by_stage=summary["calls_by_stage"],
        )
    return DeckGenerationResult(
        planning=plan,
        content=content,
        pptx_path=published_artifacts["pptx"],
        outline_path=published_artifacts["outline"],
        content_path=published_artifacts["content"],
        layout_path=published_artifacts["layout"],
        report_path=published_artifacts["report"],
        diagnostics=tuple(diagnostics),
    )


def _resolve_render_elements(
    plan: PresentationPlanningResult,
    contexts: Sequence[object],
    content: PresentationContent,
    root: Path,
    diagnostics: list[dict[str, object]],
) -> dict[str, object]:
    """Resolve every semantic element through the renderer's frozen typed API."""
    from presentation_pipeline.rendering.inputs import resolve_render_inputs

    evidence_by_identity = {
        (index.doc_id, item.evidence_id): item
        for index in plan.indexes
        for item in index.evidence
    }
    assets_by_doc: dict[str, dict[str, object]] = {}
    artifacts_by_doc = {artifact.doc_id: artifact for artifact in plan.artifacts}
    nodes_by_doc = {
        artifact.doc_id: {node.node_id: node for node in artifact.extraction.nodes}
        for artifact in plan.artifacts
    }
    source_filenames = {artifact.doc_id: artifact.filename for artifact in plan.artifacts}
    for artifact in plan.artifacts:
        source_dir = root / "assets" / "source" / artifact.job_id
        assets_by_doc[artifact.doc_id] = {
            asset.asset_id: {"path": str(source_dir / asset.path)}
            for asset in artifact.extraction.assets
            if isinstance(getattr(asset, "path", None), str) and asset.path
        }
    output: dict[str, object] = {}
    for slide in content.slides:
        render_elements: list[object] = []
        for element in slide.elements:
            if isinstance(element, TableContent):
                evidence = evidence_by_identity.get((element.doc_id, element.evidence_id))
                artifact = artifacts_by_doc.get(element.doc_id)
                table = None
                if evidence is not None and artifact is not None:
                    node_ids = getattr(evidence, "source_node_ids", [])
                    table = nodes_by_doc.get(element.doc_id, {}).get(node_ids[0]) if node_ids else None
                if table is not None and getattr(table.kind, "value", table.kind) == "table":
                    profile = profile_table(table, artifact.extraction.nodes)
                    if profile.rectangular:
                        row_count = max((cell.row + cell.row_span for cell in profile.cells), default=0)
                        rows = [["" for _ in range(profile.column_count)] for _ in range(row_count)]
                        for cell in profile.cells:
                            rows[cell.row][cell.column] = {
                                "text": cell.original,
                                "row_span": cell.row_span,
                                "column_span": cell.column_span,
                            }
                        render_elements.append({
                            **element.model_dump(mode="python"),
                            "rows": rows,
                            "header_rows": list(profile.header_rows),
                        })
                        continue
            render_elements.append(element)
        result = resolve_render_inputs(
            slide.slide_id,
            render_elements,
            source_lookup=evidence_by_identity,
            asset_root=root / "assets" / "source",
            assets_by_doc=assets_by_doc,
            source_filenames=source_filenames,
        )
        diagnostics.extend(_as_json(item) for item in result.diagnostics)
        output[slide.slide_id] = result
    return output


async def _add_generated_images(
    contexts: Sequence[object],
    content: PresentationContent,
    resolved: dict[str, object],
    generator: ImageGenerator,
    output_dir: Path,
    config: DeckGenerationConfig,
    diagnostics: list[dict[str, object]],
    *,
    audience: str,
    tone: str | None,
) -> list[dict[str, object]]:
    """Generate a stable, capped candidate list before any provider call.

    ``gather`` preserves this candidate order even when provider responses
    finish out of order.  The semaphore limits only remote calls, not the
    deterministic eligibility decision.
    """
    from presentation_pipeline.rendering.models import ResolvedElement

    by_slide = {slide.slide_id: slide for slide in content.slides}
    candidates: list[tuple[object, object]] = []
    for context in contexts:
        slide = by_slide[context.slide.slide_id]
        forms = {getattr(form, "value", form) for form in context.slide.preferred_content_forms}
        existing = getattr(resolved[slide.slide_id], "elements", resolved[slide.slide_id])
        kinds = {item.kind if hasattr(item, "kind") else item["kind"] for item in existing}
        if context.slide.purpose is not SlidePurpose.CONTENT or {"table", "chart", "image"}.intersection(kinds):
            continue
        if "image" not in forms and "mixed" not in forms:
            continue
        candidates.append((context, slide))
    candidates = candidates[:config.max_generated_images]
    semaphore = asyncio.Semaphore(config.image_concurrency)

    async def generate_one(context: object, slide: object):
        async with semaphore:
            return await generator.generate_illustration(
                slide_id=slide.slide_id,
                title=context.slide.title,
                message=context.slide.message,
                audience=audience,
                tone=tone,
                output_dir=output_dir,
            )

    attempts = await asyncio.gather(
        *(generate_one(context, slide) for context, slide in candidates),
        return_exceptions=True,
    )
    outcomes: list[dict[str, object]] = []
    for (context, slide), attempt in zip(candidates, attempts, strict=True):
        if isinstance(attempt, BaseException):
            diagnostics.append(_warning("GENERATED_IMAGE_UNAVAILABLE", slide.slide_id, f"Decorative image generation failed: {type(attempt).__name__}"))
            outcomes.append({"slide_id": slide.slide_id, "outcome": "failed", "error_type": type(attempt).__name__})
        else:
            generated = attempt
            if generated.slide_id != slide.slide_id:
                diagnostics.append(_warning("GENERATED_IMAGE_ID_MISMATCH", slide.slide_id, "Decorative image response was assigned to the requested slide."))
            elements = getattr(resolved[slide.slide_id], "elements", resolved[slide.slide_id])
            elements.append(ResolvedElement(element_index=-1, kind="image", path=str(generated.path)))
            outcomes.append({
                "slide_id": slide.slide_id,
                "outcome": "success",
                "model": generated.model,
                "prompt_hash": hashlib.sha256(generated.prompt.encode("utf-8")).hexdigest(),
                "path": str(generated.path),
            })
    return outcomes


def _warning(code: str, slide_id: str, message: str) -> dict[str, object]:
    return {"severity": "warning", "code": code, "slide_id": slide_id, "message": message}


def _as_json(value: object) -> dict[str, object] | object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


def _diagnostic_severity(value: object) -> str:
    severity = getattr(value, "severity", None)
    return getattr(severity, "value", severity) or "warning"


def _write_json(
    path: Path,
    value: object,
    *,
    overwrite: bool,
    staging_root: Path | None = None,
    published_root: Path | None = None,
) -> None:
    serializable = _as_json(value)
    if staging_root is not None and published_root is not None:
        serializable = _replace_staging_paths(serializable, staging_root, published_root)
    payload = json.dumps(serializable, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
        handle.write(payload)


def _artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "pptx": root / "deck.pptx",
        "outline": root / "outline.json",
        "content": root / "content.json",
        "layout": root / "layout.json",
        "report": root / "render-report.json",
    }


def _create_staging_root(root: Path, *, overwrite: bool) -> Path:
    """Create an empty sibling directory without touching published output."""
    if root.is_symlink():
        raise ValueError("output_dir must not be a symlink")
    if root.exists():
        if not root.is_dir():
            raise ValueError("output_dir must be a directory path")
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = root.parent / f".{root.name}.staging-{uuid4().hex}"
    staging_root.mkdir(mode=0o700)
    return staging_root


def _promote_staging_root(staging_root: Path, root: Path, *, overwrite: bool) -> None:
    """Promote a complete staging directory and restore prior output on error."""
    backup: Path | None = None
    try:
        if root.exists():
            if not overwrite:  # defensive against a racing creator
                raise FileExistsError(f"output directory already exists: {root}")
            if root.is_symlink() or not root.is_dir():
                raise DeckPromotionError("output_dir changed into an unsafe path during generation")
            backup = root.parent / f".{root.name}.backup-{uuid4().hex}"
            root.replace(backup)
        staging_root.replace(root)
    except BaseException as error:
        if backup is not None and backup.exists() and not root.exists():
            backup.replace(root)
        raise DeckPromotionError("could not promote staged deck output") from error
    if backup is not None:
        _remove_staging_root(backup)


def _remove_staging_root(path: Path) -> None:
    if path.exists() and path.is_dir() and path.name.startswith("."):
        shutil.rmtree(path, ignore_errors=True)


def _preserve_failure(staging_root: Path, root: Path, *, run_id: str, keep: bool) -> Path | None:
    if not keep:
        _remove_staging_root(staging_root)
        return None
    failure_root = root.parent / ".presentation-failures" / f"{root.name}-{run_id}"
    failure_root.parent.mkdir(parents=True, exist_ok=True)
    if staging_root.exists():
        staging_root.replace(failure_root)
    return failure_root


def _failure_report(
    *,
    root: Path,
    run_id: str,
    failed_stage: str,
    error: BaseException,
    elapsed_seconds: float,
    input_paths: Sequence[str | Path],
    log_path: str | None,
    failed_artifacts_path: Path | None,
) -> tuple[Path, dict[str, object]]:
    directory = root.parent / ".presentation-failures"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{root.name}-{run_id}.json"
    metadata = getattr(error, "metadata", None)
    report: dict[str, object] = {
        "run_id": run_id,
        "success": False,
        "failed_stage": failed_stage,
        "error_type": type(error).__name__,
        "message": str(error)[:512],
        "elapsed_seconds": round(elapsed_seconds, 3),
        "documents": [Path(item).name for item in input_paths],
        "log_path": log_path,
        "failed_artifacts_path": str(failed_artifacts_path) if failed_artifacts_path else None,
    }
    if isinstance(metadata, dict):
        # Domain errors are responsible for content-free metadata. Bound this
        # final persistence boundary defensively to simple JSON-safe values.
        report["error_metadata"] = _safe_error_metadata(metadata)
    return path, report


def _safe_error_metadata(metadata: dict[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, (bool, int, float)) or value is None:
            safe[key] = value
        elif isinstance(value, list) and all(isinstance(item, (bool, int, float)) for item in value):
            safe[key] = value[:100]
        elif isinstance(value, dict):
            safe[key] = _safe_error_metadata(value)
        elif key in {"doc_id", "reason"} and isinstance(value, str):
            safe[key] = value[:128]
    return safe


def _failure_stage(default: str, error: BaseException) -> str:
    """Expose provider/budget stages without coupling domain errors to deck code."""
    stage = getattr(error, "stage", None)
    if isinstance(stage, str) and stage.strip():
        return stage
    from presentation_pipeline.understanding.reduction import ReductionNonReductionError

    if isinstance(error, ReductionNonReductionError):
        return "document_digest_reduction"
    return default


def _replace_staging_paths(value: object, staging_root: Path, published_root: Path) -> object:
    """Ensure JSON audit artifacts only identify their eventual public paths."""
    staging = str(staging_root)
    published = str(published_root)
    if is_dataclass(value) and not isinstance(value, type):
        return _replace_staging_paths(asdict(value), staging_root, published_root)
    if isinstance(value, str):
        return published + value[len(staging):] if value == staging or value.startswith(staging + "/") else value
    if isinstance(value, dict):
        return {key: _replace_staging_paths(item, staging_root, published_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_staging_paths(item, staging_root, published_root) for item in value]
    if isinstance(value, tuple):
        return [_replace_staging_paths(item, staging_root, published_root) for item in value]
    return value
