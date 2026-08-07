"""Public API for auditable, ordered DOCX extraction."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable

from .assets import AssetStore
from .config import ExtractionConfig
from .inventory import FeatureInventory as PackageInventory
from .inventory import inventory_package
from .ir import (
    Asset,
    Diagnostic,
    DocumentMetadata,
    ExtractionResult,
    FeatureInventory,
    InventoryOccurrence,
    PartMetadata,
    SourceLocator,
    SourceNode,
)
from .ids import make_doc_id
from .normalize import build_normalized_views
from .package import DocxPackage
from .validation import build_coverage_report
from .walker import OrderedXmlWalker
from .logging import get_logger


def extract_docx(
    source: str | Path | bytes | bytearray,
    *,
    asset_output_dir: str | Path | None = None,
    config: ExtractionConfig | None = None,
    filename: str | None = None,
) -> ExtractionResult:
    """Extract a DOCX into deterministic Source IR and derived block views.

    The source package is never modified. Assets are content-addressed and are
    written only when ``asset_output_dir`` is supplied.
    """
    policy = config or ExtractionConfig()
    log = get_logger(operation="extract")
    started = perf_counter()
    log.info("Extraction started")
    if isinstance(source, (bytes, bytearray)):
        package = DocxPackage.from_bytes(source, filename=filename or "input.docx")
    else:
        package = DocxPackage.from_path(source)

    # The sanitizer interface is present, but the current package layer only
    # diagnoses repairs. Do not claim a sanitation pass until bytes change.
    stage_started = perf_counter()
    preflight = package.preflight(sanitize=False)
    log.info("Preflight completed in {:.3f}s with {} diagnostics", perf_counter() - stage_started, len(preflight.diagnostics))
    part_names = _configured_part_names(package, policy)
    log.debug("Selected {} XML parts", len(part_names))
    stage_started = perf_counter()
    package_inventory = inventory_package(package, part_names=part_names)
    log.info("Inventory completed in {:.3f}s with {} feature occurrences", perf_counter() - stage_started, len(package_inventory.occurrences))

    stage_started = perf_counter()
    walker = OrderedXmlWalker(package, asset_store=AssetStore(asset_output_dir))
    walked = walker.walk(part_names)
    log.info("Walk completed in {:.3f}s with {} nodes, {} assets, and {} diagnostics", perf_counter() - stage_started, len(walked.nodes), len(walked.assets), len(walked.diagnostics))
    nodes = [SourceNode.model_validate(item) for item in walked.nodes]
    assets = [Asset.model_validate(item.as_dict()) for item in walked.assets]

    diagnostics = _diagnostics(preflight.diagnostics, package_inventory, walked.diagnostics)
    inventory = _inventory_model(package_inventory)
    stage_started = perf_counter()
    coverage = build_coverage_report(package_inventory.counts, nodes)
    log.info("Coverage completed in {:.3f}s with {} silent losses", perf_counter() - stage_started, coverage.silent_losses)
    stage_started = perf_counter()
    views = build_normalized_views(nodes, config=policy)
    log.info("Normalization completed in {:.3f}s", perf_counter() - stage_started)
    parts = [
        PartMetadata(
            part_name=part_name,
            content_type=package.content_type_for(part_name),
            kind=_part_kind(part_name),
            relationship_count=len(package.relationships_for(part_name)),
        )
        for part_name in part_names
    ]

    result = ExtractionResult(
        extractor_version="0.2.0",
        document=DocumentMetadata(
            doc_id=make_doc_id(package.original_bytes),
            filename=filename or package.filename,
            sha256=package.document_sha256,
            sanitized_sha256=preflight.effective_sha256,
        ),
        parts=parts,
        nodes=nodes,
        assets=assets,
        diagnostics=diagnostics,
        inventory=inventory,
        coverage=coverage,
        views=views,
        config=policy,
    )
    log.info("Extraction completed in {:.3f}s with {} nodes, {} assets, and {} diagnostics", perf_counter() - started, len(result.nodes), len(result.assets), len(result.diagnostics))
    return result


def _configured_part_names(package: DocxPackage, config: ExtractionConfig) -> list[str]:
    selected: list[str] = []
    if package.has_part("word/document.xml"):
        selected.append("word/document.xml")

    def add_matching(prefix: str) -> None:
        selected.extend(
            name
            for name in package.part_names
            if name.startswith(f"word/{prefix}") and name.endswith(".xml") and name not in selected
        )

    if config.include_headers_footers:
        add_matching("header")
        add_matching("footer")
    if config.include_notes:
        for name in ("word/footnotes.xml", "word/endnotes.xml"):
            if package.has_part(name):
                selected.append(name)
    if config.include_comments:
        add_matching("comments")
    return selected


def _inventory_model(inventory: PackageInventory) -> FeatureInventory:
    occurrences = [
        InventoryOccurrence(
            feature=item.feature,
            source=SourceLocator(
                part_name=item.part_name,
                xml_path=item.xml_path,
                relationship_id=item.relationship_id,
            ),
            details={
                "value": item.value,
                "target_part_name": item.target_part_name,
            },
        )
        for item in inventory.occurrences
    ]
    return FeatureInventory(occurrences=occurrences, totals=inventory.counts)


def _diagnostics(
    preflight_diagnostics: Iterable[object],
    inventory: PackageInventory,
    walker_diagnostics: Iterable[dict[str, object]],
) -> list[Diagnostic]:
    results: list[Diagnostic] = []
    seen: set[tuple[str, str, str | None, str | None]] = set()

    def add(
        *,
        code: str,
        severity: str,
        message: str,
        part_name: str | None = None,
        xml_path: str | None = None,
        relationship_id: str | None = None,
        source_node_id: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        key = (code, message, part_name, relationship_id)
        if key in seen:
            return
        seen.add(key)
        source = None
        if part_name and xml_path:
            source = SourceLocator(
                part_name=part_name,
                xml_path=xml_path,
                relationship_id=relationship_id,
            )
        merged_details = dict(details or {})
        if part_name and source is None:
            merged_details["part_name"] = part_name
        if relationship_id and source is None:
            merged_details["relationship_id"] = relationship_id
        results.append(
            Diagnostic(
                code=code,
                severity=severity,
                message=message,
                source_node_id=source_node_id,
                source=source,
                details=merged_details,
            )
        )

    for item in [*preflight_diagnostics, *inventory.diagnostics]:
        add(
            code=item.code,
            severity=item.severity,
            message=item.message,
            part_name=item.part_name,
            relationship_id=item.relationship_id,
        )
    for item in walker_diagnostics:
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        add(
            code=str(item["code"]),
            severity=str(item["severity"]),
            message=str(item["message"]),
            part_name=source.get("part_name"),
            xml_path=source.get("xml_path"),
            source_node_id=str(item["source_node_id"]) if item.get("source_node_id") else None,
        )
    return results


def _part_kind(part_name: str) -> str:
    name = Path(part_name).name
    if name == "document.xml":
        return "body"
    if name.startswith("header"):
        return "header"
    if name.startswith("footer"):
        return "footer"
    if name in {"footnotes.xml", "endnotes.xml", "comments.xml"}:
        return name.removesuffix("s.xml").removesuffix(".xml")
    return "other"
