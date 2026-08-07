"""Reconcile an independent package inventory with emitted Source IR."""

from __future__ import annotations

from ..ir import CoverageMetric, CoverageReport, SourceNode
from ..logging import get_logger


def build_coverage_report(
    inventory_counts: dict[str, int],
    nodes: list[SourceNode],
) -> CoverageReport:
    kinds: dict[str, list[SourceNode]] = {}
    for node in nodes:
        kinds.setdefault(node.kind.value, []).append(node)

    extracted_text_nodes = [
        node for node in kinds.get("text_run", []) if node.status.value == "extracted"
    ]
    extracted_text_chars = sum(
        len(str(node.payload.get("text") or ""))
        for node in extracted_text_nodes
        if not node.payload.get("deleted")
    )
    extracted_mapping = {
        "visible_text_nodes": len(extracted_text_nodes),
        "visible_text_chars": extracted_text_chars,
        "image_blips": _successful(kinds.get("image", [])),
        "tables": _successful(kinds.get("table", [])),
        "equations": _successful(kinds.get("equation", [])),
        "hyperlinks": _successful(kinds.get("hyperlink", [])),
        "field_hyperlinks": sum(
            1
            for node in kinds.get("field", [])
            if node.status.value == "extracted"
            and "HYPERLINK" in str(node.payload.get("instruction") or "").upper()
        ),
        "charts": _successful(kinds.get("chart", [])),
        "textboxes": _successful(kinds.get("text_box", [])),
        "structured_document_tags": _successful(kinds.get("content_control", [])),
    }
    diagnosed_mapping = {
        "image_blips": _diagnosed(kinds.get("image", [])),
        "tables": _diagnosed(kinds.get("table", [])),
        "equations": _diagnosed(kinds.get("equation", [])),
        "hyperlinks": _diagnosed(kinds.get("hyperlink", [])),
        "charts": _diagnosed(kinds.get("chart", [])),
        "textboxes": _diagnosed(kinds.get("text_box", [])),
        "structured_document_tags": _diagnosed(kinds.get("content_control", [])),
    }
    metrics: dict[str, CoverageMetric] = {}
    silent_losses = 0
    for feature, extracted in extracted_mapping.items():
        source = inventory_counts.get(feature, 0)
        failed = max(0, source - extracted)
        diagnosed = min(failed, diagnosed_mapping.get(feature, 0))
        silent_losses += max(0, failed - diagnosed)
        metrics[feature] = CoverageMetric(
            source=source,
            extracted=min(extracted, source) if source else extracted,
            diagnosed=diagnosed,
        )
    report = CoverageReport(
        metrics=metrics,
        unsupported_nodes=len(kinds.get("unsupported", [])),
        silent_losses=silent_losses,
    )
    log = get_logger(component_area="coverage")
    if silent_losses:
        log.warning("Coverage found {} silent losses across {} metrics", silent_losses, len(metrics))
    else:
        log.debug("Coverage reconciled {} metrics without silent losses", len(metrics))
    return report


def _successful(nodes: list[SourceNode]) -> int:
    return sum(node.status.value == "extracted" for node in nodes)


def _diagnosed(nodes: list[SourceNode]) -> int:
    return sum(node.status.value in {"unsupported", "failed"} for node in nodes)
