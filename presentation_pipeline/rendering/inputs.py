"""Conversion at the source-to-render boundary, with safe media handling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import ChartSeries, RenderDiagnostic, RenderInputResult, ResolvedElement, SourceAttribution, TableCell, TablePayload


def resolve_render_inputs(
    slide_id: str,
    elements: Sequence[object],
    *,
    source_lookup: Mapping[tuple[str, str], object] | None = None,
    asset_root: str | Path | None = None,
    assets_by_doc: Mapping[str, Mapping[str, object]] | None = None,
    source_filenames: Mapping[str, str] | None = None,
) -> RenderInputResult:
    """Resolve semantic or already-resolved values without silently losing data.

    ``source_lookup`` is keyed by ``(doc_id, evidence_id)`` and accepts the
    pipeline's indexed evidence objects or simple mapping test fixtures.
    ``asset_root`` defines the only directory from which source images may be
    embedded; an escaping path becomes a visible text fallback plus a warning.
    """
    diagnostics: list[RenderDiagnostic] = []
    resolved: list[ResolvedElement] = []
    for index, element in enumerate(elements):
        try:
            result = _resolve_one(index, element, source_lookup or {}, asset_root, assets_by_doc or {}, source_filenames or {}, diagnostics, slide_id)
        except Exception as error:
            diagnostics.append(_warning(slide_id, index, "RENDER_INPUT_INVALID", f"Could not resolve render input: {type(error).__name__}"))
            result = ResolvedElement(element_index=index, kind="text", text="Source content is unavailable.")
        resolved.append(result)
    return RenderInputResult(slide_id=slide_id, elements=resolved, diagnostics=diagnostics)


def _resolve_one(index: int, element: object, lookup: Mapping[tuple[str, str], object], asset_root: str | Path | None, assets_by_doc: Mapping[str, Mapping[str, object]], source_filenames: Mapping[str, str], diagnostics: list[RenderDiagnostic], slide_id: str) -> ResolvedElement:
    if isinstance(element, ResolvedElement):
        return _safe_resolved_image(element, asset_root, diagnostics, slide_id)
    value = _mapping(element)
    if "element_index" in value and "kind" in value and _is_resolved_mapping(value):
        return _safe_resolved_image(ResolvedElement.model_validate(value), asset_root, diagnostics, slide_id)
    kind = str(value.get("kind", "text"))
    attributions = _attributions(value, source_filenames)
    attribution = attributions[0] if attributions else None
    source = (attribution.doc_id, attribution.evidence_ids[0]) if attribution else None
    evidence = lookup.get(source) if source else None
    if kind == "text":
        return ResolvedElement(element_index=index, kind="text", text=str(value["text"]), attribution=attributions)
    if kind == "list":
        raw_items = value.get("items", [])
        items = [str(_get(item, "text", item)) for item in raw_items]
        return ResolvedElement(element_index=index, kind="list", items=items, attribution=attributions)
    if kind == "image":
        path = _image_path(value, evidence, source[0] if source else None, assets_by_doc)
        safe = _safe_path(path, asset_root)
        if safe is not None and safe.is_file():
            return ResolvedElement(element_index=index, kind="image", path=str(safe), caption=_optional_string(value.get("caption")), attribution=attributions)
        code = "SOURCE_IMAGE_PATH_UNSAFE" if path is not None and asset_root is not None else "SOURCE_IMAGE_UNAVAILABLE"
        diagnostics.append(_warning(slide_id, index, code, "Selected source image could not be materialized safely."))
        return _fallback(index, "image", value, evidence, attributions)
    if kind == "table":
        payload = _table_payload(value, evidence)
        if payload is not None:
            return ResolvedElement(element_index=index, kind="table", table=payload, rows=payload.plain_rows(), header_rows=payload.header_rows, title=_optional_string(value.get("title")), attribution=attributions)
        diagnostics.append(_warning(slide_id, index, "TABLE_FALLBACK", "Selected table was unavailable; rendered its source summary."))
        return _fallback(index, "table", value, evidence, attributions)
    if kind == "chart":
        chart = _chart_payload(value, evidence)
        if chart is not None:
            categories, series = chart
            chart_type = str(value.get("chart_type", "bar"))
            return ResolvedElement(element_index=index, kind="chart", categories=categories, series=series, chart_type=chart_type, title=_optional_string(value.get("title")), attribution=attributions)
        diagnostics.append(_warning(slide_id, index, "NATIVE_CHART_VALUES_UNAVAILABLE", "Native chart values were unavailable; rendered a source summary."))
        return _fallback(index, "chart", value, evidence, attributions)
    if kind == "equation":
        diagnostics.append(_warning(slide_id, index, "EQUATION_FALLBACK", "Equation rendered as editable source text."))
        return ResolvedElement(element_index=index, kind="equation", text=_fallback_text("equation", value, evidence), attribution=attributions)
    raise ValueError(f"unsupported render element kind {kind!r}")


def _safe_resolved_image(element: ResolvedElement, asset_root: str | Path | None, diagnostics: list[RenderDiagnostic], slide_id: str) -> ResolvedElement:
    if element.kind != "image" or asset_root is None:
        return element
    safe = _safe_path(element.path, asset_root)
    if safe is not None and safe.is_file():
        return element.model_copy(update={"path": str(safe)})
    diagnostics.append(_warning(slide_id, element.element_index, "SOURCE_IMAGE_PATH_UNSAFE", "Image path is outside the approved source asset root."))
    return ResolvedElement(element_index=element.element_index, kind="text", text=element.caption or "Source image unavailable.", attribution=element.attribution)


def _table_payload(value: Mapping[str, Any], evidence: object | None) -> TablePayload | None:
    existing = value.get("table")
    if isinstance(existing, TablePayload):
        return existing
    rows = value.get("rows")
    header_rows = value.get("header_rows", [])
    if rows is None:
        data = _get(evidence, "structured_data", {})
        rows = _get(data, "rows", None)
        header_rows = _get(data, "header_rows", header_rows)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    try:
        cells = [[cell if isinstance(cell, TableCell) else TableCell(text=str(_get(cell, "text", cell)), row_span=int(_get(cell, "row_span", 1)), column_span=int(_get(cell, "column_span", 1))) for cell in row] for row in rows]
        return TablePayload(rows=cells, header_rows=[int(item) for item in header_rows])
    except (TypeError, ValueError):
        return None


def _chart_payload(value: Mapping[str, Any], evidence: object | None) -> tuple[list[str], list[ChartSeries]] | None:
    data = _get(evidence, "structured_data", {})
    categories = value.get("categories", _get(data, "categories", None))
    series = value.get("series", _get(data, "series", None))
    if not isinstance(categories, Sequence) or isinstance(categories, (str, bytes)) or not isinstance(series, Sequence) or isinstance(series, (str, bytes)):
        return None
    try:
        normalized_series = []
        for item in series:
            if isinstance(item, ChartSeries):
                normalized_series.append(item)
            else:
                normalized_series.append(ChartSeries(
                    name=str(_get(item, "name", "")),
                    values=[float(value) for value in _get(item, "values", [])],
                ))
        return [str(item) for item in categories], normalized_series
    except (TypeError, ValueError):
        return None


def _image_path(value: Mapping[str, Any], evidence: object | None, doc_id: str | None, assets_by_doc: Mapping[str, Mapping[str, object]]) -> str | Path | None:
    direct = value.get("path")
    if isinstance(direct, (str, Path)):
        return direct
    direct = _get(evidence, "path", None)
    if isinstance(direct, (str, Path)):
        return direct
    asset_ids = _get(evidence, "asset_ids", [])
    if doc_id and isinstance(asset_ids, Sequence) and asset_ids:
        asset = assets_by_doc.get(doc_id, {}).get(str(asset_ids[0]))
        return _get(asset, "path", None)
    return None


def _safe_path(value: str | Path | None, root: str | Path | None) -> Path | None:
    if not isinstance(value, (str, Path)):
        return None
    candidate = Path(value)
    if root is None:
        return candidate.resolve()
    root_path = Path(root).resolve()
    candidate = candidate if candidate.is_absolute() else root_path / candidate
    try:
        return candidate.resolve().relative_to(root_path) and candidate.resolve()
    except ValueError:
        return None


def _fallback(index: int, kind: str, value: Mapping[str, Any], evidence: object | None, attributions: list[SourceAttribution]) -> ResolvedElement:
    return ResolvedElement(element_index=index, kind="text", text=_fallback_text(kind, value, evidence), title=_optional_string(value.get("title")), attribution=attributions)


def _fallback_text(kind: str, value: Mapping[str, Any], evidence: object | None) -> str:
    title = _optional_string(value.get("title"))
    source_text = _get(evidence, "text", None)
    if isinstance(source_text, str) and source_text.strip():
        return source_text.strip()
    return f"Source {kind}: {title or 'data unavailable'}"


def _attributions(value: Mapping[str, Any], source_filenames: Mapping[str, str]) -> list[SourceAttribution]:
    grouped: dict[str, list[str]] = {}
    doc_id, evidence_id = value.get("doc_id"), value.get("evidence_id")
    if isinstance(doc_id, str) and doc_id.strip() and isinstance(evidence_id, str) and evidence_id.strip():
        grouped.setdefault(doc_id, []).append(evidence_id)
    evidence_refs = value.get("evidence", [])
    if isinstance(evidence_refs, Sequence) and not isinstance(evidence_refs, (str, bytes)):
        for reference in evidence_refs:
            reference_doc_id = _get(reference, "doc_id", None)
            reference_ids = _get(reference, "evidence_ids", [])
            if not isinstance(reference_doc_id, str) or not reference_doc_id.strip():
                continue
            if isinstance(reference_ids, Sequence) and not isinstance(reference_ids, (str, bytes)):
                grouped.setdefault(reference_doc_id, []).extend(str(item) for item in reference_ids if str(item).strip())
    return [
        SourceAttribution(doc_id=source_doc_id, evidence_ids=list(dict.fromkeys(evidence_ids)), filename=source_filenames.get(source_doc_id))
        for source_doc_id, evidence_ids in grouped.items()
        if evidence_ids
    ]


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    dumped = getattr(value, "model_dump", None)
    if callable(dumped):
        return dumped()
    values = getattr(value, "__dict__", None)
    if isinstance(values, Mapping):
        return values
    raise ValueError("element is not mapping-like")


def _get(value: object | None, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _is_resolved_mapping(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in ("path", "rows", "table", "categories", "text", "items")) and "doc_id" not in value


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _warning(slide_id: str, element_index: int, code: str, message: str) -> RenderDiagnostic:
    return RenderDiagnostic(severity="warning", code=code, message=message, slide_id=slide_id, element_index=element_index)
