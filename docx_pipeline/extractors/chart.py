"""Safe, provenance-preserving extraction of native OOXML chart metadata."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Protocol

from lxml import etree

C_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_C = f"{{{C_NS}}}"
_R = f"{{{R_NS}}}"
_CHART_RELATIONSHIP_SUFFIX = "/chart"
_WORKBOOK_EXTENSIONS = {".xlsx", ".xlsm", ".xlsb", ".xls"}


class _Package(Protocol):
    """The small ``DocxPackage`` surface used by :func:`extract_chart`."""

    def has_part(self, part_name: str) -> bool: ...
    def read_part(self, part_name: str) -> bytes: ...
    def content_type_for(self, part_name: str) -> str | None: ...
    def relationships_for(self, part_name: str | None): ...
    def resolve_relationship(self, part_name: str | None, rel_id: str) -> tuple[str | None, str | None]: ...


@dataclass(frozen=True, slots=True)
class ChartDiagnostic:
    """A non-fatal chart extraction finding tied to a source part or relation."""

    code: str
    severity: str
    message: str
    part_name: str | None = None
    relationship_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "part_name": self.part_name,
            "relationship_id": self.relationship_id,
        }


@dataclass(frozen=True, slots=True)
class ChartSeries:
    """Formula-level data references for a single native chart series."""

    index: int | None
    order: int | None
    name: str | None
    category_formula: str | None
    value_formula: str | None

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "index": self.index,
            "order": self.order,
            "name": self.name,
            "category_formula": self.category_formula,
            "value_formula": self.value_formula,
        }


@dataclass(frozen=True, slots=True)
class ChartExtraction:
    """Deterministic, non-executing metadata for one chart occurrence."""

    owning_part_name: str
    relationship_id: str
    chart_part_name: str | None
    raw_xml_sha256: str | None
    chart_types: tuple[str, ...]
    title: str | None
    series: tuple[ChartSeries, ...]
    embedded_workbook_parts: tuple[str, ...]
    diagnostics: tuple[ChartDiagnostic, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "owning_part_name": self.owning_part_name,
            "relationship_id": self.relationship_id,
            "chart_part_name": self.chart_part_name,
            "raw_xml_sha256": self.raw_xml_sha256,
            "chart_types": list(self.chart_types),
            "title": self.title,
            "series": [item.to_dict() for item in self.series],
            "embedded_workbook_parts": list(self.embedded_workbook_parts),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def extract_chart(package: _Package, owning_part_name: str, relationship_id: str) -> ChartExtraction:
    """Extract a native chart relation without opening workbooks or external URLs.

    The result is always safe to serialize.  An unresolved/external chart or a
    malformed chart part yields diagnostics and an otherwise empty extraction,
    allowing a walker to retain its ``chart`` source node rather than drop it.
    """

    diagnostics: list[ChartDiagnostic] = []
    relation = package.relationships_for(owning_part_name).get(relationship_id)
    if relation is None:
        diagnostics.append(ChartDiagnostic("MISSING_CHART_RELATIONSHIP", "warning", "chart relationship ID is not declared by owning part", owning_part_name, relationship_id))
        return _empty(owning_part_name, relationship_id, diagnostics)
    if not relation.relationship_type.endswith(_CHART_RELATIONSHIP_SUFFIX):
        diagnostics.append(ChartDiagnostic("NON_CHART_RELATIONSHIP", "warning", "relationship does not declare an OOXML chart type", owning_part_name, relationship_id))
    chart_part, external_url = package.resolve_relationship(owning_part_name, relationship_id)
    if external_url:
        diagnostics.append(ChartDiagnostic("EXTERNAL_CHART_NOT_FETCHED", "warning", "external chart relationship was not fetched", owning_part_name, relationship_id))
        return _empty(owning_part_name, relationship_id, diagnostics)
    if not chart_part:
        diagnostics.append(ChartDiagnostic("UNRESOLVED_CHART_RELATIONSHIP", "warning", "chart target is missing or unsafe", owning_part_name, relationship_id))
        return _empty(owning_part_name, relationship_id, diagnostics)

    raw_xml = package.read_part(chart_part)
    raw_hash = sha256(raw_xml).hexdigest()
    try:
        root = etree.fromstring(raw_xml, parser=etree.XMLParser(resolve_entities=False, no_network=True, recover=False))
    except etree.XMLSyntaxError as exc:
        diagnostics.append(ChartDiagnostic("INVALID_CHART_XML", "warning", str(exc), chart_part))
        return ChartExtraction(owning_part_name, relationship_id, chart_part, raw_hash, (), None, (), (), tuple(diagnostics))
    if root.tag != f"{_C}chartSpace":
        diagnostics.append(ChartDiagnostic("UNEXPECTED_CHART_ROOT", "warning", "chart part does not have a c:chartSpace root", chart_part))

    chart_types = _chart_types(root)
    title = _text_value(root.find(f".//{_C}title"))
    series = tuple(_series_values(series) for chart in _chart_elements(root) for series in chart.findall(f"{_C}ser"))
    workbooks = _embedded_workbooks(package, chart_part, root, diagnostics)
    return ChartExtraction(owning_part_name, relationship_id, chart_part, raw_hash, chart_types, title, series, workbooks, tuple(diagnostics))


def _empty(owning_part_name: str, relationship_id: str, diagnostics: list[ChartDiagnostic]) -> ChartExtraction:
    return ChartExtraction(owning_part_name, relationship_id, None, None, (), None, (), (), tuple(diagnostics))


def _chart_elements(root: etree._Element) -> tuple[etree._Element, ...]:
    plot_area = root.find(f".//{_C}plotArea")
    if plot_area is None:
        return ()
    return tuple(element for element in plot_area if isinstance(element.tag, str) and etree.QName(element).namespace == C_NS and etree.QName(element).localname.endswith("Chart"))


def _chart_types(root: etree._Element) -> tuple[str, ...]:
    return tuple(etree.QName(element).localname for element in _chart_elements(root))


def _series_values(series: etree._Element) -> ChartSeries:
    index = _as_int(series.find(f"{_C}idx"))
    order = _as_int(series.find(f"{_C}order"))
    name = _text_value(series.find(f"{_C}tx"))
    return ChartSeries(
        index=index,
        order=order,
        name=name,
        category_formula=_formula(series.find(f"{_C}cat")),
        value_formula=_formula(series.find(f"{_C}val")),
    )


def _formula(container: etree._Element | None) -> str | None:
    if container is None:
        return None
    formula = container.find(f".//{_C}f")
    return (formula.text or "").strip() if formula is not None and formula.text else None


def _as_int(element: etree._Element | None) -> int | None:
    if element is None or element.get("val") is None:
        return None
    try:
        return int(element.get("val", ""))
    except ValueError:
        return None


def _text_value(element: etree._Element | None) -> str | None:
    if element is None:
        return None
    values = [value.strip() for value in element.itertext() if value and value.strip()]
    return "".join(values) or None


def _embedded_workbooks(package: _Package, chart_part: str, root: etree._Element, diagnostics: list[ChartDiagnostic]) -> tuple[str, ...]:
    workbook_parts: set[str] = set()
    for element in root.findall(f".//{_C}externalData"):
        rel_id = element.get(f"{_R}id")
        if not rel_id:
            diagnostics.append(ChartDiagnostic("WORKBOOK_WITHOUT_RELATIONSHIP", "warning", "c:externalData has no r:id", chart_part))
            continue
        target_part, external_url = package.resolve_relationship(chart_part, rel_id)
        if external_url:
            diagnostics.append(ChartDiagnostic("EXTERNAL_WORKBOOK_NOT_FETCHED", "warning", "external workbook relationship was not fetched", chart_part, rel_id))
        elif not target_part:
            diagnostics.append(ChartDiagnostic("UNRESOLVED_WORKBOOK_RELATIONSHIP", "warning", "embedded workbook target is missing or unsafe", chart_part, rel_id))
        elif _is_workbook_part(package, target_part):
            workbook_parts.add(target_part)
        else:
            diagnostics.append(ChartDiagnostic("UNEXPECTED_WORKBOOK_TARGET", "warning", "c:externalData target is not a recognised workbook package", chart_part, rel_id))
    return tuple(sorted(workbook_parts))


def _is_workbook_part(package: _Package, part_name: str) -> bool:
    content_type = package.content_type_for(part_name) or ""
    return "spreadsheetml" in content_type or PurePosixPath(part_name).suffix.lower() in _WORKBOOK_EXTENSIONS
