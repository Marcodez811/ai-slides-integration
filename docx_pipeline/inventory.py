"""Independent, XML-level occurrence inventory for DOCX package coverage."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from lxml import etree

from .package import DocxPackage, PackageDiagnostic
from .markup_compatibility import iter_effective_elements

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
C_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
V_NS = "urn:schemas-microsoft-com:vml"
DGM_NS = "http://schemas.openxmlformats.org/drawingml/2006/diagram"

_PREFIXES = {W_NS: "w", A_NS: "a", R_NS: "r", M_NS: "m", C_NS: "c", WP_NS: "wp", V_NS: "v", DGM_NS: "dgm"}
_WORD_DOCUMENT = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
_WORD_PART_PREFIX = "application/vnd.openxmlformats-officedocument.wordprocessingml."
_INVENTORY_CONTENT_TYPES = {
    _WORD_DOCUMENT,
    f"{_WORD_PART_PREFIX}header+xml",
    f"{_WORD_PART_PREFIX}footer+xml",
    f"{_WORD_PART_PREFIX}footnotes+xml",
    f"{_WORD_PART_PREFIX}endnotes+xml",
    f"{_WORD_PART_PREFIX}comments+xml",
}


@dataclass(frozen=True, slots=True)
class InventoryOccurrence:
    """A countable OOXML source feature and its exact canonical location."""

    feature: str
    part_name: str
    xml_path: str
    value: int | str | None = None
    relationship_id: str | None = None
    target_part_name: str | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "feature": self.feature,
            "part_name": self.part_name,
            "xml_path": self.xml_path,
            "value": self.value,
            "relationship_id": self.relationship_id,
            "target_part_name": self.target_part_name,
        }


@dataclass(slots=True)
class FeatureInventory:
    """Deterministic feature counts plus source-level occurrence evidence."""

    occurrences: list[InventoryOccurrence] = field(default_factory=list)
    diagnostics: list[PackageDiagnostic] = field(default_factory=list)
    included_parts: tuple[str, ...] = ()

    def add(self, occurrence: InventoryOccurrence) -> None:
        self.occurrences.append(occurrence)

    @property
    def counts(self) -> dict[str, int]:
        """Feature occurrence counts (``visible_text_chars`` sums char values)."""

        totals: Counter[str] = Counter()
        for item in self.occurrences:
            totals[item.feature] += item.value if item.feature == "visible_text_chars" and isinstance(item.value, int) else 1
        return dict(sorted(totals.items()))

    def count(self, feature: str) -> int:
        return self.counts.get(feature, 0)

    def occurrences_for(self, feature: str) -> tuple[InventoryOccurrence, ...]:
        return tuple(item for item in self.occurrences if item.feature == feature)

    def to_dict(self) -> dict[str, object]:
        return {
            "included_parts": list(self.included_parts),
            "counts": self.counts,
            "occurrences": [item.to_dict() for item in self.occurrences],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def inventory_package(
    package: DocxPackage,
    *,
    part_names: Iterable[str] | None = None,
    include_linked_parts: bool = True,
) -> FeatureInventory:
    """Inventory visible/extractable features without using the pipeline walker.

    By default this inventories the document body and the selected Word parts
    (headers, footers, notes, and comments).  ``part_names`` gives callers a
    precise policy override.  Each occurrence carries a stable canonical XPath
    so coverage can be reconciled independently after extraction.
    """

    parts = _configured_parts(package, part_names, include_linked_parts)
    inventory = FeatureInventory(included_parts=tuple(parts))
    for part_name in parts:
        try:
            root = package.xml_part(part_name)
        except (KeyError, etree.XMLSyntaxError, ValueError) as exc:
            inventory.diagnostics.append(PackageDiagnostic("UNREADABLE_XML_PART", "warning", str(exc), part_name))
            continue
        _inventory_part(package, part_name, root, inventory)
    inventory.occurrences.sort(key=lambda item: (item.part_name, item.xml_path, item.feature, item.relationship_id or ""))
    inventory.diagnostics.sort(key=lambda item: (item.part_name or "", item.code, item.relationship_id or ""))
    return inventory


def canonical_xml_paths(root: etree._Element) -> dict[etree._Element, str]:
    """Return deterministic, namespace-stable paths for every XML element."""
    return _canonical_paths(root)


def _configured_parts(package: DocxPackage, specified: Iterable[str] | None, include_linked_parts: bool) -> list[str]:
    if specified is not None:
        return sorted({name.lstrip("/") for name in specified if package.has_part(name)})
    content_types = package.content_types
    selected = {name for name, content_type in content_types.items() if content_type in _INVENTORY_CONTENT_TYPES}
    # Content types are occasionally incomplete.  Relationship discovery makes
    # the default robust while keeping the Word-specific policy bounded.
    main = "word/document.xml"
    if package.has_part(main):
        selected.add(main)
    if include_linked_parts:
        queue = list(selected)
        while queue:
            owner = queue.pop()
            for relation in package.relationships_for(owner).values():
                target = relation.target_part_name
                if target and target not in selected and package.content_type_for(target) in _INVENTORY_CONTENT_TYPES:
                    selected.add(target)
                    queue.append(target)
    return _part_order(selected, content_types)


def _part_order(parts: set[str], content_types: dict[str, str]) -> list[str]:
    priority = {
        _WORD_DOCUMENT: 0,
        f"{_WORD_PART_PREFIX}header+xml": 1,
        f"{_WORD_PART_PREFIX}footer+xml": 2,
        f"{_WORD_PART_PREFIX}footnotes+xml": 3,
        f"{_WORD_PART_PREFIX}endnotes+xml": 4,
        f"{_WORD_PART_PREFIX}comments+xml": 5,
    }
    return sorted(parts, key=lambda name: (priority.get(content_types.get(name, ""), 99), name))


def _inventory_part(package: DocxPackage, part_name: str, root: etree._Element, inventory: FeatureInventory) -> None:
    paths = _canonical_paths(root)
    effective_elements = list(iter_effective_elements(root))
    for element in effective_elements:
        if not isinstance(element.tag, str):
            continue
        namespace, local = _expanded_name(element.tag)
        path = paths[element]
        if namespace == W_NS and local == "t" and not _is_deleted_text(element):
            text = element.text or ""
            inventory.add(InventoryOccurrence("visible_text_nodes", part_name, path, text))
            inventory.add(InventoryOccurrence("visible_text_chars", part_name, path, len(text)))
        elif namespace == W_NS and local == "p":
            inventory.add(InventoryOccurrence("paragraphs", part_name, path))
        elif namespace == W_NS and local == "tbl":
            inventory.add(InventoryOccurrence("tables", part_name, path))
        elif namespace == W_NS and local == "drawing":
            inventory.add(InventoryOccurrence("drawings", part_name, path))
        elif namespace == A_NS and local == "blip":
            _image_occurrence(package, part_name, path, element, inventory)
        elif namespace == V_NS and local == "imagedata":
            _image_occurrence(package, part_name, path, element, inventory)
        elif namespace == M_NS and local == "oMath":
            inventory.add(InventoryOccurrence("equations", part_name, path))
        elif namespace == W_NS and local == "hyperlink":
            inventory.add(InventoryOccurrence("hyperlinks", part_name, path, relationship_id=element.get(f"{{{R_NS}}}id")))
        elif namespace == C_NS and local == "chart":
            _relationship_occurrence(package, part_name, "charts", path, element, inventory)
        elif namespace == W_NS and local == "txbxContent":
            inventory.add(InventoryOccurrence("textboxes", part_name, path))
        elif namespace == W_NS and local == "sdt":
            inventory.add(InventoryOccurrence("structured_document_tags", part_name, path))
        elif _is_unsupported(namespace, local):
            inventory.add(InventoryOccurrence("unsupported_features", part_name, path, _qname(namespace, local)))
        elif _is_unknown_block_child(element):
            inventory.add(InventoryOccurrence("unknown_features", part_name, path, _qname(namespace, local)))

    # Complex-field hyperlinks do not have a ``w:hyperlink`` element.  Count
    # each field instruction once; its exact text node retains provenance.
    for element in effective_elements:
        if element.tag != f"{{{W_NS}}}instrText":
            continue
        if "HYPERLINK" in (element.text or "").upper():
            inventory.add(InventoryOccurrence("field_hyperlinks", part_name, paths[element]))
    for element in effective_elements:
        if element.tag != f"{{{W_NS}}}fldSimple":
            continue
        if "HYPERLINK" in (element.get(f"{{{W_NS}}}instr") or "").upper():
            inventory.add(InventoryOccurrence("field_hyperlinks", part_name, paths[element]))


def _image_occurrence(package: DocxPackage, part_name: str, path: str, element: etree._Element, inventory: FeatureInventory) -> None:
    rid = (
        element.get(f"{{{R_NS}}}embed")
        or element.get(f"{{{R_NS}}}link")
        or element.get(f"{{{R_NS}}}id")
    )
    target, _external = package.resolve_relationship(part_name, rid) if rid else (None, None)
    inventory.add(InventoryOccurrence("image_blips", part_name, path, relationship_id=rid, target_part_name=target))
    if not rid:
        inventory.diagnostics.append(PackageDiagnostic("IMAGE_WITHOUT_RELATIONSHIP", "warning", "image element has no relationship reference", part_name))
    elif target is None and not _external:
        inventory.diagnostics.append(PackageDiagnostic("UNRESOLVED_IMAGE_RELATIONSHIP", "warning", f"unable to resolve {rid}", part_name, rid))


def _relationship_occurrence(package: DocxPackage, part_name: str, feature: str, path: str, element: etree._Element, inventory: FeatureInventory) -> None:
    rid = element.get(f"{{{R_NS}}}id")
    target, _ = package.resolve_relationship(part_name, rid) if rid else (None, None)
    inventory.add(InventoryOccurrence(feature, part_name, path, relationship_id=rid, target_part_name=target))


def _canonical_paths(root: etree._Element) -> dict[etree._Element, str]:
    paths: dict[etree._Element, str] = {}

    def visit(element: etree._Element, parent_path: str, index: int) -> None:
        namespace, local = _expanded_name(element.tag)
        here = f"{parent_path}/{_qname(namespace, local)}[{index}]" if parent_path else f"/{_qname(namespace, local)}"
        paths[element] = here
        counts: Counter[str] = Counter()
        for child in element:
            if not isinstance(child.tag, str):
                continue
            namespace_child, local_child = _expanded_name(child.tag)
            key = _qname(namespace_child, local_child)
            counts[key] += 1
            visit(child, here, counts[key])

    visit(root, "", 1)
    return paths


def _expanded_name(tag: str) -> tuple[str, str]:
    name = etree.QName(tag)
    return name.namespace or "", name.localname


def _qname(namespace: str, local: str) -> str:
    return f"{_PREFIXES.get(namespace, 'ns')}:{local}" if namespace else local


def _is_deleted_text(element: etree._Element) -> bool:
    return any(_expanded_name(ancestor.tag) == (W_NS, "del") for ancestor in element.iterancestors() if isinstance(ancestor.tag, str))


def _is_unsupported(namespace: str, local: str) -> bool:
    return namespace == DGM_NS or (namespace == W_NS and local in {"altChunk", "smartTag", "customXml", "object", "pict"})


def _is_unknown_block_child(element: etree._Element) -> bool:
    parent = element.getparent()
    if parent is None or _expanded_name(parent.tag) != (W_NS, "body"):
        return False
    namespace, local = _expanded_name(element.tag)
    return namespace != W_NS or local not in {"p", "tbl", "sdt", "sectPr", "customXml", "altChunk"}
