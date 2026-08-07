"""Read DOCX/OPC packages without mutating the caller's original input."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from io import BytesIO
from pathlib import Path
import zipfile

from lxml import etree

from .relationships import Relationship, parse_relationships, relationship_part_name
from .logging import get_logger

CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
REL_CONTENT_TYPE = "application/vnd.openxmlformats-package.relationships+xml"


@dataclass(frozen=True, slots=True)
class PackageDiagnostic:
    """A preflight finding tied to an OPC member where possible."""

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
class PackagePreflight:
    """Non-destructive package validation result.

    Sanitization is deliberately diagnosed-only in this first implementation:
    ``effective_bytes`` is the original byte stream and ``repairs`` is empty.
    This gives callers a stable interface before any repair policy is approved.
    """

    original_sha256: str
    effective_sha256: str
    diagnostics: tuple[PackageDiagnostic, ...]
    repairs: tuple[PackageDiagnostic, ...] = ()
    effective_bytes: bytes | None = field(default=None, repr=False, compare=False)

    @property
    def is_valid(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)

    def to_dict(self) -> dict[str, object]:
        return {
            "original_sha256": self.original_sha256,
            "effective_sha256": self.effective_sha256,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "repairs": [item.to_dict() for item in self.repairs],
        }


class DocxPackage:
    """An immutable in-memory view of an OPC/DOCX ZIP package.

    Use :meth:`from_path` or :meth:`from_bytes`; the original bytes and their
    SHA-256 are retained so preflight processing cannot overwrite source input.
    """

    def __init__(self, source: str | Path | bytes | bytearray, *, filename: str | None = None):
        if isinstance(source, (str, Path)):
            path = Path(source)
            self.original_bytes = path.read_bytes()
            self.filename = filename or path.name
        elif isinstance(source, (bytes, bytearray)):
            self.original_bytes = bytes(source)
            self.filename = filename or "input.docx"
        else:
            raise TypeError("source must be a path or bytes-like DOCX package")
        self.document_sha256 = hashlib.sha256(self.original_bytes).hexdigest()
        self.sha256 = self.document_sha256
        try:
            with zipfile.ZipFile(BytesIO(self.original_bytes)) as archive:
                self._part_names = tuple(sorted(info.filename for info in archive.infolist() if not info.is_dir()))
        except zipfile.BadZipFile as exc:
            raise ValueError("input is not a readable ZIP/OPC package") from exc
        self._content_types: dict[str, str] | None = None
        self._relationships: dict[str | None, dict[str, Relationship]] = {}

    @classmethod
    def from_path(cls, path: str | Path) -> "DocxPackage":
        """Read a DOCX from disk while preserving its exact original bytes."""

        return cls(path)

    @classmethod
    def from_bytes(cls, data: bytes | bytearray, *, filename: str = "input.docx") -> "DocxPackage":
        """Create a package reader from caller-owned bytes."""

        return cls(data, filename=filename)

    @property
    def part_names(self) -> tuple[str, ...]:
        """All ZIP member names in deterministic lexical order."""

        return self._part_names

    @property
    def content_types(self) -> dict[str, str]:
        """Map normalised part names to declared content types."""

        if self._content_types is None:
            self._content_types = self._parse_content_types()
        return dict(self._content_types)

    def has_part(self, part_name: str) -> bool:
        return part_name.lstrip("/") in self._part_names

    def content_type_for(self, part_name: str) -> str | None:
        """Return the OPC-declared MIME/content type for a package member."""

        return self.content_types.get(part_name.lstrip("/")) or None

    def read_part(self, part_name: str) -> bytes:
        """Return exact uncompressed bytes for a package member."""

        normalised = part_name.lstrip("/")
        if normalised not in self._part_names:
            raise KeyError(f"package does not contain part: {normalised}")
        with zipfile.ZipFile(BytesIO(self.original_bytes)) as archive:
            return archive.read(normalised)

    def xml_part(self, part_name: str) -> etree._Element:
        """Parse an XML part with safe network-disabled parser settings."""

        parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
        return etree.fromstring(self.read_part(part_name), parser=parser)

    def relationships_for(self, part_name: str | None) -> dict[str, Relationship]:
        """Return relationships owned by ``part_name`` (``None`` is package root)."""

        key = part_name.lstrip("/") if part_name else None
        if key not in self._relationships:
            rels_name = relationship_part_name(key)
            if not self.has_part(rels_name):
                self._relationships[key] = {}
            else:
                self._relationships[key] = parse_relationships(self.read_part(rels_name), key)
            get_logger(component_area="package").debug("Loaded {} relationships for part {}", len(self._relationships[key]), key or "package-root")
        return dict(self._relationships[key])

    def resolve_relationship(self, part_name: str | None, rel_id: str) -> tuple[str | None, str | None]:
        """Resolve a relationship to ``(internal_part, external_url)``.

        Unsafe, malformed, and missing targets return ``(None, None)``.  They
        remain visible through :meth:`preflight` diagnostics.
        """

        relationship = self.relationships_for(part_name).get(rel_id)
        if relationship is None:
            get_logger(component_area="package").debug("Relationship was not found for part {}", part_name or "package-root")
            return None, None
        if relationship.external_url is not None:
            return None, relationship.external_url
        if relationship.target_part_name is not None and self.has_part(relationship.target_part_name):
            return relationship.target_part_name, None
        return None, None

    def preflight(self, *, sanitize: bool = False) -> PackagePreflight:
        """Validate ZIP metadata, content types, and relationship targets.

        ``sanitize`` is accepted for the future repair phase.  It currently
        performs no rewrite; callers receive diagnosed findings and the exact
        original bytes as the effective parsing input.
        """

        log = get_logger(component_area="package")
        diagnostics: list[PackageDiagnostic] = []
        log.debug("Preflight inspecting {} package members", len(self._part_names))
        # ``infolist`` alone does not validate per-member CRCs.  This is kept
        # in preflight (rather than construction) so callers can still inspect
        # a partially-corrupt package's central directory and report it.
        with zipfile.ZipFile(BytesIO(self.original_bytes)) as archive:
            corrupt_member = archive.testzip()
            names = [info.filename for info in archive.infolist() if not info.is_dir()]
        if corrupt_member:
            diagnostics.append(PackageDiagnostic("CORRUPT_ZIP_MEMBER", "error", f"CRC check failed for {corrupt_member}", corrupt_member))
        duplicates = sorted({name for name in names if names.count(name) > 1})
        for name in duplicates:
            diagnostics.append(PackageDiagnostic("DUPLICATE_ZIP_MEMBER", "warning", "package contains duplicate member name", name))
        if "[Content_Types].xml" not in self._part_names:
            diagnostics.append(PackageDiagnostic("MISSING_CONTENT_TYPES", "error", "[Content_Types].xml is missing"))
        else:
            try:
                self._parse_content_types()
            except (etree.XMLSyntaxError, ValueError) as exc:
                diagnostics.append(PackageDiagnostic("INVALID_CONTENT_TYPES", "error", str(exc), "[Content_Types].xml"))
        for rels_name in (name for name in self._part_names if name.endswith(".rels")):
            owner = self._owner_for_relationship_part(rels_name)
            try:
                relationships = parse_relationships(self.read_part(rels_name), owner)
            except (etree.XMLSyntaxError, ValueError) as exc:
                diagnostics.append(PackageDiagnostic("INVALID_RELATIONSHIPS", "error", str(exc), rels_name))
                continue
            for rel in relationships.values():
                if rel.error:
                    diagnostics.append(PackageDiagnostic("UNSAFE_RELATIONSHIP_TARGET", "warning", rel.error, owner, rel.relationship_id))
                elif rel.target_part_name and not self.has_part(rel.target_part_name):
                    diagnostics.append(PackageDiagnostic("MISSING_RELATIONSHIP_TARGET", "warning", f"target is absent: {rel.target_part_name}", owner, rel.relationship_id))
        if sanitize:
            diagnostics.append(PackageDiagnostic("SANITIZATION_NOT_APPLIED", "info", "preflight reports unsafe or missing members but does not repair this package version"))
        if diagnostics:
            log.warning("Preflight produced {} diagnostics", len(diagnostics))
        return PackagePreflight(self.document_sha256, self.document_sha256, tuple(diagnostics), effective_bytes=self.original_bytes)

    def _parse_content_types(self) -> dict[str, str]:
        root = self.xml_part("[Content_Types].xml")
        if root.tag != f"{{{CONTENT_TYPES_NS}}}Types":
            raise ValueError("content-types part does not have a Types root")
        defaults: dict[str, str] = {}
        overrides: dict[str, str] = {}
        for element in root:
            local = etree.QName(element).localname
            if local == "Default" and element.get("Extension") and element.get("ContentType"):
                defaults[element.get("Extension", "").lower()] = element.get("ContentType", "")
            elif local == "Override" and element.get("PartName") and element.get("ContentType"):
                overrides[element.get("PartName", "").lstrip("/")] = element.get("ContentType", "")
        return {name: overrides.get(name, defaults.get(name.rsplit(".", 1)[-1].lower(), "")) for name in self._part_names}

    @staticmethod
    def _owner_for_relationship_part(rels_name: str) -> str | None:
        if rels_name == "_rels/.rels":
            return None
        parent, _, filename = rels_name.rpartition("/_rels/")
        if not parent or not filename.endswith(".rels"):
            return None
        return f"{parent}/{filename[:-5]}"
