"""OOXML relationship parsing and safe, part-relative target resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit

from lxml import etree

RELATIONSHIPS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
REL_TAG = f"{{{RELATIONSHIPS_NS}}}Relationship"


@dataclass(frozen=True, slots=True)
class Relationship:
    """A relationship declared by a package or OOXML part.

    ``target_part_name`` is set only for safe internal targets.  External
    targets retain their URI in ``external_url`` and are never opened by this
    module.
    """

    relationship_id: str
    relationship_type: str
    target: str
    target_mode: str
    source_part_name: str | None
    target_part_name: str | None
    external_url: str | None
    error: str | None = None

    @property
    def is_external(self) -> bool:
        return self.target_mode.lower() == "external"


def relationship_part_name(part_name: str | None) -> str:
    """Return the package member containing relationships for ``part_name``."""

    if not part_name:
        return "_rels/.rels"
    part = _normalise_part_name(part_name)
    parent = str(PurePosixPath(part).parent)
    filename = PurePosixPath(part).name
    return f"{'' if parent == '.' else parent + '/'}_rels/{filename}.rels"


def resolve_relationship_target(source_part_name: str | None, target: str) -> str:
    """Resolve an internal relationship target without allowing ZIP traversal.

    OOXML targets are relative to the *owning part*, not to the document part.
    A ``ValueError`` signals an absolute URI/path, malformed URI, or a target
    that would escape the package root.
    """

    if not target or "\x00" in target or "\\" in target:
        raise ValueError("empty, NUL-containing, or backslash relationship target")
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith("/"):
        raise ValueError("absolute relationship target is not an internal package path")
    # Fragments name a location inside the resolved part and are not members.
    path = parsed.path
    if not path:
        raise ValueError("relationship target has no package member path")
    base = PurePosixPath(_normalise_part_name(source_part_name)).parent if source_part_name else PurePosixPath(".")
    candidate = base / PurePosixPath(path)
    pieces: list[str] = []
    for piece in candidate.parts:
        if piece in ("", "."):
            continue
        if piece == "..":
            if not pieces:
                raise ValueError("relationship target escapes package root")
            pieces.pop()
        else:
            pieces.append(piece)
    if not pieces:
        raise ValueError("relationship target resolves to package root")
    return "/".join(pieces)


def parse_relationships(data: bytes, source_part_name: str | None) -> dict[str, Relationship]:
    """Parse a ``.rels`` member into a relationship-ID keyed mapping.

    Invalid targets remain represented with ``error`` so callers can report
    them rather than silently discarding relationship declarations.
    """

    root = etree.fromstring(data)
    if root.tag != f"{{{RELATIONSHIPS_NS}}}Relationships":
        raise ValueError("relationship part does not have a Relationships root")
    result: dict[str, Relationship] = {}
    for element in root.findall(REL_TAG):
        rid = element.get("Id", "")
        target = element.get("Target", "")
        target_mode = element.get("TargetMode", "Internal")
        relation_type = element.get("Type", "")
        if not rid:
            # It cannot be resolved by a caller, but keeping a stable synthetic
            # key makes malformed packages inspectable.
            rid = f"__missing_id_{len(result) + 1}"
        if target_mode.lower() == "external":
            result[rid] = Relationship(rid, relation_type, target, target_mode, source_part_name, None, target, None)
            continue
        try:
            resolved = resolve_relationship_target(source_part_name, target)
            result[rid] = Relationship(rid, relation_type, target, target_mode, source_part_name, resolved, None, None)
        except ValueError as exc:
            result[rid] = Relationship(rid, relation_type, target, target_mode, source_part_name, None, None, str(exc))
    return result


def _normalise_part_name(part_name: str | None) -> str:
    if not part_name:
        return ""
    return part_name.lstrip("/")
