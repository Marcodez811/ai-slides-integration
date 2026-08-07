"""Pure heading/section detection based on paragraph facts in Source IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..ids import make_view_id
from ..ir.source import SourceNode, SourceNodeKind
from .lists import is_heading_paragraph


@dataclass(frozen=True, slots=True)
class DetectedSection:
    section_id: str
    source_heading_node_id: str
    title: str | None
    level: int
    parent_id: str | None
    source_node_ids: tuple[str, ...]
    confidence: float
    detected_via: str
    detector_version: str


def _heading_level(node: SourceNode) -> tuple[int, str, float] | None:
    if node.kind is not SourceNodeKind.PARAGRAPH or not is_heading_paragraph(node):
        return None
    payload = node.payload
    outline = payload.get("outline_level")
    if isinstance(outline, int) and 0 <= outline <= 8:
        return outline + 1, "outline_level", 1.0
    for key in ("style_id", "style_name"):
        normalized = str(payload.get(key, "")).replace(" ", "").lower()
        if normalized.startswith("heading"):
            suffix = normalized[7:]
            if suffix.isdigit() and 1 <= int(suffix) <= 9:
                return int(suffix), "style", 0.95
            return 1, "style", 0.85
        if normalized.startswith("titre"):
            suffix = normalized[5:]
            if suffix.isdigit() and 1 <= int(suffix) <= 9:
                return int(suffix), "style", 0.85
    return None


def _paragraph_text(node: SourceNode, by_parent: dict[str, list[SourceNode]]) -> str | None:
    direct = node.payload.get("text")
    if isinstance(direct, str) and direct:
        return direct
    text = [child.payload.get("text", "") for child in by_parent.get(node.node_id, []) if child.kind is SourceNodeKind.TEXT_RUN]
    joined = "".join(value for value in text if isinstance(value, str))
    return joined or None


def detect_sections(nodes: Iterable[SourceNode], *, detector_version: str = "1.0.0") -> list[DetectedSection]:
    """Derive trustworthy sections from outline levels or heading style IDs.

    Source IDs are memberships, not a physical re-parenting of source nodes.
    Each section contains nodes until the next heading at the same or a higher
    level; parent sections consequently retain their nested section content.
    """
    ordered = sorted(nodes, key=lambda item: item.sequence)
    by_parent: dict[str, list[SourceNode]] = {}
    for node in ordered:
        if node.parent_id is not None:
            by_parent.setdefault(node.parent_id, []).append(node)
    headings = [(index, node, info) for index, node in enumerate(ordered) if (info := _heading_level(node))]
    sections: list[DetectedSection] = []
    for heading_index, (index, node, info) in enumerate(headings):
        level, via, confidence = info
        end = len(ordered)
        for next_index, _, next_info in headings[heading_index + 1 :]:
            if next_info[0] <= level:
                end = next_index
                break
        parent_id = None
        for prior in reversed(sections):
            if prior.level < level:
                parent_id = prior.section_id
                break
        source_ids = tuple(candidate.node_id for candidate in ordered[index:end])
        section_id = make_view_id("sections", detector_version, {"level": level, "via": via}, source_ids)
        sections.append(
            DetectedSection(
                section_id=section_id,
                source_heading_node_id=node.node_id,
                title=_paragraph_text(node, by_parent),
                level=level,
                parent_id=parent_id,
                source_node_ids=source_ids,
                confidence=confidence,
                detected_via=via,
                detector_version=detector_version,
            )
        )
    return sections
