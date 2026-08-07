"""Pure numbering-aware list reconstruction from flat Source IR paragraphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..ids import make_view_id
from ..ir.source import SourceNode, SourceNodeKind


def is_heading_paragraph(node: SourceNode) -> bool:
    """Return whether paragraph facts make it a heading, without heuristics."""
    if node.kind is not SourceNodeKind.PARAGRAPH:
        return False
    payload = node.payload
    outline = payload.get("outline_level")
    if isinstance(outline, int) and 0 <= outline <= 8:
        return True
    for key in ("style_id", "style_name"):
        value = str(payload.get(key, "")).replace(" ", "").lower()
        if value.startswith("heading") or value.startswith("titre"):
            return True
    return False


@dataclass(frozen=True, slots=True)
class ListItem:
    """A paragraph-backed list item, retaining source rather than copied text."""

    source_node_id: str
    level: int
    num_id: str
    ordered: bool | None
    marker: str | None
    children: tuple["ListItem", ...] = ()


@dataclass(frozen=True, slots=True)
class ListView:
    list_id: str
    source_node_ids: tuple[str, ...]
    num_id: str
    items: tuple[ListItem, ...]


def _numbering(node: SourceNode) -> dict[str, Any] | None:
    value = node.payload.get("numbering")
    if not isinstance(value, dict) or value.get("num_id") is None:
        return None
    return value


def _item(node: SourceNode, numbering: dict[str, Any]) -> ListItem:
    level = numbering.get("ilvl", numbering.get("level", 0))
    try:
        level = max(0, int(level))
    except (TypeError, ValueError):
        level = 0
    num_id = str(numbering["num_id"])
    fmt = str(numbering.get("num_fmt", numbering.get("format", ""))).lower()
    ordered = numbering.get("ordered")
    if ordered is None and fmt:
        ordered = fmt not in {"bullet", "none"}
    marker = numbering.get("marker", numbering.get("marker_template"))
    return ListItem(node.node_id, level, num_id, ordered, str(marker) if marker is not None else None)


def _nest(items: list[ListItem]) -> tuple[ListItem, ...]:
    """Nest items by level while preserving malformed level jumps predictably."""
    roots: list[ListItem] = []
    # Mutable working tuples are rebuilt into immutable values at the end.
    working: list[dict[str, Any]] = []
    stack: list[tuple[int, list[dict[str, Any]]]] = [(-1, working)]
    for item in items:
        while len(stack) > 1 and item.level <= stack[-1][0]:
            stack.pop()
        target = stack[-1][1]
        record = {"item": item, "children": []}
        target.append(record)
        stack.append((item.level, record["children"]))

    def freeze(records: list[dict[str, Any]]) -> tuple[ListItem, ...]:
        return tuple(
            ListItem(
                source_node_id=record["item"].source_node_id,
                level=record["item"].level,
                num_id=record["item"].num_id,
                ordered=record["item"].ordered,
                marker=record["item"].marker,
                children=freeze(record["children"]),
            )
            for record in records
        )

    return freeze(working)


def build_list_views(nodes: Iterable[SourceNode], *, normalizer_version: str = "1.0.0") -> list[ListView]:
    """Build consecutive, numbering-aware list views without changing Source IR.

    A numbered paragraph that carries heading facts is explicitly excluded.  A
    list is split whenever a non-list paragraph occurs or a numbering ID
    changes, preventing accidental grouping across unrelated content.
    """
    views: list[ListView] = []
    pending: list[ListItem] = []
    ordered_nodes = sorted(nodes, key=lambda item: item.sequence)
    parent_by_id = {node.node_id: node.parent_id for node in ordered_nodes}

    def flush() -> None:
        if not pending:
            return
        source_ids = tuple(item.source_node_id for item in pending)
        views.append(
            ListView(
                list_id=make_view_id("lists", normalizer_version, {}, source_ids),
                source_node_ids=source_ids,
                num_id=pending[0].num_id,
                items=_nest(pending),
            )
        )
        pending.clear()

    for node in ordered_nodes:
        # Inline children do not terminate a list; they occur between every
        # pair of paragraph nodes in the flat ledger. Tables do terminate the
        # surrounding paragraph sequence, while their cell paragraphs are
        # grouped independently by parent context.
        if node.kind not in {SourceNodeKind.PARAGRAPH, SourceNodeKind.TABLE}:
            continue
        if node.kind is SourceNodeKind.TABLE:
            flush()
            continue
        numbering = _numbering(node)
        if node.kind is not SourceNodeKind.PARAGRAPH or numbering is None or is_heading_paragraph(node):
            flush()
            continue
        candidate = _item(node, numbering)
        if pending and (
            candidate.num_id != pending[0].num_id
            or node.parent_id != parent_by_id.get(pending[-1].source_node_id)
        ):
            flush()
        pending.append(candidate)
    flush()
    return views
