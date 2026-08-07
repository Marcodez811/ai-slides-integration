"""Pure caption-to-image/table relationship detection."""

from __future__ import annotations

import re
from typing import Iterable

from ..ir.normalized import CaptionLink
from ..ir.source import SourceNode, SourceNodeKind


_CAPTION_PREFIX = re.compile(
    r"^\s*(?:figure|fig\.?|table|chart|image|圖|表)\s*[0-9一二三四五六七八九十]*[.:：、\-\s]",
    re.IGNORECASE,
)


def _paragraph_text(node: SourceNode, descendants: dict[str, list[SourceNode]]) -> str:
    text = node.payload.get("text")
    if isinstance(text, str):
        return text
    return "".join(
        str(child.payload.get("text", ""))
        for child in descendants.get(node.node_id, [])
        if child.kind is SourceNodeKind.TEXT_RUN
    )


def _caption_strength(node: SourceNode, descendants: dict[str, list[SourceNode]]) -> float | None:
    if node.kind is not SourceNodeKind.PARAGRAPH:
        return None
    style = " ".join(str(node.payload.get(key, "")) for key in ("style_id", "style_name")).lower()
    if "caption" in style or "圖說" in style or "圖表標題" in style:
        return 0.95
    if _CAPTION_PREFIX.match(_paragraph_text(node, descendants)):
        return 0.75
    return None


def link_captions(nodes: Iterable[SourceNode], *, detector_version: str = "1.0.0") -> list[CaptionLink]:
    """Link retained caption paragraphs to their adjacent image/table occurrence.

    A Caption-styled paragraph has stronger evidence than a prefix-only match.
    The only candidates are immediately adjacent image/table occurrences in the
    ledger, so distant false-prefix text is never attached speculatively.
    """
    ordered = sorted(nodes, key=lambda item: item.sequence)
    descendants: dict[str, list[SourceNode]] = {}
    for node in ordered:
        if node.parent_id is not None:
            descendants.setdefault(node.parent_id, []).append(node)
    by_id = {node.node_id: node for node in ordered}

    def content_block(node: SourceNode) -> SourceNode:
        """Find a paragraph/table-level block without changing the source tree."""
        current = node
        while current.parent_id is not None:
            parent = by_id.get(current.parent_id)
            if parent is None or parent.kind in {SourceNodeKind.DOCUMENT_BODY, SourceNodeKind.HEADER, SourceNodeKind.FOOTER}:
                return current
            current = parent
        return current

    blocks: list[SourceNode] = []
    for node in ordered:
        block = content_block(node)
        if block.node_id == node.node_id and block.kind in {SourceNodeKind.PARAGRAPH, SourceNodeKind.TABLE}:
            blocks.append(block)

    def target_in(block: SourceNode) -> SourceNode | None:
        if block.kind is SourceNodeKind.TABLE:
            return block
        for node in ordered:
            if node.kind is SourceNodeKind.IMAGE and content_block(node).node_id == block.node_id:
                return node
        return None

    links: list[CaptionLink] = []
    for caption in ordered:
        strength = _caption_strength(caption, descendants)
        if strength is None:
            continue
        caption_block = content_block(caption)
        try:
            index = next(i for i, block in enumerate(blocks) if block.node_id == caption_block.node_id)
        except StopIteration:
            continue
        candidates: list[tuple[int, SourceNode]] = []
        for direction in (-1, 1):
            cursor = index + direction
            if 0 <= cursor < len(blocks):
                target = target_in(blocks[cursor])
                if target is not None:
                    candidates.append((1, target))
        if not candidates:
            continue
        distance, target = min(candidates, key=lambda item: (item[0], item[1].sequence))
        confidence = max(0.0, min(1.0, strength - (0.05 * max(0, distance - 1))))
        links.append(
            CaptionLink(
                caption_node_id=caption.node_id,
                target_node_id=target.node_id,
                confidence=confidence,
                detector_name="adjacent-caption",
                detector_version=detector_version,
            )
        )
    return links
