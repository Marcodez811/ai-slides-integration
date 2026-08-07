"""Build a compact MinerU-like block view without mutating Source IR."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Iterable

from ..config import ExtractionConfig
from ..ids import make_view_id
from ..ir import NormalizedBlock, NormalizedBlockKind, NormalizedViews, SectionView, SourceNode
from .captions import link_captions
from .lists import build_list_views
from .sections import detect_sections


def build_normalized_views(
    nodes: Iterable[SourceNode],
    *,
    config: ExtractionConfig,
) -> NormalizedViews:
    """Return presentation-friendly blocks that reference immutable source IDs."""
    ordered = sorted(nodes, key=lambda item: item.sequence)
    children: dict[str, list[SourceNode]] = defaultdict(list)
    for node in ordered:
        if node.parent_id is not None:
            children[node.parent_id].append(node)

    blocks: list[NormalizedBlock] = []
    for node in ordered:
        kind = node.kind.value
        if kind == "paragraph":
            descendants = _descendants(node.node_id, children)
            content = _inline_content(descendants)
            style_id = str(node.payload.get("style_id") or "")
            numbering = node.payload.get("numbering")
            if "heading" in style_id.lower() or style_id.lower() == "title":
                block_kind = NormalizedBlockKind.TITLE
            elif numbering:
                # Numbered paragraphs are emitted as one grouped list view
                # below, retaining one source ID per item.
                continue
            else:
                block_kind = NormalizedBlockKind.TEXT
            source_ids = [node.node_id, *(item.node_id for item in descendants)]
            blocks.append(
                _block(
                    block_kind,
                    source_ids,
                    config,
                    {
                        "content": content,
                        "style_id": node.payload.get("style_id"),
                        "outline_level": node.payload.get("outline_level"),
                        "numbering": numbering,
                    },
                )
            )
        elif kind in {"table", "image", "equation", "chart", "unsupported"}:
            mapping = {
                "table": NormalizedBlockKind.TABLE,
                "image": NormalizedBlockKind.IMAGE,
                "equation": NormalizedBlockKind.EQUATION,
                "chart": NormalizedBlockKind.CHART,
                "unsupported": NormalizedBlockKind.UNSUPPORTED,
            }
            descendants = _descendants(node.node_id, children) if kind == "table" else []
            source_ids = [node.node_id, *(item.node_id for item in descendants)]
            payload = dict(node.payload)
            if kind == "table":
                payload["text"] = _inline_content(descendants)
            blocks.append(_block(mapping[kind], source_ids, config, payload))

    for list_view in build_list_views(ordered, normalizer_version=config.normalizer_version):
        blocks.append(
            _block(
                NormalizedBlockKind.LIST,
                list(list_view.source_node_ids),
                config,
                {
                    "num_id": list_view.num_id,
                    "items": [asdict(item) for item in list_view.items],
                },
            )
        )

    sections: list[SectionView] = []
    for section in detect_sections(ordered, detector_version=config.normalizer_version):
        member_ids = set(section.source_node_ids)
        sections.append(
            SectionView(
                section_id=section.section_id,
                source_heading_node_id=section.source_heading_node_id,
                title=section.title,
                block_ids=[
                    block.block_id
                    for block in blocks
                    if member_ids.intersection(block.source_node_ids)
                ],
                parent_id=section.parent_id,
                level=section.level,
                confidence=section.confidence,
                detector_name=section.detected_via,
                detector_version=section.detector_version,
            )
        )

    sequence_by_id = {node.node_id: node.sequence for node in ordered}
    return NormalizedViews(
        blocks=sorted(
            blocks,
            key=lambda block: min(sequence_by_id[node_id] for node_id in block.source_node_ids),
        ),
        sections=sections,
        caption_links=link_captions(ordered, detector_version=config.normalizer_version),
    )


def _block(
    kind: NormalizedBlockKind,
    source_ids: list[str],
    config: ExtractionConfig,
    payload: dict[str, object],
) -> NormalizedBlock:
    block_id = make_view_id(
        "semantic-blocks",
        config.normalizer_version,
        config.model_dump(mode="json"),
        source_ids,
    )
    return NormalizedBlock(
        block_id=block_id,
        kind=kind,
        source_node_ids=source_ids,
        payload=payload,
    )


def _descendants(root_id: str, children: dict[str, list[SourceNode]]) -> list[SourceNode]:
    result: list[SourceNode] = []
    pending = list(children.get(root_id, []))
    while pending:
        node = pending.pop(0)
        result.append(node)
        pending[0:0] = children.get(node.node_id, [])
    return sorted(result, key=lambda item: item.sequence)


def _inline_content(nodes: Iterable[SourceNode]) -> str:
    parts: list[str] = []
    for node in sorted(nodes, key=lambda item: item.sequence):
        kind = node.kind.value
        if kind == "text_run" and not node.payload.get("deleted"):
            parts.append(str(node.payload.get("text") or ""))
        elif kind == "equation":
            latex = node.payload.get("latex")
            if latex:
                parts.append(f"<eq>{latex}</eq>")
        elif kind == "line_break":
            parts.append("\n")
        elif kind == "tab":
            parts.append("\t")
    return "".join(parts)
