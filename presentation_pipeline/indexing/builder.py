"""Build deterministic presentation evidence indexes from extracted DOCX artifacts."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from docx_pipeline.enrichment.table_profiler import profile_tables
from docx_pipeline.ir.source import SourceNodeKind

from presentation_pipeline.common.ids import make_pipeline_id
from presentation_pipeline.corpus.models import DocumentArtifact

from .models import DocumentIndex, EvidenceItem, EvidenceKind, IndexedSection


INDEXER_VERSION = "1.0.0"

_BLOCK_KIND_TO_EVIDENCE_KIND: dict[str, EvidenceKind] = {
    "text": EvidenceKind.TEXT,
    "title": EvidenceKind.TITLE,
    "caption": EvidenceKind.CAPTION,
    "list": EvidenceKind.LIST,
    "table": EvidenceKind.TABLE,
    "image": EvidenceKind.IMAGE,
    "equation": EvidenceKind.EQUATION,
    "chart": EvidenceKind.CHART,
}


def build_document_index(artifact: DocumentArtifact) -> DocumentIndex:
    """Index supported normalized blocks without changing the extraction result.

    References are deliberately validated again here.  The index may be loaded
    from persisted artifacts that bypassed Pydantic's extraction-time checks;
    producing evidence with dangling provenance would be harder to diagnose
    downstream than failing at this boundary.
    """
    result = artifact.extraction
    if artifact.doc_id != result.document.doc_id:
        raise ValueError("artifact doc_id must match extraction.document.doc_id")
    if artifact.filename != result.document.filename:
        raise ValueError("artifact filename must match extraction.document.filename")
    nodes = list(result.nodes)
    blocks = list(result.views.blocks)
    sections = list(result.views.sections)
    assets = list(result.assets)

    node_by_id = _unique_by_id(nodes, "node_id", "SourceNode")
    block_by_id = _unique_by_id(blocks, "block_id", "NormalizedBlock")
    asset_by_id = _unique_by_id(assets, "asset_id", "Asset")
    section_by_id = _unique_by_id(sections, "section_id", "SectionView")

    _validate_blocks(blocks, node_by_id, asset_by_id)
    section_by_block = _build_section_by_block(sections, block_by_id, node_by_id, section_by_id)
    table_block_by_node_id = _table_block_mapping(blocks, node_by_id)
    candidate_by_table_node_id = _chart_candidates_by_table_node(nodes, table_block_by_node_id, node_by_id)

    evidence: list[EvidenceItem] = []
    evidence_ids_by_block: dict[str, list[str]] = defaultdict(list)
    for block in blocks:
        kind = _BLOCK_KIND_TO_EVIDENCE_KIND.get(_enum_value(block.kind))
        if kind is None:
            continue
        item = _primary_evidence(
            result.document.doc_id,
            block,
            kind,
            section_by_block[block.block_id],
            node_by_id,
            asset_by_id,
        )
        evidence.append(item)
        evidence_ids_by_block[block.block_id].append(item.evidence_id)
        if kind is EvidenceKind.TABLE and (
            candidate := candidate_by_table_node_id.get(_source_ids(block)[0])
        ) is not None:
            candidate_item = _chart_candidate_evidence(
                result.document.doc_id,
                block.block_id,
                section_by_block[block.block_id],
                candidate,
            )
            evidence.append(candidate_item)
            evidence_ids_by_block[block.block_id].append(candidate_item.evidence_id)

    indexed_sections = [
        IndexedSection(
            section_id=section.section_id,
            source_heading_node_id=section.source_heading_node_id,
            title=section.title,
            block_ids=list(section.block_ids),
            evidence_ids=[
                evidence_id
                for block_id in section.block_ids
                for evidence_id in evidence_ids_by_block.get(block_id, [])
            ],
            parent_id=section.parent_id,
            level=section.level,
            confidence=section.confidence,
            detector_name=section.detector_name,
            detector_version=section.detector_version,
        )
        for section in sections
    ]
    return DocumentIndex(
        doc_id=artifact.doc_id,
        filename=artifact.filename,
        extraction_schema_version=result.schema_version,
        extractor_version=result.extractor_version,
        indexer_version=INDEXER_VERSION,
        evidence=evidence,
        sections=indexed_sections,
    )


def _unique_by_id(items: Iterable[Any], attribute: str, label: str) -> dict[str, Any]:
    by_id: dict[str, Any] = {}
    for item in items:
        value = getattr(item, attribute, None)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} has a missing or invalid {attribute}")
        if value in by_id:
            raise ValueError(f"duplicate {label} {attribute}: {value!r}")
        by_id[value] = item
    return by_id


def _validate_blocks(blocks: Sequence[Any], node_by_id: Mapping[str, Any], asset_by_id: Mapping[str, Any]) -> None:
    for block in blocks:
        source_ids = _source_ids(block)
        unknown = [source_id for source_id in source_ids if source_id not in node_by_id]
        if unknown:
            raise ValueError(f"block {block.block_id!r} references unknown source nodes: {unknown!r}")
        kind = _enum_value(block.kind)
        if kind == "table" and _enum_value(node_by_id[source_ids[0]].kind) != SourceNodeKind.TABLE.value:
            raise ValueError(f"table block {block.block_id!r} must start with a structural table node")
        if kind == "image":
            if _enum_value(node_by_id[source_ids[0]].kind) != SourceNodeKind.IMAGE.value:
                raise ValueError(f"image block {block.block_id!r} must start with an image source node")
            payload = _payload(block)
            asset_id = payload.get("asset_id")
            if asset_id is not None and (not isinstance(asset_id, str) or not asset_id):
                raise ValueError(f"image block {block.block_id!r} has an invalid asset_id")
            if asset_id is not None and asset_id not in asset_by_id:
                raise ValueError(f"image block {block.block_id!r} references unknown asset {asset_id!r}")
            source_asset_ids = {
                source_asset_id
                for source_id in source_ids
                if (source_asset_id := node_by_id[source_id].payload.get("asset_id")) is not None
            }
            if asset_id is not None and source_asset_ids and source_asset_ids != {asset_id}:
                raise ValueError(
                    f"image block {block.block_id!r} asset_id {asset_id!r} does not match its image source node"
                )


def _build_section_by_block(
    sections: Sequence[Any],
    block_by_id: Mapping[str, Any],
    node_by_id: Mapping[str, Any],
    section_by_id: Mapping[str, Any],
) -> dict[str, list[str]]:
    membership: dict[str, list[str]] = {block_id: [] for block_id in block_by_id}
    for section in sections:
        if section.source_heading_node_id is not None and section.source_heading_node_id not in node_by_id:
            raise ValueError(
                f"section {section.section_id!r} references unknown heading node {section.source_heading_node_id!r}"
            )
        if section.parent_id is not None and section.parent_id not in section_by_id:
            raise ValueError(f"section {section.section_id!r} references unknown parent section {section.parent_id!r}")
        for block_id in section.block_ids:
            if block_id not in block_by_id:
                raise ValueError(f"section {section.section_id!r} references unknown block {block_id!r}")
            membership[block_id].append(section.section_id)
    return membership


def _table_block_mapping(blocks: Sequence[Any], node_by_id: Mapping[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for block in blocks:
        if _enum_value(block.kind) != "table":
            continue
        table_node_id = _source_ids(block)[0]
        if _enum_value(node_by_id[table_node_id].kind) != SourceNodeKind.TABLE.value:
            # _validate_blocks gives the clearer public error, but retain this
            # guard so this helper is safe if it is reused independently.
            raise ValueError(f"table block {block.block_id!r} has no structural table source")
        if table_node_id in mapping:
            raise ValueError(f"multiple normalized table blocks reference table node {table_node_id!r}")
        mapping[table_node_id] = block.block_id
    return mapping


def _chart_candidates_by_table_node(
    nodes: Sequence[Any], table_block_by_node_id: Mapping[str, str], node_by_id: Mapping[str, Any]
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for profile in profile_tables(nodes):
        candidate = profile.chart_candidate
        if candidate is None:
            continue
        if candidate.table_node_id not in table_block_by_node_id:
            raise ValueError(
                "chart candidate references table node "
                f"{candidate.table_node_id!r} without a normalized table block"
            )
        if candidate.table_node_id in candidates:
            raise ValueError(f"multiple chart candidates reference table node {candidate.table_node_id!r}")
        unknown = [source_id for source_id in candidate.source_node_ids if source_id not in node_by_id]
        if unknown:
            raise ValueError(
                f"chart candidate for block {table_block_by_node_id[candidate.table_node_id]!r} "
                f"references unknown nodes: {unknown!r}"
            )
        candidates[candidate.table_node_id] = candidate
    return candidates


def _chart_candidate_evidence(
    document_id: str, block_id: str, section_ids: Sequence[str], candidate: Any
) -> EvidenceItem:
    source_ids = list(candidate.source_node_ids)
    return EvidenceItem(
        doc_id=document_id,
        evidence_id=_evidence_id(
            document_id, EvidenceKind.CHART_CANDIDATE, block_id, source_ids, "table-profile"
        ),
        kind=EvidenceKind.CHART_CANDIDATE,
        block_ids=[block_id],
        source_node_ids=source_ids,
        section_ids=list(section_ids),
        structured_data=_json_safe(
            {
                "candidate_id": candidate.candidate_id,
                "table_node_id": candidate.table_node_id,
                "category_column": candidate.category_column,
                "categories": candidate.categories,
                "series": candidate.series,
                "source_node_ids": candidate.source_node_ids,
            }
        ),
    )


def _primary_evidence(
    document_id: str,
    block: Any,
    kind: EvidenceKind,
    section_ids: Sequence[str],
    node_by_id: Mapping[str, Any],
    asset_by_id: Mapping[str, Any],
) -> EvidenceItem:
    source_ids = _source_ids(block)
    payload = _payload(block)
    text: str | None = None
    asset_ids: list[str] = []
    structured_data: dict[str, Any] = {"payload": _json_safe(payload)}
    if kind in {EvidenceKind.TEXT, EvidenceKind.TITLE, EvidenceKind.CAPTION, EvidenceKind.TABLE, EvidenceKind.EQUATION}:
        content_key = "text" if kind is EvidenceKind.TABLE else "latex" if kind is EvidenceKind.EQUATION else "content"
        value = payload.get(content_key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{kind.value} block {block.block_id!r} has non-string content")
        text = value
    elif kind is EvidenceKind.LIST:
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError(f"list block {block.block_id!r} has malformed items payload")
        _validate_list_items(items, block.block_id, source_ids, node_by_id)
        text = _list_text(source_ids, node_by_id)
        structured_data["items"] = _json_safe(items)
    elif kind is EvidenceKind.IMAGE:
        candidate_asset_id = payload.get("asset_id")
        if candidate_asset_id is not None:
            # _validate_blocks already ensures both type and existence.
            asset_ids = [candidate_asset_id]
            structured_data["asset"] = _json_safe(asset_by_id[candidate_asset_id])
    return EvidenceItem(
        doc_id=document_id,
        evidence_id=_evidence_id(document_id, kind, block.block_id, source_ids, "primary"),
        kind=kind,
        block_ids=[block.block_id],
        source_node_ids=source_ids,
        section_ids=list(section_ids),
        text=text,
        asset_ids=asset_ids,
        structured_data=structured_data,
    )


def _list_text(source_ids: Sequence[str], node_by_id: Mapping[str, Any]) -> str:
    """Recover list text from paragraph descendants in source order."""
    child_nodes: dict[str, list[Any]] = defaultdict(list)
    for node in node_by_id.values():
        if node.parent_id is not None:
            child_nodes[node.parent_id].append(node)
    for children in child_nodes.values():
        children.sort(key=lambda node: node.sequence)

    parts: list[str] = []
    for source_id in source_ids:
        root = node_by_id[source_id]
        if _enum_value(root.kind) != SourceNodeKind.PARAGRAPH.value:
            continue
        for node in _descendants(root.node_id, child_nodes):
            kind = _enum_value(node.kind)
            if kind == SourceNodeKind.TEXT_RUN.value and not node.payload.get("deleted"):
                parts.append(str(node.payload.get("text") or ""))
            elif kind == SourceNodeKind.LINE_BREAK.value:
                parts.append("\n")
            elif kind == SourceNodeKind.TAB.value:
                parts.append("\t")
        parts.append("\n")
    return "".join(parts).rstrip("\n")


def _descendants(root_id: str, children: Mapping[str, Sequence[Any]]) -> list[Any]:
    descendants: list[Any] = []
    pending = list(children.get(root_id, []))
    while pending:
        node = pending.pop(0)
        descendants.append(node)
        pending[0:0] = children.get(node.node_id, [])
    return sorted(descendants, key=lambda node: node.sequence)


def _validate_list_items(
    items: Sequence[Any], block_id: str, block_source_ids: Sequence[str], node_by_id: Mapping[str, Any]
) -> None:
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("source_node_id"), str):
            raise ValueError(f"list block {block_id!r} has malformed item structure")
        source_node_id = item["source_node_id"]
        if source_node_id not in node_by_id:
            raise ValueError(f"list block {block_id!r} item references unknown source node {source_node_id!r}")
        if source_node_id not in block_source_ids:
            raise ValueError(f"list block {block_id!r} item is not linked by block source_node_ids")
        if _enum_value(node_by_id[source_node_id].kind) != SourceNodeKind.PARAGRAPH.value:
            raise ValueError(f"list block {block_id!r} item source {source_node_id!r} is not a paragraph")
        children = item.get("children", [])
        if not isinstance(children, list):
            raise ValueError(f"list block {block_id!r} has malformed nested item children")
        _validate_list_items(children, block_id, block_source_ids, node_by_id)


def _source_ids(block: Any) -> list[str]:
    source_ids = getattr(block, "source_node_ids", None)
    if not isinstance(source_ids, list) or not source_ids or any(not isinstance(value, str) or not value for value in source_ids):
        raise ValueError(f"block {getattr(block, 'block_id', '<unknown>')!r} has malformed source_node_ids")
    return list(source_ids)


def _payload(block: Any) -> dict[str, Any]:
    payload = getattr(block, "payload", None)
    if not isinstance(payload, dict):
        raise ValueError(f"block {block.block_id!r} has malformed payload")
    return payload


def _enum_value(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def _evidence_id(
    document_id: str,
    kind: EvidenceKind,
    block_id: str,
    source_node_ids: Sequence[str],
    derivation: str,
) -> str:
    """Create one central, reproducible evidence identity for all evidence types."""
    return make_pipeline_id(
        "evidence",
        document_id,
        INDEXER_VERSION,
        kind.value,
        block_id,
        derivation,
        list(source_node_ids),
    )


def _json_safe(value: Any) -> Any:
    """Convert Pydantic/profile values to JSON-safe builtins without mutation."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("structured evidence data must not contain non-finite floats")
        return value
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise ValueError(f"structured evidence data is not JSON-safe: {type(value).__name__}")
