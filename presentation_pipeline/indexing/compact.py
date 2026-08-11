"""Provider-safe semantic projections of deterministic document indexes."""

from __future__ import annotations


def compact_evidence(index: object) -> list[dict[str, object]]:
    """Return prompt-safe evidence: IDs plus human-readable indexed content only."""
    items = getattr(index, "evidence", None)
    if not isinstance(items, list):
        raise ValueError("document index does not expose evidence")
    return [compact_evidence_item(item) for item in items]


def compact_evidence_item(item: object) -> dict[str, object]:
    """Return semantic evidence content without mechanical provenance fields."""
    evidence_id = getattr(item, "evidence_id", None)
    text = getattr(item, "text", None)
    kind = getattr(item, "kind", None)
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError("index contains evidence without a non-empty evidence ID")
    if kind is None:
        raise ValueError(f"evidence {evidence_id!r} has no kind")
    compact: dict[str, object] = {
        "evidence_id": evidence_id,
        "kind": str(getattr(kind, "value", kind)),
    }
    section_ids = getattr(item, "section_ids", [])
    if not isinstance(section_ids, list) or any(
        not isinstance(section_id, str) or not section_id for section_id in section_ids
    ):
        raise ValueError(f"evidence {evidence_id!r} has malformed section IDs")
    if section_ids:
        compact["section_ids"] = list(section_ids)
    if isinstance(text, str) and text.strip():
        compact["text"] = text
    structured_data = getattr(item, "structured_data", None)
    if isinstance(structured_data, dict):
        content = without_provenance_internals(structured_data)
        if content:
            compact["content"] = content
    return compact


def compact_sections(index: object) -> list[dict[str, object]]:
    """Return semantic section metadata without extraction internals."""
    sections = getattr(index, "sections", [])
    if not isinstance(sections, list):
        raise ValueError("document index has malformed sections")
    compact: list[dict[str, object]] = []
    for section in sections:
        section_id = getattr(section, "section_id", None)
        if not isinstance(section_id, str) or not section_id:
            raise ValueError("document index contains a section without an ID")
        item: dict[str, object] = {"section_id": section_id}
        for field in ("title", "level", "parent_id"):
            value = getattr(section, field, None)
            if value is not None:
                item[field] = value
        compact.append(item)
    return compact


def without_provenance_internals(value: object) -> object:
    """Recursively remove mechanical provenance fields from semantic payloads."""
    if isinstance(value, list):
        return [without_provenance_internals(item) for item in value]
    if not isinstance(value, dict):
        return value
    compact: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("structured evidence data has a non-string key")
        if not is_provenance_key(key):
            compact[key] = without_provenance_internals(item)
    return compact


def is_provenance_key(key: str) -> bool:
    """Whether a structured-data key describes extraction implementation detail."""
    normalized = key.lower()
    return normalized in {
        "asset_id",
        "candidate_id",
        "source_node_id",
        "source_node_ids",
        "table_node_id",
    } or any(token in normalized for token in ("xml", "relationship", "block_id", "path"))
