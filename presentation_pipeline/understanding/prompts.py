"""Small deterministic inputs and prompts for document understanding."""

from __future__ import annotations

DOCUMENT_DIGEST_PROMPT = """You are producing an auditable document digest for presentation planning.
Use only the supplied evidence IDs and supplied text. Do not fabricate facts, mutate evidence,
or alter numbers. Every topic and key fact must cite one or more supplied evidence IDs from its
own document. Return only the requested structured response."""


def document_id(artifact: object) -> str:
    value = getattr(artifact, "doc_id", None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("artifact does not expose a non-empty document ID")
    return value


def compact_evidence(index: object) -> list[dict[str, object]]:
    """Return prompt-safe evidence: IDs plus human-readable indexed content only.

    Source node IDs, normalized block IDs, XML locations, and extraction payloads
    intentionally never cross this provider boundary.
    """
    items = getattr(index, "evidence", None)
    if not isinstance(items, list):
        raise ValueError("document index does not expose evidence")
    result: list[dict[str, object]] = []
    for item in items:
        result.append(compact_evidence_item(item))
    return result


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
        content = _without_provenance_internals(structured_data)
        if content:
            compact["content"] = content
    return compact


def _without_provenance_internals(value: object) -> object:
    if isinstance(value, list):
        return [_without_provenance_internals(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _without_provenance_internals(item)
        for key, item in value.items()
        if not _is_provenance_key(key)
    }


def _is_provenance_key(key: str) -> bool:
    normalized = key.lower()
    return normalized in {
        "asset_id",
        "candidate_id",
        "source_node_id",
        "source_node_ids",
        "table_node_id",
    } or any(token in normalized for token in ("xml", "relationship", "block_id", "path"))


def build_document_digest_input(artifact: object, index: object) -> dict[str, object]:
    filename = getattr(artifact, "filename", None)
    document_input: dict[str, str] = {"doc_id": document_id(artifact)}
    if isinstance(filename, str) and filename:
        document_input["filename"] = filename
    return {
        "document": document_input,
        "sections": compact_sections(index),
        "evidence": compact_evidence(index),
    }


def compact_sections(index: object) -> list[dict[str, object]]:
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
