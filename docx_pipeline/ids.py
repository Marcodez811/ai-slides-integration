"""Deterministic identities and canonical JSON primitives for the Source IR."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping, Sequence


def sha256_hex(value: bytes | str) -> str:
    """Return the SHA-256 digest of bytes (or UTF-8 text)."""
    if isinstance(value, str):
        value = value.encode("utf-8")
    return sha256(value).hexdigest()


def _identity_digest(*components: object) -> str:
    # Length-prefixing prevents ambiguous inputs such as ("ab", "c") and
    # ("a", "bc").  JSON gives mappings/lists a deterministic representation.
    encoded = bytearray()
    for component in components:
        if isinstance(component, (Mapping, Sequence)) and not isinstance(component, str):
            text = json.dumps(component, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        else:
            text = str(component)
        value = text.encode("utf-8")
        encoded.extend(len(value).to_bytes(8, "big"))
        encoded.extend(value)
    return sha256(encoded).hexdigest()


def make_doc_id(package_bytes: bytes, *, prefix_length: int = 16) -> str:
    """Build the stable, human-readable document identifier from original bytes."""
    if prefix_length < 1 or prefix_length > 64:
        raise ValueError("prefix_length must be between 1 and 64")
    return f"doc-{sha256_hex(package_bytes)[:prefix_length]}"


def make_node_id(doc_id: str, part_name: str, xml_path: str, occurrence_kind: str) -> str:
    """Build an occurrence identity; XML position, not extracted text, is its key."""
    return f"node-{_identity_digest(doc_id, part_name, xml_path, occurrence_kind)[:24]}"


def make_asset_id(asset_bytes: bytes) -> str:
    """Build a content-addressed asset ID from its original bytes."""
    return f"asset-{sha256_hex(asset_bytes)}"


def make_view_id(
    normalizer_name: str,
    normalizer_version: str,
    config: Mapping[str, Any] | None,
    source_node_ids: Sequence[str],
) -> str:
    """Build a deterministic ID for a derived view object."""
    return "view-" + _identity_digest(
        normalizer_name, normalizer_version, config or {}, list(source_node_ids)
    )[:24]


# Short aliases are intentionally public: extractors read more clearly with them.
document_id = make_doc_id
node_id = make_node_id
asset_id = make_asset_id
view_id = make_view_id


def canonical_json(value: Any) -> str:
    """Serialize models and ordinary JSON values to stable, compact UTF-8 JSON.

    Pydantic v2 values are converted in JSON mode first so enums, paths, and
    datetimes never depend on the standard encoder's implementation details.
    """
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True, exclude_none=False)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")
