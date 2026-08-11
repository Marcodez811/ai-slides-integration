"""General identifiers for orchestration and persistence boundaries.

Semantic identities belong to their owning domain.  These IDs deliberately
identify execution and storage records, so random UUIDs are appropriate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from uuid import uuid4


def new_id(prefix: str) -> str:
    """Return a namespaced UUID identifier for a non-semantic record."""
    normalized_prefix = prefix.strip().replace("_", "-")
    if not normalized_prefix or not normalized_prefix.replace("-", "").isalnum():
        raise ValueError("ID prefix must contain only letters, digits, and hyphens")
    return f"{normalized_prefix}-{uuid4()}"


def make_pipeline_id(kind: str, *components: object) -> str:
    """Build a deterministic, length-safe ID from stable pipeline facts."""
    normalized_kind = kind.strip().replace("_", "-")
    if not normalized_kind or not normalized_kind.replace("-", "").isalnum():
        raise ValueError("ID kind must contain only letters, digits, and hyphens")

    encoded = bytearray()
    for component in components:
        if isinstance(component, (Mapping, Sequence)) and not isinstance(component, (str, bytes, bytearray)):
            text = json.dumps(component, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        elif isinstance(component, (bytes, bytearray)):
            text = bytes(component).hex()
        else:
            text = str(component)
        value = text.encode("utf-8")
        encoded.extend(len(value).to_bytes(8, "big"))
        encoded.extend(value)
    return f"{normalized_kind}-{sha256(encoded).hexdigest()[:24]}"


def make_job_id() -> str:
    return new_id("job")


def make_batch_id() -> str:
    return new_id("batch")


def make_corpus_id() -> str:
    return new_id("corpus")


def make_document_artifact_id() -> str:
    return new_id("artifact")
