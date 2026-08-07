"""Offline contracts and cache primitives for post-extraction image classification."""

from __future__ import annotations

from hashlib import sha256
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..ir.source import Asset


class ImageClassification(BaseModel):
    """A versioned, structured classifier result; extraction never creates it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str
    asset_sha256: str
    classifier_version: str
    prompt_version: str
    relevance: str
    decorative: bool
    description: str | None = None
    likely_caption: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    error: str | None = None

    @property
    def cache_key(self) -> str:
        return classification_cache_key(
            self.asset_sha256, self.classifier_version, self.prompt_version
        )


class ImageClassifier(Protocol):
    """Boundary for a caller-supplied classifier (local or remote)."""

    classifier_version: str
    prompt_version: str

    def classify(self, asset: Asset) -> ImageClassification:
        """Return a result for ``asset`` without modifying extraction records."""


def classification_cache_key(asset_sha256: str, classifier_version: str, prompt_version: str) -> str:
    """Return a stable cache key independent of asset filenames or output paths."""
    encoded = "\0".join((asset_sha256, classifier_version, prompt_version)).encode("utf-8")
    return "image-classification-" + sha256(encoded).hexdigest()


class InMemoryClassificationCache:
    """Tiny explicit cache useful to adapters and tests; it performs no I/O."""

    def __init__(self) -> None:
        self._entries: dict[str, ImageClassification] = {}

    def get(self, asset: Asset, classifier_version: str, prompt_version: str) -> ImageClassification | None:
        return self._entries.get(classification_cache_key(asset.sha256, classifier_version, prompt_version))

    def put(self, classification: ImageClassification) -> None:
        self._entries[classification.cache_key] = classification
