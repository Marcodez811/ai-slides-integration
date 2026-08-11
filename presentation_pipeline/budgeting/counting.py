"""Provider-neutral token counting interfaces and offline estimators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .serialization import serialize_provider_input


@runtime_checkable
class TokenCounter(Protocol):
    """Measure text and deterministically serialized provider payloads."""

    def count_text(self, text: str) -> int: ...

    def count_payload(self, payload: object) -> int: ...


@dataclass(frozen=True, slots=True)
class CharacterTokenEstimator:
    """Deterministic conservative-enough fallback when no tokenizer is present."""

    chars_per_token: float = 4.0

    def __post_init__(self) -> None:
        if isinstance(self.chars_per_token, bool) or not isinstance(self.chars_per_token, (int, float)):
            raise TypeError("chars_per_token must be a positive number")
        if not math.isfinite(self.chars_per_token) or self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be a positive finite number")

    def count_text(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return math.ceil(len(text) / self.chars_per_token)

    def count_payload(self, payload: object) -> int:
        if not isinstance(payload, dict):
            raise TypeError("provider payload must be a dictionary")
        return self.count_text(serialize_provider_input(payload))


@dataclass(frozen=True, slots=True)
class Utf8ByteTokenEstimator:
    """Conservative offline estimator based on UTF-8 bytes.

    One token per byte intentionally overestimates most provider tokenizers,
    including for CJK-heavy corpora.  It is the safe choice for preflight
    scale configuration when an exact provider tokenizer is unavailable.
    """

    bytes_per_token: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.bytes_per_token, bool) or not isinstance(self.bytes_per_token, (int, float)):
            raise TypeError("bytes_per_token must be a positive number")
        if not math.isfinite(self.bytes_per_token) or self.bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be a positive finite number")

    def count_text(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return math.ceil(len(text.encode("utf-8")) / self.bytes_per_token)

    def count_payload(self, payload: object) -> int:
        if not isinstance(payload, dict):
            raise TypeError("provider payload must be a dictionary")
        return self.count_text(serialize_provider_input(payload))
