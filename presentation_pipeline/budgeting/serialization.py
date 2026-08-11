"""Canonical, provider-facing input serialization."""

from __future__ import annotations

import json


def serialize_provider_input(input_data: dict[str, object]) -> str:
    """Serialize structured provider input deterministically.

    This deliberately matches the OpenAI adapter's wire representation.  Keep
    budgeting on this helper so preflight estimates cannot drift from requests.
    """
    return json.dumps(
        input_data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
