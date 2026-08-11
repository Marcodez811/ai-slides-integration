"""Deterministic, local artifact persistence for generated outlines."""

from __future__ import annotations

import json
from pathlib import Path

from presentation_pipeline.planning.models import PresentationOutline


def write_outline_json(
    outline: PresentationOutline,
    target: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a schema-valid outline as deterministic, UTF-8 pretty JSON.

    ``target`` is deliberately explicit: this helper does not choose a storage
    directory or manage generation runs.  Parent directories are created for
    the supplied target.  Existing files are protected by default; callers
    must opt into replacement with ``overwrite=True``.
    """
    if not isinstance(outline, PresentationOutline):
        raise TypeError("outline must be a PresentationOutline")

    output_path = Path(target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        outline.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"

    # Exclusive creation preserves the no-silent-overwrite contract even when
    # two generation processes happen to choose the same explicit target.
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return output_path
