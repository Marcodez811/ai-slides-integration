"""Deterministic local persistence for outline and semantic-content artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from presentation_pipeline.planning.models import PresentationOutline
from presentation_pipeline.synthesis.models import PresentationContent


TArtifact = TypeVar("TArtifact", bound=BaseModel)


def _write_json_artifact(
    artifact: TArtifact,
    expected_type: type[TArtifact],
    target: str | Path,
    *,
    overwrite: bool,
) -> Path:
    if not isinstance(artifact, expected_type):
        raise TypeError(f"artifact must be a {expected_type.__name__}")

    output_path = Path(target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        artifact.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    # Exclusive creation preserves the no-silent-overwrite contract even when
    # two generation processes choose the same explicit target.
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return output_path


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
    return _write_json_artifact(outline, PresentationOutline, target, overwrite=overwrite)


def write_presentation_content_json(
    content: PresentationContent,
    target: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write validated semantic slide content as deterministic, protected JSON."""
    return _write_json_artifact(content, PresentationContent, target, overwrite=overwrite)
