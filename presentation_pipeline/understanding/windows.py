"""Deterministic, provider-safe evidence windows for document understanding."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
import re
from presentation_pipeline.budgeting import InputBudget, TokenCounter, estimate_request_tokens

from presentation_pipeline.indexing.compact import compact_evidence_item, compact_sections

from .prompts import CHUNK_DIGEST_PROMPT, build_chunk_digest_input, document_id


class EvidenceWindowingError(ValueError):
    """A document cannot be represented as safe evidence windows."""


class OversizedEvidenceError(EvidenceWindowingError):
    """An atomic structured evidence item cannot safely be transported."""


@dataclass(frozen=True, slots=True)
class EvidenceWindow:
    """Ephemeral transport data; IDs inside it remain canonical evidence IDs."""

    doc_id: str
    window_id: str
    ordinal: int
    evidence_ids: tuple[str, ...]
    payload: dict[str, object]
    estimated_tokens: int


def _limit(budget: InputBudget) -> int:
    if not isinstance(budget, InputBudget):
        raise TypeError("window budget must be an InputBudget")
    return budget.target_input_tokens_or_usable


def _count(counter: TokenCounter, payload: dict[str, object]) -> int:
    # Count the full request shape, including the fixed instruction text, rather
    # than adding up evidence text alone.
    value = estimate_request_tokens(CHUNK_DIGEST_PROMPT, payload, counter)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token counter returned an invalid token count")
    return value


def _sections_for(
    evidence: list[dict[str, object]], sections: list[dict[str, object]]
) -> list[dict[str, object]]:
    used = {
        section_id
        for item in evidence
        for section_id in item.get("section_ids", [])
        if isinstance(section_id, str)
    }
    return [section for section in sections if section.get("section_id") in used]


def _window_payload(
    artifact: object, *, ordinal: int, evidence: list[dict[str, object]], sections: list[dict[str, object]]
) -> dict[str, object]:
    return build_chunk_digest_input(
        artifact,
        window_id=f"window-{ordinal:04d}",
        ordinal=ordinal,
        sections=_sections_for(evidence, sections),
        evidence=list(evidence),
    )


def _text_units(text: str) -> list[str]:
    paragraphs = [part for part in re.split(r"(\n\s*\n+)", text) if part]
    return paragraphs or [text]


def _sentence_units(text: str) -> list[str]:
    # End each unit *after* its separator so joining the slices recreates the
    # source exactly. ``re.split`` without retaining separators silently drops
    # whitespace between sentences.
    pieces: list[str] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        pieces.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        pieces.append(text[start:])
    return pieces or [text]


def _slice_text_item(
    item: dict[str, object], *, fits: Callable[[dict[str, object]], bool]
) -> list[dict[str, object]]:
    text = item.get("text")
    if not isinstance(text, str) or not text:
        raise AssertionError("text slice requested without text")

    def clone(part: str, number: int, total: int = 0) -> dict[str, object]:
        result = dict(item)
        result["text"] = part
        # This is deliberately transport metadata, not an evidence identifier.
        result["slice"] = {"slice_id": f"slice-{number:04d}", "index": number, "count": total}
        return result

    raw: list[str] = []
    for paragraph in _text_units(text):
        if fits(clone(paragraph, len(raw), 1_000_000_000)):
            raw.append(paragraph)
            continue
        for sentence in _sentence_units(paragraph):
            if fits(clone(sentence, len(raw), 1_000_000_000)):
                raw.append(sentence)
                continue
            start = 0
            while start < len(sentence):
                # Binary search gives a deterministic character fallback and
                # guarantees no silent truncation.
                low, high, best = 1, len(sentence) - start, 0
                while low <= high:
                    middle = (low + high) // 2
                    if fits(
                        clone(
                            sentence[start : start + middle],
                            len(raw),
                            1_000_000_000,
                        )
                    ):
                        best, low = middle, middle + 1
                    else:
                        high = middle - 1
                if best == 0:
                    evidence_id = item.get("evidence_id", "<unknown>")
                    raise OversizedEvidenceError(
                        f"text evidence {evidence_id!r} cannot fit an otherwise empty window"
                    )
                raw.append(sentence[start : start + best])
                start += best
    return [clone(part, number, len(raw)) for number, part in enumerate(raw)]


def _slice_table_item(
    item: dict[str, object], *, fits: Callable[[dict[str, object]], bool]
) -> list[dict[str, object]]:
    """Slice a text-backed table on row boundaries, then exact character spans."""
    text = item.get("text")
    if not isinstance(text, str) or not text:
        raise OversizedEvidenceError(
            f"table evidence {item.get('evidence_id', '<unknown>')!r} has no row text to slice"
        )

    def clone(part: str, number: int, total: int = 0) -> dict[str, object]:
        result = deepcopy(item)
        result["text"] = part
        content = result.get("content")
        if isinstance(content, dict):
            payload = content.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("text"), str):
                payload["text"] = part
        result["slice"] = {
            "slice_id": f"slice-{number:04d}",
            "index": number,
            "count": total,
            "unit": "table_rows",
        }
        return result

    raw: list[str] = []
    for row in text.splitlines(keepends=True) or [text]:
        if fits(clone(row, len(raw), 1_000_000_000)):
            raw.append(row)
            continue
        start = 0
        while start < len(row):
            low, high, best = 1, len(row) - start, 0
            while low <= high:
                middle = (low + high) // 2
                if fits(
                    clone(
                        row[start : start + middle],
                        len(raw),
                        1_000_000_000,
                    )
                ):
                    best, low = middle, middle + 1
                else:
                    high = middle - 1
            if best == 0:
                raise OversizedEvidenceError(
                    f"table evidence {item.get('evidence_id', '<unknown>')!r} cannot fit an otherwise empty window"
                )
            raw.append(row[start : start + best])
            start += best
    return [clone(part, number, len(raw)) for number, part in enumerate(raw)]


def build_evidence_windows(
    artifact: object | None,
    index: object,
    *,
    token_counter: TokenCounter,
    budget: InputBudget,
) -> list[EvidenceWindow]:
    """Pack ordered evidence into bounded section-aware windows.

    Text, title, and caption evidence can be sliced without changing their
    canonical ID.  Other oversized structured evidence fails explicitly.
    """
    if artifact is None:
        artifact = index
    doc_id = document_id(artifact)
    if getattr(index, "doc_id", None) != doc_id:
        raise ValueError("artifact and index document IDs do not match")
    limit = _limit(budget)
    source = getattr(index, "evidence", None)
    if not isinstance(source, list) or not source:
        raise ValueError(f"document {doc_id!r} has no evidence to window")
    sections = compact_sections(index)
    compact = [compact_evidence_item(item) for item in source]
    windows: list[EvidenceWindow] = []
    current: list[dict[str, object]] = []

    def fits(candidate: list[dict[str, object]], ordinal: int) -> tuple[bool, int, dict[str, object]]:
        payload = _window_payload(artifact, ordinal=ordinal, evidence=candidate, sections=sections)
        estimated = _count(token_counter, payload)
        return estimated <= limit, estimated, payload

    def flush() -> None:
        if not current:
            return
        ordinal = len(windows)
        ok, estimated, payload = fits(current, ordinal)
        if not ok:  # Defensive: all additions are tested before entering current.
            raise OversizedEvidenceError("window exceeded its configured input budget")
        ids = tuple(str(value["evidence_id"]) for value in current)
        windows.append(
            EvidenceWindow(
                doc_id,
                f"window-{ordinal:04d}",
                ordinal,
                ids,
                payload,
                estimated,
            )
        )
        current.clear()

    for item_index, item in enumerate(compact):
        # Prefer a section boundary only when the next adjacent section cannot
        # fit with the current window. Small neighboring sections may share a
        # window, avoiding one provider call per tiny section.
        current_sections = {
            section_id
            for value in current
            for section_id in value.get("section_ids", [])
            if isinstance(section_id, str)
        }
        item_sections = {
            section_id
            for section_id in item.get("section_ids", [])
            if isinstance(section_id, str)
        }
        if (
            current
            and current_sections
            and item_sections
            and current_sections.isdisjoint(item_sections)
        ):
            section_run = [item]
            for following in compact[item_index + 1 :]:
                following_sections = {
                    section_id
                    for section_id in following.get("section_ids", [])
                    if isinstance(section_id, str)
                }
                if following_sections != item_sections:
                    break
                section_run.append(following)
            if not fits([*current, *section_run], len(windows))[0]:
                flush()
        ordinal = len(windows)
        candidate = [*current, item]
        ok, _, _ = fits(candidate, ordinal)
        if ok:
            current.append(item)
            continue
        if current:
            flush()
            ordinal = len(windows)
            ok, _, _ = fits([item], ordinal)
            if ok:
                current.append(item)
                continue
        kind = item.get("kind")
        if kind == "table":
            slices = _slice_table_item(
                item, fits=lambda sliced: fits([sliced], len(windows))[0]
            )
        elif kind in {"text", "title", "caption"} and isinstance(item.get("text"), str):
            slices = _slice_text_item(
                item, fits=lambda sliced: fits([sliced], len(windows))[0]
            )
        else:
            raise OversizedEvidenceError(
                f"unsupported oversized structured evidence {item.get('evidence_id')!r} ({kind!r})"
            )
        for sliced in slices:
            if current and not fits([*current, sliced], len(windows))[0]:
                flush()
            if not fits([sliced], len(windows))[0]:
                raise OversizedEvidenceError(f"slice for {item.get('evidence_id')!r} does not fit")
            current.append(sliced)
    flush()
    return windows
