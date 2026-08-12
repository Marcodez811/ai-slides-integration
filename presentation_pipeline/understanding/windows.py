"""Deterministic, provider-safe evidence windows for document understanding."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
import re
from presentation_pipeline.budgeting import InputBudget, TokenCounter, estimate_request_tokens
from presentation_pipeline.observability import safe_debug, safe_event

from presentation_pipeline.indexing.compact import compact_evidence_item, compact_sections

from .prompts import CHUNK_DIGEST_PROMPT, build_chunk_digest_input, document_id


class EvidenceWindowingError(ValueError):
    """A document cannot be represented as safe evidence windows."""


class OversizedEvidenceError(EvidenceWindowingError):
    """An atomic structured evidence item cannot safely be transported."""

    def __init__(self, message: str, **metadata: object) -> None:
        self.metadata = {
            key: value for key, value in metadata.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
        super().__init__(message)


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


def _list_items_to_text(items: list[dict[str, object]]) -> str:
    lines: list[str] = []
    def visit(node: dict[str, object], depth: int) -> None:
        text = node.get("text")
        if isinstance(text, str) and text:
            lines.append("  " * depth + text)
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                if not isinstance(child, dict):  # defensive; validated before slicing
                    raise ValueError("list descendants must be dictionaries")
                visit(child, depth + 1)
    for item in items:
        visit(item, 0)
    return "\n".join(lines)


def _validate_list_tree(nodes: list[object]) -> None:
    """Reject malformed descendants before any packing can skip them."""
    for node in nodes:
        if not isinstance(node, dict):
            raise OversizedEvidenceError(
                "list evidence contains a malformed descendant",
                kind="list",
                reason="malformed_list_tree",
            )
        children = node.get("children")
        if children is None:
            continue
        if not isinstance(children, list):
            raise OversizedEvidenceError(
                "list evidence contains malformed children",
                kind="list",
                reason="malformed_list_tree",
            )
        _validate_list_tree(children)


def _slice_list_item(
    item: dict[str, object], *, fits: Callable[[dict[str, object]], bool]
) -> list[dict[str, object]]:
    """Greedily slice ordered list subtrees, preserving the canonical identity."""
    content = item.get("content")
    items = content.get("items") if isinstance(content, dict) else None
    if not isinstance(items, list):
        raise OversizedEvidenceError(
            "list evidence has no sliceable structured items",
            evidence_id=item.get("evidence_id"), kind="list", reason="malformed_list_items",
        )
    _validate_list_tree(items)

    def clone(nodes: list[dict[str, object]], number: int, total: int = 0, unit: str = "list_items") -> dict[str, object]:
        result = deepcopy(item)
        result_content = result.get("content")
        assert isinstance(result_content, dict)
        result_content["items"] = deepcopy(nodes)
        payload = result_content.get("payload")
        if isinstance(payload, dict):
            payload.pop("items", None)
        result["text"] = _list_items_to_text(nodes)
        result["slice"] = {"slice_id": f"slice-{number:04d}", "index": number, "count": total, "unit": unit}
        return result

    def leaf_text_slices(node: dict[str, object]) -> list[tuple[list[dict[str, object]], str]]:
        text = node.get("text")
        if not isinstance(text, str) or not text:
            raise OversizedEvidenceError(
                "list leaf cannot fit an otherwise empty window",
                evidence_id=item.get("evidence_id"), kind="list", reason="atomic_list_leaf",
            )
        parts: list[tuple[list[dict[str, object]], str]] = []
        start = 0
        while start < len(text):
            low, high, best = 1, len(text) - start, 0
            while low <= high:
                middle = (low + high) // 2
                candidate = deepcopy(node)
                candidate["text"] = text[start:start + middle]
                candidate.pop("children", None)
                if fits(clone([candidate], len(parts), 1_000_000, "list_item_text")):
                    best, low = middle, middle + 1
                else:
                    high = middle - 1
            if best == 0:
                raise OversizedEvidenceError(
                    "list leaf cannot fit an otherwise empty window",
                    evidence_id=item.get("evidence_id"), kind="list", reason="atomic_list_leaf",
                )
            candidate = deepcopy(node)
            candidate["text"] = text[start:start + best]
            candidate.pop("children", None)
            parts.append(([candidate], "list_item_text"))
            start += best
        return parts

    def split_subtree(node: dict[str, object]) -> list[tuple[list[dict[str, object]], str]]:
        if fits(clone([node], 0, 1)):
            return [([node], "list_items")]
        children = node.get("children")
        if not isinstance(children, list) or not children:
            return leaf_text_slices(node)
        base = deepcopy(node)
        base["children"] = []
        groups: list[tuple[list[dict[str, object]], str]] = []
        current: list[dict[str, object]] = []
        parent_text_emitted = False
        for child in children:
            assert isinstance(child, dict)  # _validate_list_tree above
            candidate = [*current, child]
            parent = deepcopy(base)
            parent["children"] = candidate
            if fits(clone([parent], len(groups), 1_000_000)):
                current = candidate
                continue
            if current:
                parent = deepcopy(base); parent["children"] = current
                groups.append(([parent], "list_items")); current = []
                parent_text_emitted = True
            parent = deepcopy(base); parent["children"] = [child]
            if fits(clone([parent], len(groups), 1_000_000)):
                current = [child]
            else:
                # Keep the oversized parent's own semantic text before its
                # descendant fragments. It cannot be silently sacrificed just
                # because it does not fit beside one child.
                if not current and not parent_text_emitted:
                    if fits(clone([base], len(groups), 1_000_000)):
                        groups.append(([deepcopy(base)], "list_items"))
                    else:
                        groups.extend(leaf_text_slices(base))
                    parent_text_emitted = True
                # Repeat only the enclosing parent needed to preserve the
                # hierarchy of an oversized child fragment.  This is transport
                # context, never a new canonical evidence item.
                for child_fragment, unit in split_subtree(child):
                    parent_fragment = deepcopy(base)
                    parent_fragment["children"] = child_fragment
                    if fits(clone([parent_fragment], len(groups), 1_000_000, unit)):
                        groups.append(([parent_fragment], unit))
                    else:
                        # A parent which itself cannot fit is represented by
                        # its bounded descendants; its text is retained by the
                        # leaf fallback when it is the oversized unit.
                        groups.append((child_fragment, unit))
        if current:
            parent = deepcopy(base); parent["children"] = current
            groups.append(([parent], "list_items"))
        return groups

    raw: list[tuple[list[dict[str, object]], str]] = []
    current: list[dict[str, object]] = []
    for node in items:
        candidate = [*current, node]
        if fits(clone(candidate, len(raw), 1_000_000)):
            current = candidate
            continue
        if current:
            raw.append((current, "list_items")); current = []
        if fits(clone([node], len(raw), 1_000_000)):
            current = [node]
        else:
            raw.extend(split_subtree(node))
    if current:
        raw.append((current, "list_items"))
    return [clone(nodes, number, len(raw), unit) for number, (nodes, unit) in enumerate(raw)]


def build_evidence_windows(
    artifact: object | None,
    index: object,
    *,
    token_counter: TokenCounter,
    budget: InputBudget,
) -> list[EvidenceWindow]:
    """Pack ordered evidence into bounded section-aware windows.

    Text, title, caption, table, and list evidence can be sliced without
    changing canonical IDs. Other oversized structured evidence fails
    explicitly.
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
        if item.get("kind") == "list":
            list_content = item.get("content")
            list_items = list_content.get("items") if isinstance(list_content, dict) else None
            if isinstance(list_items, list):
                try:
                    _validate_list_tree(list_items)
                except OversizedEvidenceError as error:
                    _, estimate, _ = fits([item], len(windows))
                    error.metadata = {
                        **error.metadata,
                        "doc_id": doc_id,
                        "evidence_id": item.get("evidence_id"),
                        "kind": "list",
                        "estimated_tokens": estimate,
                        "limit_tokens": limit,
                    }
                    raise
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
        _, original_estimate, _ = fits([item], ordinal)
        safe_event(
            "oversized_evidence_detected",
            doc_id=doc_id,
            evidence_id=item.get("evidence_id"),
            kind=kind,
            estimated_tokens=original_estimate,
            input_limit_tokens=limit,
        )
        if kind == "table":
            slices = _slice_table_item(
                item, fits=lambda sliced: fits([sliced], len(windows))[0]
            )
        elif kind in {"text", "title", "caption"} and isinstance(item.get("text"), str):
            slices = _slice_text_item(
                item, fits=lambda sliced: fits([sliced], len(windows))[0]
            )
        elif kind == "list":
            try:
                slices = _slice_list_item(
                    item, fits=lambda sliced: fits([sliced], len(windows))[0]
                )
            except OversizedEvidenceError as error:
                # Sub-slicers know the semantic reason; enrich that safe
                # metadata with the window context used by failure reports.
                error.metadata = {
                    **error.metadata,
                    "doc_id": doc_id,
                    "evidence_id": item.get("evidence_id"),
                    "kind": "list",
                    "estimated_tokens": original_estimate,
                    "limit_tokens": limit,
                }
                raise
        else:
            raise OversizedEvidenceError(
                f"unsupported oversized structured evidence {item.get('evidence_id')!r} ({kind!r})",
                doc_id=doc_id,
                evidence_id=item.get("evidence_id"),
                kind=str(kind),
                estimated_tokens=original_estimate,
                limit_tokens=limit,
                reason="unsupported_structured_evidence",
            )
        estimates = [fits([sliced], len(windows))[1] for sliced in slices]
        if kind == "list":
            safe_event(
                "oversized_evidence_sliced",
                doc_id=doc_id,
                evidence_id=item.get("evidence_id"),
                kind="list",
                slice_count=len(slices),
                original_estimated_tokens=original_estimate,
                slice_token_estimates=estimates,
                max_slice_tokens=max(estimates, default=0),
            )
            safe_debug(
                "oversized_list_slice_detail",
                doc_id=doc_id,
                evidence_id=item.get("evidence_id"),
                slice_item_counts=[
                    len((sliced.get("content") or {}).get("items", []))
                    if isinstance(sliced.get("content"), dict) else 0
                    for sliced in slices
                ],
            )
        for sliced in slices:
            if current and not fits([*current, sliced], len(windows))[0]:
                flush()
            if not fits([sliced], len(windows))[0]:
                raise OversizedEvidenceError(
                    f"slice for {item.get('evidence_id')!r} does not fit",
                    doc_id=doc_id,
                    evidence_id=item.get("evidence_id"),
                    kind=str(kind),
                    estimated_tokens=fits([sliced], len(windows))[1],
                    limit_tokens=limit,
                    reason="slice_does_not_fit",
                )
            current.append(sliced)
    flush()
    return windows
