"""Deterministic layout selection, pagination, and physical-slide placement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import floor
import unicodedata

from presentation_pipeline.planning.models import SlidePurpose
from presentation_pipeline.synthesis.context import SlideContext
from presentation_pipeline.synthesis.models import BulletListContent, SlideContent, TextContent

from .models import (
    Box,
    ExecutivePolicyTheme,
    LayoutArchetype,
    PhysicalSlideLayout,
    PositionedElement,
    PresentationLayout,
    RenderDiagnostic,
    RenderInputResult,
    ResolvedElement,
    SourceAttribution,
    TableCell,
    TablePayload,
)


class LayoutValidationError(ValueError):
    """Raised when a layout has errors that make it unsafe to render."""


def select_layout(
    context: SlideContext,
    content: SlideContent,
    resolved_elements: Sequence[ResolvedElement | Mapping[str, object]] = (),
) -> LayoutArchetype:
    """Choose from materialized payloads, never stale semantic intent alone."""
    resolved = _coerce_resolved(resolved_elements) or _semantic_as_resolved(content)
    if context.slide.purpose == SlidePurpose.TITLE:
        return LayoutArchetype.TITLE
    if context.slide.purpose == SlidePurpose.SECTION:
        return LayoutArchetype.SECTION_DIVIDER
    kinds = [element.kind for element in resolved]
    if "image" in kinds:
        return LayoutArchetype.VISUAL_TEXT
    if "chart" in kinds:
        return LayoutArchetype.CHART_FOCUS
    if "table" in kinds:
        return LayoutArchetype.TABLE_FOCUS
    if len(resolved) >= 2:
        return LayoutArchetype.TWO_COLUMN
    if "list" in kinds:
        return LayoutArchetype.KEY_POINTS
    return LayoutArchetype.HEADLINE_BODY


def build_slide_layout(
    context: SlideContext,
    content: SlideContent,
    resolved_elements: Sequence[ResolvedElement | Mapping[str, object]] = (),
    *,
    theme: ExecutivePolicyTheme | None = None,
) -> PhysicalSlideLayout:
    """Build the first physical page for compatibility; use the deck builder for pagination."""
    pages = _build_slide_pages(context, content, resolved_elements, theme=theme)
    return pages[0]


def build_presentation_layout(
    contexts: Sequence[SlideContext],
    contents: Sequence[SlideContent],
    resolved_by_slide: Mapping[str, Sequence[ResolvedElement | Mapping[str, object]] | RenderInputResult] | None = None,
    *,
    theme: ExecutivePolicyTheme | None = None,
) -> PresentationLayout:
    """Build deterministically paginated physical slides in semantic-context order."""
    content_by_id = {content.slide_id: content for content in contents}
    if len(content_by_id) != len(contents):
        raise LayoutValidationError("content slide IDs must be unique")
    expected_ids = [context.slide.slide_id for context in contexts]
    if set(content_by_id) != set(expected_ids) or len(expected_ids) != len(set(expected_ids)):
        raise LayoutValidationError("contexts and contents must have exactly matching unique slide IDs")
    active_theme = theme or ExecutivePolicyTheme()
    resolved_by_slide = resolved_by_slide or {}
    slides: list[PhysicalSlideLayout] = []
    for context in contexts:
        value = resolved_by_slide.get(context.slide.slide_id, ())
        values = value.elements if isinstance(value, RenderInputResult) else value
        slides.extend(_build_slide_pages(context, content_by_id[context.slide.slide_id], values, theme=active_theme))
    return PresentationLayout(slides=slides, theme=active_theme)


def text_width_units(text: str) -> float:
    """Deterministic CJK-aware text width estimate for pagination decisions."""
    width = 0.0
    for character in text:
        if character.isspace():
            width += 0.30
        elif unicodedata.east_asian_width(character) in {"W", "F"}:
            width += 1.0
        elif unicodedata.east_asian_width(character) == "A":
            width += 0.92
        elif character.isalnum():
            width += 0.52
        elif character in ".,;:!?()[]{}":
            width += 0.42
        else:
            width += 0.62
    return width


def text_capacity(box: Box, font_size_pt: float, *, bullet: bool = False) -> int:
    """Conservative character capacity used for deterministic pagination diagnostics."""
    if font_size_pt <= 0:
        raise ValueError("font size must be positive")
    usable_width = max(box.width * 72 - (font_size_pt * 1.8 if bullet else 0), 1)
    characters_per_line = max(floor(usable_width / (font_size_pt * 0.52)), 1)
    lines = max(floor((box.height * 72) / (font_size_pt * 1.35)), 1)
    return characters_per_line * lines


def validate_layout(layout: PhysicalSlideLayout) -> list[RenderDiagnostic]:
    """Check physical bounds, collisions, font minima and conservative capacity."""
    diagnostics: list[RenderDiagnostic] = []
    theme = layout.theme
    role_minimums = {"title": 28, "section_title": 28, "subtitle": 16, "body": 18, "bullets": 18, "caption": 12, "footer": 10, "table": 12, "chart": 10}
    for element in layout.elements:
        box = element.box
        if box.right > theme.slide_width_inches + 0.001 or box.bottom > theme.slide_height_inches + 0.001:
            diagnostics.append(_diagnostic("error", "bounds", "element extends outside the 16:9 slide", layout, element))
        if element.font_size_pt is not None:
            minimum = role_minimums.get(element.role, theme.minimum_body_font_pt)
            if element.font_size_pt < minimum or element.font_size_pt > 72:
                diagnostics.append(_diagnostic("error", "font_limit", f"font size {element.font_size_pt:g}pt is outside permitted limits", layout, element))
            text = element.text or (element.payload.text if element.payload else None)
            items = element.items or (element.payload.items if element.payload else None)
            capacity_units = text_capacity(box, element.font_size_pt) * 0.52
            if text and text_width_units(text) > capacity_units:
                diagnostics.append(_diagnostic("error", "text_capacity", "text exceeds its allocated box after pagination", layout, element))
            if items and sum(text_width_units(item) + 0.6 for item in items) > text_capacity(box, element.font_size_pt, bullet=True) * 0.52:
                diagnostics.append(_diagnostic("error", "text_capacity", "bullets exceed their allocated box after pagination", layout, element))
    for index, first in enumerate(layout.elements):
        for second in layout.elements[index + 1:]:
            if first.box.intersects(second.box):
                diagnostics.append(_diagnostic("error", "overlap", "elements overlap", layout, first))
    return diagnostics


def _build_slide_pages(context: SlideContext, content: SlideContent, values: Sequence[ResolvedElement | Mapping[str, object]], *, theme: ExecutivePolicyTheme | None) -> list[PhysicalSlideLayout]:
    if content.slide_id != context.slide.slide_id:
        raise LayoutValidationError("content slide_id must match the slide context")
    active_theme = theme or ExecutivePolicyTheme()
    resolved = _fill_missing_semantic(content, _coerce_resolved(values))
    if len({item.element_index for item in resolved}) != len(resolved):
        raise LayoutValidationError("resolved element indexes must be unique per slide")
    page_payloads = _paginate_payloads(resolved)
    pages: list[PhysicalSlideLayout] = []
    page_count = len(page_payloads)
    for page_number, payloads in enumerate(page_payloads, start=1):
        archetype = select_layout(context, content, payloads)
        physical_id = content.slide_id if page_count == 1 else f"{content.slide_id}--p{page_number:02d}"
        attributions = _sources(payloads)
        positions = _layout_positions(context, payloads, archetype, active_theme, attributions)
        if page_number > 1:
            positions = [
                item.model_copy(update={"text": f"{item.text} (continued)"})
                if item.role in {"title", "section_title"} and item.text else item
                for item in positions
            ]
        layout = PhysicalSlideLayout(slide_id=physical_id, physical_slide_id=physical_id, semantic_slide_id=content.slide_id, continuation_index=page_number - 1, is_continuation=page_number > 1, page_number=page_number, page_count=page_count, archetype=archetype, elements=positions, source_attributions=attributions, theme=active_theme)
        errors = [item for item in validate_layout(layout) if item.severity == "error"]
        if errors:
            raise LayoutValidationError("; ".join(item.message for item in errors))
        pages.append(layout)
    return pages


def _paginate_payloads(resolved: list[ResolvedElement]) -> list[list[ResolvedElement]]:
    """Split overflowing prose/lists/tables and all visual payloads deterministically."""
    units: list[list[ResolvedElement]] = []
    text_page: list[ResolvedElement] = []
    for element in resolved:
        chunks = _element_chunks(element)
        visual = element.kind in {"image", "table", "chart"}
        if visual:
            if text_page:
                units.append(text_page)
                text_page = []
            units.extend([[chunk] for chunk in chunks])
        else:
            for chunk in chunks:
                text_page.append(chunk)
                if len(text_page) == 3:
                    units.append(text_page)
                    text_page = []
    if text_page:
        units.append(text_page)
    # A visual-text page intentionally combines one visual with short prose.
    # Keep charts/tables informative without introducing a layout solver.
    if units and len(units[0]) == 1 and units[0][0].kind in {"image", "table", "chart"}:
        text_page_index = next((index for index, page in enumerate(units[1:], start=1) if all(item.kind not in {"image", "table", "chart"} for item in page)), None)
        if text_page_index is not None:
            units[0].extend(units.pop(text_page_index))
    return units or [[]]


def _element_chunks(element: ResolvedElement) -> list[ResolvedElement]:
    if element.kind in {"text", "equation"} and element.text is not None:
        pieces = _split_text(element.text, 240)
        return [element.model_copy(update={"text": piece}) for piece in pieces]
    if element.kind == "list" and element.items is not None:
        return [element.model_copy(update={"items": element.items[index:index + 5]}) for index in range(0, len(element.items), 5)]
    if element.kind == "table" and element.table_payload is not None:
        return _table_chunks(element)
    return [element]


def _table_chunks(element: ResolvedElement) -> list[ResolvedElement]:
    table = element.table_payload
    assert table is not None
    header_count = max(table.header_rows, default=-1) + 1 if table.header_rows else 0
    header = table.rows[:header_count]
    data = table.rows[header_count:] or []
    row_groups = [data[index:index + 8] for index in range(0, len(data), 8)] or [[]]
    if table.column_count <= 6:
        col_groups = [list(range(table.column_count))]
    else:
        col_groups = [[0, *range(index, min(index + 5, table.column_count))] for index in range(1, table.column_count, 5)]
    chunks: list[ResolvedElement] = []
    for columns in col_groups:
        for group in row_groups:
            rows = [[row[column] for column in columns] for row in [*header, *group]]
            payload = TablePayload(rows=rows, header_rows=list(range(len(header))))
            chunks.append(element.model_copy(update={"table": payload, "rows": payload.plain_rows(), "header_rows": payload.header_rows}))
    return chunks


def _split_text(value: str, limit: int) -> list[str]:
    if text_width_units(value) <= limit * 0.52:
        return [value]
    sentences = _sentence_units(value)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if text_width_units(sentence) > limit * 0.52:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_split_overlong(sentence, limit))
            continue
        candidate = f"{current} {sentence}".strip()
        if current and text_width_units(candidate) > limit * 0.52:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _sentence_units(value: str) -> list[str]:
    """Prefer Chinese sentence punctuation while retaining punctuation in output."""
    units: list[str] = []
    current = ""
    terminal = set("。！？；：.!?;")
    for character in value:
        current += character
        if character in terminal:
            units.append(current.strip())
            current = ""
    if current.strip():
        units.append(current.strip())
    return units or [value]


def _split_overlong(value: str, limit: int) -> list[str]:
    """Use CJK punctuation then safe character boundaries, avoiding bad starts."""
    result: list[str] = []
    current = ""
    preferred = set("。！？；：")
    secondary = set("，、")
    prohibited_start = set("，。！？；：、）》」』】％")
    prohibited_end = set("（《「『【")
    target = limit * 0.52
    for character in value:
        candidate = current + character
        if current and text_width_units(candidate) > target:
            split_at = max(
                (index + 1 for index, item in enumerate(current) if item in preferred),
                default=0,
            )
            if not split_at:
                split_at = max(
                    (index + 1 for index, item in enumerate(current) if item in secondary),
                    default=0,
                )
            split_at = split_at or len(current)
            while split_at > 1 and current[split_at - 1] in prohibited_end:
                split_at -= 1
            part, current = current[:split_at].strip(), current[split_at:]
            while current and current[0] in prohibited_start:
                part += current[0]
                current = current[1:]
            if part:
                result.append(part)
            current += character
        else:
            current = candidate
    if current.strip():
        result.append(current.strip())
    return result


def _layout_positions(context: SlideContext, resolved: list[ResolvedElement], archetype: LayoutArchetype, theme: ExecutivePolicyTheme, attributions: list[SourceAttribution]) -> list[PositionedElement]:
    title = context.slide.title
    if archetype == LayoutArchetype.TITLE:
        positions = [_text_position(None, "title", "title", title, Box(x=0.85, y=1.2, width=11.55, height=0.95), 38), _text_position(None, "subtitle", "subtitle", context.presentation_objective, Box(x=0.88, y=2.35, width=8.5, height=0.7), 18), _text_position(None, "text", "body", context.slide.message, Box(x=0.88, y=3.25, width=8.2, height=1.25), 18)]
    elif archetype == LayoutArchetype.SECTION_DIVIDER:
        positions = [_text_position(None, "title", "section_title", title, Box(x=0.9, y=2.15, width=10.7, height=0.85), 34), _text_position(None, "subtitle", "subtitle", context.section_purpose, Box(x=0.92, y=3.15, width=9.0, height=0.65), 18)]
    else:
        positions = [_text_position(None, "title", "title", title, Box(x=0.55, y=0.42, width=12.15, height=0.62), 28)]
        images = [item for item in resolved if item.kind == "image"]
        focus = next((item for item in resolved if item.kind in {"table", "chart"}), None)
        remaining = [item for item in resolved if item is not focus and item not in images]
        if focus is not None:
            if focus.title:
                positions.append(_text_position(None, "caption", "caption", focus.title, Box(x=0.65, y=1.18, width=12.0, height=0.22), 12))
                focus_box = Box(x=0.65, y=1.48, width=12.0, height=4.55)
            else:
                focus_box = Box(x=0.65, y=1.25, width=12.0, height=4.8)
            positions.append(_payload_position(focus, focus.kind, focus_box, 12 if focus.kind == "table" else 10))
            if remaining:
                positions.extend(_payload_positions(remaining[:1], Box(x=0.85, y=6.18, width=11.65, height=0.62), theme))
        elif images:
            image = images[0]
            positions.append(_payload_position(image, "image", Box(x=6.55, y=1.25, width=6.2, height=5.1), None))
            if image.caption:
                positions.append(_text_position(None, "caption", "caption", image.caption, Box(x=6.55, y=6.39, width=6.2, height=0.28), 12))
            positions.extend(_payload_positions(remaining, Box(x=0.65, y=1.35, width=5.35, height=5.3), theme))
        elif archetype == LayoutArchetype.TWO_COLUMN and len(remaining) >= 2:
            midpoint = (len(remaining) + 1) // 2
            positions.extend(_payload_positions(remaining[:midpoint], Box(x=0.65, y=1.35, width=5.8, height=5.35), theme))
            positions.extend(_payload_positions(remaining[midpoint:], Box(x=6.85, y=1.35, width=5.8, height=5.35), theme))
        else:
            positions.extend(_payload_positions(remaining, Box(x=0.85, y=1.35, width=11.65, height=5.35), theme))
    if attributions:
        footer = _compact_footer(attributions, resolved)
        positions.append(_text_position(None, "footer", "footer", footer, Box(x=0.55, y=7.03, width=12.2, height=0.22), 10))
    return positions


def _compact_footer(
    attributions: list[SourceAttribution], resolved: Sequence[ResolvedElement]
) -> str:
    """Keep on-slide provenance readable; detailed IDs remain in audit JSON."""
    labels = [item.filename or item.label or item.doc_id for item in attributions]
    cjk = any(_contains_cjk(label) for label in labels) or any(
        _contains_cjk(value)
        for item in resolved
        for value in ([item.text, item.title, item.caption] + (item.items or []))
        if isinstance(value, str)
    )
    visible = labels[:2]
    if cjk:
        suffix = f"（另 {len(labels) - len(visible)} 份）" if len(labels) > len(visible) else ""
        return f"來源：{'；'.join(visible)}{suffix}"
    suffix = f" (+{len(labels) - len(visible)} more)" if len(labels) > len(visible) else ""
    return f"Source: {'; '.join(visible)}{suffix}"


def _contains_cjk(value: str) -> bool:
    return any("\u3400" <= character <= "\u9fff" for character in value)


def _payload_positions(payloads: list[ResolvedElement], area: Box, theme: ExecutivePolicyTheme) -> list[PositionedElement]:
    if not payloads:
        return []
    height = area.height / len(payloads)
    return [_payload_position(payload, payload.kind, Box(x=area.x, y=area.y + ordinal * height, width=area.width, height=height - (0.12 if len(payloads) > 1 else 0)), _body_font(len(payloads), theme)) for ordinal, payload in enumerate(payloads)]


def _body_font(count: int, theme: ExecutivePolicyTheme) -> float:
    return max(18, theme.minimum_body_font_pt, min(theme.maximum_body_font_pt, 20 - (count - 1) * 2))


def _text_position(element_index: int | None, kind: str, role: str, text: str, box: Box, font_size: float) -> PositionedElement:
    return PositionedElement(element_index=element_index, kind=kind, role=role, box=box, font_size_pt=font_size, text=text)


def _payload_position(payload: ResolvedElement, kind: str, box: Box, font_size: float | None) -> PositionedElement:
    return PositionedElement(element_index=payload.element_index, kind=kind, role=kind if kind not in {"text", "list"} else ("bullets" if kind == "list" else "body"), box=box, font_size_pt=font_size, payload=payload)


def _coerce_resolved(values: Sequence[ResolvedElement | Mapping[str, object]]) -> list[ResolvedElement]:
    return [item if isinstance(item, ResolvedElement) else ResolvedElement.model_validate(item) for item in values]


def _semantic_as_resolved(content: SlideContent) -> list[ResolvedElement]:
    output: list[ResolvedElement] = []
    for index, element in enumerate(content.elements):
        if isinstance(element, TextContent):
            output.append(ResolvedElement(element_index=index, kind="text", text=element.text))
        elif isinstance(element, BulletListContent):
            output.append(ResolvedElement(element_index=index, kind="list", items=[item.text for item in element.items]))
        else:
            output.append(ResolvedElement(element_index=index, kind="text", text=f"[{element.kind} source data unavailable]"))
    return output


def _fill_missing_semantic(content: SlideContent, resolved: list[ResolvedElement]) -> list[ResolvedElement]:
    """Keep source/decorative payloads additive without dropping semantic prose."""
    existing = {item.element_index for item in resolved if item.element_index >= 0}
    return [*resolved, *(item for item in _semantic_as_resolved(content) if item.element_index not in existing)]


def _sources(payloads: Sequence[ResolvedElement]) -> list[SourceAttribution]:
    result: list[SourceAttribution] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for payload in payloads:
        for source in payload.attribution:
            identity = (source.doc_id, tuple(source.evidence_ids))
            if identity not in seen:
                seen.add(identity)
                result.append(source)
    return result


def _diagnostic(severity: str, code: str, message: str, layout: PhysicalSlideLayout, element: PositionedElement) -> RenderDiagnostic:
    return RenderDiagnostic(severity=severity, code=code, message=message, slide_id=layout.semantic_slide_id or layout.slide_id, physical_slide_id=layout.slide_id, element_index=element.element_index, source_attribution=element.payload.source_attribution if element.payload else None)
