"""Pure structural table profiling for native slide tables and chart candidates."""

from __future__ import annotations

from datetime import date
import re
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from ..ids import make_view_id
from ..ir.source import SourceNode, SourceNodeKind


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProfiledCell(_FrozenModel):
    source_node_id: str
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    original: str
    numeric_value: float | None = None
    date_value: date | None = None


class ChartSeries(_FrozenModel):
    name: str
    column: int = Field(ge=0)
    values: tuple[float, ...]


class ChartCandidate(_FrozenModel):
    candidate_id: str
    table_node_id: str
    category_column: int = Field(ge=0)
    categories: tuple[str, ...]
    series: tuple[ChartSeries, ...]
    source_node_ids: tuple[str, ...]


class TableProfile(_FrozenModel):
    table_node_id: str
    cells: tuple[ProfiledCell, ...]
    header_rows: tuple[int, ...]
    column_count: int = Field(ge=0)
    rectangular: bool
    warnings: tuple[str, ...] = ()
    chart_candidate: ChartCandidate | None = None


_NUMBER = re.compile(r"^\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?:\s*%|\s*[A-Za-z$€£¥]+)?\s*$")


def _parse_number(value: str) -> float | None:
    match = _NUMBER.match(value)
    if not match:
        return None
    try:
        parsed = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    return parsed / 100 if value.strip().endswith("%") else parsed


def _parse_date(value: str) -> date | None:
    normalized = value.strip().replace("/", "-")
    try:
        return date.fromisoformat(normalized)
    except ValueError:
        return None


def _descendants(root_id: str, children: dict[str, list[SourceNode]]) -> list[SourceNode]:
    result: list[SourceNode] = []
    pending = list(children.get(root_id, []))
    while pending:
        node = pending.pop(0)
        result.append(node)
        pending[0:0] = children.get(node.node_id, [])
    return sorted(result, key=lambda node: node.sequence)


def _cell_text(cell: SourceNode, children: dict[str, list[SourceNode]]) -> str:
    chunks: list[str] = []
    for node in _descendants(cell.node_id, children):
        if node.kind is SourceNodeKind.TEXT_RUN and not node.payload.get("deleted"):
            chunks.append(str(node.payload.get("text") or ""))
        elif node.kind is SourceNodeKind.LINE_BREAK:
            chunks.append("\n")
        elif node.kind is SourceNodeKind.TAB:
            chunks.append("\t")
    return "".join(chunks)


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def profile_table(table: SourceNode, nodes: Iterable[SourceNode], *, profiler_version: str = "1.0.0") -> TableProfile:
    """Profile one table node and return derived values only.

    Structural row/cell coordinates are authoritative; text is collected from
    cell descendants and is never written back to the Source IR payload.
    """
    if table.kind is not SourceNodeKind.TABLE:
        raise ValueError("profile_table requires a table SourceNode")
    ordered = sorted(nodes, key=lambda node: node.sequence)
    children: dict[str, list[SourceNode]] = {}
    for node in ordered:
        if node.parent_id is not None:
            children.setdefault(node.parent_id, []).append(node)
    rows = [node for node in children.get(table.node_id, []) if node.kind is SourceNodeKind.TABLE_ROW]
    cells: list[ProfiledCell] = []
    header_rows: set[int] = set()
    for row_position, row in enumerate(rows):
        row_index = _as_int(row.payload.get("row_index"), row_position)
        if row.payload.get("is_header"):
            header_rows.add(row_index)
        for cell_position, cell in enumerate(children.get(row.node_id, [])):
            if cell.kind is not SourceNodeKind.TABLE_CELL:
                continue
            original = _cell_text(cell, children)
            cells.append(
                ProfiledCell(
                    source_node_id=cell.node_id,
                    row=row_index,
                    column=_as_int(cell.payload.get("column_index"), cell_position),
                    row_span=_as_int(cell.payload.get("row_span"), 1),
                    column_span=_as_int(cell.payload.get("column_span"), 1),
                    original=original,
                    numeric_value=_parse_number(original),
                    date_value=_parse_date(original),
                )
            )
    column_count = _as_int(table.payload.get("grid_column_count"), 0)
    if not column_count and cells:
        column_count = max(cell.column + cell.column_span for cell in cells)
    warnings: list[str] = []
    row_indices = sorted({cell.row for cell in cells})
    occupied = {(cell.row, column) for cell in cells for column in range(cell.column, cell.column + cell.column_span)}
    rectangular = bool(row_indices and column_count) and all(
        all((row, column) in occupied for column in range(column_count)) for row in row_indices
    )
    if not rectangular:
        warnings.append("table grid is sparse or contains unsupported row/column spans")
    if not header_rows and len(row_indices) >= 2:
        first = min(row_indices)
        first_cells = [cell for cell in cells if cell.row == first]
        following = [cell for cell in cells if cell.row != first]
        if first_cells and any(cell.numeric_value is not None for cell in following) and all(
            cell.numeric_value is None for cell in first_cells
        ):
            header_rows.add(first)
    profile = TableProfile(
        table_node_id=table.node_id,
        cells=tuple(sorted(cells, key=lambda cell: (cell.row, cell.column, cell.source_node_id))),
        header_rows=tuple(sorted(header_rows)),
        column_count=column_count,
        rectangular=rectangular,
        warnings=tuple(warnings),
    )
    return profile.model_copy(update={"chart_candidate": _chart_candidate(profile, profiler_version)})


def _chart_candidate(profile: TableProfile, profiler_version: str) -> ChartCandidate | None:
    if not profile.rectangular or len(profile.header_rows) != 1 or profile.column_count < 2:
        return None
    header_row = profile.header_rows[0]
    rows = sorted({cell.row for cell in profile.cells})
    data_rows = [row for row in rows if row != header_row]
    if not data_rows:
        return None
    index = {(cell.row, cell.column): cell for cell in profile.cells if cell.column_span == 1 and cell.row_span == 1}
    headers = [index.get((header_row, column)) for column in range(profile.column_count)]
    if any(header is None or not header.original.strip() for header in headers):
        return None
    category_column = next(
        (
            column
            for column in range(profile.column_count)
            if all((cell := index.get((row, column))) is not None and cell.numeric_value is None and cell.original.strip() for row in data_rows)
        ),
        None,
    )
    if category_column is None:
        return None
    categories = tuple(index[(row, category_column)].original for row in data_rows)
    if len(set(categories)) != len(categories):
        return None
    series: list[ChartSeries] = []
    for column in range(profile.column_count):
        if column == category_column:
            continue
        values = [index.get((row, column)) for row in data_rows]
        if all(value is not None and value.numeric_value is not None for value in values):
            series.append(ChartSeries(name=headers[column].original, column=column, values=tuple(value.numeric_value for value in values)))
    if not series:
        return None
    source_ids = tuple(cell.source_node_id for cell in profile.cells)
    return ChartCandidate(
        candidate_id=make_view_id("table-profiler", profiler_version, {"category_column": category_column}, source_ids),
        table_node_id=profile.table_node_id,
        category_column=category_column,
        categories=categories,
        series=tuple(series),
        source_node_ids=source_ids,
    )


def profile_tables(nodes: Iterable[SourceNode], *, profiler_version: str = "1.0.0") -> list[TableProfile]:
    """Profile every structural table in source order."""
    ordered = sorted(nodes, key=lambda node: node.sequence)
    return [profile_table(node, ordered, profiler_version=profiler_version) for node in ordered if node.kind is SourceNodeKind.TABLE]
