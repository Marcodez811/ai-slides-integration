"""Feature-accounting and diagnostics records for auditable extraction."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from .source import IRModel, SourceLocator


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Diagnostic(IRModel):
    code: str
    severity: DiagnosticSeverity
    message: str
    source_node_id: str | None = None
    source: SourceLocator | None = None
    details: dict[str, object] = Field(default_factory=dict)


class InventoryOccurrence(IRModel):
    feature: str
    source: SourceLocator
    status: str = "discovered"
    details: dict[str, object] = Field(default_factory=dict)


class FeatureInventory(IRModel):
    """Independent occurrence ledger, used to prove extraction conservation."""

    occurrences: list[InventoryOccurrence] = Field(default_factory=list)
    totals: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _derive_or_validate_totals(self) -> "FeatureInventory":
        computed: dict[str, int] = {}
        for occurrence in self.occurrences:
            computed[occurrence.feature] = computed.get(occurrence.feature, 0) + 1
        if not self.totals:
            self.totals = computed
        elif any(value < 0 for value in self.totals.values()):
            raise ValueError("inventory totals must not be negative")
        return self


class CoverageMetric(IRModel):
    source: int = Field(default=0, ge=0)
    extracted: int = Field(default=0, ge=0)
    diagnosed: int = Field(default=0, ge=0)
    ignored: int = Field(default=0, ge=0)

    @property
    def ratio(self) -> float:
        return 1.0 if self.source == 0 else self.extracted / self.source


class CoverageReport(IRModel):
    metrics: dict[str, CoverageMetric] = Field(default_factory=dict)
    unsupported_nodes: int = Field(default=0, ge=0)
    silent_losses: int = Field(default=0, ge=0)
