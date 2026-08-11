"""Explicit finite configuration for corpus-scale presentation planning."""

from __future__ import annotations

from dataclasses import dataclass, field

from presentation_pipeline.budgeting import (
    PlanningBudgets,
    TokenCounter,
    Utf8ByteTokenEstimator,
)
from presentation_pipeline.indexing.compact import compact_evidence_item
from presentation_pipeline.retrieval.protocol import EvidenceRetriever
from presentation_pipeline.understanding.prompts import build_document_digest_input
from presentation_pipeline.understanding.windows import build_evidence_windows


@dataclass(frozen=True, slots=True)
class PlanningScaleConfig:
    """Budget and dependency injection boundary for all planning provider calls."""

    token_counter: TokenCounter = field(default_factory=Utf8ByteTokenEstimator)
    budgets: PlanningBudgets = field(default_factory=PlanningBudgets)
    retriever: EvidenceRetriever | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.budgets, PlanningBudgets):
            raise TypeError("budgets must be a PlanningBudgets")
        if not isinstance(self.token_counter, TokenCounter):
            raise TypeError("token_counter must implement TokenCounter")
        if self.retriever is not None and not hasattr(self.retriever, "retrieve"):
            raise TypeError("retriever must implement retrieve()")


def document_window_diagnostics(
    artifact: object, index: object, *, token_counter: TokenCounter, budgets: PlanningBudgets
) -> dict[str, int]:
    """Deterministic, content-free corpus-scale diagnostics for one index."""
    windows = build_evidence_windows(
        artifact, index, token_counter=token_counter, budget=budgets.understanding.window
    )
    estimates = [int(getattr(window, "estimated_tokens")) for window in windows]
    atoms = [
        token_counter.count_payload({"evidence": [compact_evidence_item(item)]})
        for item in getattr(index, "evidence", [])
    ]
    legacy = token_counter.count_payload(build_document_digest_input(artifact, index))
    target = budgets.understanding.window.target_input_tokens_or_usable
    return {
        "window_count": len(windows),
        "max_window_token_estimate": max(estimates, default=0),
        "total_window_token_estimate": sum(estimates),
        "largest_atomic_evidence_token_estimate": max(atoms, default=0),
        "oversized_atomic_evidence_count": sum(value > target for value in atoms),
        "legacy_full_document_token_estimate": legacy,
    }
