"""Focused contracts for compact, provenance-aware retrieval candidates."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from presentation_pipeline.budgeting import RetrievalBudget
from presentation_pipeline.retrieval.models import (
    CandidateDescriptor,
    CandidateEvidence,
    CandidateHit,
)


def test_candidate_hits_are_private_and_merge_immutably() -> None:
    candidate = CandidateEvidence(doc_id="doc-a", evidence_id="ev-a", reason="relevant")
    first = CandidateHit(
        doc_id="doc-a",
        evidence_id="ev-a",
        window_id="window-0",
        reason="matching topic",
        score=0.8,
        slice_ids=("slice-0",),
    )
    second = CandidateHit(
        doc_id="doc-a",
        evidence_id="ev-a",
        window_id="window-1",
        reason="supporting detail",
        slice_ids=("slice-1",),
    )

    merged = candidate.with_hit(first).with_hits((second, first))

    assert candidate.hits() == ()
    assert merged.hits() == (first, second)
    assert "_hits" not in merged.model_dump(mode="json")
    assert merged.model_dump(mode="json") == {
        "doc_id": "doc-a",
        "evidence_id": "ev-a",
        "reason": "relevant",
        "score": None,
    }
    with pytest.raises(ValueError, match="identity"):
        candidate.with_hit(
            CandidateHit(
                doc_id="doc-b",
                evidence_id="ev-a",
                window_id="window-0",
                reason="wrong candidate",
            )
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"doc_id": " ", "evidence_id": "ev", "window_id": "w", "reason": "reason"}, ValueError),
        ({"doc_id": "doc", "evidence_id": "ev", "window_id": "w", "reason": "reason", "score": True}, TypeError),
        ({"doc_id": "doc", "evidence_id": "ev", "window_id": "w", "reason": "reason", "score": math.nan}, ValueError),
        ({"doc_id": "doc", "evidence_id": "ev", "window_id": "w", "reason": "reason", "slice_ids": ["slice"]}, TypeError),
        ({"doc_id": "doc", "evidence_id": "ev", "window_id": "w", "reason": "reason", "slice_ids": ("slice", "slice")}, ValueError),
    ],
)
def test_candidate_hit_validates_immutable_provenance_fields(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        CandidateHit(**kwargs)  # type: ignore[arg-type]


def test_candidate_descriptor_is_strict_and_excludes_source_fields() -> None:
    descriptor = CandidateDescriptor(
        doc_id="doc-a",
        evidence_id="ev-a",
        kind="text",
        reason="relevant",
        score=0.75,
        hit_count=2,
        slice_count=3,
    )

    assert descriptor.model_dump(mode="json") == {
        "doc_id": "doc-a",
        "evidence_id": "ev-a",
        "kind": "text",
        "reason": "relevant",
        "score": 0.75,
        "hit_count": 2,
        "slice_count": 3,
    }
    with pytest.raises(ValidationError):
        CandidateDescriptor(
            doc_id="doc-a",
            evidence_id="ev-a",
            kind="text",
            reason="relevant",
            source_text="must not be accepted",
        )
    with pytest.raises(ValidationError):
        CandidateDescriptor(
            doc_id="doc-a",
            evidence_id="ev-a",
            kind="text",
            reason="relevant",
            hit_count=True,
        )


@pytest.mark.parametrize("value", [True, "0.5", 0, 1, -0.1, math.inf, math.nan])
def test_retrieval_budget_rejects_invalid_reduction_keep_ratio(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        RetrievalBudget(reduction_keep_ratio=value)  # type: ignore[arg-type]


def test_retrieval_budget_has_conservative_default_reduction_keep_ratio() -> None:
    assert RetrievalBudget().reduction_keep_ratio == 0.5
    assert RetrievalBudget(reduction_keep_ratio=0.25).reduction_keep_ratio == 0.25
