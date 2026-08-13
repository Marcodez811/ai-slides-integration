"""Runnable unit contracts for document-digest scope reduction."""

from __future__ import annotations

import pytest

from presentation_pipeline.common.references import EvidenceRef
from presentation_pipeline.understanding.models import DocumentDigest, KeyFact, TopicDigest


def _digest() -> DocumentDigest:
    return DocumentDigest(
        doc_id="document-a",
        summary="Document summary",
        topics=[
            TopicDigest(
                topic="mixed topic",
                summary="Mixed summary",
                evidence=[
                    EvidenceRef(doc_id="document-a", evidence_ids=["e-1", "e-2", "e-3"]),
                    EvidenceRef(doc_id="document-a", evidence_ids=["e-4"]),
                ],
            ),
            TopicDigest(
                topic="out of scope topic",
                summary="Out of scope summary",
                evidence=[EvidenceRef(doc_id="document-a", evidence_ids=["e-5"])],
            ),
        ],
        key_facts=[
            KeyFact(
                claim="mixed fact",
                evidence=[
                    EvidenceRef(doc_id="document-a", evidence_ids=["e-3", "e-6"]),
                    EvidenceRef(doc_id="document-a", evidence_ids=["e-7"]),
                ],
            ),
            KeyFact(
                claim="out of scope fact",
                evidence=[EvidenceRef(doc_id="document-a", evidence_ids=["e-8"])],
            ),
        ],
    )


def test_scoped_to_full_coverage_returns_an_equal_independent_copy() -> None:
    digest = _digest()

    scoped = digest.scoped_to({"e-1", "e-2", "e-3", "e-4", "e-5", "e-6", "e-7", "e-8"})

    assert scoped == digest
    assert scoped is not digest
    assert scoped.topics[0] is not digest.topics[0]
    assert scoped.topics[0].evidence[0] is not digest.topics[0].evidence[0]
    assert scoped.key_facts[0] is not digest.key_facts[0]


def test_scoped_to_empty_or_disjoint_scope_preserves_summary_and_removes_grounding() -> None:
    digest = _digest()

    for allowed in (set(), {"not-present"}):
        scoped = digest.scoped_to(allowed)
        assert scoped.doc_id == "document-a"
        assert scoped.summary == "Document summary"
        assert scoped.topics == []
        assert scoped.key_facts == []


def test_scoped_to_narrows_partial_references_in_original_order() -> None:
    scoped = _digest().scoped_to(["e-3", "e-1", "e-6"])

    assert [topic.topic for topic in scoped.topics] == ["mixed topic"]
    assert [reference.evidence_ids for reference in scoped.topics[0].evidence] == [["e-1", "e-3"]]
    assert [fact.claim for fact in scoped.key_facts] == ["mixed fact"]
    assert [reference.evidence_ids for reference in scoped.key_facts[0].evidence] == [["e-3", "e-6"]]


def test_scoped_to_drops_empty_references_and_empty_topics_or_facts() -> None:
    scoped = _digest().scoped_to(["e-2", "e-4", "e-7"])

    assert [topic.topic for topic in scoped.topics] == ["mixed topic"]
    assert [reference.evidence_ids for reference in scoped.topics[0].evidence] == [["e-2"], ["e-4"]]
    assert [fact.claim for fact in scoped.key_facts] == ["mixed fact"]
    assert [reference.evidence_ids for reference in scoped.key_facts[0].evidence] == [["e-7"]]


def test_scoped_to_does_not_mutate_original_nested_references() -> None:
    digest = _digest()
    original = digest.model_dump(mode="json")

    scoped = digest.scoped_to(["e-1", "e-3"])
    scoped.topics[0].evidence[0].evidence_ids.append("new-id")

    assert digest.model_dump(mode="json") == original
    assert digest.topics[0].evidence[0].evidence_ids == ["e-1", "e-2", "e-3"]


def test_scoped_to_rejects_a_bare_string_scope() -> None:
    with pytest.raises(TypeError, match="not a string"):
        _digest().scoped_to("e-1")
