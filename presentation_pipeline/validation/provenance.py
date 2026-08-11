"""Resolve every generated evidence reference back to extracted source facts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from presentation_pipeline.planning.models import EvidenceSelection, PresentationOutline, SlidePurpose
from presentation_pipeline.understanding.models import DocumentDigest, EvidenceRef


class ProvenanceValidationError(ValueError):
    """Generated data references evidence that is absent or not deeply traceable."""


@dataclass(frozen=True)
class _DocumentSources:
    artifact: object
    blocks: frozenset[str]
    source_nodes: frozenset[str]
    assets: frozenset[str]
    sections: frozenset[str]
    evidence: dict[str, object]


class CorpusLookup:
    """Read-only lookup built from the corpus artifacts and indexes.

    It deliberately retains original objects rather than materializing or
    repairing them, making the validation boundary non-mutating.
    """

    def __init__(self, documents: dict[str, _DocumentSources]):
        self._documents = documents

    @classmethod
    def from_artifacts_indexes(
        cls, artifacts: Iterable[object], indexes: Iterable[object]
    ) -> "CorpusLookup":
        index_by_doc: dict[str, object] = {}
        for index in indexes:
            doc_id = getattr(index, "doc_id", None)
            if not isinstance(doc_id, str) or not doc_id:
                raise ProvenanceValidationError("document index has no doc_id")
            if doc_id in index_by_doc:
                raise ProvenanceValidationError(f"duplicate document index for {doc_id!r}")
            index_by_doc[doc_id] = index

        documents: dict[str, _DocumentSources] = {}
        for artifact in artifacts:
            doc_id = getattr(artifact, "doc_id", None)
            extraction = getattr(artifact, "extraction", None)
            if not isinstance(doc_id, str) or not doc_id or extraction is None:
                raise ProvenanceValidationError("artifact must expose doc_id and extraction")
            if doc_id in documents:
                raise ProvenanceValidationError(f"duplicate artifact for {doc_id!r}")
            if getattr(getattr(extraction, "document", None), "doc_id", None) != doc_id:
                raise ProvenanceValidationError(f"artifact {doc_id!r} disagrees with extraction document ID")
            index = index_by_doc.pop(doc_id, None)
            if index is None:
                raise ProvenanceValidationError(f"artifact {doc_id!r} has no document index")
            blocks = frozenset(_ids(getattr(getattr(extraction, "views", None), "blocks", []), "block_id"))
            nodes = frozenset(_ids(getattr(extraction, "nodes", []), "node_id"))
            assets = frozenset(_ids(getattr(extraction, "assets", []), "asset_id"))
            sections = frozenset(
                _ids(getattr(getattr(extraction, "views", None), "sections", []), "section_id")
            )
            evidence: dict[str, object] = {}
            for item in getattr(index, "evidence", []):
                evidence_id = getattr(item, "evidence_id", None)
                if not isinstance(evidence_id, str) or not evidence_id:
                    raise ProvenanceValidationError(f"document {doc_id!r} contains malformed evidence")
                if evidence_id in evidence:
                    raise ProvenanceValidationError(f"document {doc_id!r} contains duplicate evidence {evidence_id!r}")
                _validate_item_provenance(doc_id, item, blocks, nodes, assets, sections)
                evidence[evidence_id] = item
            documents[doc_id] = _DocumentSources(artifact, blocks, nodes, assets, sections, evidence)
        if index_by_doc:
            raise ProvenanceValidationError(
                f"indexes without an artifact: {sorted(index_by_doc)!r}"
            )
        return cls(documents)

    def validate_ref(self, reference: EvidenceRef) -> None:
        document = self._documents.get(reference.doc_id)
        if document is None:
            raise ProvenanceValidationError(f"unknown document ID {reference.doc_id!r}")
        for evidence_id in reference.evidence_ids:
            if evidence_id not in document.evidence:
                raise ProvenanceValidationError(
                    f"unknown evidence ID {evidence_id!r} for document {reference.doc_id!r}"
                )

    def validate_refs(self, references: Iterable[EvidenceRef]) -> None:
        for reference in references:
            self.validate_ref(reference)


def _ids(items: Iterable[object], attribute: str) -> list[str]:
    values: list[str] = []
    for item in items:
        value = getattr(item, attribute, None)
        if not isinstance(value, str) or not value:
            raise ProvenanceValidationError(f"extraction has an invalid {attribute}")
        values.append(value)
    if len(values) != len(set(values)):
        raise ProvenanceValidationError(f"extraction has duplicate {attribute} values")
    return values


def _validate_item_provenance(
    doc_id: str,
    item: object,
    blocks: frozenset[str],
    nodes: frozenset[str],
    assets: frozenset[str],
    sections: frozenset[str],
) -> None:
    if getattr(item, "doc_id", doc_id) != doc_id:
        raise ProvenanceValidationError(f"evidence document ID does not match {doc_id!r}")
    block_ids = getattr(item, "block_ids", None)
    if not isinstance(block_ids, list) or not block_ids or any(not isinstance(value, str) or not value for value in block_ids):
        raise ProvenanceValidationError(f"evidence in {doc_id!r} has malformed block_ids")
    unknown_blocks = sorted(set(block_ids) - blocks)
    if unknown_blocks:
        raise ProvenanceValidationError(
            f"evidence in {doc_id!r} references unknown block IDs {unknown_blocks!r}"
        )
    source_node_ids = getattr(item, "source_node_ids", None)
    if not isinstance(source_node_ids, list) or not source_node_ids or any(
        not isinstance(value, str) or not value for value in source_node_ids
    ):
        raise ProvenanceValidationError(f"evidence in {doc_id!r} has malformed source_node_ids")
    unknown_nodes = sorted(set(source_node_ids) - nodes)
    if unknown_nodes:
        raise ProvenanceValidationError(
            f"evidence in {doc_id!r} references unknown source node IDs {unknown_nodes!r}"
        )
    _validate_optional_ids(doc_id, item, "asset_ids", assets, "asset")
    _validate_optional_ids(doc_id, item, "section_ids", sections, "section")


def _validate_optional_ids(
    doc_id: str,
    item: object,
    field: str,
    valid_ids: frozenset[str],
    label: str,
) -> None:
    values = getattr(item, field, [])
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise ProvenanceValidationError(f"evidence in {doc_id!r} has malformed {field}")
    unknown = sorted(set(values) - valid_ids)
    if unknown:
        raise ProvenanceValidationError(
            f"evidence in {doc_id!r} references unknown {label} IDs {unknown!r}"
        )


def validate_document_digests(digests: Iterable[DocumentDigest], lookup: CorpusLookup) -> None:
    seen: set[str] = set()
    for digest in digests:
        if digest.doc_id in seen:
            raise ProvenanceValidationError(f"duplicate digest for document {digest.doc_id!r}")
        seen.add(digest.doc_id)
        # A digest must name a real document even when it happens to contain no
        # topics/facts; no generated document identity is accepted.
        if digest.doc_id not in lookup._documents:
            raise ProvenanceValidationError(f"unknown document ID {digest.doc_id!r}")
        references = digest.evidence_refs()
        if any(reference.doc_id != digest.doc_id for reference in references):
            raise ProvenanceValidationError(
                f"digest {digest.doc_id!r} references evidence from another document"
            )
        lookup.validate_refs(references)


def validate_evidence_selection(selection: EvidenceSelection, lookup: CorpusLookup) -> None:
    lookup.validate_refs(item.reference for item in selection.selected)


def validate_presentation_outline(outline: PresentationOutline, lookup: CorpusLookup) -> None:
    for slide in outline.all_slides():
        if slide.purpose in {SlidePurpose.CONTENT, SlidePurpose.SUMMARY} and not slide.evidence:
            raise ProvenanceValidationError(
                f"{slide.purpose.value} slide {slide.slide_id!r} must include evidence"
            )
        lookup.validate_refs(slide.evidence)
