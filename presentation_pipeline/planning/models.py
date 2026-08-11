"""Strict contracts for presentation planning."""

from __future__ import annotations

from enum import Enum

from pydantic import AliasChoices, ConfigDict, Field, field_validator, model_validator

from presentation_pipeline.common.models import PipelineModel
from presentation_pipeline.understanding.models import EvidenceRef


class PlanningModel(PipelineModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _nonempty(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


class PresentationRequirements(PlanningModel):
    goal: str
    audience: str
    target_slide_count: int = Field(ge=1, le=100)
    tone: str | None = None
    presentation_type: str | None = None
    must_include: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)

    _goal_is_nonempty = field_validator("goal")(_nonempty)
    _audience_is_nonempty = field_validator("audience")(_nonempty)
    @field_validator("tone", "presentation_type")
    @classmethod
    def _optional_text_is_nonempty(cls, value: str | None) -> str | None:
        return _nonempty(value) if value is not None else value

    @field_validator("must_include", "must_avoid")
    @classmethod
    def _requirements_are_nonempty(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("requirement entries must not be blank")
        return values


class SelectedEvidence(PlanningModel):
    doc_id: str
    evidence_id: str
    reason: str

    _doc_id_is_nonempty = field_validator("doc_id")(_nonempty)
    _evidence_id_is_nonempty = field_validator("evidence_id")(_nonempty)
    _reason_is_nonempty = field_validator("reason")(_nonempty)

    @property
    def reference(self) -> EvidenceRef:
        return EvidenceRef(doc_id=self.doc_id, evidence_ids=[self.evidence_id])


class EvidenceSelection(PlanningModel):
    selected: list[SelectedEvidence] = Field(
        min_length=1,
        validation_alias=AliasChoices("selected", "selected_evidence", "items", "selections"),
    )
    strategy: str | None = None

    @field_validator("strategy")
    @classmethod
    def _optional_strategy_is_nonempty(cls, value: str | None) -> str | None:
        return _nonempty(value) if value is not None else value

    @model_validator(mode="after")
    def _selected_evidence_is_unique(self) -> "EvidenceSelection":
        identities = [(item.doc_id, item.evidence_id) for item in self.selected]
        if len(identities) != len(set(identities)):
            raise ValueError("selected evidence references must be unique")
        return self

    @property
    def items(self) -> list[SelectedEvidence]:
        return self.selected


class SlidePurpose(str, Enum):
    TITLE = "title"
    SECTION = "section"
    CONTENT = "content"
    SUMMARY = "summary"


class ContentForm(str, Enum):
    TEXT = "text"
    LIST = "list"
    TABLE = "table"
    CHART = "chart"
    IMAGE = "image"
    MIXED = "mixed"


class SlideOutline(PlanningModel):
    slide_id: str
    title: str
    purpose: SlidePurpose
    message: str
    content_requirements: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    preferred_content_forms: list[ContentForm] = Field(default_factory=list)

    _slide_id_is_nonempty = field_validator("slide_id")(_nonempty)
    _title_is_nonempty = field_validator("title")(_nonempty)
    _message_is_nonempty = field_validator("message")(_nonempty)

    @field_validator("content_requirements")
    @classmethod
    def _content_requirements_are_nonempty(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("content requirements must not be blank")
        return values


class OutlineSection(PlanningModel):
    section_id: str
    title: str
    purpose: str
    slides: list[SlideOutline] = Field(min_length=1)

    _section_id_is_nonempty = field_validator("section_id")(_nonempty)
    _title_is_nonempty = field_validator("title")(_nonempty)
    _purpose_is_nonempty = field_validator("purpose")(_nonempty)


class PresentationOutline(PlanningModel):
    schema_version: str = "1.0.0"
    title: str
    objective: str
    narrative: str
    sections: list[OutlineSection] = Field(min_length=1)

    _title_is_nonempty = field_validator("title")(_nonempty)
    _objective_is_nonempty = field_validator("objective")(_nonempty)
    _narrative_is_nonempty = field_validator("narrative")(_nonempty)
    _schema_version_is_nonempty = field_validator("schema_version")(_nonempty)

    def all_slides(self) -> list[SlideOutline]:
        """Slides flattened in the authored section order."""
        return [slide for section in self.sections for slide in section.slides]

    @model_validator(mode="after")
    def _slide_ids_are_unique(self) -> "PresentationOutline":
        slide_ids = [slide.slide_id for slide in self.all_slides()]
        if len(slide_ids) != len(set(slide_ids)):
            raise ValueError("slide IDs must be unique across the presentation outline")
        return self
