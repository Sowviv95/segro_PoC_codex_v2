"""Classifier result contracts for evidence-first pipeline stages."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator

from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.common import ProvenanceRef


class ClassificationStage(StrEnum):
    SOURCE_DOCUMENT = "source_document"
    PAGE_EVIDENCE_TYPE = "page_evidence_type"
    DOMAIN_COMPONENT = "domain_component"
    EXTRACTION_ROUTE = "extraction_route"


class ClassificationSubjectType(StrEnum):
    SOURCE = "source"
    DOCUMENT = "document"
    SECTION = "section"
    PAGE = "page"
    SHEET = "sheet"
    EVIDENCE_UNIT = "evidence_unit"
    EVIDENCE_BUNDLE = "evidence_bundle"
    TARGET_SPECIFICATION = "target_specification"


class ClassificationMethodType(StrEnum):
    RULE = "rule"
    MODEL = "model"
    HYBRID = "hybrid"
    HUMAN = "human"


class AlternativeLabel(StrictBaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    rationale: str | None = None


class HumanOverride(StrictBaseModel):
    override_label: str
    override_reason: str
    reviewer_id: str | None = None
    timestamp: datetime


class ClassificationResult(StrictBaseModel):
    classification_id: str
    classifier_name: str
    classifier_version: str
    classification_stage: ClassificationStage
    subject_type: ClassificationSubjectType
    subject_id: str
    primary_label: str
    secondary_labels: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    alternative_labels: list[AlternativeLabel] = Field(default_factory=list)
    evidence_references: list[ProvenanceRef] = Field(default_factory=list)
    rationale: str
    method_type: ClassificationMethodType
    model_name: str | None = None
    prompt_version: str | None = None
    warnings: list[str] = Field(default_factory=list)
    human_override: HumanOverride | None = None
    timestamp: datetime
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator(
        "classification_id",
        "classifier_name",
        "classifier_version",
        "subject_id",
        "primary_label",
        "rationale",
    )
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        if not value:
            msg = "value must not be blank"
            raise ValueError(msg)
        return value

    @property
    def effective_label(self) -> str:
        if self.human_override is not None:
            return self.human_override.override_label
        return self.primary_label
