"""Candidate extraction result contracts."""

from pydantic import Field, field_validator

from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.common import ProvenanceRef


class CandidateExtractionResult(StrictBaseModel):
    target_row_id: str
    requirement_id: str
    candidate_value: str | int | float | bool | list[str] | None
    normalized_value: str | int | float | bool | list[str] | None = None
    unit: str | None = None
    source_id: str | None = None
    page_or_sheet: str | None = None
    evidence_references: list[ProvenanceRef] = Field(default_factory=list)
    capability_id: str
    model_name: str | None = None
    confidence: float = Field(ge=0, le=1)
    reasoning_summary: str | None = None
    warnings: list[str] = Field(default_factory=list)
    raw_structured_response_ref: str | None = None

    @field_validator("target_row_id", "requirement_id", "capability_id")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        if not value:
            msg = "value must not be blank"
            raise ValueError(msg)
        return value
