"""Independent validation result contracts."""

from enum import StrEnum

from pydantic import Field

from segro_evidence_extraction.models.base import StrictBaseModel


class ValidationStatus(StrEnum):
    ACCEPTED_EXACT = "accepted_exact"
    ACCEPTED_NORMALIZED = "accepted_normalized"
    REJECTED_UNSUPPORTED = "rejected_unsupported"
    REJECTED_WRONG_COMPONENT = "rejected_wrong_component"
    REJECTED_WRONG_FIELD = "rejected_wrong_field"
    REJECTED_SCHEMA = "rejected_schema"
    REJECTED_UNIT = "rejected_unit"
    REVIEW_REQUIRED = "review_required"


class ValidationFinding(StrictBaseModel):
    check: str
    status: ValidationStatus
    message: str
    evidence_refs: list[str] = Field(default_factory=list)


class IndependentValidationResult(StrictBaseModel):
    target_row_id: str
    candidate_id: str | None = None
    status: ValidationStatus
    evidence_grounding_result: ValidationFinding | None = None
    component_identity_result: ValidationFinding | None = None
    field_semantics_result: ValidationFinding | None = None
    data_type_result: ValidationFinding | None = None
    unit_result: ValidationFinding | None = None
    cardinality_result: ValidationFinding | None = None
    reference_list_result: ValidationFinding | None = None
    conflict_result: ValidationFinding | None = None
    duplicate_result: ValidationFinding | None = None
    reviewer_notes: str | None = None
    findings: list[ValidationFinding] = Field(default_factory=list)
