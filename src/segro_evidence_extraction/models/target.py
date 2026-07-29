"""Normalized target extraction specification models."""

from pydantic import Field, field_validator

from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.common import Cardinality, EvidenceType, ExpectedDataType


class DictionaryProvenance(StrictBaseModel):
    dictionary_id: str | None = None
    dictionary_path: str | None = None
    sheet_name: str | None = None
    row_number: int | None = Field(default=None, ge=1)
    column_map: dict[str, str] = Field(default_factory=dict)
    mapping_version: str | None = None
    ingestion_version: str | None = None
    raw_requirement_id: str | None = None
    normalization_warnings: list[str] = Field(default_factory=list)
    raw_value_provenance: dict[str, str] = Field(default_factory=dict)


class TargetSpecification(StrictBaseModel):
    target_row_id: str
    requirement_id: str
    sub_domain: str
    requirement_text: str
    expected_field: str
    expected_data_type: ExpectedDataType = ExpectedDataType.UNKNOWN
    unit: str | None = None
    cardinality: Cardinality = Cardinality.UNKNOWN
    component_type: str | None = None
    component_subtype: str | None = None
    source_guidance: str | None = None
    accepted_values: list[str] = Field(default_factory=list)
    likely_evidence_types: list[EvidenceType] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    source_dictionary_provenance: DictionaryProvenance | None = None

    @field_validator(
        "target_row_id",
        "requirement_id",
        "sub_domain",
        "requirement_text",
        "expected_field",
    )
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        if not value:
            msg = "value must not be blank"
            raise ValueError(msg)
        return value

    @field_validator("accepted_values")
    @classmethod
    def accepted_values_must_be_unique(cls, values: list[str]) -> list[str]:
        normalized = [value.lower() for value in values]
        if len(normalized) != len(set(normalized)):
            msg = "accepted_values must be unique case-insensitively"
            raise ValueError(msg)
        return values


class TargetSpecificationSet(StrictBaseModel):
    dictionary_id: str | None = None
    specifications: list[TargetSpecification]

    @field_validator("specifications")
    @classmethod
    def target_row_ids_must_be_unique(
        cls, specifications: list[TargetSpecification]
    ) -> list[TargetSpecification]:
        ids = [spec.target_row_id for spec in specifications]
        if len(ids) != len(set(ids)):
            msg = "target_row_id values must be unique"
            raise ValueError(msg)
        return specifications
