"""Models for dictionary ingestion, normalization and reporting."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator

from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.target import TargetSpecification

INGESTION_VERSION = "dictionary-ingestion-v1"


class DictionaryFormat(StrEnum):
    XLSX = "xlsx"
    CSV = "csv"


class RowDisposition(StrEnum):
    NORMALIZED = "normalized"
    NORMALIZED_WITH_WARNINGS = "normalized_with_warnings"
    REJECTED = "rejected"
    IGNORED_BLANK = "ignored_blank"


class IssueSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationIssue(StrictBaseModel):
    issue_code: str
    severity: IssueSeverity
    source_row: int | None = None
    source_column: str | None = None
    message: str
    raw_value: str | None = None
    suggested_action: str | None = None


class DictionaryMetadata(StrictBaseModel):
    dictionary_path: str
    dictionary_filename: str
    dictionary_id: str
    source_name: str
    format: DictionaryFormat
    sheet_names: list[str] = Field(default_factory=list)
    selected_sheet: str | None = None
    header_row: int
    columns: list[str]
    mapping_version: str
    ingestion_version: str = INGESTION_VERSION
    ingested_at: datetime


class RawDictionaryRow(StrictBaseModel):
    physical_row_number: int
    values: dict[str, str | int | float | bool | None]

    @property
    def is_blank(self) -> bool:
        return all(_is_blank(value) for value in self.values.values())


class ColumnMapping(StrictBaseModel):
    mapping_version: str
    dictionary_id: str
    header_row: int = Field(ge=1)
    required_source_columns: list[str]
    field_to_source: dict[str, str]
    metadata_columns: list[str] = Field(default_factory=list)
    accepted_values_columns: list[str] = Field(default_factory=list)
    likely_evidence_type_rules: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    unused_columns: list[str] = Field(default_factory=list)
    ambiguous_mappings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RowNormalizationResult(StrictBaseModel):
    physical_row_number: int
    disposition: RowDisposition
    target: TargetSpecification | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)
    raw_values: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class DictionaryTimings(StrictBaseModel):
    file_read_ms: float = 0
    worksheet_read_ms: float = 0
    normalization_ms: float = 0
    validation_ms: float = 0
    artifact_writing_ms: float = 0


class DictionarySummary(StrictBaseModel):
    physical_rows: int
    blank_rows: int
    candidate_rows: int
    normalized_rows: int
    rows_with_warnings: int
    rejected_rows: int
    unique_ids: int
    duplicate_ids: int
    counts_by_sub_domain: dict[str, int]
    counts_by_expected_type: dict[str, int]
    counts_by_likely_evidence_type: dict[str, int]
    unmapped_columns: list[str]
    ambiguous_mappings: list[str]


class DictionaryIngestionResult(StrictBaseModel):
    metadata: DictionaryMetadata
    mapping: ColumnMapping
    summary: DictionarySummary
    timings: DictionaryTimings
    normalized_targets: list[TargetSpecification]
    warnings: list[ValidationIssue] = Field(default_factory=list)
    rejected_rows: list[RowNormalizationResult] = Field(default_factory=list)
    ignored_rows: list[RowNormalizationResult] = Field(default_factory=list)
    duplicate_findings: list[ValidationIssue] = Field(default_factory=list)
    output_paths: dict[str, str] = Field(default_factory=dict)


class RawDictionary(StrictBaseModel):
    metadata: DictionaryMetadata
    rows: list[RawDictionaryRow]
    duplicate_or_blank_columns: list[str] = Field(default_factory=list)
    mixed_type_columns: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("rows")
    @classmethod
    def physical_rows_must_be_unique(cls, rows: list[RawDictionaryRow]) -> list[RawDictionaryRow]:
        row_numbers = [row.physical_row_number for row in rows]
        if len(row_numbers) != len(set(row_numbers)):
            msg = "physical row numbers must be unique"
            raise ValueError(msg)
        return rows


class DictionaryInspectReport(StrictBaseModel):
    sheet_names: list[str]
    selected_source: str
    header_row: int
    columns: list[str]
    physical_rows: int
    blank_rows: int
    non_target_rows: int
    duplicate_or_blank_columns: list[str]
    mixed_type_columns: dict[str, list[str]]


def _is_blank(value: object) -> bool:
    return value is None or str(value).strip() == ""


def source_name(path: Path, sheet_name: str | None) -> str:
    return sheet_name if sheet_name is not None else path.name
