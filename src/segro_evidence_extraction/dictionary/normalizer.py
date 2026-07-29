"""Conservative target specification normalization."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256

from pydantic import ValidationError

from segro_evidence_extraction.dictionary.models import (
    INGESTION_VERSION,
    ColumnMapping,
    IssueSeverity,
    RawDictionaryRow,
    RowDisposition,
    RowNormalizationResult,
    ValidationIssue,
)
from segro_evidence_extraction.models.common import Cardinality, EvidenceType, ExpectedDataType
from segro_evidence_extraction.models.target import DictionaryProvenance, TargetSpecification


def normalize_row(
    row: RawDictionaryRow,
    *,
    dictionary_filename: str,
    source_name: str,
    dictionary_id: str,
    mapping: ColumnMapping,
) -> RowNormalizationResult:
    if row.is_blank:
        return RowNormalizationResult(
            physical_row_number=row.physical_row_number,
            disposition=RowDisposition.IGNORED_BLANK,
            raw_values=row.values,
        )
    issues: list[ValidationIssue] = []
    get = _value_getter(row, mapping)
    expected_field = _text(get("expected_field"))
    sub_domain = _text(get("sub_domain"))
    requirement_name = _text(get("requirement_name"))
    field_label = _text(get("field_label"))
    business_definition = _text(get("requirement_text"))
    raw_requirement_id = _text(get("requirement_id"))
    if not expected_field:
        issues.append(
            _issue("BLANK_MANDATORY_COLUMN", row, "expected_field", get("expected_field"))
        )
    if not sub_domain:
        issues.append(_issue("BLANK_MANDATORY_COLUMN", row, "sub_domain", get("sub_domain")))
    if not requirement_name and not business_definition:
        issues.append(
            _issue(
                "MISSING_DESCRIPTIVE_FIELD",
                row,
                "requirement_text",
                get("requirement_text"),
            )
        )
    if any(issue.severity == IssueSeverity.ERROR for issue in issues):
        return RowNormalizationResult(
            physical_row_number=row.physical_row_number,
            disposition=RowDisposition.REJECTED,
            issues=issues,
            raw_values=row.values,
        )
    requirement_id = raw_requirement_id or f"row_{row.physical_row_number}"
    if not raw_requirement_id:
        issues.append(
            ValidationIssue(
                issue_code="MISSING_REQUIREMENT_ID",
                severity=IssueSeverity.WARNING,
                source_row=row.physical_row_number,
                source_column=mapping.field_to_source.get("requirement_id"),
                message=(
                    "Requirement ID is blank; using physical-row fallback in normalized record."
                ),
                suggested_action="Populate Data Requirement id in the source dictionary.",
            )
        )
    cardinality = _normalize_cardinality(get("cardinality"), row, mapping, issues)
    expected_type = _infer_expected_type(
        field_label,
        business_definition,
        get("source_guidance"),
        issues,
        row,
    )
    accepted_values = _accepted_values(get("source_guidance"), row, mapping, issues)
    evidence_types = _likely_evidence_types(row, mapping, issues)
    requirement_text = _compose_requirement_text(requirement_name, field_label, business_definition)
    target_row_id = stable_target_row_id(
        dictionary_id=dictionary_id,
        source_name=source_name,
        physical_row=row.physical_row_number,
        requirement_id=requirement_id,
        expected_field=expected_field,
    )
    try:
        target = TargetSpecification(
            target_row_id=target_row_id,
            requirement_id=requirement_id,
            sub_domain=sub_domain or "Unknown",
            requirement_text=requirement_text,
            expected_field=expected_field or f"field_row_{row.physical_row_number}",
            expected_data_type=expected_type,
            unit=_normalize_unit(get("source_guidance")),
            cardinality=cardinality,
            component_type=_text(get("component_type")) or None,
            component_subtype=_text(get("component_subtype")) or None,
            source_guidance=_text(get("source_guidance")) or None,
            accepted_values=accepted_values,
            likely_evidence_types=evidence_types,
            metadata=_metadata(row, mapping),
            source_dictionary_provenance=DictionaryProvenance(
                dictionary_id=dictionary_id,
                dictionary_path=dictionary_filename,
                sheet_name=source_name,
                row_number=row.physical_row_number,
                column_map=mapping.field_to_source,
                mapping_version=mapping.mapping_version,
                ingestion_version=INGESTION_VERSION,
                raw_requirement_id=raw_requirement_id,
                normalization_warnings=[issue.issue_code for issue in issues],
                raw_value_provenance=_compact_raw_values(row.values),
            ),
        )
    except ValidationError as exc:
        issues.append(
            ValidationIssue(
                issue_code="TARGET_SPEC_VALIDATION_FAILED",
                severity=IssueSeverity.ERROR,
                source_row=row.physical_row_number,
                message=str(exc),
                suggested_action="Review source row and mapping config.",
            )
        )
        return RowNormalizationResult(
            physical_row_number=row.physical_row_number,
            disposition=RowDisposition.REJECTED,
            issues=issues,
            raw_values=row.values,
        )
    disposition = (
        RowDisposition.NORMALIZED_WITH_WARNINGS if issues else RowDisposition.NORMALIZED
    )
    return RowNormalizationResult(
        physical_row_number=row.physical_row_number,
        disposition=disposition,
        target=target,
        issues=issues,
        raw_values=row.values,
    )


def stable_target_row_id(
    *,
    dictionary_id: str,
    source_name: str,
    physical_row: int,
    requirement_id: str,
    expected_field: str,
) -> str:
    """Stable for repeated ingestion of the same physical workbook structure."""

    payload = "|".join(
        [
            dictionary_id,
            source_name,
            str(physical_row),
            requirement_id,
            expected_field,
        ]
    )
    return f"trg_{sha256(payload.encode()).hexdigest()[:16]}"


def _value_getter(row: RawDictionaryRow, mapping: ColumnMapping) -> Callable[[str], object]:
    def get(field_name: str) -> object:
        source_column = mapping.field_to_source.get(field_name)
        if source_column is None:
            return None
        return row.values.get(source_column)

    return get


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _issue(
    code: str,
    row: RawDictionaryRow,
    source_column: str,
    raw_value: object,
    *,
    severity: IssueSeverity = IssueSeverity.ERROR,
) -> ValidationIssue:
    return ValidationIssue(
        issue_code=code,
        severity=severity,
        source_row=row.physical_row_number,
        source_column=source_column,
        message=f"{source_column} is required but blank or unsupported.",
        raw_value=None if raw_value is None else str(raw_value),
        suggested_action="Populate the source cell or update the mapping config.",
    )


def _normalize_cardinality(
    value: object,
    row: RawDictionaryRow,
    mapping: ColumnMapping,
    issues: list[ValidationIssue],
) -> Cardinality:
    text = _text(value).lower()
    if text in {"y", "yes", "true", "1", "multiple"}:
        return Cardinality.MULTIPLE
    if text in {"n", "no", "false", "0", "single"}:
        return Cardinality.SINGLE
    if text == "":
        issues.append(
            ValidationIssue(
                issue_code="MISSING_CARDINALITY",
                severity=IssueSeverity.WARNING,
                source_row=row.physical_row_number,
                source_column=mapping.field_to_source.get("cardinality"),
                message="Cardinality source value is blank; using unknown.",
                suggested_action="Populate Multiple Records Possible (Y/N).",
            )
        )
        return Cardinality.UNKNOWN
    issues.append(
        ValidationIssue(
            issue_code="INVALID_CARDINALITY",
            severity=IssueSeverity.WARNING,
            source_row=row.physical_row_number,
            source_column=mapping.field_to_source.get("cardinality"),
            message="Unsupported cardinality label; retaining unknown.",
            raw_value=text,
            suggested_action="Use Y/N or add an explicit mapping.",
        )
    )
    return Cardinality.UNKNOWN


def _infer_expected_type(
    field_label: str,
    definition: str,
    notes: object,
    issues: list[ValidationIssue],
    row: RawDictionaryRow,
) -> ExpectedDataType:
    haystack = " ".join([field_label, definition, _text(notes)]).lower()
    if "y / n" in haystack or "yes/no" in haystack or "yes / no" in haystack:
        return ExpectedDataType.BOOLEAN
    if "iso 8601" in haystack or "date" in field_label.lower():
        return ExpectedDataType.DATE
    if field_label.lower() == "count" or "number of" in haystack:
        return ExpectedDataType.INTEGER
    decimal_terms = ["area", "height", "width", "length", "capacity", "rating"]
    if any(term in haystack for term in decimal_terms):
        return ExpectedDataType.DECIMAL
    if any(term in haystack for term in ["e.g.", "category", "class"]):
        return ExpectedDataType.ENUM
    if not haystack.strip():
        issues.append(
            ValidationIssue(
                issue_code="UNSUPPORTED_EXPECTED_TYPE",
                severity=IssueSeverity.WARNING,
                source_row=row.physical_row_number,
                message="No usable type signal found; using unknown.",
            )
        )
        return ExpectedDataType.UNKNOWN
    return ExpectedDataType.STRING


def _accepted_values(
    value: object,
    row: RawDictionaryRow,
    mapping: ColumnMapping,
    issues: list[ValidationIssue],
) -> list[str]:
    text = _text(value)
    if not text:
        return []
    markers = ["e.g.", "eg.", "e.g"]
    lower = text.lower()
    if not any(marker in lower for marker in markers) and " / " not in text:
        return []
    candidate_text = text
    for marker in markers:
        marker_index = lower.find(marker)
        if marker_index >= 0:
            candidate_text = text[marker_index + len(marker) :].strip()
            break
    normalized_candidates = candidate_text.replace(" / ", ",").replace(";", ",")
    values = [part.strip() for part in normalized_candidates.split(",")]
    cleaned = [item for item in dict.fromkeys(values) if item]
    if len(cleaned) > 20:
        issues.append(
            ValidationIssue(
                issue_code="MALFORMED_ACCEPTED_VALUES",
                severity=IssueSeverity.WARNING,
                source_row=row.physical_row_number,
                source_column=mapping.field_to_source.get("source_guidance"),
                message="Accepted values parse produced too many values; dropping parsed list.",
                raw_value=text,
                suggested_action="Represent accepted values in a structured dictionary column.",
            )
        )
        return []
    return cleaned


def _likely_evidence_types(
    row: RawDictionaryRow,
    mapping: ColumnMapping,
    issues: list[ValidationIssue],
) -> list[EvidenceType]:
    _ = issues
    labels: list[EvidenceType] = []
    for source_field, rules in mapping.likely_evidence_type_rules.items():
        source_column = mapping.field_to_source.get(source_field)
        text = _text(row.values.get(source_column or "")).lower()
        for label, terms in rules.items():
            if any(term.lower() in text for term in terms):
                try:
                    evidence_type = EvidenceType(label)
                except ValueError:
                    continue
                if evidence_type not in labels:
                    labels.append(evidence_type)
    if not labels:
        labels.append(EvidenceType.TEXT)
    return labels


def _compose_requirement_text(
    requirement_name: str,
    field_label: str,
    business_definition: str,
) -> str:
    parts = [part for part in [requirement_name, field_label, business_definition] if part]
    return " - ".join(parts) if parts else "Unspecified requirement"


def _normalize_unit(value: object) -> str | None:
    text = _text(value).lower()
    unit_markers = {
        "sqm": "sqm",
        "m2": "m2",
        "kw": "kW",
        "kva": "kVA",
        "kg": "kg",
        "mm": "mm",
        "m ": "m",
    }
    for marker, unit in unit_markers.items():
        if marker in text:
            return unit
    return None


def _metadata(
    row: RawDictionaryRow,
    mapping: ColumnMapping,
) -> dict[str, str | int | float | bool | None]:
    metadata: dict[str, str | int | float | bool | None] = {}
    for field_name in mapping.metadata_columns:
        source_column = mapping.field_to_source.get(field_name)
        if source_column is not None and row.values.get(source_column) is not None:
            metadata[field_name] = row.values[source_column]
    return metadata


def _compact_raw_values(
    values: dict[str, str | int | float | bool | None],
) -> dict[str, str]:
    compact: dict[str, str] = {}
    for key, value in values.items():
        if value is not None and str(value).strip():
            compact[key] = str(value)[:200]
    return compact
