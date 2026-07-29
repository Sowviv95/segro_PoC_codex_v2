"""Configurable dictionary column mapping."""

from pathlib import Path
from typing import Any

import yaml

from segro_evidence_extraction.dictionary.models import (
    ColumnMapping,
    IssueSeverity,
    ValidationIssue,
)


class ColumnMappingError(ValueError):
    """Raised when a mapping config cannot map required source columns."""


def normalize_header(header: str) -> str:
    return " ".join(header.strip().lower().replace("_", " ").split())


def load_column_mapping(config_path: Path, source_columns: list[str]) -> ColumnMapping:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Column mapping config is malformed: {config_path}"
        raise ColumnMappingError(msg)
    return build_column_mapping(raw, source_columns)


def build_column_mapping(raw: dict[str, Any], source_columns: list[str]) -> ColumnMapping:
    normalized_source = {normalize_header(column): column for column in source_columns}
    mapping_version = str(raw.get("mapping_version", "unknown"))
    dictionary_id = str(raw.get("dictionary_id", "unknown"))
    header_row = int(raw.get("header_row", 1))
    field_to_source: dict[str, str] = {}
    warnings: list[str] = []
    ambiguous: list[str] = []
    configured_columns = raw.get("columns", {})
    if not isinstance(configured_columns, dict):
        configured_columns = {}
    for target_field, config in configured_columns.items():
        aliases = _aliases(config)
        matches = [
            normalized_source[normalize_header(alias)]
            for alias in aliases
            if normalize_header(alias) in normalized_source
        ]
        unique_matches = list(dict.fromkeys(matches))
        if unique_matches:
            field_to_source[str(target_field)] = unique_matches[0]
            if len(unique_matches) > 1:
                ambiguous.append(str(target_field))
                warnings.append(
                    f"Ambiguous aliases for {target_field}; using {unique_matches[0]}"
                )
    required_source_columns = [str(item) for item in raw.get("required_source_columns", [])]
    missing_required = [
        column
        for column in required_source_columns
        if normalize_header(column) not in normalized_source
    ]
    if missing_required:
        msg = f"Missing required source columns: {', '.join(missing_required)}"
        raise ColumnMappingError(msg)
    used_source_columns = set(field_to_source.values())
    unused_columns = [column for column in source_columns if column not in used_source_columns]
    return ColumnMapping(
        mapping_version=mapping_version,
        dictionary_id=dictionary_id,
        header_row=header_row,
        required_source_columns=required_source_columns,
        field_to_source=field_to_source,
        metadata_columns=[str(item) for item in raw.get("metadata_columns", [])],
        accepted_values_columns=[str(item) for item in raw.get("accepted_values_columns", [])],
        likely_evidence_type_rules=_evidence_rules(raw.get("likely_evidence_type_rules", {})),
        unused_columns=unused_columns,
        ambiguous_mappings=ambiguous,
        warnings=warnings,
    )


def mapping_warnings(mapping: ColumnMapping) -> list[ValidationIssue]:
    return [
        ValidationIssue(
            issue_code="COLUMN_MAPPING_WARNING",
            severity=IssueSeverity.WARNING,
            message=warning,
            suggested_action="Review dictionary mapping config aliases.",
        )
        for warning in mapping.warnings
    ]


def _aliases(config: object) -> list[str]:
    if isinstance(config, dict):
        raw_aliases = config.get("aliases", [])
        if isinstance(raw_aliases, list):
            return [str(alias) for alias in raw_aliases]
    return []


def _evidence_rules(raw: object) -> dict[str, dict[str, list[str]]]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict[str, list[str]]] = {}
    for source_name, labels in raw.items():
        if not isinstance(labels, dict):
            continue
        result[str(source_name)] = {
            str(label): [str(term) for term in terms]
            for label, terms in labels.items()
            if isinstance(terms, list)
        }
    return result
