"""Dictionary ingestion orchestration and artifact writing."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from segro_evidence_extraction.dictionary.column_mapping import (
    load_column_mapping,
    mapping_warnings,
)
from segro_evidence_extraction.dictionary.models import (
    DictionaryIngestionResult,
    DictionaryInspectReport,
    DictionaryMetadata,
    DictionaryTimings,
    RawDictionary,
    RowDisposition,
    RowNormalizationResult,
    ValidationIssue,
)
from segro_evidence_extraction.dictionary.normalizer import normalize_row
from segro_evidence_extraction.dictionary.readers import available_sheets, read_dictionary
from segro_evidence_extraction.dictionary.validation import validate_normalized_rows


def inspect_dictionary(
    dictionary_path: Path,
    *,
    sheet_name: str | None,
    header_row: int = 1,
) -> DictionaryInspectReport:
    raw = read_dictionary(dictionary_path, sheet_name=sheet_name, header_row=header_row)
    return DictionaryInspectReport(
        sheet_names=raw.metadata.sheet_names,
        selected_source=raw.metadata.source_name,
        header_row=raw.metadata.header_row,
        columns=raw.metadata.columns,
        physical_rows=len(raw.rows),
        blank_rows=sum(1 for row in raw.rows if row.is_blank),
        non_target_rows=sum(1 for row in raw.rows if row.is_blank),
        duplicate_or_blank_columns=raw.duplicate_or_blank_columns,
        mixed_type_columns=raw.mixed_type_columns,
    )


def ingest_dictionary(
    dictionary_path: Path,
    *,
    sheet_name: str | None,
    mapping_config: Path,
    output_dir: Path | None = None,
) -> DictionaryIngestionResult:
    file_read_start = perf_counter()
    sheets = available_sheets(dictionary_path)
    file_read_ms = (perf_counter() - file_read_start) * 1000
    mapping_header_row = _mapping_header_row(mapping_config)
    worksheet_read_start = perf_counter()
    raw = read_dictionary(dictionary_path, sheet_name=sheet_name, header_row=mapping_header_row)
    worksheet_read_ms = (perf_counter() - worksheet_read_start) * 1000
    mapping = load_column_mapping(mapping_config, raw.metadata.columns)
    raw.metadata.mapping_version = mapping.mapping_version
    raw.metadata.dictionary_id = mapping.dictionary_id
    raw.metadata.sheet_names = sheets or raw.metadata.sheet_names
    normalized_start = perf_counter()
    row_results = [
        normalize_row(
            row,
            dictionary_filename=raw.metadata.dictionary_filename,
            source_name=raw.metadata.source_name,
            dictionary_id=mapping.dictionary_id,
            mapping=mapping,
        )
        for row in raw.rows
    ]
    normalization_ms = (perf_counter() - normalized_start) * 1000
    validation_start = perf_counter()
    duplicate_findings, summary = validate_normalized_rows(row_results, mapping)
    validation_ms = (perf_counter() - validation_start) * 1000
    normalized_targets = [row.target for row in row_results if row.target is not None]
    warnings = [
        issue
        for row in row_results
        for issue in row.issues
        if issue.severity == "warning"
    ]
    warnings.extend(mapping_warnings(mapping))
    result = DictionaryIngestionResult(
        metadata=_metadata_with_now(raw),
        mapping=mapping,
        summary=summary,
        timings=DictionaryTimings(
            file_read_ms=file_read_ms,
            worksheet_read_ms=worksheet_read_ms,
            normalization_ms=normalization_ms,
            validation_ms=validation_ms,
        ),
        normalized_targets=normalized_targets,
        warnings=warnings,
        rejected_rows=[
            row for row in row_results if row.disposition == RowDisposition.REJECTED
        ],
        ignored_rows=[
            row for row in row_results if row.disposition == RowDisposition.IGNORED_BLANK
        ],
        duplicate_findings=duplicate_findings,
    )
    if output_dir is not None:
        write_start = perf_counter()
        output_paths = write_artifacts(result, row_results, output_dir)
        result.output_paths = output_paths
        result.timings.artifact_writing_ms = (perf_counter() - write_start) * 1000
        _rewrite_manifest_with_output_paths(result, output_dir)
    return result


def write_artifacts(
    result: DictionaryIngestionResult,
    row_results: list[RowNormalizationResult],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "dictionary_manifest": output_dir / "dictionary_manifest.json",
        "normalized_targets_jsonl": output_dir / "normalized_targets.jsonl",
        "normalized_targets_csv": output_dir / "normalized_targets.csv",
        "validation_issues_csv": output_dir / "validation_issues.csv",
        "dictionary_summary": output_dir / "dictionary_summary.json",
        "dictionary_summary_md": output_dir / "dictionary_summary.md",
        "column_mapping_report": output_dir / "column_mapping_report.json",
    }
    paths["dictionary_manifest"].write_text(
        result.metadata.model_dump_json(indent=2), encoding="utf-8"
    )
    with paths["normalized_targets_jsonl"].open("w", encoding="utf-8", newline="\n") as handle:
        for target in result.normalized_targets:
            handle.write(target.model_dump_json() + "\n")
    _write_targets_csv(result, paths["normalized_targets_csv"])
    _write_issues_csv(row_results, result.duplicate_findings, paths["validation_issues_csv"])
    paths["dictionary_summary"].write_text(
        result.summary.model_dump_json(indent=2), encoding="utf-8"
    )
    paths["dictionary_summary_md"].write_text(_summary_markdown(result), encoding="utf-8")
    paths["column_mapping_report"].write_text(
        result.mapping.model_dump_json(indent=2), encoding="utf-8"
    )
    return {name: str(path) for name, path in paths.items()}


def _write_targets_csv(result: DictionaryIngestionResult, path: Path) -> None:
    fields = [
        "target_row_id",
        "requirement_id",
        "sub_domain",
        "expected_field",
        "expected_data_type",
        "cardinality",
        "unit",
        "component_type",
        "component_subtype",
        "source_row",
        "likely_evidence_types",
        "warning_codes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for target in result.normalized_targets:
            provenance = target.source_dictionary_provenance
            writer.writerow(
                {
                    "target_row_id": target.target_row_id,
                    "requirement_id": target.requirement_id,
                    "sub_domain": target.sub_domain,
                    "expected_field": target.expected_field,
                    "expected_data_type": target.expected_data_type,
                    "cardinality": target.cardinality,
                    "unit": target.unit,
                    "component_type": target.component_type,
                    "component_subtype": target.component_subtype,
                    "source_row": provenance.row_number if provenance else None,
                    "likely_evidence_types": "|".join(
                        str(item) for item in target.likely_evidence_types
                    ),
                    "warning_codes": "|".join(
                        provenance.normalization_warnings if provenance else []
                    ),
                }
            )


def _write_issues_csv(
    rows: list[RowNormalizationResult],
    duplicate_findings: list[ValidationIssue],
    path: Path,
) -> None:
    fields = [
        "issue_code",
        "severity",
        "source_row",
        "source_column",
        "message",
        "raw_value",
        "suggested_action",
    ]
    issues = [issue for row in rows for issue in row.issues] + duplicate_findings
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue.model_dump(mode="json"))


def _summary_markdown(result: DictionaryIngestionResult) -> str:
    summary = result.summary
    subdomains = "\n".join(
        f"- {name}: {count}" for name, count in summary.counts_by_sub_domain.items()
    )
    expected_types = "\n".join(
        f"- {name}: {count}" for name, count in summary.counts_by_expected_type.items()
    )
    evidence_types = "\n".join(
        f"- {name}: {count}" for name, count in summary.counts_by_likely_evidence_type.items()
    )
    return "\n".join(
        [
            "# Dictionary Summary",
            "",
            f"- Physical rows: {summary.physical_rows}",
            f"- Blank rows: {summary.blank_rows}",
            f"- Candidate rows: {summary.candidate_rows}",
            f"- Normalized rows: {summary.normalized_rows}",
            f"- Rows with warnings: {summary.rows_with_warnings}",
            f"- Rejected rows: {summary.rejected_rows}",
            f"- Unique IDs: {summary.unique_ids}",
            f"- Duplicate IDs: {summary.duplicate_ids}",
            "",
            "## Counts By Sub-Domain",
            subdomains,
            "",
            "## Expected Types",
            expected_types,
            "",
            "## Likely Evidence Types",
            evidence_types,
        ]
    )


def _rewrite_manifest_with_output_paths(
    result: DictionaryIngestionResult,
    output_dir: Path,
) -> None:
    manifest_path = output_dir / "dictionary_manifest.json"
    manifest = result.metadata.model_dump(mode="json")
    manifest["output_paths"] = result.output_paths
    manifest["timings"] = result.timings.model_dump(mode="json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _metadata_with_now(raw: RawDictionary) -> DictionaryMetadata:
    metadata = raw.metadata.model_copy(deep=True)
    metadata.ingested_at = datetime.now(UTC)
    return metadata


def _mapping_header_row(mapping_config: Path) -> int:
    import yaml

    raw = yaml.safe_load(mapping_config.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return int(raw.get("header_row", 1))
    return 1


def distribution_for_report(result: DictionaryIngestionResult) -> dict[str, Counter[str]]:
    return {
        "sub_domain": Counter(result.summary.counts_by_sub_domain),
        "expected_type": Counter(result.summary.counts_by_expected_type),
        "evidence_type": Counter(result.summary.counts_by_likely_evidence_type),
    }
