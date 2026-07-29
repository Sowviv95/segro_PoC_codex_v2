"""Dictionary normalization validation and summary helpers."""

from collections import Counter
from collections.abc import Iterable

from segro_evidence_extraction.dictionary.models import (
    ColumnMapping,
    DictionarySummary,
    IssueSeverity,
    RowDisposition,
    RowNormalizationResult,
    ValidationIssue,
)


def validate_normalized_rows(
    rows: list[RowNormalizationResult],
    mapping: ColumnMapping,
) -> tuple[list[ValidationIssue], DictionarySummary]:
    duplicate_issues = _duplicate_identity_issues(rows)
    physical_duplicate_issues = _duplicate_physical_row_issues(rows)
    issues = duplicate_issues + physical_duplicate_issues
    normalized = [row for row in rows if row.target is not None]
    warning_rows = [
        row
        for row in rows
        if row.target is not None
        and any(issue.severity == IssueSeverity.WARNING for issue in row.issues)
    ]
    duplicate_ids = {
        issue.raw_value
        for issue in duplicate_issues
        if issue.raw_value is not None
    }
    summary = DictionarySummary(
        physical_rows=len(rows),
        blank_rows=sum(1 for row in rows if row.disposition == RowDisposition.IGNORED_BLANK),
        candidate_rows=sum(1 for row in rows if row.disposition != RowDisposition.IGNORED_BLANK),
        normalized_rows=len(normalized),
        rows_with_warnings=len(warning_rows),
        rejected_rows=sum(1 for row in rows if row.disposition == RowDisposition.REJECTED),
        unique_ids=len({row.target.target_row_id for row in normalized if row.target is not None}),
        duplicate_ids=len(duplicate_ids),
        counts_by_sub_domain=_counts(row.target.sub_domain for row in normalized if row.target),
        counts_by_expected_type=_counts(
            str(row.target.expected_data_type) for row in normalized if row.target
        ),
        counts_by_likely_evidence_type=_evidence_counts(normalized),
        unmapped_columns=mapping.unused_columns,
        ambiguous_mappings=mapping.ambiguous_mappings,
    )
    return issues, summary


def _duplicate_identity_issues(rows: list[RowNormalizationResult]) -> list[ValidationIssue]:
    by_id: dict[str, list[int]] = {}
    for row in rows:
        if row.target is not None:
            by_id.setdefault(row.target.target_row_id, []).append(row.physical_row_number)
    issues: list[ValidationIssue] = []
    for target_id, row_numbers in sorted(by_id.items()):
        if len(row_numbers) > 1:
            issues.append(
                ValidationIssue(
                    issue_code="DUPLICATE_TARGET_ID",
                    severity=IssueSeverity.ERROR,
                    source_row=row_numbers[0],
                    message=f"Duplicate target identity across physical rows {row_numbers}.",
                    raw_value=target_id,
                    suggested_action="Review deterministic identity inputs and source rows.",
                )
            )
    return issues


def _duplicate_physical_row_issues(rows: list[RowNormalizationResult]) -> list[ValidationIssue]:
    counts = Counter(row.physical_row_number for row in rows)
    return [
        ValidationIssue(
            issue_code="DUPLICATE_PHYSICAL_ROW",
            severity=IssueSeverity.ERROR,
            source_row=row_number,
            message="Physical row appeared more than once in reader output.",
            raw_value=str(row_number),
            suggested_action="Review reader implementation.",
        )
        for row_number, count in sorted(counts.items())
        if count > 1
    ]


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _evidence_counts(rows: list[RowNormalizationResult]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.target is None:
            continue
        for evidence_type in row.target.likely_evidence_types:
            counts[str(evidence_type)] += 1
    return dict(sorted(counts.items()))
