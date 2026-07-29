"""Source ingestion artifact writing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from segro_evidence_extraction.source_ingestion.models import SourceIngestionResult


def write_source_artifacts(result: SourceIngestionResult, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source_pack_manifest": output_dir / "source_pack_manifest.json",
        "source_registry_jsonl": output_dir / "source_registry.jsonl",
        "source_registry_csv": output_dir / "source_registry.csv",
        "document_classifications_csv": output_dir / "document_classifications.csv",
        "duplicate_groups": output_dir / "duplicate_groups.json",
        "archive_manifest": output_dir / "archive_manifest.json",
        "ingestion_issues_csv": output_dir / "ingestion_issues.csv",
        "source_summary": output_dir / "source_summary.json",
        "source_summary_md": output_dir / "source_summary.md",
        "stage_timings": output_dir / "stage_timings.json",
    }
    manifest = result.model_dump(mode="json", exclude={"registered_sources", "classifications"})
    paths["source_pack_manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with paths["source_registry_jsonl"].open("w", encoding="utf-8", newline="\n") as handle:
        for source in result.registered_sources:
            handle.write(source.model_dump_json() + "\n")
    _write_registry_csv(result, paths["source_registry_csv"])
    _write_classifications_csv(result, paths["document_classifications_csv"])
    paths["duplicate_groups"].write_text(
        json.dumps([group.model_dump(mode="json") for group in result.duplicate_groups], indent=2),
        encoding="utf-8",
    )
    paths["archive_manifest"].write_text(
        json.dumps(
            [archive.model_dump(mode="json") for archive in result.archive_summaries],
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_issues_csv(result, paths["ingestion_issues_csv"])
    paths["source_summary"].write_text(result.summary.model_dump_json(indent=2), encoding="utf-8")
    paths["source_summary_md"].write_text(_summary_md(result), encoding="utf-8")
    paths["stage_timings"].write_text(result.timings.model_dump_json(indent=2), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _write_registry_csv(result: SourceIngestionResult, path: Path) -> None:
    fields = [
        "source_id",
        "content_hash",
        "logical_path",
        "file_type",
        "extension",
        "size_bytes",
        "page_count",
        "sheet_count",
        "parent_archive_source_id",
        "archive_member_path",
        "classification",
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in result.registered_sources:
            writer.writerow(
                {
                    "source_id": source.source_id,
                    "content_hash": source.file_hash,
                    "logical_path": source.logical_path,
                    "file_type": source.file_type,
                    "extension": source.extension,
                    "size_bytes": source.size_bytes,
                    "page_count": source.page_count,
                    "sheet_count": source.sheet_count,
                    "parent_archive_source_id": source.parent_archive_source_id,
                    "archive_member_path": source.archive_member_path,
                    "classification": source.classification,
                    "warnings": "|".join(source.warnings),
                }
            )


def _write_classifications_csv(result: SourceIngestionResult, path: Path) -> None:
    fields = [
        "classification_id",
        "source_id",
        "primary_label",
        "secondary_labels",
        "confidence",
        "alternatives",
        "rationale",
        "method_type",
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for classification in result.classifications:
            writer.writerow(
                {
                    "classification_id": classification.classification_id,
                    "source_id": classification.subject_id,
                    "primary_label": classification.primary_label,
                    "secondary_labels": "|".join(classification.secondary_labels),
                    "confidence": classification.confidence,
                    "alternatives": "|".join(
                        item.label for item in classification.alternative_labels
                    ),
                    "rationale": classification.rationale,
                    "method_type": classification.method_type,
                    "warnings": "|".join(classification.warnings),
                }
            )


def _write_issues_csv(result: SourceIngestionResult, path: Path) -> None:
    fields = [
        "issue_code",
        "severity",
        "source_path",
        "archive_member_path",
        "message",
        "raw_value",
        "suggested_action",
    ]
    issues = result.warnings + result.errors
    for file in result.discovered_files:
        issues.extend(file.warnings)
    for archive in result.archive_summaries:
        issues.extend(archive.warnings)
        for member in archive.members:
            issues.extend(member.warnings)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue.model_dump(mode="json"))


def _summary_md(result: SourceIngestionResult) -> str:
    lines = [
        "# Source Ingestion Summary",
        "",
        f"- Files discovered: {result.summary.files_discovered}",
        f"- Files registered: {result.summary.files_registered}",
        f"- Files rejected: {result.summary.files_rejected}",
        f"- Files ignored: {result.summary.files_ignored}",
        f"- Total source bytes: {result.summary.total_source_bytes}",
        f"- Archives: {result.summary.archives}",
        f"- Archive members: {result.summary.archive_members}",
        f"- Duplicate groups: {len(result.duplicate_groups)}",
        f"- Unknown classifications: {result.summary.unknown_classifications}",
        "",
        "## Files By Type",
    ]
    lines.extend(
        f"- {name}: {count}" for name, count in result.summary.files_by_detected_type.items()
    )
    lines.append("")
    lines.append("## Classification Counts")
    lines.extend(
        f"- {name}: {count}" for name, count in result.summary.classification_counts.items()
    )
    return "\n".join(lines)
