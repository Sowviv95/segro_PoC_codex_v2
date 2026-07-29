"""Source ingestion orchestration."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from segro_evidence_extraction.models.classification import ClassificationResult
from segro_evidence_extraction.models.source import ExtractionStatus, FileType, SourceRegistryEntry
from segro_evidence_extraction.source_ingestion.archives import inspect_and_extract_zip
from segro_evidence_extraction.source_ingestion.classification import classify_source
from segro_evidence_extraction.source_ingestion.discovery import discover_sources
from segro_evidence_extraction.source_ingestion.duplicates import detect_duplicates
from segro_evidence_extraction.source_ingestion.file_types import detect_file_type
from segro_evidence_extraction.source_ingestion.hashing import (
    content_identity,
    hash_file,
    source_instance_id,
)
from segro_evidence_extraction.source_ingestion.metadata import inspect_metadata
from segro_evidence_extraction.source_ingestion.models import (
    SOURCE_INGESTION_VERSION,
    ArchiveLimits,
    ArchiveSummary,
    DiscoveredFile,
    DuplicateGroup,
    SourceDisposition,
    SourceFileTiming,
    SourceIngestionResult,
    SourceIngestionSummary,
    SourceIngestionTimings,
    SourceIssue,
    SourceIssueSeverity,
)
from segro_evidence_extraction.source_ingestion.reporting import write_source_artifacts


def inspect_source_pack(
    source_path: Path,
    *,
    progress: Callable[[str, str], None] | None = None,
) -> dict[str, object]:
    discovered, discovery_ms = discover_sources(source_path, progress=progress)
    extensions = Counter(item.extension or "<none>" for item in discovered)
    duplicate_names = [
        name
        for name, count in Counter(item.file_name.casefold() for item in discovered).items()
        if count > 1
    ]
    return {
        "root_path": str(source_path),
        "recursive_file_count": len(discovered),
        "discovery_ms": discovery_ms,
        "extensions": dict(sorted(extensions.items())),
        "zip_archives": [item.relative_path for item in discovered if item.detected_type == "zip"],
        "unsupported_files": [
            item.relative_path
            for item in discovered
            if item.disposition == SourceDisposition.REGISTERED_WITH_WARNINGS
        ],
        "duplicate_names": sorted(duplicate_names),
        "files": [item.model_dump(mode="json") for item in discovered],
    }


def ingest_source_pack(
    source_path: Path,
    *,
    output_dir: Path | None,
    archive_cache_dir: Path | None = None,
    limits: ArchiveLimits | None = None,
    fail_on_unreadable: bool = False,
    progress: Callable[[str, str], None] | None = None,
) -> SourceIngestionResult:
    active_limits = limits or ArchiveLimits()
    discovered, discovery_ms = discover_sources(source_path, progress=progress)
    timings = SourceIngestionTimings(discovery_ms=discovery_ms)
    registered: list[SourceRegistryEntry] = []
    rejected = [item for item in discovered if item.disposition == SourceDisposition.REJECTED]
    ignored = [item for item in discovered if item.disposition == SourceDisposition.IGNORED]
    errors: list[SourceIssue] = []
    archive_summaries: list[ArchiveSummary] = []
    file_timings: list[SourceFileTiming] = []
    base_output_dir = output_dir or Path("output/source_ingestion")
    cache_dir = archive_cache_dir or base_output_dir / "archive_cache"
    for item in discovered:
        if item.disposition in {SourceDisposition.REJECTED, SourceDisposition.IGNORED}:
            file_timings.append(
                SourceFileTiming(
                    relative_path=item.relative_path,
                    file_size=item.size_bytes,
                    detected_type=item.detected_type,
                    status=str(item.disposition),
                )
            )
            continue
        file_start = perf_counter()
        if progress is not None:
            progress("register", item.relative_path)
        source, archive_summary, stage_times = _register_path(
            Path(item.original_path),
            logical_path=item.relative_path,
            parent_archive_source_id=None,
            archive_member_path=None,
            archive_depth=0,
            archive_cache_dir=cache_dir,
            limits=active_limits,
        )
        _add_times(timings, stage_times)
        if source is None:
            rejected.append(item)
            issue = SourceIssue(
                issue_code="SOURCE_REGISTRATION_FAILED",
                severity=SourceIssueSeverity.ERROR,
                source_path=item.relative_path,
                message="Source could not be registered.",
                suggested_action="Inspect file permissions or corruption.",
            )
            errors.append(issue)
            if fail_on_unreadable:
                raise ValueError(issue.message)
            file_timings.append(
                _file_timing_from_item(
                    item,
                    stage_times,
                    "rejected",
                    (perf_counter() - file_start) * 1000,
                )
            )
            continue
        registered.append(source)
        if archive_summary is not None:
            archive_summaries.append(archive_summary)
            member_sources, nested_summaries, member_times = _register_archive_members(
                archive_summary,
                cache_dir,
                active_limits,
                depth=1,
            )
            _add_times(timings, member_times)
            registered.extend(member_sources)
            archive_summaries.extend(nested_summaries)
        file_timings.append(
            _file_timing_from_item(
                item,
                stage_times,
                str(source.extraction_status),
                (perf_counter() - file_start) * 1000,
            )
        )
    classifications = []
    classification_total_start = perf_counter()
    classification_durations: dict[str, float] = {}
    for source in registered:
        if progress is not None:
            progress("classify", source.logical_path)
        start = perf_counter()
        classification = classify_source(source)
        classification_durations[source.source_id] = (perf_counter() - start) * 1000
        classifications.append(classification)
        source.classification = classification.primary_label
    timings.classification_ms = (perf_counter() - classification_total_start) * 1000
    _apply_classification_timings(file_timings, registered, classification_durations)
    start = perf_counter()
    duplicate_groups = detect_duplicates(registered)
    timings.duplicate_detection_ms = (perf_counter() - start) * 1000
    result = SourceIngestionResult(
        source_pack_id=_source_pack_id(source_path),
        root_path=str(source_path),
        created_at=datetime.now(UTC),
        discovered_files=discovered,
        registered_sources=registered,
        rejected_sources=rejected,
        ignored_files=ignored,
        archive_summaries=archive_summaries,
        duplicate_groups=duplicate_groups,
        classifications=classifications,
        warnings=[],
        errors=errors,
        timings=timings,
        file_timings=file_timings,
        summary=_summary(
            discovered,
            registered,
            ignored,
            rejected,
            archive_summaries,
            duplicate_groups,
            classifications,
        ),
    )
    if output_dir is not None:
        if progress is not None:
            progress("write_artifacts", str(output_dir))
        start = perf_counter()
        result.output_paths = write_source_artifacts(result, output_dir)
        result.timings.artifact_writing_ms = (perf_counter() - start) * 1000
        _rewrite_timed_manifest(result, output_dir)
    return result


def _register_path(
    path: Path,
    *,
    logical_path: str,
    parent_archive_source_id: str | None,
    archive_member_path: str | None,
    archive_depth: int,
    archive_cache_dir: Path,
    limits: ArchiveLimits,
) -> tuple[SourceRegistryEntry | None, ArchiveSummary | None, dict[str, float]]:
    timings = {
        "file_type_detection_ms": 0.0,
        "hashing_ms": 0.0,
        "metadata_ms": 0.0,
        "archive_ms": 0.0,
    }
    start = perf_counter()
    file_type, mime_type = detect_file_type(path)
    timings["file_type_detection_ms"] += (perf_counter() - start) * 1000
    hash_result = hash_file(path)
    timings["hashing_ms"] += hash_result.duration_ms
    if hash_result.failed:
        return None, None, timings
    start = perf_counter()
    metadata, metadata_warnings = inspect_metadata(path, file_type)
    timings["metadata_ms"] += (perf_counter() - start) * 1000
    source_id = source_instance_id(
        content_hash=hash_result.content_hash,
        logical_path=logical_path,
        parent_archive_source_id=parent_archive_source_id,
        archive_member_path=archive_member_path,
    )
    status = (
        ExtractionStatus.REGISTERED_WITH_WARNINGS
        if metadata_warnings
        else ExtractionStatus.REGISTERED
    )
    source = SourceRegistryEntry(
        source_id=source_id,
        original_path=str(path),
        logical_path=logical_path,
        file_type=file_type,
        extension=path.suffix.lower(),
        mime_type=mime_type,
        file_hash=hash_result.content_hash,
        size_bytes=path.stat().st_size,
        bytes_read=hash_result.bytes_read,
        hash_duration_ms=hash_result.duration_ms,
        page_count=_int_or_none(metadata.get("page_count")),
        sheet_count=_int_or_none(metadata.get("sheet_count")),
        extraction_status=status,
        warnings=metadata_warnings,
        parent_archive_source_id=parent_archive_source_id,
        archive_member_path=archive_member_path,
        archive_depth=archive_depth,
        content_identity=content_identity(hash_result.content_hash),
        ingestion_version=SOURCE_INGESTION_VERSION,
        metadata=metadata,
    )
    archive_summary = None
    if file_type == FileType.ZIP and archive_depth <= limits.max_archive_depth:
        archive_summary, archive_ms = inspect_and_extract_zip(
            path,
            archive_source_id=source_id,
            cache_dir=archive_cache_dir,
            limits=limits,
            depth=archive_depth,
        )
        timings["archive_ms"] += archive_ms
    return source, archive_summary, timings


def _register_archive_members(
    archive_summary: ArchiveSummary,
    cache_dir: Path,
    limits: ArchiveLimits,
    *,
    depth: int,
) -> tuple[list[SourceRegistryEntry], list[ArchiveSummary], dict[str, float]]:
    sources: list[SourceRegistryEntry] = []
    summaries: list[ArchiveSummary] = []
    timings = {
        "file_type_detection_ms": 0.0,
        "hashing_ms": 0.0,
        "metadata_ms": 0.0,
        "archive_ms": 0.0,
    }
    if depth > limits.max_archive_depth:
        return sources, summaries, timings
    for member in archive_summary.members:
        if member.status != "registered" or member.extracted_path is None:
            continue
        source, nested_archive, stage_times = _register_path(
            Path(member.extracted_path),
            logical_path=f"{archive_summary.archive_path}!/{member.member_path}",
            parent_archive_source_id=archive_summary.archive_source_id,
            archive_member_path=member.member_path,
            archive_depth=depth,
            archive_cache_dir=cache_dir,
            limits=limits,
        )
        _add_dict_times(timings, stage_times)
        if source is not None:
            sources.append(source)
        if nested_archive is not None:
            summaries.append(nested_archive)
            nested_sources, nested_summaries, nested_times = _register_archive_members(
                nested_archive,
                cache_dir,
                limits,
                depth=depth + 1,
            )
            sources.extend(nested_sources)
            summaries.extend(nested_summaries)
            _add_dict_times(timings, nested_times)
    return sources, summaries, timings


def _summary(
    discovered: list[DiscoveredFile],
    registered: list[SourceRegistryEntry],
    ignored: list[DiscoveredFile],
    rejected: list[DiscoveredFile],
    archives: list[ArchiveSummary],
    duplicates: list[DuplicateGroup],
    classifications: list[ClassificationResult],
) -> SourceIngestionSummary:
    classification_counts = Counter(item.primary_label for item in classifications)
    warning_counts = Counter(issue.issue_code for item in discovered for issue in item.warnings)
    for archive in archives:
        warning_counts.update(issue.issue_code for issue in archive.warnings)
        for member in archive.members:
            warning_counts.update(issue.issue_code for issue in member.warnings)
    return SourceIngestionSummary(
        files_discovered=len(discovered),
        files_registered=len(registered),
        files_rejected=len(rejected),
        files_ignored=len(ignored),
        files_by_detected_type=dict(
            sorted(Counter(item.detected_type for item in discovered).items())
        ),
        archives=len(archives),
        archive_members=sum(archive.member_count for archive in archives),
        total_source_bytes=sum(source.size_bytes for source in registered),
        identical_content_groups=sum(
            1 for group in duplicates if group.duplicate_type == "identical_content"
        ),
        same_name_different_content_findings=sum(
            1 for group in duplicates if group.duplicate_type == "same_filename_different_content"
        ),
        classification_counts=dict(sorted(classification_counts.items())),
        unknown_classifications=classification_counts.get("unknown", 0),
        warnings_by_code=dict(sorted(warning_counts.items())),
    )


def _add_times(timings: SourceIngestionTimings, values: dict[str, float]) -> None:
    timings.hashing_ms += values["hashing_ms"]
    timings.metadata_ms += values["metadata_ms"]
    timings.archive_ms += values["archive_ms"]


def _add_dict_times(target: dict[str, float], values: dict[str, float]) -> None:
    for key, value in values.items():
        target[key] += value


def _source_pack_id(source_path: Path) -> str:
    return f"sp_{sha256(str(source_path.resolve()).casefold().encode()).hexdigest()[:16]}"


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _rewrite_timed_manifest(result: SourceIngestionResult, output_dir: Path) -> None:
    (output_dir / "stage_timings.json").write_text(
        result.timings.model_dump_json(indent=2),
        encoding="utf-8",
    )
    manifest = result.model_dump(mode="json", exclude={"registered_sources", "classifications"})
    (output_dir / "source_pack_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _file_timing_from_item(
    item: DiscoveredFile,
    stage_times: dict[str, float],
    status: str,
    total_ms: float,
) -> SourceFileTiming:
    return SourceFileTiming(
        relative_path=item.relative_path,
        file_size=item.size_bytes,
        detected_type=item.detected_type,
        file_type_detection_ms=stage_times["file_type_detection_ms"],
        hashing_duration_ms=stage_times["hashing_ms"],
        metadata_inspection_ms=stage_times["metadata_ms"],
        archive_duration_ms=stage_times["archive_ms"],
        total_duration_ms=total_ms,
        status=status,
    )


def _apply_classification_timings(
    file_timings: list[SourceFileTiming],
    sources: list[SourceRegistryEntry],
    classification_durations: dict[str, float],
) -> None:
    by_logical_path = {timing.relative_path: timing for timing in file_timings}
    for source in sources:
        timing = by_logical_path.get(source.logical_path)
        if timing is not None:
            duration = classification_durations.get(source.source_id, 0.0)
            timing.classification_ms += duration
            timing.total_duration_ms += duration
