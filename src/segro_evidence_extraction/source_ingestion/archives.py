"""Safe bounded ZIP archive inspection and extraction."""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath
from time import perf_counter

from segro_evidence_extraction.source_ingestion.file_types import detect_file_type
from segro_evidence_extraction.source_ingestion.models import (
    ArchiveLimits,
    ArchiveMemberRecord,
    ArchiveSummary,
    SourceIssue,
    SourceIssueSeverity,
)


def inspect_and_extract_zip(
    archive_path: Path,
    *,
    archive_source_id: str,
    cache_dir: Path,
    limits: ArchiveLimits,
    depth: int = 0,
) -> tuple[ArchiveSummary, float]:
    start = perf_counter()
    cache_root = cache_dir / archive_source_id
    cache_root.mkdir(parents=True, exist_ok=True)
    members: list[ArchiveMemberRecord] = []
    warnings: list[SourceIssue] = []
    extracted_count = 0
    rejected_count = 0
    ignored_count = 0
    total_uncompressed = 0
    seen_member_paths: set[str] = set()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) > limits.max_member_count:
                warnings.append(_issue("ARCHIVE_MEMBER_LIMIT", str(archive_path), None))
            running_uncompressed = 0
            for info in infos[: limits.max_member_count]:
                member_path = info.filename.replace("\\", "/")
                record_warnings = _member_warnings(
                    info,
                    member_path,
                    archive_path,
                    limits,
                    seen_member_paths,
                )
                if running_uncompressed + info.file_size > limits.max_uncompressed_bytes:
                    record_warnings.append(
                        SourceIssue(
                            issue_code="ARCHIVE_UNCOMPRESSED_LIMIT",
                            severity=SourceIssueSeverity.ERROR,
                            source_path=str(archive_path),
                            archive_member_path=member_path,
                            message="Archive exceeds total uncompressed byte limit.",
                            suggested_action="Increase limit only after manual review.",
                        )
                    )
                status = "registered"
                extracted_path: str | None = None
                if record_warnings:
                    status = "rejected"
                    rejected_count += 1
                elif info.is_dir():
                    status = "ignored"
                    ignored_count += 1
                else:
                    total_uncompressed += info.file_size
                    running_uncompressed += info.file_size
                    target_path = cache_root / member_path
                    try:
                        _ensure_within(cache_root, target_path)
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info) as source, target_path.open("wb") as target:
                            for block in iter(lambda: source.read(1024 * 1024), b""):
                                target.write(block)
                        extracted_path = str(target_path)
                        extracted_count += 1
                    except (OSError, zipfile.BadZipFile, RuntimeError, ValueError) as exc:
                        record_warnings.append(
                            SourceIssue(
                                issue_code="ARCHIVE_EXTRACTION_FAILED",
                                severity=SourceIssueSeverity.ERROR,
                                source_path=str(archive_path),
                                archive_member_path=member_path,
                                message="Archive member could not be extracted safely.",
                                raw_value=str(exc),
                                suggested_action="Inspect archive member manually.",
                            )
                        )
                        status = "rejected"
                        rejected_count += 1
                members.append(
                    ArchiveMemberRecord(
                        parent_source_id=archive_source_id,
                        member_path=member_path,
                        archive_depth=depth,
                        compressed_size=info.compress_size,
                        uncompressed_size=info.file_size,
                        crc=info.CRC,
                        extracted_path=extracted_path,
                        status=status,
                        warnings=record_warnings,
                    )
                )
                seen_member_paths.add(member_path)
    except zipfile.BadZipFile as exc:
        warnings.append(
            SourceIssue(
                issue_code="CORRUPT_ARCHIVE",
                severity=SourceIssueSeverity.ERROR,
                source_path=str(archive_path),
                message="ZIP archive is corrupt or unreadable.",
                raw_value=str(exc),
                suggested_action="Replace or repair the archive.",
            )
        )
    summary = ArchiveSummary(
        archive_source_id=archive_source_id,
        archive_path=str(archive_path),
        member_count=len(members),
        extracted_count=extracted_count,
        rejected_count=rejected_count,
        ignored_count=ignored_count,
        total_uncompressed_bytes=total_uncompressed,
        warnings=warnings,
        members=members,
    )
    return summary, (perf_counter() - start) * 1000


def member_detected_type(path: Path) -> str:
    file_type, _ = detect_file_type(path)
    return str(file_type)


def _member_warnings(
    info: zipfile.ZipInfo,
    member_path: str,
    archive_path: Path,
    limits: ArchiveLimits,
    seen_member_paths: set[str],
) -> list[SourceIssue]:
    warnings: list[SourceIssue] = []
    normalized = PurePosixPath(member_path)
    compression_ratio = (
        (info.file_size / info.compress_size) if info.compress_size else float(info.file_size)
    )
    checks = [
        (normalized.is_absolute(), "ARCHIVE_ABSOLUTE_PATH", "Archive member uses absolute path."),
        (".." in normalized.parts, "ARCHIVE_PATH_TRAVERSAL", "Archive member contains '..'."),
        (member_path in seen_member_paths, "ARCHIVE_DUPLICATE_MEMBER", "Duplicate member path."),
        (bool(info.flag_bits & 0x1), "ARCHIVE_ENCRYPTED_MEMBER", "Encrypted member."),
        (
            info.file_size > limits.max_member_size,
            "ARCHIVE_MEMBER_SIZE_LIMIT",
            "Archive member exceeds size limit.",
        ),
        (
            compression_ratio > limits.max_compression_ratio,
            "ARCHIVE_COMPRESSION_RATIO_LIMIT",
            "Archive member exceeds compression ratio limit.",
        ),
    ]
    for failed, code, message in checks:
        if failed:
            warnings.append(
                SourceIssue(
                    issue_code=code,
                    severity=SourceIssueSeverity.ERROR,
                    source_path=str(archive_path),
                    archive_member_path=member_path,
                    message=message,
                    suggested_action="Review archive before ingestion.",
                )
            )
    return warnings


def _ensure_within(root: Path, target: Path) -> None:
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    if root_resolved != target_resolved and root_resolved not in target_resolved.parents:
        msg = f"Archive extraction target escaped cache root: {target}"
        raise ValueError(msg)


def _issue(code: str, source_path: str, member_path: str | None) -> SourceIssue:
    return SourceIssue(
        issue_code=code,
        severity=SourceIssueSeverity.ERROR,
        source_path=source_path,
        archive_member_path=member_path,
        message="Archive safety limit was exceeded.",
        suggested_action="Adjust limits only after manual review.",
    )
