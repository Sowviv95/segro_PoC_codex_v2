"""Recursive source pack discovery."""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from segro_evidence_extraction.source_ingestion.file_types import (
    detect_file_type,
    is_supported_file_type,
)
from segro_evidence_extraction.source_ingestion.models import (
    DiscoveredFile,
    SourceDisposition,
    SourceIssue,
    SourceIssueSeverity,
)

_IGNORED_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}


def discover_sources(
    root: Path,
    progress: Callable[[str, str], None] | None = None,
) -> tuple[list[DiscoveredFile], float]:
    start = perf_counter()
    files: list[DiscoveredFile] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=_sort_key):
        relative = path.relative_to(root).as_posix()
        if progress is not None:
            progress("discover", relative)
        ignored = _is_ignored(path)
        detected_type, _ = detect_file_type(path)
        warnings: list[SourceIssue] = []
        disposition = SourceDisposition.REGISTERED
        if ignored:
            disposition = SourceDisposition.IGNORED
        elif not is_supported_file_type(detected_type):
            disposition = SourceDisposition.REGISTERED_WITH_WARNINGS
            warnings.append(
                SourceIssue(
                    issue_code="UNSUPPORTED_FILE_TYPE",
                    severity=SourceIssueSeverity.WARNING,
                    source_path=relative,
                    message="File type is unsupported by Sprint 3 ingestion.",
                    raw_value=detected_type,
                    suggested_action="Add a file-type handler in a future sprint if needed.",
                )
            )
        try:
            size = path.stat().st_size
        except OSError as exc:
            size = 0
            disposition = SourceDisposition.REJECTED
            warnings.append(
                SourceIssue(
                    issue_code="UNREADABLE_FILE",
                    severity=SourceIssueSeverity.ERROR,
                    source_path=relative,
                    message="File could not be statted.",
                    raw_value=str(exc),
                    suggested_action="Check file permissions or corruption.",
                )
            )
        files.append(
            DiscoveredFile(
                original_path=str(path),
                relative_path=relative,
                file_name=path.name,
                extension=path.suffix.lower(),
                detected_type=detected_type,
                size_bytes=size,
                disposition=disposition,
                warnings=warnings,
            )
        )
    return files, (perf_counter() - start) * 1000


def _is_ignored(path: Path) -> bool:
    name = path.name.casefold()
    return name.startswith("~$") or name in _IGNORED_NAMES


def _sort_key(path: Path) -> str:
    return path.as_posix().casefold()
