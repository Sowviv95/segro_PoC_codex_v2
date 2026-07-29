"""Source ingestion result models."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.classification import ClassificationResult
from segro_evidence_extraction.models.source import SourceRegistryEntry

SOURCE_INGESTION_VERSION = "source-ingestion-v1"


class SourceDisposition(StrEnum):
    REGISTERED = "registered"
    REGISTERED_WITH_WARNINGS = "registered_with_warnings"
    REJECTED = "rejected"
    IGNORED = "ignored"


class SourceIssueSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class SourceIssue(StrictBaseModel):
    issue_code: str
    severity: SourceIssueSeverity
    source_path: str | None = None
    archive_member_path: str | None = None
    message: str
    raw_value: str | None = None
    suggested_action: str | None = None


class DiscoveredFile(StrictBaseModel):
    original_path: str
    relative_path: str
    file_name: str
    extension: str
    detected_type: str
    size_bytes: int = Field(ge=0)
    disposition: SourceDisposition
    warnings: list[SourceIssue] = Field(default_factory=list)


class HashResult(StrictBaseModel):
    algorithm: str = "sha256"
    content_hash: str
    bytes_read: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
    failed: bool = False
    error: str | None = None


class ArchiveLimits(StrictBaseModel):
    max_archive_depth: int = Field(default=2, ge=0)
    max_member_count: int = Field(default=500, ge=1)
    max_uncompressed_bytes: int = Field(default=500_000_000, ge=1)
    max_member_size: int = Field(default=100_000_000, ge=1)
    max_compression_ratio: float = Field(default=100.0, gt=0)


class ArchiveMemberRecord(StrictBaseModel):
    parent_source_id: str
    member_path: str
    archive_depth: int
    compressed_size: int = Field(ge=0)
    uncompressed_size: int = Field(ge=0)
    crc: int | None = None
    extracted_path: str | None = None
    status: str
    warnings: list[SourceIssue] = Field(default_factory=list)


class ArchiveSummary(StrictBaseModel):
    archive_source_id: str
    archive_path: str
    member_count: int
    extracted_count: int
    rejected_count: int
    ignored_count: int
    total_uncompressed_bytes: int = Field(ge=0)
    warnings: list[SourceIssue] = Field(default_factory=list)
    members: list[ArchiveMemberRecord] = Field(default_factory=list)


class DuplicateType(StrEnum):
    IDENTICAL_CONTENT = "identical_content"
    DUPLICATE_RELATIVE_PATH = "duplicate_relative_path"
    DUPLICATE_ARCHIVE_MEMBER_PATH = "duplicate_archive_member_path"
    SAME_FILENAME_DIFFERENT_CONTENT = "same_filename_different_content"
    SAME_CONTENT_DIFFERENT_FILENAMES = "same_content_different_filenames"


class DuplicateGroup(StrictBaseModel):
    group_id: str
    duplicate_type: DuplicateType
    source_ids: list[str]
    content_hash: str | None = None
    recommended_canonical_source_id: str
    rationale: str
    warnings: list[str] = Field(default_factory=list)


class SourceIngestionTimings(StrictBaseModel):
    discovery_ms: float = 0
    hashing_ms: float = 0
    archive_ms: float = 0
    metadata_ms: float = 0
    classification_ms: float = 0
    duplicate_detection_ms: float = 0
    artifact_writing_ms: float = 0


class SourceFileTiming(StrictBaseModel):
    relative_path: str
    file_size: int = Field(ge=0)
    detected_type: str
    file_type_detection_ms: float = Field(default=0, ge=0)
    hashing_duration_ms: float = Field(default=0, ge=0)
    metadata_inspection_ms: float = Field(default=0, ge=0)
    classification_ms: float = Field(default=0, ge=0)
    archive_duration_ms: float = Field(default=0, ge=0)
    total_duration_ms: float = Field(default=0, ge=0)
    status: str


class SourceIngestionSummary(StrictBaseModel):
    files_discovered: int
    files_registered: int
    files_rejected: int
    files_ignored: int
    files_by_detected_type: dict[str, int]
    archives: int
    archive_members: int
    total_source_bytes: int
    identical_content_groups: int
    same_name_different_content_findings: int
    classification_counts: dict[str, int]
    unknown_classifications: int
    warnings_by_code: dict[str, int]


class SourceIngestionResult(StrictBaseModel):
    source_pack_id: str
    root_path: str
    ingestion_version: str = SOURCE_INGESTION_VERSION
    created_at: datetime
    discovered_files: list[DiscoveredFile]
    registered_sources: list[SourceRegistryEntry]
    rejected_sources: list[DiscoveredFile] = Field(default_factory=list)
    ignored_files: list[DiscoveredFile] = Field(default_factory=list)
    archive_summaries: list[ArchiveSummary] = Field(default_factory=list)
    duplicate_groups: list[DuplicateGroup] = Field(default_factory=list)
    classifications: list[ClassificationResult] = Field(default_factory=list)
    warnings: list[SourceIssue] = Field(default_factory=list)
    errors: list[SourceIssue] = Field(default_factory=list)
    timings: SourceIngestionTimings
    file_timings: list[SourceFileTiming] = Field(default_factory=list)
    summary: SourceIngestionSummary
    output_paths: dict[str, str] = Field(default_factory=dict)
