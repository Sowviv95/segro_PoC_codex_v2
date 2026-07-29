"""Source registry contracts for arbitrary asset source packs."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator

from segro_evidence_extraction.models.base import StrictBaseModel


class FileType(StrEnum):
    PDF = "pdf"
    XLSX = "xlsx"
    CSV = "csv"
    TEXT = "text"
    ZIP = "zip"
    IMAGE = "image"
    UNKNOWN = "unknown"


class ExtractionStatus(StrEnum):
    REGISTERED = "registered"
    INDEX_PENDING = "index_pending"
    INDEXED = "indexed"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class SourceRegistryEntry(StrictBaseModel):
    source_id: str
    original_path: str
    logical_path: str
    logical_role: str | None = None
    file_type: FileType = FileType.UNKNOWN
    file_hash: str
    size_bytes: int = Field(ge=0)
    page_count: int | None = Field(default=None, ge=0)
    sheet_count: int | None = Field(default=None, ge=0)
    extraction_status: ExtractionStatus = ExtractionStatus.REGISTERED
    classification: str | None = None
    warnings: list[str] = Field(default_factory=list)
    parent_archive_source_id: str | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("source_id", "original_path", "logical_path", "file_hash")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        if not value:
            msg = "value must not be blank"
            raise ValueError(msg)
        return value


class SourceRegistry(StrictBaseModel):
    registry_id: str
    root_path: str | None = None
    entries: list[SourceRegistryEntry]
    created_at: datetime
    warnings: list[str] = Field(default_factory=list)

    @field_validator("entries")
    @classmethod
    def source_ids_must_be_unique(
        cls, entries: list[SourceRegistryEntry]
    ) -> list[SourceRegistryEntry]:
        ids = [entry.source_id for entry in entries]
        if len(ids) != len(set(ids)):
            msg = "source_id values must be unique"
            raise ValueError(msg)
        return entries


class SourceHasher:
    """Deterministic hash interface for source registration."""

    algorithm = "sha256"

    def hash_path(self, path: Path) -> str:
        raise NotImplementedError
