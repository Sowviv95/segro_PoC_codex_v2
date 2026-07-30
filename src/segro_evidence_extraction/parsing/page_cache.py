"""Canonical parsed-page cache for bounded batch parsing."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field, ValidationError

from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.common import ProvenanceRef
from segro_evidence_extraction.models.source import SourceRegistryEntry
from segro_evidence_extraction.parsing.batch_worker import (
    BatchRequest,
    BatchRunResult,
    BatchWorkerConfig,
    ParserFactory,
    run_bounded_batch_worker,
)
from segro_evidence_extraction.parsing.cache import config_hash
from segro_evidence_extraction.parsing.models import (
    OcrRouting,
    ParsedPage,
    ParserWarning,
    ParsingConfig,
    ParsingStatus,
)
from segro_evidence_extraction.parsing.pdf import PyMuPdfPageParser, bounded_page_numbers
from segro_evidence_extraction.parsing.quality import assess_text_quality
from segro_evidence_extraction.parsing.service import ParsingError

PARSED_PAGE_SCHEMA_VERSION = "canonical-parsed-page-v1"


class PageCacheIdentity(StrictBaseModel):
    schema_version: str
    source_content_hash: str
    page_number: int = Field(ge=1)
    parser_name: str
    parser_version: str
    parser_config_fingerprint: str


class CanonicalParsedPage(StrictBaseModel):
    schema_version: str
    source_id: str
    source_content_hash: str
    source_path: str
    page_number: int = Field(ge=1)
    parser_name: str
    parser_version: str
    parser_config_fingerprint: str
    extracted_text: str
    text_character_count: int = Field(ge=0)
    non_whitespace_character_count: int = Field(ge=0)
    page_parse_status: str
    parser_warnings: list[ParserWarning] = Field(default_factory=list)
    page_parse_duration_ms: float = Field(ge=0)
    parsed_timestamp: str
    artifact_identity: str
    cache_key: str


class PageCacheWarning(StrictBaseModel):
    code: str
    message: str
    page_number: int | None = None
    artifact_path: str | None = None


class PageCacheLookupResult(StrictBaseModel):
    pages: dict[int, ParsedPage] = Field(default_factory=dict)
    missing_pages: list[int] = Field(default_factory=list)
    invalid_warnings: list[PageCacheWarning] = Field(default_factory=list)
    read_ms: float = Field(default=0, ge=0)
    hits: int = Field(default=0, ge=0)
    misses: int = Field(default=0, ge=0)
    invalid_entries: int = Field(default=0, ge=0)


class PageCacheWriteResult(StrictBaseModel):
    writes: int = Field(default=0, ge=0)
    write_ms: float = Field(default=0, ge=0)
    artifact_paths: list[str] = Field(default_factory=list)


class CachedBatchParseResult(StrictBaseModel):
    source_id: str
    source_hash: str
    parser_name: str
    parser_version: str
    parser_config_fingerprint: str
    requested_page_start: int
    requested_page_end: int
    cache_root: str
    cache_hits: int
    cache_misses: int
    invalid_cache_entries: int
    pages_loaded_from_cache: list[int]
    pages_newly_parsed: list[int]
    page_artifacts_written: int
    missing_ranges_sent_to_workers: list[str]
    worker_invocation_count: int
    restart_count: int
    total_wall_time_ms: float = Field(ge=0)
    cache_read_time_ms: float = Field(ge=0)
    parse_time_ms: float = Field(ge=0)
    cache_write_time_ms: float = Field(ge=0)
    active_child_count_after_cleanup: int = Field(ge=0)
    pages: list[ParsedPage]
    cache_warnings: list[PageCacheWarning] = Field(default_factory=list)
    batch_results: list[BatchRunResult] = Field(default_factory=list)


class CanonicalParsedPageCache:
    def __init__(
        self,
        root: Path,
        *,
        schema_version: str = PARSED_PAGE_SCHEMA_VERSION,
    ) -> None:
        self.root = root
        self.schema_version = schema_version
        self.root.mkdir(parents=True, exist_ok=True)

    def identity(
        self,
        *,
        source: SourceRegistryEntry,
        page_number: int,
        parser_name: str,
        parser_version: str,
        parser_config_fingerprint: str,
    ) -> PageCacheIdentity:
        return PageCacheIdentity(
            schema_version=self.schema_version,
            source_content_hash=source.file_hash,
            page_number=page_number,
            parser_name=parser_name,
            parser_version=parser_version,
            parser_config_fingerprint=parser_config_fingerprint,
        )

    def cache_key(self, identity: PageCacheIdentity) -> str:
        payload = identity.model_dump(mode="json")
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def parser_fingerprint(
        self,
        *,
        parser_name: str,
        parser_version: str,
        parser_config_fingerprint: str,
    ) -> str:
        payload = {
            "parser_name": parser_name,
            "parser_version": parser_version,
            "parser_config_fingerprint": parser_config_fingerprint,
            "schema_version": self.schema_version,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def artifact_path(
        self,
        *,
        source: SourceRegistryEntry,
        page_number: int,
        parser_name: str,
        parser_version: str,
        parser_config_fingerprint: str,
    ) -> Path:
        parser_fingerprint = self.parser_fingerprint(
            parser_name=parser_name,
            parser_version=parser_version,
            parser_config_fingerprint=parser_config_fingerprint,
        )
        return (
            self.root
            / source.source_id
            / parser_fingerprint
            / f"page_{page_number:06d}.json"
        )

    def read_page(
        self,
        *,
        source: SourceRegistryEntry,
        source_path: str,
        page_number: int,
        parser_name: str,
        parser_version: str,
        parser_config_fingerprint: str,
    ) -> tuple[ParsedPage | None, PageCacheWarning | None, float]:
        start = time.perf_counter()
        artifact_path = self.artifact_path(
            source=source,
            page_number=page_number,
            parser_name=parser_name,
            parser_version=parser_version,
            parser_config_fingerprint=parser_config_fingerprint,
        )
        if not artifact_path.exists():
            return None, None, (time.perf_counter() - start) * 1000
        try:
            artifact = CanonicalParsedPage.model_validate_json(
                artifact_path.read_text(encoding="utf-8")
            )
            expected_identity = self.identity(
                source=source,
                page_number=page_number,
                parser_name=parser_name,
                parser_version=parser_version,
                parser_config_fingerprint=parser_config_fingerprint,
            )
            expected_key = self.cache_key(expected_identity)
            self._validate_artifact(
                artifact,
                expected_key=expected_key,
                source=source,
                source_path=source_path,
                page_number=page_number,
                parser_name=parser_name,
                parser_version=parser_version,
                parser_config_fingerprint=parser_config_fingerprint,
            )
            return (
                _artifact_to_page(artifact, source),
                None,
                (time.perf_counter() - start) * 1000,
            )
        except (OSError, ValidationError, ValueError, json.JSONDecodeError) as exc:
            warning = PageCacheWarning(
                code="invalid_cache_artifact",
                message=(
                    "Cache artifact was invalid and will be reparsed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                page_number=page_number,
                artifact_path=str(artifact_path),
            )
            return None, warning, (time.perf_counter() - start) * 1000

    def read_range(
        self,
        *,
        source: SourceRegistryEntry,
        source_path: str,
        page_numbers: list[int],
        parser_name: str,
        parser_version: str,
        parser_config_fingerprint: str,
    ) -> PageCacheLookupResult:
        result = PageCacheLookupResult()
        for page_number in page_numbers:
            page, warning, read_ms = self.read_page(
                source=source,
                source_path=source_path,
                page_number=page_number,
                parser_name=parser_name,
                parser_version=parser_version,
                parser_config_fingerprint=parser_config_fingerprint,
            )
            result.read_ms += read_ms
            if page is None:
                result.misses += 1
                result.missing_pages.append(page_number)
                if warning is not None:
                    result.invalid_entries += 1
                    result.invalid_warnings.append(warning)
            else:
                result.hits += 1
                result.pages[page_number] = page
        return result

    def write_page(
        self,
        *,
        source: SourceRegistryEntry,
        source_path: str,
        page: ParsedPage,
        parser_config_fingerprint: str,
    ) -> str | None:
        if page.status not in {ParsingStatus.PARSED, ParsingStatus.PARSED_WITH_WARNINGS}:
            return None
        identity = self.identity(
            source=source,
            page_number=page.page_number,
            parser_name=page.parser_name,
            parser_version=page.parser_version,
            parser_config_fingerprint=parser_config_fingerprint,
        )
        key = self.cache_key(identity)
        artifact_path = self.artifact_path(
            source=source,
            page_number=page.page_number,
            parser_name=page.parser_name,
            parser_version=page.parser_version,
            parser_config_fingerprint=parser_config_fingerprint,
        )
        text = page.text or ""
        artifact = CanonicalParsedPage(
            schema_version=self.schema_version,
            source_id=source.source_id,
            source_content_hash=source.file_hash,
            source_path=source_path,
            page_number=page.page_number,
            parser_name=page.parser_name,
            parser_version=page.parser_version,
            parser_config_fingerprint=parser_config_fingerprint,
            extracted_text=text,
            text_character_count=len(text),
            non_whitespace_character_count=sum(1 for char in text if not char.isspace()),
            page_parse_status=str(page.status),
            parser_warnings=page.warnings,
            page_parse_duration_ms=page.parsing_duration_ms,
            parsed_timestamp=datetime.now(UTC).isoformat(),
            artifact_identity=key,
            cache_key=key,
        )
        _atomic_write_json(artifact_path, artifact.model_dump(mode="json"))
        return str(artifact_path)

    def write_pages(
        self,
        *,
        source: SourceRegistryEntry,
        source_path: str,
        pages: list[ParsedPage],
        parser_config_fingerprint: str,
    ) -> PageCacheWriteResult:
        start = time.perf_counter()
        result = PageCacheWriteResult()
        for page in pages:
            path = self.write_page(
                source=source,
                source_path=source_path,
                page=page,
                parser_config_fingerprint=parser_config_fingerprint,
            )
            if path is not None:
                result.writes += 1
                result.artifact_paths.append(path)
        result.write_ms = (time.perf_counter() - start) * 1000
        return result

    def _validate_artifact(
        self,
        artifact: CanonicalParsedPage,
        *,
        expected_key: str,
        source: SourceRegistryEntry,
        source_path: str,
        page_number: int,
        parser_name: str,
        parser_version: str,
        parser_config_fingerprint: str,
    ) -> None:
        checks = {
            "schema_version": artifact.schema_version == self.schema_version,
            "source_id": artifact.source_id == source.source_id,
            "source_content_hash": artifact.source_content_hash == source.file_hash,
            "source_path": artifact.source_path == source_path,
            "page_number": artifact.page_number == page_number,
            "parser_name": artifact.parser_name == parser_name,
            "parser_version": artifact.parser_version == parser_version,
            "parser_config_fingerprint": artifact.parser_config_fingerprint
            == parser_config_fingerprint,
            "cache_key": artifact.cache_key == expected_key,
            "artifact_identity": artifact.artifact_identity == expected_key,
            "status": artifact.page_parse_status
            in {ParsingStatus.PARSED, ParsingStatus.PARSED_WITH_WARNINGS},
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"cache identity mismatch: {', '.join(failed)}")


class CachedBatchParsingService:
    def __init__(
        self,
        *,
        cache: CanonicalParsedPageCache,
        batch_config: BatchWorkerConfig | None = None,
        parser_factory: ParserFactory | None = None,
        parser_name: str | None = None,
        parser_version: str | None = None,
        parser_config: ParsingConfig | None = None,
    ) -> None:
        parser = parser_factory() if parser_factory else PyMuPdfPageParser()
        self.cache = cache
        self.batch_config = batch_config or BatchWorkerConfig()
        self.parser_factory = parser_factory
        self.parser_name = parser_name or parser.parser_name
        self.parser_version = parser_version or parser.parser_version
        self.parser_config = parser_config or ParsingConfig()
        self.parser_config_fingerprint = config_hash(self.parser_config)

    def parse(self, request: BatchRequest) -> CachedBatchParseResult:
        started = time.perf_counter()
        requested_pages = bounded_page_numbers(request.page_start, request.page_end)
        lookup = self.cache.read_range(
            source=request.source,
            source_path=request.source_path,
            page_numbers=requested_pages,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            parser_config_fingerprint=self.parser_config_fingerprint,
        )
        cached_pages = dict(lookup.pages)
        missing_ranges = _contiguous_ranges(lookup.missing_pages)
        batch_results: list[BatchRunResult] = []
        newly_parsed_pages: list[ParsedPage] = []
        cache_write_ms = 0.0
        writes = 0
        parse_start = time.perf_counter()
        for start_page, end_page in missing_ranges:
            batch_request = request.model_copy(
                update={
                    "page_start": start_page,
                    "page_end": end_page,
                    "output_dir": str(
                        Path(request.output_dir)
                        / ".w"
                        / f"p{start_page:04d}_{end_page:04d}_{uuid.uuid4().hex[:8]}"
                    ),
                }
            )
            batch_result = run_bounded_batch_worker(
                batch_request,
                config=self.batch_config,
                parser_factory=self.parser_factory,
            )
            batch_results.append(batch_result)
            newly_parsed_pages.extend(batch_result.parsed_pages)
            write_result = self.cache.write_pages(
                source=request.source,
                source_path=request.source_path,
                pages=batch_result.parsed_pages,
                parser_config_fingerprint=self.parser_config_fingerprint,
            )
            cache_write_ms += write_result.write_ms
            writes += write_result.writes
            for page in batch_result.parsed_pages:
                cached_pages[page.page_number] = page
        parse_ms = (time.perf_counter() - parse_start) * 1000
        ordered_pages = [cached_pages[page] for page in requested_pages if page in cached_pages]
        return CachedBatchParseResult(
            source_id=request.source.source_id,
            source_hash=request.source.file_hash,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            parser_config_fingerprint=self.parser_config_fingerprint,
            requested_page_start=request.page_start,
            requested_page_end=request.page_end,
            cache_root=str(self.cache.root),
            cache_hits=lookup.hits,
            cache_misses=lookup.misses,
            invalid_cache_entries=lookup.invalid_entries,
            pages_loaded_from_cache=sorted(lookup.pages),
            pages_newly_parsed=sorted(page.page_number for page in newly_parsed_pages),
            page_artifacts_written=writes,
            missing_ranges_sent_to_workers=[
                f"{start_page}-{end_page}" for start_page, end_page in missing_ranges
            ],
            worker_invocation_count=len(batch_results),
            restart_count=sum(result.restart_count for result in batch_results),
            total_wall_time_ms=(time.perf_counter() - started) * 1000,
            cache_read_time_ms=lookup.read_ms,
            parse_time_ms=parse_ms,
            cache_write_time_ms=cache_write_ms,
            active_child_count_after_cleanup=len(multiprocessing.active_children()),
            pages=ordered_pages,
            cache_warnings=lookup.invalid_warnings,
            batch_results=batch_results,
        )


def _artifact_to_page(artifact: CanonicalParsedPage, source: SourceRegistryEntry) -> ParsedPage:
    quality = assess_text_quality(artifact.extracted_text)
    return ParsedPage(
        page_id=f"{source.source_id}:p{artifact.page_number}",
        source_id=source.source_id,
        content_hash=source.file_hash,
        page_number=artifact.page_number,
        physical_page_index=artifact.page_number - 1,
        parser_name=artifact.parser_name,
        parser_version=artifact.parser_version,
        text=artifact.extracted_text,
        character_count=artifact.text_character_count,
        word_count=quality.word_count,
        quality=quality,
        scan_likelihood=quality.scan_likelihood,
        ocr_routing=OcrRouting.NOT_REQUIRED,
        ocr_rationale="Loaded from canonical parsed-page cache.",
        status=ParsingStatus.CACHE_HIT,
        warnings=artifact.parser_warnings,
        parsing_duration_ms=artifact.page_parse_duration_ms,
        provenance=[
            ProvenanceRef(
                source_id=source.source_id,
                page_or_sheet=f"page-{artifact.page_number}",
                notes="Loaded from canonical parsed-page cache.",
            )
        ],
    )


def _contiguous_ranges(pages: list[int]) -> list[tuple[int, int]]:
    if not pages:
        return []
    sorted_pages = sorted(pages)
    ranges: list[tuple[int, int]] = []
    start = sorted_pages[0]
    end = start
    for page in sorted_pages[1:]:
        if page == end + 1:
            end = page
            continue
        ranges.append((start, end))
        start = page
        end = page
    ranges.append((start, end))
    return ranges


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def cache_key_composition() -> list[str]:
    return [
        "schema_version",
        "source_content_hash",
        "page_number",
        "parser_name",
        "parser_version",
        "parser_config_fingerprint",
    ]


def run_cached_bounded_parse(
    *,
    request: BatchRequest,
    cache_root: Path,
    batch_config: BatchWorkerConfig | None = None,
) -> CachedBatchParseResult:
    service = CachedBatchParsingService(
        cache=CanonicalParsedPageCache(cache_root),
        batch_config=batch_config,
    )
    return service.parse(request)


def run_manifest_cached_bounded_parse(
    *,
    source: SourceRegistryEntry,
    source_path: str,
    output_dir: Path,
    cache_root: Path,
    page_start: int,
    page_end: int,
    batch_config: BatchWorkerConfig | None = None,
) -> CachedBatchParseResult:
    if page_end < page_start:
        raise ParsingError("page_end must be greater than or equal to page_start")
    request = BatchRequest(
        source=source,
        source_path=source_path,
        output_dir=str(output_dir),
        page_start=page_start,
        page_end=page_end,
    )
    return run_cached_bounded_parse(
        request=request,
        cache_root=cache_root,
        batch_config=batch_config,
    )
