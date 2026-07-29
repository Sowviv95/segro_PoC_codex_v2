"""Parsing orchestration over Sprint 3 source registry artifacts."""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any

from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry
from segro_evidence_extraction.parsing.cache import JsonParseCache
from segro_evidence_extraction.parsing.interfaces import ProgressCallback
from segro_evidence_extraction.parsing.models import (
    DocumentTiming,
    DrawingCandidate,
    ImageRegion,
    OcrRouting,
    PageTiming,
    ParsedDocument,
    ParsedPage,
    ParsedSheet,
    ParseOptions,
    ParserWarning,
    ParsingConfig,
    ParsingResult,
    ParsingStatus,
    ParsingSummary,
    TableCandidate,
)
from segro_evidence_extraction.parsing.registry import ParserRegistry, default_parser_registry
from segro_evidence_extraction.parsing.reporting import write_parsing_artifacts


class ParsingError(Exception):
    """Raised when parsing inputs are malformed."""


def inspect_parse_manifest(source_manifest: Path) -> dict[str, Any]:
    sources = load_source_registry(source_manifest)
    counts = Counter(str(source.file_type) for source in sources)
    return {
        "source_manifest": str(source_manifest),
        "source_registry": str(_registry_path_from_manifest(source_manifest)),
        "sources": len(sources),
        "files_by_type": dict(sorted(counts.items())),
        "parseable_sources": sum(1 for source in sources if source.file_type != FileType.ZIP),
        "unsupported_for_parsing": [
            {
                "source_id": source.source_id,
                "logical_path": source.logical_path,
                "file_type": source.file_type,
            }
            for source in sources
            if source.file_type in {FileType.ZIP, FileType.XLS, FileType.UNKNOWN}
        ],
    }


def parse_sources(
    source_manifest: Path,
    *,
    output_dir: Path,
    cache_dir: Path,
    config: ParsingConfig | None = None,
    options: ParseOptions | None = None,
    registry: ParserRegistry | None = None,
    progress: ProgressCallback | None = None,
    write_artifacts: bool = True,
) -> ParsingResult:
    config = config or ParsingConfig()
    options = options or ParseOptions()
    registry = registry or default_parser_registry()
    started = datetime.now(UTC)
    run_start = time.perf_counter()
    sources = load_source_registry(source_manifest)
    if options.source_id:
        sources = [source for source in sources if source.source_id == options.source_id]
    cache = JsonParseCache(cache_dir) if options.use_cache else None
    documents: list[ParsedDocument] = []
    pages: list[ParsedPage] = []
    sheets: list[ParsedSheet] = []
    tables: list[TableCandidate] = []
    drawings: list[DrawingCandidate] = []
    issues: list[ParserWarning] = []
    document_timings: list[DocumentTiming] = []
    page_timings: list[PageTiming] = []
    stage_timings: dict[str, float] = {}
    parse_loop_start = time.perf_counter()
    for doc_index, source in enumerate(sources, start=1):
        parser = registry.get(source)
        doc_started = datetime.now(UTC)
        doc_start = time.perf_counter()
        source_path = Path(source.original_path)
        if progress:
            progress(
                "parse:document",
                f"Document {doc_index}/{len(sources)}: {source.logical_path}",
            )
        if parser is None:
            warning = ParserWarning(
                code="unsupported_parser",
                message=f"No Sprint 4 parser is available for {source.file_type}.",
                source_id=source.source_id,
            )
            issues.append(warning)
            documents.append(
                ParsedDocument(
                    document_id=f"doc_{source.source_id}",
                    source_id=source.source_id,
                    logical_path=source.logical_path,
                    file_type=str(source.file_type),
                    parser_name="unsupported",
                    parser_version="unsupported-v1",
                    status=ParsingStatus.UNSUPPORTED,
                    pages_parsed=0,
                    sheets_parsed=0,
                    cache_hits=0,
                    cache_misses=0,
                    warnings=[warning],
                    started_at=doc_started,
                    completed_at=datetime.now(UTC),
                    duration_ms=(time.perf_counter() - doc_start) * 1000,
                )
            )
            continue
        before_hits = cache.hits if cache else 0
        before_misses = cache.misses if cache else 0
        doc_warnings: list[ParserWarning] = []
        doc_pages = 0
        doc_sheets = 0
        for item in parser.parse(
            source,
            source_path=source_path,
            config=config,
            options=options,
            cache=cache,
            progress=progress,
        ):
            if isinstance(item, ParsedPage):
                pages.append(item)
                doc_pages += 1
                if item.table_indicator:
                    tables.append(
                        TableCandidate(
                            table_id=f"{item.page_id}:table-1",
                            source_id=item.source_id,
                            page_or_sheet=f"page-{item.page_number}",
                            confidence=0.65,
                            rationale="Table-like page signal detected during parsing.",
                        )
                    )
                if item.drawing_indicator:
                    drawings.append(
                        DrawingCandidate(
                            drawing_id=f"{item.page_id}:drawing-1",
                            source_id=item.source_id,
                            page_or_sheet=f"page-{item.page_number}",
                            confidence=0.62,
                            rationale="Drawing-like page signal detected during parsing.",
                        )
                    )
                page_timings.append(_page_timing(source, item))
            elif isinstance(item, ParsedSheet):
                sheets.append(item)
                doc_sheets += 1
                if item.table_indicator:
                    tables.append(
                        TableCandidate(
                            table_id=f"{item.sheet_id}:table-1",
                            source_id=item.source_id,
                            page_or_sheet=f"sheet-{item.sheet_number}",
                            confidence=0.8,
                            rationale="Worksheet has multiple columns and rows.",
                        )
                    )
                page_timings.append(_sheet_timing(source, item))
            elif isinstance(item, ImageRegion):
                continue
            elif isinstance(item, DocumentTiming):
                document_timings.append(item)
            elif isinstance(item, ParserWarning):
                issues.append(item)
                doc_warnings.append(item)
        completed = datetime.now(UTC)
        documents.append(
            ParsedDocument(
                document_id=f"doc_{source.source_id}",
                source_id=source.source_id,
                logical_path=source.logical_path,
                file_type=str(source.file_type),
                parser_name=parser.parser_name,
                parser_version=parser.parser_version,
                status=ParsingStatus.PARSED_WITH_WARNINGS
                if doc_warnings
                else ParsingStatus.PARSED,
                pages_parsed=doc_pages,
                sheets_parsed=doc_sheets,
                cache_hits=(cache.hits - before_hits) if cache else 0,
                cache_misses=(cache.misses - before_misses) if cache else 0,
                warnings=doc_warnings,
                started_at=doc_started,
                completed_at=completed,
                duration_ms=(time.perf_counter() - doc_start) * 1000,
            )
        )
    stage_timings["parse_loop_ms"] = (time.perf_counter() - parse_loop_start) * 1000
    _add_aggregate_stage_timings(stage_timings, document_timings)
    summary = _summarize(
        documents=documents,
        pages=pages,
        sheets=sheets,
        tables=tables,
        drawings=drawings,
        issues=issues,
        cache=cache,
    )
    result = ParsingResult(
        parsing_run_id=_run_id(str(source_manifest), started),
        source_manifest_path=str(source_manifest),
        started_at=started,
        completed_at=datetime.now(UTC),
        parsed_documents=documents,
        pages=pages,
        sheets=sheets,
        table_candidates=tables,
        drawing_candidates=drawings,
        issues=issues,
        document_timings=document_timings,
        page_timings=page_timings,
        stage_timings=stage_timings,
        summary=summary,
    )
    if write_artifacts:
        output_paths = write_parsing_artifacts(result, output_dir)
        result.output_paths = output_paths
        result.summary.artifact_paths = output_paths
    else:
        result.stage_timings["artifact_writing_ms"] = 0.0
        result.stage_timings["total_ms"] = (time.perf_counter() - run_start) * 1000
        result.summary.total_runtime_ms = result.stage_timings["total_ms"]
        result.summary.total_page_sheet_units = result.summary.pages_parsed + result.summary.sheets_parsed
    return result


def load_source_registry(source_manifest: Path) -> list[SourceRegistryEntry]:
    registry_path = _registry_path_from_manifest(source_manifest)
    if not registry_path.exists():
        raise ParsingError(f"Source registry not found beside manifest: {registry_path}")
    sources: list[SourceRegistryEntry] = []
    with registry_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    sources.append(SourceRegistryEntry.model_validate_json(line))
                except ValueError as exc:
                    raise ParsingError(
                        f"Malformed source registry row {line_number}: {exc}"
                    ) from exc
    return sources


def _registry_path_from_manifest(source_manifest: Path) -> Path:
    try:
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParsingError(f"Could not read source manifest: {source_manifest}") from exc
    output_paths = manifest.get("output_paths", {})
    if isinstance(output_paths, dict):
        candidate = output_paths.get("source_registry_jsonl")
        if isinstance(candidate, str) and Path(candidate).exists():
            return Path(candidate)
    return source_manifest.with_name("source_registry.jsonl")


def _page_timing(source: SourceRegistryEntry, page: ParsedPage) -> PageTiming:
    return PageTiming(
        source_id=source.source_id,
        logical_path=source.logical_path,
        page_or_sheet=f"page-{page.page_number}",
        status=page.status,
        cache_hit=page.status == ParsingStatus.CACHE_HIT,
        total_ms=page.parsing_duration_ms,
        warnings=[warning.code for warning in page.warnings],
    )


def _sheet_timing(source: SourceRegistryEntry, sheet: ParsedSheet) -> PageTiming:
    return PageTiming(
        source_id=source.source_id,
        logical_path=source.logical_path,
        page_or_sheet=f"sheet-{sheet.sheet_number}",
        status=sheet.status,
        cache_hit=sheet.status == ParsingStatus.CACHE_HIT,
        total_ms=sheet.parsing_duration_ms,
        warnings=[warning.code for warning in sheet.warnings],
    )


def _summarize(
    *,
    documents: list[ParsedDocument],
    pages: list[ParsedPage],
    sheets: list[ParsedSheet],
    tables: list[TableCandidate],
    drawings: list[DrawingCandidate],
    issues: list[ParserWarning],
    cache: JsonParseCache | None,
) -> ParsingSummary:
    classifications: Counter[str] = Counter()
    for page in pages:
        if page.classification:
            classifications[page.classification.primary_label] += 1
    for sheet in sheets:
        if sheet.classification:
            classifications[sheet.classification.primary_label] += 1
    warning_counts = Counter(issue.code for issue in issues)
    for page in pages:
        warning_counts.update(warning.code for warning in page.warnings)
    for sheet in sheets:
        warning_counts.update(warning.code for warning in sheet.warnings)
    return ParsingSummary(
        documents_seen=len(documents),
        documents_parsed=sum(
            1
            for doc in documents
            if doc.status not in {ParsingStatus.FAILED, ParsingStatus.UNSUPPORTED}
        ),
        documents_failed=sum(1 for doc in documents if doc.status == ParsingStatus.FAILED),
        pages_parsed=len(pages),
        sheets_parsed=len(sheets),
        total_page_sheet_units=len(pages) + len(sheets),
        total_runtime_ms=0.0,
        cache_hits=cache.hits if cache else 0,
        cache_misses=cache.misses if cache else 0,
        text_characters=sum(page.character_count for page in pages)
        + sum(sheet.character_count for sheet in sheets),
        table_candidates=len(tables),
        drawing_candidates=len(drawings),
        ocr_required_pages=sum(1 for page in pages if page.ocr_routing == OcrRouting.REQUIRED),
        classifications=dict(sorted(classifications.items())),
        warnings_by_code=dict(sorted(warning_counts.items())),
        failures=len([issue for issue in issues if issue.severity == "error"]),
        slow_pages=warning_counts.get("slow_page", 0),
    )


def _run_id(source_manifest: str, started: datetime) -> str:
    digest = sha1(f"{source_manifest}:{started.isoformat()}".encode()).hexdigest()[:12]
    return f"parse_{digest}"


def _add_aggregate_stage_timings(
    stage_timings: dict[str, float],
    document_timings: list[DocumentTiming],
) -> None:
    stage_timings["cache_read_ms"] = sum(timing.cache_read_ms for timing in document_timings)
    stage_timings["cache_write_ms"] = sum(timing.cache_write_ms for timing in document_timings)
    stage_timings["document_open_ms"] = sum(timing.open_ms for timing in document_timings)
    stage_timings["text_extraction_ms"] = sum(
        timing.text_extraction_ms for timing in document_timings
    )
    stage_timings["normalization_ms"] = sum(timing.normalization_ms for timing in document_timings)
    stage_timings["quality_assessment_ms"] = sum(timing.quality_ms for timing in document_timings)
    stage_timings["classification_ms"] = sum(timing.classification_ms for timing in document_timings)
