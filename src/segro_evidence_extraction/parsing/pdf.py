"""Provider-neutral PDF parser adapters."""

from __future__ import annotations

import importlib.metadata
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf
from pypdf import PdfReader

from segro_evidence_extraction.models.classification import ClassificationSubjectType
from segro_evidence_extraction.models.common import ProvenanceRef
from segro_evidence_extraction.models.source import SourceRegistryEntry
from segro_evidence_extraction.parsing.cache import JsonParseCache, cache_key, config_hash
from segro_evidence_extraction.parsing.classification import (
    CLASSIFIER_VERSION,
    classify_page_evidence,
)
from segro_evidence_extraction.parsing.interfaces import ProgressCallback
from segro_evidence_extraction.parsing.models import (
    NORMALIZATION_VERSION,
    DocumentTiming,
    ParsedPage,
    ParseOptions,
    ParserWarning,
    ParsingConfig,
    ParsingStatus,
)
from segro_evidence_extraction.parsing.normalization import normalize_text, truncate_text
from segro_evidence_extraction.parsing.quality import assess_text_quality, route_ocr

PDF_PARSER_NAME = "pypdf-incremental"
PYMUPDF_PARSER_NAME = "pymupdf"


def installed_pypdf_version() -> str:
    return importlib.metadata.version("pypdf")


def installed_pymupdf_version() -> str:
    return importlib.metadata.version("PyMuPDF")


PDF_PARSER_VERSION = f"pypdf-{installed_pypdf_version()}"
PYMUPDF_PARSER_VERSION = f"pymupdf-{installed_pymupdf_version()}"


class PdfPageParser:
    parser_name = PDF_PARSER_NAME
    parser_version = PDF_PARSER_VERSION

    def parse(
        self,
        source: SourceRegistryEntry,
        *,
        source_path: Path,
        config: ParsingConfig,
        options: ParseOptions,
        cache: JsonParseCache | None,
        progress: ProgressCallback | None,
    ) -> Iterator[ParsedPage | ParserWarning | DocumentTiming]:
        started = datetime.now(UTC)
        doc_start = time.perf_counter()
        timing = DocumentTiming(
            source_id=source.source_id,
            logical_path=source.logical_path,
            parser_name=self.parser_name,
        )
        pages_parsed = 0
        failures = 0
        try:
            open_start = time.perf_counter()
            with source_path.open("rb") as handle:
                reader = PdfReader(handle, strict=False)
                timing.open_ms += (time.perf_counter() - open_start) * 1000
                if reader.is_encrypted:
                    warning = ParserWarning(
                        code="pdf_encrypted",
                        message="Encrypted PDF cannot be parsed without credentials.",
                        source_id=source.source_id,
                    )
                    yield warning
                    timing.failures = 1
                    return
                if progress:
                    progress("parse:document_opened", source.logical_path)
                page_count = None
                if options.page_end is None:
                    page_count = len(reader.pages)
                    page_numbers = _selected_page_numbers(page_count, options)
                else:
                    page_numbers = bounded_page_numbers(options.page_start or 1, options.page_end)
                    if options.max_pages is not None:
                        page_numbers = page_numbers[: options.max_pages]
                total_selected = len(page_numbers)
                for ordinal, page_number in enumerate(page_numbers, start=1):
                    if progress and _should_report_progress(ordinal, total_selected, options):
                        progress(
                            "parse:page",
                            _page_progress_message(
                                source.logical_path,
                                ordinal,
                                total_selected,
                                page_number,
                                page_count,
                            ),
                        )
                    page_label = f"page-{page_number}"
                    key = _page_cache_key(
                        source,
                        page_label,
                        config,
                        self.parser_name,
                        self.parser_version,
                    )
                    cache_start = time.perf_counter()
                    cached = cache.read_page(key) if cache and options.use_cache else None
                    timing.cache_read_ms += (time.perf_counter() - cache_start) * 1000
                    if cached is not None:
                        cached.status = ParsingStatus.CACHE_HIT
                        pages_parsed += 1
                        yield cached
                        continue
                    try:
                        parsed, page_timing = _parse_pypdf_page(
                            reader.pages[page_number - 1],
                            source=source,
                            page_number=page_number,
                            config=config,
                            parser_name=self.parser_name,
                            parser_version=self.parser_version,
                        )
                    except Exception as exc:  # noqa: BLE001 - recover per page
                        failures += 1
                        yield ParserWarning(
                            code="pdf_page_parse_failed",
                            message=f"Page parsing failed: {type(exc).__name__}: {exc}",
                            source_id=source.source_id,
                            page_or_sheet=page_label,
                        )
                        if failures >= config.max_consecutive_page_failures:
                            yield ParserWarning(
                                code="max_consecutive_page_failures",
                                message="Stopping PDF after repeated page failures.",
                                source_id=source.source_id,
                                page_or_sheet=page_label,
                            )
                            break
                        continue
                    _add_page_timing(timing, page_timing)
                    cache_write_start = time.perf_counter()
                    if cache and options.use_cache:
                        cache.write_page(key, parsed)
                    timing.cache_write_ms += (time.perf_counter() - cache_write_start) * 1000
                    pages_parsed += 1
                    failures = 0
                    yield parsed
        except Exception as exc:  # noqa: BLE001
            yield ParserWarning(
                code="pdf_document_open_failed",
                message=f"PDF could not be opened: {type(exc).__name__}: {exc}",
                source_id=source.source_id,
            )
            timing.failures += 1
        finally:
            completed = datetime.now(UTC)
            timing.pages_or_sheets = pages_parsed
            timing.failures += failures
            timing.total_ms = (time.perf_counter() - doc_start) * 1000
            if (completed - started).total_seconds() > config.max_seconds_per_document_warning:
                yield ParserWarning(
                    code="slow_document",
                    message=f"Document exceeded {config.max_seconds_per_document_warning:.2f}s.",
                    source_id=source.source_id,
                )
            yield timing


class PyMuPdfPageParser:
    parser_name = PYMUPDF_PARSER_NAME
    parser_version = PYMUPDF_PARSER_VERSION

    def parse(
        self,
        source: SourceRegistryEntry,
        *,
        source_path: Path,
        config: ParsingConfig,
        options: ParseOptions,
        cache: JsonParseCache | None,
        progress: ProgressCallback | None,
    ) -> Iterator[ParsedPage | ParserWarning | DocumentTiming]:
        started = datetime.now(UTC)
        doc_start = time.perf_counter()
        timing = DocumentTiming(
            source_id=source.source_id,
            logical_path=source.logical_path,
            parser_name=self.parser_name,
        )
        pages_parsed = 0
        failures = 0
        try:
            open_start = time.perf_counter()
            with pymupdf.open(source_path) as document:  # type: ignore[no-untyped-call]
                timing.open_ms += (time.perf_counter() - open_start) * 1000
                if document.is_encrypted:
                    warning = ParserWarning(
                        code="pdf_encrypted",
                        message="Encrypted PDF cannot be parsed without credentials.",
                        source_id=source.source_id,
                    )
                    yield warning
                    timing.failures = 1
                    return
                if progress:
                    progress("parse:document_opened", source.logical_path)
                page_count = None
                if options.page_end is None:
                    page_count = document.page_count
                    page_numbers = _selected_page_numbers(page_count, options)
                else:
                    page_numbers = bounded_page_numbers(options.page_start or 1, options.page_end)
                    if options.max_pages is not None:
                        page_numbers = page_numbers[: options.max_pages]
                total_selected = len(page_numbers)
                for ordinal, page_number in enumerate(page_numbers, start=1):
                    if progress and _should_report_progress(ordinal, total_selected, options):
                        progress(
                            "parse:page",
                            _page_progress_message(
                                source.logical_path,
                                ordinal,
                                total_selected,
                                page_number,
                                page_count,
                            ),
                        )
                    page_label = f"page-{page_number}"
                    key = _page_cache_key(
                        source,
                        page_label,
                        config,
                        self.parser_name,
                        self.parser_version,
                    )
                    cache_start = time.perf_counter()
                    cached = cache.read_page(key) if cache and options.use_cache else None
                    timing.cache_read_ms += (time.perf_counter() - cache_start) * 1000
                    if cached is not None:
                        cached.status = ParsingStatus.CACHE_HIT
                        pages_parsed += 1
                        yield cached
                        continue
                    try:
                        page = document.load_page(page_number - 1)
                        parsed, page_timing = _parse_pymupdf_page(
                            page,
                            source=source,
                            page_number=page_number,
                            config=config,
                            parser_name=self.parser_name,
                            parser_version=self.parser_version,
                        )
                    except Exception as exc:  # noqa: BLE001 - recover per page
                        failures += 1
                        yield ParserWarning(
                            code="pdf_page_parse_failed",
                            message=f"Page parsing failed: {type(exc).__name__}: {exc}",
                            source_id=source.source_id,
                            page_or_sheet=page_label,
                        )
                        if failures >= config.max_consecutive_page_failures:
                            yield ParserWarning(
                                code="max_consecutive_page_failures",
                                message="Stopping PDF after repeated page failures.",
                                source_id=source.source_id,
                                page_or_sheet=page_label,
                            )
                            break
                        continue
                    _add_page_timing(timing, page_timing)
                    cache_write_start = time.perf_counter()
                    if cache and options.use_cache:
                        cache.write_page(key, parsed)
                    timing.cache_write_ms += (time.perf_counter() - cache_write_start) * 1000
                    pages_parsed += 1
                    failures = 0
                    yield parsed
        except Exception as exc:  # noqa: BLE001
            yield ParserWarning(
                code="pdf_document_open_failed",
                message=f"PDF could not be opened: {type(exc).__name__}: {exc}",
                source_id=source.source_id,
            )
            timing.failures += 1
        finally:
            completed = datetime.now(UTC)
            timing.pages_or_sheets = pages_parsed
            timing.failures += failures
            timing.total_ms = (time.perf_counter() - doc_start) * 1000
            if (completed - started).total_seconds() > config.max_seconds_per_document_warning:
                yield ParserWarning(
                    code="slow_document",
                    message=f"Document exceeded {config.max_seconds_per_document_warning:.2f}s.",
                    source_id=source.source_id,
                )
            yield timing


def _parse_pypdf_page(
    page: object,
    *,
    source: SourceRegistryEntry,
    page_number: int,
    config: ParsingConfig,
    parser_name: str,
    parser_version: str,
) -> tuple[ParsedPage, dict[str, float]]:
    page_start = time.perf_counter()
    iteration_ms = 0.0
    text_start = time.perf_counter()
    raw_text = page.extract_text() or ""  # type: ignore[attr-defined]
    text_extraction_ms = (time.perf_counter() - text_start) * 1000
    width, height = _pypdf_page_dimensions(page)
    rotation = int(page.get("/Rotate", 0) or 0) if hasattr(page, "get") else None
    parsed, timings = _build_parsed_page(
        source=source,
        page_number=page_number,
        raw_text=raw_text,
        width=width,
        height=height,
        rotation=rotation,
        config=config,
        parser_name=parser_name,
        parser_version=parser_version,
        page_start=page_start,
        provenance_note="Parsed incrementally from PDF page with pypdf.",
    )
    timings["iteration_ms"] = iteration_ms
    timings["text_extraction_ms"] = text_extraction_ms
    return parsed, timings


def _parse_pymupdf_page(
    page: pymupdf.Page,
    *,
    source: SourceRegistryEntry,
    page_number: int,
    config: ParsingConfig,
    parser_name: str,
    parser_version: str,
) -> tuple[ParsedPage, dict[str, float]]:
    page_start = time.perf_counter()
    text_start = time.perf_counter()
    raw_text = page.get_text("text") or ""  # type: ignore[no-untyped-call]
    text_extraction_ms = (time.perf_counter() - text_start) * 1000
    rect = page.rect
    parsed, timings = _build_parsed_page(
        source=source,
        page_number=page_number,
        raw_text=raw_text,
        width=float(rect.width),
        height=float(rect.height),
        rotation=int(page.rotation),
        config=config,
        parser_name=parser_name,
        parser_version=parser_version,
        page_start=page_start,
        provenance_note="Parsed incrementally from PDF page with PyMuPDF.",
    )
    timings["text_extraction_ms"] = text_extraction_ms
    return parsed, timings


def _build_parsed_page(
    *,
    source: SourceRegistryEntry,
    page_number: int,
    raw_text: str,
    width: float | None,
    height: float | None,
    rotation: int | None,
    config: ParsingConfig,
    parser_name: str,
    parser_version: str,
    page_start: float,
    provenance_note: str,
) -> tuple[ParsedPage, dict[str, float]]:
    page_label = f"page-{page_number}"
    norm_start = time.perf_counter()
    text = normalize_text(raw_text)
    text, truncated = truncate_text(text, config.max_extracted_chars_per_page)
    normalization_ms = (time.perf_counter() - norm_start) * 1000
    warnings: list[ParserWarning] = []
    if truncated:
        warnings.append(
            ParserWarning(
                code="page_text_truncated",
                message="Extracted page text exceeded configured character limit.",
                source_id=source.source_id,
                page_or_sheet=page_label,
            )
        )
    quality_start = time.perf_counter()
    quality = assess_text_quality(text, page_area=(width * height) if width and height else None)
    ocr_routing, ocr_rationale, _ocr_confidence = route_ocr(quality, "pdf")
    table_indicator = _looks_like_table(text)
    drawing_indicator = _looks_like_drawing(text, quality, width, height)
    certificate_indicator = _looks_like_certificate_or_test(text)
    quality_ms = (time.perf_counter() - quality_start) * 1000
    class_start = time.perf_counter()
    classification = classify_page_evidence(
        source_id=source.source_id,
        page_or_sheet=page_label,
        subject_type=ClassificationSubjectType.PAGE,
        text=text,
        quality=quality,
        file_type="pdf",
        table_indicator=table_indicator,
        drawing_indicator=drawing_indicator,
    )
    classification_ms = (time.perf_counter() - class_start) * 1000
    duration_ms = (time.perf_counter() - page_start) * 1000
    if duration_ms / 1000 > config.max_seconds_per_page_warning:
        warnings.append(
            ParserWarning(
                code="slow_page",
                message=f"Page exceeded {config.max_seconds_per_page_warning:.2f}s.",
                source_id=source.source_id,
                page_or_sheet=page_label,
                raw_value=f"{duration_ms:.2f}ms",
            )
        )
    page_result = ParsedPage(
        page_id=f"{source.source_id}:p{page_number}",
        source_id=source.source_id,
        content_hash=source.file_hash,
        page_number=page_number,
        physical_page_index=page_number - 1,
        page_label=None,
        parser_name=parser_name,
        parser_version=parser_version,
        text=text,
        character_count=quality.character_count,
        word_count=quality.word_count,
        quality=quality,
        width=width,
        height=height,
        rotation=rotation,
        table_indicator=table_indicator,
        drawing_indicator=drawing_indicator,
        certificate_or_test_indicator=certificate_indicator,
        scan_likelihood=quality.scan_likelihood,
        ocr_routing=ocr_routing,
        ocr_rationale=ocr_rationale,
        status=ParsingStatus.PARSED_WITH_WARNINGS if warnings else ParsingStatus.PARSED,
        warnings=warnings,
        parsing_duration_ms=duration_ms,
        provenance=[
            ProvenanceRef(
                source_id=source.source_id,
                page_or_sheet=page_label,
                notes=provenance_note,
            )
        ],
        classification=classification,
    )
    return page_result, {
        "iteration_ms": 0.0,
        "text_extraction_ms": 0.0,
        "normalization_ms": normalization_ms,
        "quality_ms": quality_ms,
        "classification_ms": classification_ms,
    }


def _selected_page_numbers(page_count: int, options: ParseOptions) -> list[int]:
    start = options.page_start or 1
    end = min(options.page_end or page_count, page_count)
    if start > page_count:
        return []
    numbers = list(range(start, end + 1))
    if options.max_pages is not None:
        numbers = numbers[: options.max_pages]
    return numbers


def bounded_page_numbers(page_start: int, page_end: int) -> list[int]:
    if page_end < page_start:
        msg = "page_end must be greater than or equal to page_start"
        raise ValueError(msg)
    return list(range(page_start, page_end + 1))


def _should_report_progress(ordinal: int, total_selected: int, options: ParseOptions) -> bool:
    return ordinal == 1 or ordinal == total_selected or ordinal % options.progress_every == 0


def _page_progress_message(
    logical_path: str,
    ordinal: int,
    total_selected: int,
    page_number: int,
    page_count: int | None,
) -> str:
    if page_count is None:
        return f"{logical_path} Page {ordinal}/{total_selected} ({page_number})"
    return f"{logical_path} Page {ordinal}/{total_selected} ({page_number}/{page_count})"


def _page_cache_key(
    source: SourceRegistryEntry,
    page_label: str,
    config: ParsingConfig,
    parser_name: str,
    parser_version: str,
) -> str:
    return cache_key(
        source_id=source.source_id,
        content_hash=source.file_hash,
        page_or_sheet=page_label,
        parser_name=parser_name,
        parser_version=parser_version,
        normalization_version=NORMALIZATION_VERSION,
        classification_version=CLASSIFIER_VERSION,
        config_digest=config_hash(config),
    )


def _pypdf_page_dimensions(page: object) -> tuple[float | None, float | None]:
    mediabox = getattr(page, "mediabox", None)
    if mediabox is None:
        return None, None
    try:
        return float(mediabox.width), float(mediabox.height)
    except (TypeError, ValueError, AttributeError):
        return None, None


def _looks_like_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    pipe_or_tab = sum(1 for line in lines if "|" in line or "\t" in line)
    numeric_rows = sum(1 for line in lines if sum(char.isdigit() for char in line) >= 3)
    return pipe_or_tab >= 2 or (numeric_rows >= 5 and len(lines) >= 8)


def _looks_like_certificate_or_test(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in ["certificate", "certification", "test report"])


def _looks_like_drawing(
    text: str,
    quality: Any,
    width: float | None,
    height: float | None,
) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in ["drawing no", "scale:", "revision", "title block"]):
        return True
    wide_page = width is not None and height is not None and width > height * 1.25
    return bool(wide_page and getattr(quality, "character_count", 0) < 3_000)


def _add_page_timing(timing: DocumentTiming, page_timing: dict[str, float]) -> None:
    timing.iteration_ms += page_timing["iteration_ms"]
    timing.text_extraction_ms += page_timing["text_extraction_ms"]
    timing.normalization_ms += page_timing["normalization_ms"]
    timing.quality_ms += page_timing["quality_ms"]
    timing.classification_ms += page_timing["classification_ms"]
