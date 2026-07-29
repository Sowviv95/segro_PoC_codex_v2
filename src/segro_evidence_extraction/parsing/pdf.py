"""Incremental PDF parsing using pypdf."""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

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
PDF_PARSER_VERSION = "pypdf-incremental-v1"


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
                page_count = len(reader.pages)
                page_numbers = _selected_page_numbers(page_count, options)
                total_selected = len(page_numbers)
                for ordinal, page_number in enumerate(page_numbers, start=1):
                    page_label = f"page-{page_number}"
                    if progress and (
                        ordinal == 1
                        or ordinal == total_selected
                        or ordinal % options.progress_every == 0
                    ):
                        progress(
                            "parse:page",
                            (
                                f"{source.logical_path} Page {ordinal}/{total_selected} "
                                f"({page_number}/{page_count})"
                            ),
                        )
                    key = _page_cache_key(source, page_label, config)
                    cache_start = time.perf_counter()
                    cached = cache.read_page(key) if cache and options.use_cache else None
                    timing.cache_read_ms += (time.perf_counter() - cache_start) * 1000
                    if cached is not None:
                        cached.status = ParsingStatus.CACHE_HIT
                        pages_parsed += 1
                        yield cached
                        continue
                    page_start = time.perf_counter()
                    page_timing_text = 0.0
                    page_timing_norm = 0.0
                    page_timing_quality = 0.0
                    page_timing_class = 0.0
                    warnings: list[ParserWarning] = []
                    try:
                        iteration_start = time.perf_counter()
                        page = reader.pages[page_number - 1]
                        timing.iteration_ms += (time.perf_counter() - iteration_start) * 1000
                        text_start = time.perf_counter()
                        raw_text = page.extract_text() or ""
                        page_timing_text = (time.perf_counter() - text_start) * 1000
                        timing.text_extraction_ms += page_timing_text
                        norm_start = time.perf_counter()
                        text = normalize_text(raw_text)
                        text, truncated = truncate_text(text, config.max_extracted_chars_per_page)
                        page_timing_norm = (time.perf_counter() - norm_start) * 1000
                        timing.normalization_ms += page_timing_norm
                        if truncated:
                            warnings.append(
                                ParserWarning(
                                    code="page_text_truncated",
                                    message=(
                                        "Extracted page text exceeded configured character limit."
                                    ),
                                    source_id=source.source_id,
                                    page_or_sheet=page_label,
                                )
                            )
                        width, height = _page_dimensions(page)
                        rotation = int(page.get("/Rotate", 0) or 0)
                        quality_start = time.perf_counter()
                        quality = assess_text_quality(
                            text,
                            page_area=(width * height) if width and height else None,
                        )
                        ocr_routing, ocr_rationale, _ocr_confidence = route_ocr(quality, "pdf")
                        table_indicator = _looks_like_table(text)
                        drawing_indicator = _looks_like_drawing(text, quality, width, height)
                        certificate_indicator = _looks_like_certificate_or_test(text)
                        page_timing_quality = (time.perf_counter() - quality_start) * 1000
                        timing.quality_ms += page_timing_quality
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
                        page_timing_class = (time.perf_counter() - class_start) * 1000
                        timing.classification_ms += page_timing_class
                        duration_ms = (time.perf_counter() - page_start) * 1000
                        if duration_ms / 1000 > config.max_seconds_per_page_warning:
                            warnings.append(
                                ParserWarning(
                                    code="slow_page",
                                    message=(
                                        f"Page exceeded "
                                        f"{config.max_seconds_per_page_warning:.2f}s."
                                    ),
                                    source_id=source.source_id,
                                    page_or_sheet=page_label,
                                    raw_value=f"{duration_ms:.2f}ms",
                                )
                            )
                        parsed = ParsedPage(
                            page_id=f"{source.source_id}:p{page_number}",
                            source_id=source.source_id,
                            content_hash=source.file_hash,
                            page_number=page_number,
                            physical_page_index=page_number - 1,
                            page_label=None,
                            parser_name=self.parser_name,
                            parser_version=self.parser_version,
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
                            status=ParsingStatus.PARSED_WITH_WARNINGS
                            if warnings
                            else ParsingStatus.PARSED,
                            warnings=warnings,
                            parsing_duration_ms=duration_ms,
                            provenance=[
                                ProvenanceRef(
                                    source_id=source.source_id,
                                    page_or_sheet=page_label,
                                    notes="Parsed incrementally from PDF page.",
                                )
                            ],
                            classification=classification,
                        )
                        cache_write_start = time.perf_counter()
                        if cache and options.use_cache:
                            cache.write_page(key, parsed)
                        timing.cache_write_ms += (time.perf_counter() - cache_write_start) * 1000
                        pages_parsed += 1
                        failures = 0
                        yield parsed
                    except Exception as exc:  # noqa: BLE001 - recover per page
                        failures += 1
                        yield ParserWarning(
                            code="pdf_page_parse_failed",
                            message=f"Page parsing failed: {exc}",
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
                    finally:
                        _ = (
                            page_timing_text,
                            page_timing_norm,
                            page_timing_quality,
                            page_timing_class,
                        )
        except Exception as exc:  # noqa: BLE001
            yield ParserWarning(
                code="pdf_document_open_failed",
                message=f"PDF could not be opened: {exc}",
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


def _selected_page_numbers(page_count: int, options: ParseOptions) -> list[int]:
    start = options.page_start or 1
    end = min(options.page_end or page_count, page_count)
    if start > page_count:
        return []
    numbers = list(range(start, end + 1))
    if options.max_pages is not None:
        numbers = numbers[: options.max_pages]
    return numbers


def _page_cache_key(source: SourceRegistryEntry, page_label: str, config: ParsingConfig) -> str:
    return cache_key(
        source_id=source.source_id,
        content_hash=source.file_hash,
        page_or_sheet=page_label,
        parser_name=PDF_PARSER_NAME,
        parser_version=PDF_PARSER_VERSION,
        normalization_version=NORMALIZATION_VERSION,
        classification_version=CLASSIFIER_VERSION,
        config_digest=config_hash(config),
    )


def _page_dimensions(page: object) -> tuple[float | None, float | None]:
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
    quality: object,
    width: float | None,
    height: float | None,
) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in ["drawing no", "scale:", "revision", "title block"]):
        return True
    wide_page = width is not None and height is not None and width > height * 1.25
    return bool(wide_page and getattr(quality, "character_count", 0) < 500)
