"""Streaming parsers for text-like sources."""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

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
from segro_evidence_extraction.parsing.normalization import normalize_text
from segro_evidence_extraction.parsing.quality import assess_text_quality, route_ocr

TEXT_PARSER_NAME = "text-stream"
TEXT_PARSER_VERSION = "text-stream-v1"


class TextDocumentParser:
    parser_name = TEXT_PARSER_NAME
    parser_version = TEXT_PARSER_VERSION

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
        _ = options
        start = time.perf_counter()
        timing = DocumentTiming(
            source_id=source.source_id,
            logical_path=source.logical_path,
            parser_name=self.parser_name,
        )
        label = "page-1"
        try:
            if progress:
                progress("parse:text", source.logical_path)
            key = cache_key(
                source_id=source.source_id,
                content_hash=source.file_hash,
                page_or_sheet=label,
                parser_name=self.parser_name,
                parser_version=self.parser_version,
                normalization_version=NORMALIZATION_VERSION,
                classification_version=CLASSIFIER_VERSION,
                config_digest=config_hash(config),
            )
            cached = cache.read_page(key) if cache and options.use_cache else None
            if cached is not None:
                cached.status = ParsingStatus.CACHE_HIT
                yield cached
                return
            open_start = time.perf_counter()
            chunks: list[str] = []
            chars_seen = 0
            with source_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                timing.open_ms = (time.perf_counter() - open_start) * 1000
                iter_start = time.perf_counter()
                for line in handle:
                    if chars_seen >= config.max_extracted_chars_per_page:
                        break
                    remaining = config.max_extracted_chars_per_page - chars_seen
                    chunks.append(line[:remaining])
                    chars_seen += len(chunks[-1])
                timing.iteration_ms = (time.perf_counter() - iter_start) * 1000
            norm_start = time.perf_counter()
            text = normalize_text("".join(chunks))
            timing.normalization_ms = (time.perf_counter() - norm_start) * 1000
            quality_start = time.perf_counter()
            quality = assess_text_quality(text)
            ocr_routing, ocr_rationale, _ = route_ocr(quality, "text")
            timing.quality_ms = (time.perf_counter() - quality_start) * 1000
            class_start = time.perf_counter()
            classification = classify_page_evidence(
                source_id=source.source_id,
                page_or_sheet=label,
                subject_type=ClassificationSubjectType.PAGE,
                text=text,
                quality=quality,
                file_type="text",
                table_indicator=False,
                drawing_indicator=False,
            )
            timing.classification_ms = (time.perf_counter() - class_start) * 1000
            page = ParsedPage(
                page_id=f"{source.source_id}:p1",
                source_id=source.source_id,
                content_hash=source.file_hash,
                page_number=1,
                physical_page_index=0,
                parser_name=self.parser_name,
                parser_version=self.parser_version,
                text=text,
                character_count=quality.character_count,
                word_count=quality.word_count,
                quality=quality,
                scan_likelihood=quality.scan_likelihood,
                ocr_routing=ocr_routing,
                ocr_rationale=ocr_rationale,
                status=ParsingStatus.PARSED,
                parsing_duration_ms=(time.perf_counter() - start) * 1000,
                provenance=[
                    ProvenanceRef(
                        source_id=source.source_id,
                        page_or_sheet=label,
                        notes="Text stream.",
                    )
                ],
                classification=classification,
            )
            if cache and options.use_cache:
                cache.write_page(key, page)
            yield page
        except Exception as exc:  # noqa: BLE001
            timing.failures = 1
            yield ParserWarning(
                code="text_parse_failed",
                message=f"Text parsing failed: {exc}",
                source_id=source.source_id,
            )
        finally:
            timing.pages_or_sheets = 1 if timing.failures == 0 else 0
            timing.total_ms = (time.perf_counter() - start) * 1000
            _ = datetime.now(UTC)
            yield timing
