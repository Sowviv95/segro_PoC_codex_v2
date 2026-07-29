"""Image parser contracts without OCR execution."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

from segro_evidence_extraction.models.classification import ClassificationSubjectType
from segro_evidence_extraction.models.common import ProvenanceRef
from segro_evidence_extraction.models.source import SourceRegistryEntry
from segro_evidence_extraction.parsing.cache import JsonParseCache
from segro_evidence_extraction.parsing.classification import classify_page_evidence
from segro_evidence_extraction.parsing.interfaces import ProgressCallback
from segro_evidence_extraction.parsing.models import (
    DocumentTiming,
    ImageRegion,
    OcrRouting,
    ParsedPage,
    ParseOptions,
    ParserWarning,
    ParsingConfig,
    ParsingStatus,
)
from segro_evidence_extraction.parsing.quality import assess_text_quality

IMAGE_PARSER_NAME = "image-metadata-only"
IMAGE_PARSER_VERSION = "image-metadata-only-v1"


class ImageMetadataParser:
    parser_name = IMAGE_PARSER_NAME
    parser_version = IMAGE_PARSER_VERSION

    def parse(
        self,
        source: SourceRegistryEntry,
        *,
        source_path: Path,
        config: ParsingConfig,
        options: ParseOptions,
        cache: JsonParseCache | None,
        progress: ProgressCallback | None,
    ) -> Iterator[ParsedPage | ImageRegion | ParserWarning | DocumentTiming]:
        _ = (source_path, config, options, cache)
        start = time.perf_counter()
        if progress:
            progress("parse:image", source.logical_path)
        quality = assess_text_quality("")
        label = "page-1"
        classification = classify_page_evidence(
            source_id=source.source_id,
            page_or_sheet=label,
            subject_type=ClassificationSubjectType.PAGE,
            text="",
            quality=quality,
            file_type="image",
            table_indicator=False,
            drawing_indicator=False,
        )
        page = ParsedPage(
            page_id=f"{source.source_id}:p1",
            source_id=source.source_id,
            content_hash=source.file_hash,
            page_number=1,
            physical_page_index=0,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            text=None,
            character_count=0,
            word_count=0,
            quality=quality,
            image_coverage=1.0,
            scan_likelihood=0.9,
            ocr_routing=OcrRouting.RECOMMENDED,
            ocr_rationale="Image source is registered for future OCR/VLM routing; OCR is not run.",
            status=ParsingStatus.PARSED_WITH_WARNINGS,
            warnings=[
                ParserWarning(
                    code="image_no_ocr",
                    message="Image metadata registered without OCR execution.",
                    source_id=source.source_id,
                    page_or_sheet=label,
                )
            ],
            parsing_duration_ms=(time.perf_counter() - start) * 1000,
            provenance=[ProvenanceRef(source_id=source.source_id, page_or_sheet=label)],
            classification=classification,
        )
        yield page
        yield ImageRegion(
            region_id=f"{source.source_id}:image-region-1",
            source_id=source.source_id,
            page_or_sheet=label,
            confidence=1.0,
            rationale="Whole image registered as a future OCR/VLM region.",
        )
        yield DocumentTiming(
            source_id=source.source_id,
            logical_path=source.logical_path,
            parser_name=self.parser_name,
            pages_or_sheets=1,
            total_ms=(time.perf_counter() - start) * 1000,
        )
