"""Serializable contracts for bounded document parsing."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator

from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.classification import ClassificationResult
from segro_evidence_extraction.models.common import BoundingBox, ProvenanceRef

PARSING_VERSION = "parsing-v1"
NORMALIZATION_VERSION = "text-normalization-v1"
PAGE_CLASSIFIER_VERSION = "page-evidence-classifier-v1"


class ParsingStatus(StrEnum):
    PARSED = "parsed"
    PARSED_WITH_WARNINGS = "parsed_with_warnings"
    CACHE_HIT = "cache_hit"
    SKIPPED = "skipped"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class OcrRouting(StrEnum):
    NOT_REQUIRED = "not_required"
    RECOMMENDED = "recommended"
    REQUIRED = "required"
    UNSUITABLE_UNKNOWN = "unsuitable_unknown"


class ParserWarning(StrictBaseModel):
    code: str
    message: str
    severity: str = "warning"
    source_id: str | None = None
    page_or_sheet: str | None = None
    raw_value: str | None = None


class TextQualityMetrics(StrictBaseModel):
    character_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    line_count: int = Field(ge=0)
    average_line_length: float = Field(ge=0)
    alphabetic_ratio: float = Field(ge=0, le=1)
    numeric_ratio: float = Field(ge=0, le=1)
    repeated_character_ratio: float = Field(ge=0, le=1)
    text_density: float = Field(ge=0)
    extraction_error_indicators: list[str] = Field(default_factory=list)
    likely_text_order_degradation: bool = False
    scan_likelihood: float = Field(ge=0, le=1)


class TextBlock(StrictBaseModel):
    block_id: str
    source_id: str
    page_or_sheet: str
    text: str | None = None
    text_ref: str | None = None
    character_count: int = Field(ge=0)
    bounding_box: BoundingBox | None = None
    provenance: list[ProvenanceRef] = Field(default_factory=list)


class TableCandidate(StrictBaseModel):
    table_id: str
    source_id: str
    page_or_sheet: str
    confidence: float = Field(ge=0, le=1)
    bounding_box: BoundingBox | None = None
    structural_ref: str | None = None
    rationale: str


class DrawingCandidate(StrictBaseModel):
    drawing_id: str
    source_id: str
    page_or_sheet: str
    confidence: float = Field(ge=0, le=1)
    region_ref: str | None = None
    title_block_region: BoundingBox | None = None
    rationale: str


class ImageRegion(StrictBaseModel):
    region_id: str
    source_id: str
    page_or_sheet: str
    bounding_box: BoundingBox | None = None
    image_ref: str | None = None
    confidence: float = Field(ge=0, le=1)
    rationale: str


class ParsedPage(StrictBaseModel):
    page_id: str
    source_id: str
    content_hash: str
    page_number: int = Field(ge=1)
    physical_page_index: int = Field(ge=0)
    page_label: str | None = None
    parser_name: str
    parser_version: str
    text: str | None = None
    text_ref: str | None = None
    character_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    quality: TextQualityMetrics
    width: float | None = Field(default=None, gt=0)
    height: float | None = Field(default=None, gt=0)
    rotation: int | None = None
    image_coverage: float | None = Field(default=None, ge=0, le=1)
    table_indicator: bool = False
    drawing_indicator: bool = False
    certificate_or_test_indicator: bool = False
    scan_likelihood: float = Field(ge=0, le=1)
    ocr_routing: OcrRouting
    ocr_rationale: str
    status: ParsingStatus
    warnings: list[ParserWarning] = Field(default_factory=list)
    parsing_duration_ms: float = Field(ge=0)
    provenance: list[ProvenanceRef] = Field(default_factory=list)
    classification: ClassificationResult | None = None

    @field_validator("page_id", "source_id", "content_hash", "parser_name", "parser_version")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        if not value:
            msg = "value must not be blank"
            raise ValueError(msg)
        return value


class ParsedSheet(StrictBaseModel):
    sheet_id: str
    source_id: str
    content_hash: str
    sheet_number: int = Field(ge=1)
    sheet_name: str
    parser_name: str
    parser_version: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    preview_text: str | None = None
    character_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    table_indicator: bool = False
    ocr_routing: OcrRouting = OcrRouting.NOT_REQUIRED
    status: ParsingStatus
    warnings: list[ParserWarning] = Field(default_factory=list)
    parsing_duration_ms: float = Field(ge=0)
    provenance: list[ProvenanceRef] = Field(default_factory=list)
    classification: ClassificationResult | None = None


class ParsedDocument(StrictBaseModel):
    document_id: str
    source_id: str
    logical_path: str
    file_type: str
    parser_name: str
    parser_version: str
    status: ParsingStatus
    pages_parsed: int = Field(ge=0)
    sheets_parsed: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    cache_misses: int = Field(ge=0)
    warnings: list[ParserWarning] = Field(default_factory=list)
    started_at: datetime
    completed_at: datetime
    duration_ms: float = Field(ge=0)


class ParsingConfig(StrictBaseModel):
    max_extracted_chars_per_page: int = Field(default=40_000, ge=1)
    max_text_block_chars: int = Field(default=20_000, ge=1)
    max_seconds_per_page_warning: float = Field(default=5.0, gt=0)
    max_seconds_per_document_warning: float = Field(default=300.0, gt=0)
    max_consecutive_page_failures: int = Field(default=10, ge=1)
    spreadsheet_preview_rows: int = Field(default=200, ge=1)
    csv_preview_rows: int = Field(default=500, ge=1)
    config_version: str = "default-v1"


class ParseOptions(StrictBaseModel):
    source_id: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    max_pages: int | None = Field(default=None, ge=1)
    use_cache: bool = True
    resume: bool = True
    progress_every: int = Field(default=25, ge=1)

    @field_validator("page_end")
    @classmethod
    def end_must_not_precede_start(cls, value: int | None, info: object) -> int | None:
        data = getattr(info, "data", {})
        start = data.get("page_start")
        if value is not None and isinstance(start, int) and value < start:
            msg = "page_end must be greater than or equal to page_start"
            raise ValueError(msg)
        return value


class DocumentTiming(StrictBaseModel):
    source_id: str
    logical_path: str
    parser_name: str
    open_ms: float = Field(default=0, ge=0)
    iteration_ms: float = Field(default=0, ge=0)
    text_extraction_ms: float = Field(default=0, ge=0)
    normalization_ms: float = Field(default=0, ge=0)
    quality_ms: float = Field(default=0, ge=0)
    classification_ms: float = Field(default=0, ge=0)
    cache_read_ms: float = Field(default=0, ge=0)
    cache_write_ms: float = Field(default=0, ge=0)
    total_ms: float = Field(default=0, ge=0)
    pages_or_sheets: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)


class PageTiming(StrictBaseModel):
    source_id: str
    logical_path: str
    page_or_sheet: str
    status: ParsingStatus
    cache_hit: bool = False
    text_extraction_ms: float = Field(default=0, ge=0)
    normalization_ms: float = Field(default=0, ge=0)
    quality_ms: float = Field(default=0, ge=0)
    classification_ms: float = Field(default=0, ge=0)
    total_ms: float = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class ParsingSummary(StrictBaseModel):
    documents_seen: int
    documents_parsed: int
    documents_failed: int
    pages_parsed: int
    sheets_parsed: int
    total_page_sheet_units: int
    total_runtime_ms: float = Field(ge=0)
    cache_hits: int
    cache_misses: int
    text_characters: int
    table_candidates: int
    drawing_candidates: int
    ocr_required_pages: int
    classifications: dict[str, int]
    warnings_by_code: dict[str, int]
    failures: int
    slow_pages: int
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class ParsingResult(StrictBaseModel):
    parsing_run_id: str
    source_manifest_path: str
    parser_version: str = PARSING_VERSION
    started_at: datetime
    completed_at: datetime
    parsed_documents: list[ParsedDocument]
    pages: list[ParsedPage] = Field(default_factory=list)
    sheets: list[ParsedSheet] = Field(default_factory=list)
    table_candidates: list[TableCandidate] = Field(default_factory=list)
    drawing_candidates: list[DrawingCandidate] = Field(default_factory=list)
    issues: list[ParserWarning] = Field(default_factory=list)
    document_timings: list[DocumentTiming] = Field(default_factory=list)
    page_timings: list[PageTiming] = Field(default_factory=list)
    stage_timings: dict[str, float] = Field(default_factory=dict)
    summary: ParsingSummary
    output_paths: dict[str, str] = Field(default_factory=dict)
