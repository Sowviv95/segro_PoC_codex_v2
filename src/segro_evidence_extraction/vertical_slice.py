"""Evidence-first vertical slice over bounded cached pages."""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, ValidationError, field_validator

from segro_evidence_extraction.config import Settings, load_settings
from segro_evidence_extraction.dictionary.service import ingest_dictionary
from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.common import ExpectedDataType, PageSheetRange, ProvenanceRef
from segro_evidence_extraction.models.evidence_index import (
    EvidenceIndex,
    EvidenceIndexNode,
    EvidenceNodeType,
)
from segro_evidence_extraction.models.source import SourceRegistryEntry
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.parsing.batch_worker import BatchRequest, BatchWorkerConfig
from segro_evidence_extraction.parsing.models import ParsedPage
from segro_evidence_extraction.parsing.page_cache import (
    CachedBatchParseResult,
    CachedBatchParsingService,
    CanonicalParsedPageCache,
)
from segro_evidence_extraction.parsing.service import load_source_registry

VERTICAL_SLICE_VERSION = "evidence-first-vertical-slice-v1"
VERTICAL_SLICE_V2_VERSION = "evidence-first-vertical-slice-v2"
VERTICAL_SLICE_V3_VERSION = "evidence-first-vertical-slice-v3"
DEFAULT_OUTPUT_DIR = Path("output/enfield_unit1_evidence_first_vertical_slice_v1")
DEFAULT_CACHE_ROOT = Path("output/enfield_unit1_evidence_first_vertical_slice_v1/page_cache")
DEFAULT_V2_OUTPUT_DIR = Path("output/enfield_unit1_evidence_first_vertical_slice_v2")
DEFAULT_V3_OUTPUT_DIR = Path("output/enfield_unit1_evidence_first_vertical_slice_v3")
MAX_BATCH_PAGES = 10
MAX_EVIDENCE_CHARS = 4500
MIN_RELEVANCE_SCORE = 7.0
WEAK_RELEVANCE_SCORE = 3.0


ExtractionStatus = Literal[
    "extracted",
    "multiple_candidates",
    "insufficient_evidence",
    "conflicting_evidence",
    "invalid_format",
    "model_error",
]

ValueShapeFamily = Literal[
    "categorical",
    "integer_count",
    "decimal_measurement",
    "date",
    "identifier_or_reference",
    "boolean_or_presence",
    "short_text",
    "descriptive_text",
    "ordered_or_unordered_list",
    "unsupported_or_unknown",
]

LayerStatus = Literal["valid", "valid_after_normalization", "review_required", "invalid"]

OverallReviewStatus = Literal[
    "valid",
    "valid_after_normalization",
    "valid_with_dictionary_caveat",
    "review_required",
    "invalid_evidence",
    "invalid_shape",
    "invalid_value",
    "insufficient_evidence",
    "model_error",
]


class SourceRange(StrictBaseModel):
    source_id: str
    logical_path: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    reason: str

    @field_validator("page_end")
    @classmethod
    def end_must_not_precede_start(cls, value: int, info: object) -> int:
        start = getattr(info, "data", {}).get("page_start")
        if isinstance(start, int) and value < start:
            msg = "page_end must be greater than or equal to page_start"
            raise ValueError(msg)
        return value


class SelectedTarget(StrictBaseModel):
    target: TargetSpecification
    selection_reason: str


class HierarchyNode(StrictBaseModel):
    node_id: str
    source_id: str
    node_type: Literal["document", "section", "subsection", "page", "text_block"]
    title: str
    parent_node_id: str | None = None
    page_start: int
    page_end: int
    text_summary: str
    child_node_ids: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRef] = Field(default_factory=list)


class LightweightHierarchy(StrictBaseModel):
    hierarchy_id: str
    nodes: list[HierarchyNode]
    warnings: list[str] = Field(default_factory=list)


class RetrievalExpectation(StrictBaseModel):
    target_row_id: str
    expected_source_id: str | None = None
    expected_pages: list[int] = Field(default_factory=list)
    expected_section_contains: str | None = None
    no_supported_evidence_expected: bool = False
    notes: str


class RetrievalScoreBreakdown(StrictBaseModel):
    exact_phrase: float = 0
    token_overlap: float = 0
    title_match: float = 0
    domain_match: float = 0
    pattern_match: float = 0
    hierarchy_proximity: float = 0
    component_attribute_proximity: float = 0
    domain_subdomain_match: float = 0
    datatype_unit_pattern: float = 0
    certificate_date_pattern: float = 0
    negative_penalty: float = 0
    status_context_penalty: float = 0
    cross_reference_penalty: float = 0
    final_score: float = 0


class QueryConcepts(StrictBaseModel):
    exact_field_phrase: str
    normalized_field_tokens: list[str]
    component_phrases: list[str]
    component_tokens: list[str]
    attribute_tokens: list[str]
    domain_tokens: list[str]
    datatype_unit_hints: list[str]
    certificate_date_reference_indicators: list[str]
    negative_terms: list[str] = Field(default_factory=list)


class RetrievedEvidence(StrictBaseModel):
    target_row_id: str
    rank: int
    node_id: str
    source_id: str
    source_file: str
    page_start: int
    page_end: int
    score: float
    score_components: RetrievalScoreBreakdown
    matched_terms: list[str] = Field(default_factory=list)
    hierarchy_path: list[str] = Field(default_factory=list)
    excerpt: str


class RetrievalResult(StrictBaseModel):
    target_row_id: str
    query: str
    retrieval_status: Literal["evidence_found", "weak_evidence", "no_relevant_evidence"] = (
        "evidence_found"
    )
    query_concepts: QueryConcepts | None = None
    results: list[RetrievedEvidence]
    retrieval_time_ms: float
    top_score: float


class RetrievalEvaluationItem(StrictBaseModel):
    target_row_id: str
    top1_hit: bool = False
    top3_hit: bool = False
    absent_correct: bool = False
    irrelevant_retrieval: bool = False
    notes: str


class EvidenceBundleRecord(StrictBaseModel):
    target_row_id: str
    evidence_items: list[RetrievedEvidence]
    retrieval_status: Literal["evidence_found", "weak_evidence", "no_relevant_evidence"] = (
        "evidence_found"
    )
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    combined_text: str
    character_count: int
    token_estimate: int
    truncated: bool = False


class EvidenceSpan(StrictBaseModel):
    span_id: str
    source_id: str
    source_file: str
    page_number: int = Field(ge=1)
    hierarchy_node_id: str
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    retrieval_rank: int = Field(ge=1)
    score: float


class ModelUsage(StrictBaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class ExtractionResult(StrictBaseModel):
    target_row_id: str
    requirement_id: str
    raw_model_value: str | int | float | bool | list[str] | None = None
    evidence_value: str | int | float | bool | list[str] | None = None
    extracted_value: str | int | float | bool | list[str] | None = None
    normalized_value: str | int | float | bool | list[str] | None = None
    display_value: str | None = None
    proposed_value_shape: ValueShapeFamily | None = None
    value_bearing_quote: str | None = None
    unit: str | None = None
    status: ExtractionStatus
    confidence: float = Field(ge=0, le=1)
    supporting_span_ids: list[str] = Field(default_factory=list)
    supporting_evidence_excerpt: str | None = None
    source_id: str | None = None
    source_file: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    page_range: str | None = None
    hierarchy_node_id: str | None = None
    reasoning_summary: str | None = None
    ambiguity_or_caveat: str | None = None
    model_provider: str
    model_name: str
    model_usage: ModelUsage = Field(default_factory=ModelUsage)


class ValidationResult(StrictBaseModel):
    target_row_id: str
    status: OverallReviewStatus | Literal["invalid"]
    evidence_valid: bool = False
    value_format_valid: bool = False
    issues: list[str] = Field(default_factory=list)


class SchemaCompatibilityResult(StrictBaseModel):
    target_row_id: str
    dictionary_datatype: str
    dictionary_unit: str | None = None
    observed_value: str | int | float | bool | list[str] | None = None
    observed_unit: str | None = None
    compatibility: Literal[
        "compatible",
        "compatible_after_normalization",
        "dictionary_metadata_ambiguous",
        "narrative_value_against_structured_constraint",
        "unit_not_applicable",
        "unit_not_applicable_to_observed_value",
        "suspected_dictionary_metadata_mismatch",
        "incompatible",
        "incompatible_value",
        "not_evaluated",
    ]
    evidence_validity: Literal["valid", "invalid", "not_evaluated"]
    value_format_validity: Literal["valid", "invalid", "not_evaluated"]
    overall_review_status: Literal[
        "valid",
        "valid_after_normalization",
        "valid_with_dictionary_caveat",
        "review_required",
        "invalid_evidence",
        "invalid_shape",
        "invalid_value",
        "insufficient_evidence",
        "model_error",
    ]
    issues: list[str] = Field(default_factory=list)


class ValueShapeAssignment(StrictBaseModel):
    target_row_id: str
    field_name: str
    dictionary_declared_datatype: str
    dictionary_declared_unit: str | None = None
    value_shape_family: ValueShapeFamily
    inference_basis: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    ambiguity_or_mismatch_warning: str | None = None


class NormalizedValueRecord(StrictBaseModel):
    target_row_id: str
    value_shape_family: ValueShapeFamily
    raw_model_value: str | int | float | bool | list[str] | None = None
    evidence_value: str | int | float | bool | list[str] | None = None
    normalized_value: str | int | float | bool | list[str] | None = None
    display_value: str | None = None
    unit: str | None = None
    normalization_status: Literal[
        "normalized",
        "already_normalized",
        "ambiguous",
        "not_applicable",
        "failed",
    ]
    issues: list[str] = Field(default_factory=list)


class EvidenceValidationResult(StrictBaseModel):
    target_row_id: str
    status: LayerStatus
    selected_span_ids: list[str] = Field(default_factory=list)
    canonical_evidence_available: bool = False
    evidence_value_present: bool = False
    source_page_node_valid: bool = False
    issues: list[str] = Field(default_factory=list)


class ShapeValidationResult(StrictBaseModel):
    target_row_id: str
    value_shape_family: ValueShapeFamily
    status: LayerStatus
    issues: list[str] = Field(default_factory=list)


class V2V3ComparisonItem(StrictBaseModel):
    target_row_id: str
    field_name: str
    v2_validation_status: str | None
    v2_schema_compatibility: str | None
    v3_overall_review_status: str
    v3_schema_compatibility: str
    v3_value_shape: ValueShapeFamily
    v3_normalized_value: Any = None
    diagnosis: str
    v3_change: str


class V2V3Comparison(StrictBaseModel):
    baseline_dir: str
    v3_dir: str
    v2_status_counts: dict[str, int]
    v3_status_counts: dict[str, int]
    items: list[V2V3ComparisonItem]


class V1V2ComparisonItem(StrictBaseModel):
    target_row_id: str
    field_name: str
    v1_top1_hit: bool
    v1_top3_hit: bool
    v2_top1_hit: bool
    v2_top3_hit: bool
    v1_extraction_status: str | None
    v1_extraction_value: Any = None
    v2_extraction_status: str | None
    v2_extraction_value: Any = None
    v1_validation_status: str | None
    v2_evidence_validation: str | None
    v2_schema_compatibility: str | None
    outcome: Literal["improved", "regressed", "unchanged"]


class V1V2Comparison(StrictBaseModel):
    baseline_dir: str
    v2_dir: str
    v1_top1_hits: int
    v1_top3_hits: int
    v2_top1_hits: int
    v2_top3_hits: int
    items: list[V1V2ComparisonItem]


class VerticalSliceTelemetry(StrictBaseModel):
    source_ranges_requested: list[SourceRange]
    parser_cache_hits: int
    parser_cache_misses: int
    parser_worker_invocations: int
    parser_active_children_after_cleanup: int
    hierarchy_node_counts_by_type: dict[str, int]
    retrieval_time_ms_by_target: dict[str, float]
    top_retrieval_scores: dict[str, float]
    evidence_bundle_character_count: dict[str, int]
    evidence_bundle_token_estimate: dict[str, int]
    llm_provider: str
    llm_model: str
    llm_calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    extraction_time_ms_by_target: dict[str, float]
    validation_status_by_target: dict[str, str]
    total_wall_time_ms: float


class VerticalSliceRunResult(StrictBaseModel):
    output_dir: str
    selected_targets: list[SelectedTarget]
    source_ranges: list[SourceRange]
    hierarchy: LightweightHierarchy
    retrieval_expectations: list[RetrievalExpectation]
    retrieval_results: list[RetrievalResult]
    retrieval_evaluation: list[RetrievalEvaluationItem]
    value_shape_assignments: list[ValueShapeAssignment] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    evidence_bundles: list[EvidenceBundleRecord]
    extraction_results: list[ExtractionResult]
    normalized_values: list[NormalizedValueRecord] = Field(default_factory=list)
    evidence_validation: list[EvidenceValidationResult] = Field(default_factory=list)
    shape_validation: list[ShapeValidationResult] = Field(default_factory=list)
    validation_results: list[ValidationResult]
    schema_compatibility_results: list[SchemaCompatibilityResult] = Field(default_factory=list)
    v1_v2_comparison: V1V2Comparison | None = None
    v2_v3_comparison: V2V3Comparison | None = None
    telemetry: VerticalSliceTelemetry


class ExtractionClient(Protocol):
    provider: str
    model_name: str

    def extract(
        self,
        *,
        target: TargetSpecification,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> ExtractionResult: ...


class JsonExtractionClient:
    provider = "openai"

    def __init__(self, *, api_key: str, model_name: str) -> None:
        self.api_key = api_key
        self.model_name = model_name

    def extract(
        self,
        *,
        target: TargetSpecification,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> ExtractionResult:
        prompt = _extraction_prompt(target, bundle)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Extract only from the supplied evidence. Return strict JSON only. "
                        "Abstain with insufficient_evidence when unsupported."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
                response_payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            return _model_error(target, self.provider, self.model_name, str(exc))
        try:
            content = response_payload["choices"][0]["message"]["content"]
            usage_payload = response_payload.get("usage", {})
            result = parse_extraction_response(
                target=target,
                bundle=bundle,
                provider=self.provider,
                model_name=self.model_name,
                response_text=str(content),
            )
            result.model_usage.input_tokens = int(usage_payload.get("prompt_tokens", 0) or 0)
            result.model_usage.output_tokens = int(usage_payload.get("completion_tokens", 0) or 0)
            result.model_usage.estimated_cost_usd = estimate_cost_usd(
                self.model_name,
                result.model_usage.input_tokens,
                result.model_usage.output_tokens,
            )
            return result
        except (KeyError, IndexError, TypeError, ValidationError, ValueError) as exc:
            return _model_error(target, self.provider, self.model_name, str(exc))


class AbstainingExtractionClient:
    provider = "mock"
    model_name = "abstaining-mock"

    def extract(
        self,
        *,
        target: TargetSpecification,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> ExtractionResult:
        _ = (bundle, max_output_tokens)
        return ExtractionResult(
            target_row_id=target.target_row_id,
            requirement_id=target.requirement_id,
            status="insufficient_evidence",
            confidence=0,
            model_provider=self.provider,
            model_name=self.model_name,
            ambiguity_or_caveat="Mock client abstained; no hosted LLM call was made.",
        )


class EvidenceFirstVerticalSliceService:
    def __init__(
        self,
        *,
        source_manifest: Path,
        dictionary_path: Path,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        settings: Settings | None = None,
        extraction_client: ExtractionClient | None = None,
        batch_config: BatchWorkerConfig | None = None,
        max_evidence_chars: int = MAX_EVIDENCE_CHARS,
        baseline_dir: Path | None = None,
        v2_baseline_dir: Path | None = None,
    ) -> None:
        self.source_manifest = source_manifest
        self.dictionary_path = dictionary_path
        self.output_dir = output_dir
        self.cache_root = cache_root
        self.settings = settings or load_settings(Path("configs/default.yaml"))
        self.extraction_client = extraction_client or self._build_extraction_client()
        self.batch_config = batch_config or BatchWorkerConfig()
        self.max_evidence_chars = max_evidence_chars
        self.baseline_dir = baseline_dir
        self.v2_baseline_dir = v2_baseline_dir

    def run(self) -> VerticalSliceRunResult:
        started = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        selected_targets = select_vertical_slice_targets(self.dictionary_path, self.output_dir)
        source_registry = {
            source.source_id: source for source in load_source_registry(self.source_manifest)
        }
        source_ranges = select_source_ranges(source_registry)
        pages_by_source, cache_results = self._load_pages(source_registry, source_ranges)
        hierarchy = build_lightweight_hierarchy(pages_by_source, source_registry)
        expectations = build_retrieval_expectations(selected_targets)
        value_shape_assignments = infer_value_shape_assignments(selected_targets)
        retrieval_results: list[RetrievalResult] = []
        bundles: list[EvidenceBundleRecord] = []
        evidence_spans: list[EvidenceSpan] = []
        for selected in selected_targets:
            retrieval = retrieve_evidence(
                selected.target,
                hierarchy,
                pages_by_source,
                source_registry,
                max_evidence_chars=self.max_evidence_chars,
            )
            retrieval_results.append(retrieval)
            bundle = build_evidence_bundle(
                selected.target,
                retrieval,
                self.max_evidence_chars,
                pages_by_source,
            )
            bundles.append(bundle)
            evidence_spans.extend(bundle.evidence_spans)
        retrieval_evaluation = evaluate_retrieval(expectations, retrieval_results)
        extraction_results: list[ExtractionResult] = []
        extraction_time_ms_by_target: dict[str, float] = {}
        for selected, bundle in zip(selected_targets, bundles, strict=True):
            extract_start = time.perf_counter()
            if bundle.retrieval_status in {"no_relevant_evidence", "weak_evidence"}:
                extraction = _insufficient_evidence_result(
                    selected.target,
                    self.extraction_client.provider,
                    self.extraction_client.model_name,
                    f"Retriever returned {bundle.retrieval_status} for canonical evidence.",
                )
            else:
                extraction = self.extraction_client.extract(
                    target=selected.target,
                    bundle=bundle,
                    max_output_tokens=600,
                )
                extraction = materialize_selected_spans(extraction, bundle)
            assignment = next(
                item
                for item in value_shape_assignments
                if item.target_row_id == selected.target.target_row_id
            )
            extraction = calibrate_extraction_value(
                extraction=extraction,
                target=selected.target,
                assignment=assignment,
                bundle=bundle,
            )
            extraction_results.append(extraction)
            extraction_time_ms_by_target[selected.target.target_row_id] = (
                time.perf_counter() - extract_start
            ) * 1000
        normalized_values = [
            NormalizedValueRecord(
                target_row_id=item.target_row_id,
                value_shape_family=next(
                    assignment.value_shape_family
                    for assignment in value_shape_assignments
                    if assignment.target_row_id == item.target_row_id
                ),
                raw_model_value=item.raw_model_value,
                evidence_value=item.evidence_value,
                normalized_value=item.normalized_value,
                display_value=item.display_value,
                unit=item.unit,
                normalization_status=_normalization_status(item),
                issues=[],
            )
            for item in extraction_results
        ]
        evidence_validation = validate_evidence_layer(
            extraction_results,
            selected_targets,
            pages_by_source,
            hierarchy,
            bundles,
        )
        shape_validation = validate_shape_layer(
            extraction_results,
            value_shape_assignments,
        )
        schema_results = assess_schema_compatibility_v3(
            extraction_results,
            selected_targets,
            value_shape_assignments,
            evidence_validation,
            shape_validation,
        )
        validation_results = build_overall_validation_results(
            extraction_results,
            selected_targets,
            evidence_validation,
            shape_validation,
            schema_results,
        )
        comparison = (
            build_v1_v2_comparison(
                self.baseline_dir,
                self.output_dir,
                selected_targets,
                retrieval_evaluation,
                extraction_results,
                validation_results,
                schema_results,
            )
            if self.baseline_dir is not None
            else None
        )
        v2_v3_comparison = (
            build_v2_v3_comparison(
                self.v2_baseline_dir,
                self.output_dir,
                selected_targets,
                value_shape_assignments,
                extraction_results,
                validation_results,
                schema_results,
            )
            if self.v2_baseline_dir is not None
            else None
        )
        telemetry = _build_telemetry(
            source_ranges=source_ranges,
            cache_results=cache_results,
            hierarchy=hierarchy,
            retrieval_results=retrieval_results,
            bundles=bundles,
            extraction_results=extraction_results,
            validation_results=validation_results,
            extraction_time_ms_by_target=extraction_time_ms_by_target,
            provider=self.extraction_client.provider,
            model=self.extraction_client.model_name,
            total_wall_time_ms=(time.perf_counter() - started) * 1000,
        )
        result = VerticalSliceRunResult(
            output_dir=str(self.output_dir),
            selected_targets=selected_targets,
            source_ranges=source_ranges,
            hierarchy=hierarchy,
            retrieval_expectations=expectations,
            retrieval_results=retrieval_results,
            retrieval_evaluation=retrieval_evaluation,
            value_shape_assignments=value_shape_assignments,
            evidence_spans=evidence_spans,
            evidence_bundles=bundles,
            extraction_results=extraction_results,
            normalized_values=normalized_values,
            evidence_validation=evidence_validation,
            shape_validation=shape_validation,
            validation_results=validation_results,
            schema_compatibility_results=schema_results,
            v1_v2_comparison=comparison,
            v2_v3_comparison=v2_v3_comparison,
            telemetry=telemetry,
        )
        write_vertical_slice_artifacts(result, self.output_dir)
        return result

    def _load_pages(
        self,
        source_registry: dict[str, SourceRegistryEntry],
        source_ranges: list[SourceRange],
    ) -> tuple[dict[str, list[ParsedPage]], list[CachedBatchParseResult]]:
        cache_service = CachedBatchParsingService(
            cache=CanonicalParsedPageCache(self.cache_root),
            batch_config=self.batch_config,
        )
        pages_by_source: dict[str, dict[int, ParsedPage]] = defaultdict(dict)
        results: list[CachedBatchParseResult] = []
        for source_range in source_ranges:
            source = source_registry[source_range.source_id]
            for start, end in split_range(source_range.page_start, source_range.page_end):
                result = cache_service.parse(
                    BatchRequest(
                        source=source,
                        source_path=source.original_path,
                        output_dir=str(
                            self.output_dir
                            / "worker_runs"
                            / f"{source.source_id}_{start:04d}_{end:04d}"
                        ),
                        page_start=start,
                        page_end=end,
                    )
                )
                results.append(result)
                for page in result.pages:
                    pages_by_source[page.source_id][page.page_number] = page
        return {
            source_id: [pages[page] for page in sorted(pages)]
            for source_id, pages in sorted(pages_by_source.items())
        }, results

    def _build_extraction_client(self) -> ExtractionClient:
        if not self.settings.hosted_llm_enabled or self.settings.openai_api_key is None:
            return AbstainingExtractionClient()
        model_name = self.settings.text_model_name or "gpt-4o-mini"
        return JsonExtractionClient(
            api_key=self.settings.openai_api_key.get_secret_value(),
            model_name=model_name,
        )


def select_vertical_slice_targets(dictionary_path: Path, output_dir: Path) -> list[SelectedTarget]:
    dictionary_result = ingest_dictionary(
        dictionary_path,
        sheet_name="Extraction Template",
        mapping_config=Path("configs/dictionaries/segro_extraction_template_v1.yaml"),
        output_dir=output_dir / "dictionary_selection_ingestion",
    )
    by_field = {target.expected_field: target for target in dictionary_result.normalized_targets}
    target_fields = [
        ("frame_primary_construction_type", "Building frame type is text-evidenced in Part 1."),
        ("frame_construction_description", "Building description covers steel frame warehouse."),
        ("roof_construction_description", "Roof construction appears in description and drawings."),
        (
            "external_cladding_description",
            "External cladding appears in building description/elevations.",
        ),
        (
            "wall_construction_description",
            "Wall construction appears in building description/elevations.",
        ),
        ("dock_leveller_count", "Dock leveller count appears in Part 1 plan text."),
        ("pv_panel_component_description", "PV evidence appears in roof plan/maintenance text."),
        (
            "floor_construction_description",
            "Concrete floor/slab evidence appears in external works text.",
        ),
        (
            "fire_alarm_system_description",
            "Fire alarm evidence appears in bounded Part 6 safety text.",
        ),
        ("approved_use_classes", "Planning consent page includes approved industrial use classes."),
        (
            "landlord_planning_consent_obligations",
            "Planning approval conditions are present in Part 1.",
        ),
        (
            "pv_certification_component_description",
            "PV/certificate evidence is expected in Part 6 appendix index.",
        ),
        (
            "construction_date",
            "Construction/practical completion evidence appears in Part 1 certificates.",
        ),
        (
            "office_area_value",
            "Office area is a representative size target; bounded evidence may be absent.",
        ),
    ]
    selected: list[SelectedTarget] = []
    missing: list[str] = []
    for expected_field, reason in target_fields:
        target = by_field.get(expected_field)
        if target is None:
            missing.append(expected_field)
            continue
        selected.append(SelectedTarget(target=target, selection_reason=reason))
    if len(selected) < 12:
        msg = f"Only {len(selected)} vertical-slice targets found; missing {missing}"
        raise ValueError(msg)
    return selected[:15]


def select_source_ranges(
    source_registry: dict[str, SourceRegistryEntry],
) -> list[SourceRange]:
    by_path = {source.logical_path: source for source in source_registry.values()}
    desired = [
        (
            "Building Manual - Part 1 General.pdf",
            4,
            11,
            "Index, building description, plans/elevations, roof/PV/dock plan text.",
        ),
        (
            "Building Manual - Part 1 General.pdf",
            14,
            22,
            "Planning approval notice and approved use classes.",
        ),
        (
            "Building Manual - Part 1 General.pdf",
            41,
            46,
            "Building regulation final certificate and practical completion certificate.",
        ),
        (
            "Building Manual - Part 4 External Works.pdf",
            4,
            14,
            "External works index and concrete yard/slab specification.",
        ),
        (
            "Building Manual - Part 5 The Health & Safety File.pdf",
            1,
            10,
            "Health and safety file pages covering fire strategy/access/maintenance summary.",
        ),
        (
            "Building Manual - Part 6 Appendices.pdf",
            4,
            10,
            "Appendices index, maintenance recommendations, PV and certificate headings.",
        ),
    ]
    ranges: list[SourceRange] = []
    for logical_path, start, end, reason in desired:
        source = by_path[logical_path]
        ranges.append(
            SourceRange(
                source_id=source.source_id,
                logical_path=logical_path,
                page_start=start,
                page_end=end,
                reason=reason,
            )
        )
    return ranges


def split_range(start: int, end: int, max_pages: int = MAX_BATCH_PAGES) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    current = start
    while current <= end:
        range_end = min(end, current + max_pages - 1)
        ranges.append((current, range_end))
        current = range_end + 1
    return ranges


def build_lightweight_hierarchy(
    pages_by_source: dict[str, list[ParsedPage]],
    source_registry: dict[str, SourceRegistryEntry],
) -> LightweightHierarchy:
    nodes: list[HierarchyNode] = []
    child_map: dict[str, list[str]] = defaultdict(list)
    for source_id, pages in sorted(pages_by_source.items()):
        source = source_registry[source_id]
        doc_id = stable_node_id(source_id, "document", 1, 1, source.logical_path)
        doc_node = HierarchyNode(
            node_id=doc_id,
            source_id=source_id,
            node_type="document",
            title=source.logical_path,
            page_start=min(page.page_number for page in pages),
            page_end=max(page.page_number for page in pages),
            text_summary=source.logical_path,
            provenance=[ProvenanceRef(source_id=source_id, notes="source_registry")],
        )
        nodes.append(doc_node)
        current_section_id = doc_id
        for page in sorted(pages, key=lambda item: item.page_number):
            text = page.text or ""
            heading = infer_heading(text, page.page_number, source.logical_path)
            section_id = stable_node_id(
                source_id, "section", page.page_number, page.page_number, heading
            )
            section_node = HierarchyNode(
                node_id=section_id,
                source_id=source_id,
                node_type="section",
                title=heading,
                parent_node_id=doc_id,
                page_start=page.page_number,
                page_end=page.page_number,
                text_summary=bounded_text(text, 350),
                provenance=[
                    ProvenanceRef(
                        source_id=source_id,
                        page_or_sheet=f"page-{page.page_number}",
                        notes="heading inference",
                    )
                ],
            )
            nodes.append(section_node)
            child_map[doc_id].append(section_id)
            current_section_id = section_id
            page_id = stable_node_id(
                source_id, "page", page.page_number, page.page_number, str(page.page_number)
            )
            page_node = HierarchyNode(
                node_id=page_id,
                source_id=source_id,
                node_type="page",
                title=f"Page {page.page_number}",
                parent_node_id=current_section_id,
                page_start=page.page_number,
                page_end=page.page_number,
                text_summary=bounded_text(text, 450),
                provenance=[
                    ProvenanceRef(
                        source_id=source_id,
                        page_or_sheet=f"page-{page.page_number}",
                        text_ref=page.text_ref,
                    )
                ],
            )
            nodes.append(page_node)
            child_map[current_section_id].append(page_id)
            block_id = stable_node_id(
                source_id, "text_block", page.page_number, page.page_number, text[:80]
            )
            block_node = HierarchyNode(
                node_id=block_id,
                source_id=source_id,
                node_type="text_block",
                title=f"{heading} text",
                parent_node_id=page_id,
                page_start=page.page_number,
                page_end=page.page_number,
                text_summary=bounded_text(text, 2200),
                provenance=[
                    ProvenanceRef(
                        source_id=source_id,
                        node_id=page_id,
                        page_or_sheet=f"page-{page.page_number}",
                    )
                ],
            )
            nodes.append(block_node)
            child_map[page_id].append(block_id)
    hydrated = [
        node.model_copy(update={"child_node_ids": sorted(child_map.get(node.node_id, []))})
        for node in nodes
    ]
    hierarchy_id = hashlib.sha256(
        json.dumps([node.model_dump(mode="json") for node in hydrated], sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    return LightweightHierarchy(hierarchy_id=f"hier_{hierarchy_id}", nodes=hydrated)


def to_evidence_index(hierarchy: LightweightHierarchy, source_pack_id: str) -> EvidenceIndex:
    converted = [
        EvidenceIndexNode(
            node_id=node.node_id,
            parent_id=node.parent_node_id,
            source_id=node.source_id,
            node_type=EvidenceNodeType(node.node_type),
            title=node.title,
            page_or_sheet_range=PageSheetRange(start=node.page_start, end=node.page_end),
            summary=node.text_summary,
            provenance=node.provenance,
        )
        for node in hierarchy.nodes
    ]
    return EvidenceIndex(
        index_id=hierarchy.hierarchy_id, source_pack_id=source_pack_id, nodes=converted
    )


def infer_heading(text: str, page_number: int, logical_path: str) -> str:
    lines = [line.strip(" :\t") for line in text.splitlines() if line.strip()]
    joined = " ".join(lines[:12])
    patterns = [
        r"(\d+\.\d+(?:\.\d+)?\s*-\s*[A-Z][A-Z0-9 &/()'.,-]+)",
        r"(ELEMENT:\s*\d+\.\d+\.\d+\s+[A-Z0-9 &/()'.,-]+)",
        r"(Certificate of Practical Completion)",
        r"(Final Certificate)",
        r"(PLANNING GRANTED)",
        r"(PART \d+\s*-\s*INDEX\s*-\s*[A-Z ]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_space(match.group(1))[:160]
    if lines:
        return normalize_space(joined)[:160]
    return f"{logical_path} page {page_number}"


def stable_node_id(
    source_id: str,
    node_type: str,
    page_start: int,
    page_end: int,
    label: str,
) -> str:
    raw = {
        "source_id": source_id,
        "node_type": node_type,
        "page_start": page_start,
        "page_end": page_end,
        "label": normalize_space(label).lower(),
        "version": VERTICAL_SLICE_VERSION,
    }
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return f"vs_{digest}"


def retrieve_evidence(
    target: TargetSpecification,
    hierarchy: LightweightHierarchy,
    pages_by_source: dict[str, list[ParsedPage]],
    source_registry: dict[str, SourceRegistryEntry],
    *,
    max_evidence_chars: int,
) -> RetrievalResult:
    started = time.perf_counter()
    concepts = query_concepts_for_target(target)
    query_terms = query_terms_for_concepts(concepts)
    page_text = {
        (page.source_id, page.page_number): page.text or ""
        for pages in pages_by_source.values()
        for page in pages
    }
    node_by_id = {node.node_id: node for node in hierarchy.nodes}
    scores: list[RetrievedEvidence] = []
    for node in hierarchy.nodes:
        if node.node_type not in {"section", "page", "text_block"}:
            continue
        source_text = page_text.get((node.source_id, node.page_start), node.text_summary)
        haystack = f"{node.title}\n{node.text_summary}".lower()
        full_page_lower = source_text.lower()
        title = node.title.lower()
        components = score_node_for_target(target, concepts, node, haystack, title)
        score = components.final_score
        if score <= 0:
            continue
        source = source_registry[node.source_id]
        matched = sorted(
            {
                term
                for term in query_terms
                if term and (term in haystack or term in full_page_lower)
            }
        )
        excerpt = best_excerpt(
            source_text, query_terms, max_chars=min(900, max_evidence_chars)
        )
        scores.append(
            RetrievedEvidence(
                target_row_id=target.target_row_id,
                rank=0,
                node_id=node.node_id,
                source_id=node.source_id,
                source_file=source.logical_path,
                page_start=node.page_start,
                page_end=node.page_end,
                score=round(score, 4),
                score_components=components,
                matched_terms=matched[:20],
                hierarchy_path=hierarchy_path(node_by_id, node.node_id),
                excerpt=excerpt,
            )
        )
    ordered = sorted(
        scores, key=lambda item: (-item.score, item.source_file, item.page_start, item.node_id)
    )
    deduped = deduplicate_overlapping_retrievals(ordered)
    deduped = prefer_explicit_quantity_evidence(target, deduped, page_text)
    strong = [item for item in deduped if item.score >= MIN_RELEVANCE_SCORE]
    if strong:
        selected = strong[:3]
        status: Literal["evidence_found", "weak_evidence", "no_relevant_evidence"] = (
            "evidence_found"
        )
    elif deduped and deduped[0].score >= WEAK_RELEVANCE_SCORE:
        selected = []
        status = "weak_evidence"
    else:
        selected = []
        status = "no_relevant_evidence"
    ranked = [item.model_copy(update={"rank": rank}) for rank, item in enumerate(selected, start=1)]
    return RetrievalResult(
        target_row_id=target.target_row_id,
        query=" ".join(query_terms),
        retrieval_status=status,
        query_concepts=concepts,
        results=ranked,
        retrieval_time_ms=(time.perf_counter() - started) * 1000,
        top_score=deduped[0].score if deduped else 0,
    )


def prefer_explicit_quantity_evidence(
    target: TargetSpecification,
    results: list[RetrievedEvidence],
    page_text: dict[tuple[str, int], str],
) -> list[RetrievedEvidence]:
    field = target.expected_field.lower()
    quantity_target = target.expected_data_type == ExpectedDataType.INTEGER or any(
        term in field for term in ["count", "quantity"]
    )
    if not quantity_target:
        return results
    explicit_installed = [
        item
        for item in results
        if "number installed" in page_text.get((item.source_id, item.page_start), "").lower()
    ]
    if explicit_installed:
        return explicit_installed
    explicit_quantity = [
        item
        for item in results
        if re.search(
            r"\b(?:qty|quantity|count)\b|\b\d+\s*no\.?\b",
            page_text.get((item.source_id, item.page_start), "").lower(),
        )
    ]
    return explicit_quantity or results


def query_concepts_for_target(target: TargetSpecification) -> QueryConcepts:
    raw = " ".join(
        [
            target.expected_field,
            target.requirement_text,
            target.sub_domain,
            target.component_type or "",
            target.component_subtype or "",
            target.unit or "",
        ]
    )
    field_tokens = tokenize(target.expected_field.replace("_", " "))
    raw_tokens = set(tokenize(raw))
    attribute_tokens = {
        token
        for token in field_tokens
        if token
        in {
            "area",
            "capacity",
            "category",
            "classes",
            "colour",
            "consent",
            "count",
            "date",
            "description",
            "finish",
            "installed",
            "installation",
            "loading",
            "manufacturer",
            "model",
            "number",
            "quantity",
            "reference",
            "type",
            "unit",
            "use",
            "value",
        }
    }
    phrase_map = {
        "dock_leveller": ["dock leveller", "dock levellers"],
        "ev_charger": ["ev charger", "ev charging", "electric vehicle charging"],
        "pv": ["photovoltaic", "pv", "solar panel", "solar panels"],
        "cladding": ["cladding", "profiled metal clad", "wall cladding"],
        "roof": ["roof", "rooflights", "roof construction", "roofing system", "roof finish"],
        "frame": ["steel frame", "portal frame", "frame"],
        "planning": ["planning", "permission", "approved use", "use classes"],
        "fire_alarm": [
            "fire alarm",
            "fire alarms",
            "fire detection",
            "fire detection and alarm system",
        ],
        "floor": ["floor slab", "concrete slab", "floor construction"],
        "completion": ["practical completion", "completion certificate"],
        "building_control": ["building control", "final certificate", "assent ref"],
        "construction_date": ["practical completion", "construction", "certificate"],
        "office_area": ["office area", "office floor area", "offices"],
        "wall": ["wall construction", "external wall", "wall cladding"],
        "hard_landscaping": ["hard landscaping", "macadam surfacing", "concrete surfacing"],
        "external_yard": ["external yard", "yard concrete", "concrete slabs"],
        "soft_landscaping": ["soft landscaping", "grass seeding", "turfing", "trees", "shrubs"],
        "drainage": ["drainage", "stormwater", "attenuation tanks", "aco road", "qmax"],
        "bollard": ["bollard", "bollards", "telescopic bollard"],
        "fencing": ["fencing", "fence", "mesh panel", "gates"],
        "gate": ["gate", "gates", "manual sliding", "swing gates"],
        "cycle_shelter": ["cycle shelter", "cycle shelters", "circonomy"],
        "barrier": ["barrier", "armco barrier", "handrail"],
        "line_marking": ["line marking", "line markings", "road markings", "thermoplastic"],
        "retaining_wall": ["retaining wall", "precast", "concrete pour"],
        "bin_store": ["bin store", "perforated metal sheet", "planters"],
        "external_steps": ["external steps", "steel stairs", "durbar tread"],
        "roof_access": ["roof access", "cat ladder", "roof hatch", "man-safe", "fall restraint"],
        "access_method": ["access method", "cat ladder", "roof hatch", "mewp"],
        "structural_frame": ["structural frame", "steel frame", "s355"],
        "foundation": [
            "foundation",
            "piled foundation",
            "piled solution",
            "pile caps",
            "ground beams",
            "piles",
        ],
        "warehouse_floor_loading": ["warehouse slab", "live warehouse 50kn/m2", "imposed load"],
        "office_floor_loading": ["office slab", "7.5kn/m2", "imposed load"],
        "concrete_grade": ["concrete grade", "c32/40", "dc3", "dc-1"],
        "asbestos": ["asbestos", "asbestos containing products", "asbestos statement"],
        "emergency_utility": [
            "emergency contact",
            "emergency number",
            "national grid",
            "uk power networks",
            "thames water",
        ],
        "fire_door": ["fire door", "fd60", "fd30", "fire rating"],
        "fire_rating": ["fd60", "fd30", "fire resistance", "fire compartment"],
        "disabled_refuge": ["disabled refuge", "emergency voice", "evc", "bs5839-9", "bs 5839-9"],
        "evc": ["emergency voice", "evc", "installation certificate", "bs 5839-9"],
        "pv_inverter": ["pv commissioning form", "inverter", "solis", "photovoltaic", "unit 1"],
        "pv_module": ["solar pv", "pv commissioning form", "array module", "modules"],
        "pv_system_capacity": ["solar pv", "system consists", "modules", "inverters", "kwh", "kw"],
        "air_conditioning": ["air conditioning", "model number", "unit serial number"],
        "outdoor_unit": ["outdoor unit", "model number", "unit serial number", "pury-p"],
        "indoor_unit": ["indoor unit", "model number", "unit serial number", "location"],
        "cold_water_booster": ["cold water", "booster", "model no", "serial number", "commissioning report"],
        "lift": ["lift no", "owner documentation", "lift"],
        "electrical_distribution": ["mccb switch panel", "distribution board", "mp1"],
        "mechanical_contractor": ["installations completed by", "company", "contractor"],
        "bms_controller": ["bms", "outstation", "trend iq", "points schedule", "controller"],
        "wc_extract_fan": ["wc extract fan", "measured volume", "design volume", "fan"],
        "mechanical_fan": ["fan", "measured volume", "design volume", "commissioning report"],
        "water_test": ["water", "test report", "certificate of conformity", "sample date", "unit 1"],
        "horizontal_lifeline": ["horizontal lifeline", "lifeline system", "soter", "system length", "test loads"],
    }
    field = target.expected_field.lower()
    component_phrases: set[str] = set()
    for key, values in phrase_map.items():
        if key in field or key in raw.lower():
            component_phrases.update(values)
    component_tokens = raw_tokens - attribute_tokens
    domain_tokens = set(
        tokenize(
            f"{target.sub_domain} {target.component_type or ''} "
            f"{target.component_subtype or ''}"
        )
    )
    datatype_hints = {str(target.expected_data_type)}
    if target.unit:
        datatype_hints.add(target.unit)
    indicators: set[str] = set()
    if target.expected_data_type == ExpectedDataType.DATE:
        indicators.update({"date", "dated", "certificate"})
    if target.expected_data_type == ExpectedDataType.INTEGER:
        indicators.update({"no", "number", "quantity"})
    if target.expected_data_type == ExpectedDataType.DECIMAL:
        indicators.update({"area", "sqm", "m2", "m²", "sq m"})
    if "certificate" in raw.lower():
        indicators.add("certificate")
    if "planning" in raw.lower():
        indicators.update({"planning", "reference", "permission"})
    if "fire_alarm" in field:
        attribute_tokens.update({"system", "systems", "category"})
        indicators.update({"certificate", "certification", "annually", "systems"})
    negative_terms: set[str] = set()
    if field == "office_area_value":
        negative_terms.update(
            {
                "concrete",
                "slab",
                "dock",
                "leveller",
                "roof",
                "pv",
                "photovoltaic",
                "fire",
                "external",
                "yard",
            }
        )
    elif "fire_alarm" in field:
        negative_terms.update({"fire door", "fire strategy", "fire stopping"})
    elif "dock_leveller" in field:
        negative_terms.update({"photovoltaic", "pv", "maintenance", "roof"})
    return QueryConcepts(
        exact_field_phrase=target.expected_field.replace("_", " "),
        normalized_field_tokens=sorted(token for token in field_tokens if len(token) > 2),
        component_phrases=sorted(component_phrases),
        component_tokens=sorted(token for token in component_tokens if len(token) > 2),
        attribute_tokens=sorted(attribute_tokens),
        domain_tokens=sorted(token for token in domain_tokens if len(token) > 2),
        datatype_unit_hints=sorted(datatype_hints),
        certificate_date_reference_indicators=sorted(indicators),
        negative_terms=sorted(negative_terms),
    )


def query_terms_for_concepts(concepts: QueryConcepts) -> list[str]:
    terms = set(concepts.normalized_field_tokens)
    terms.update(concepts.component_phrases)
    terms.update(concepts.component_tokens)
    terms.update(concepts.attribute_tokens)
    terms.update(concepts.domain_tokens)
    terms.update(concepts.datatype_unit_hints)
    terms.update(concepts.certificate_date_reference_indicators)
    return sorted(term for term in terms if len(term) > 1)


def query_terms_for_target(target: TargetSpecification) -> list[str]:
    return query_terms_for_concepts(query_concepts_for_target(target))


def score_node_for_target(
    target: TargetSpecification,
    concepts: QueryConcepts,
    node: HierarchyNode,
    haystack: str,
    title: str,
) -> RetrievalScoreBreakdown:
    hay_tokens = set(tokenize(haystack))
    component_hits = [phrase for phrase in concepts.component_phrases if phrase in haystack]
    field_phrase = concepts.exact_field_phrase.lower()
    components = RetrievalScoreBreakdown()
    components.exact_phrase = 4.0 * len(component_hits)
    if field_phrase and field_phrase in haystack:
        components.exact_phrase += 2.0
    meaningful_tokens = set(concepts.component_tokens) | set(concepts.attribute_tokens)
    generic = {"description", "area", "system", "type", "value"}
    components.token_overlap = 0.45 * len((hay_tokens & meaningful_tokens) - generic)
    components.title_match = 4.0 * len([phrase for phrase in component_hits if phrase in title])
    components.title_match += 1.5 * len(set(concepts.attribute_tokens) & set(tokenize(title)))
    components.domain_match = 0.35 * len(hay_tokens & set(concepts.domain_tokens))
    components.domain_subdomain_match = components.domain_match
    components.pattern_match = pattern_score(target, haystack)
    components.datatype_unit_pattern = datatype_unit_score(target, concepts, haystack)
    components.certificate_date_pattern = certificate_date_reference_score(
        target, concepts, haystack
    )
    components.component_attribute_proximity = proximity_score(
        haystack,
        concepts.component_phrases,
        [*concepts.attribute_tokens, *concepts.certificate_date_reference_indicators],
    )
    components.hierarchy_proximity = 0.6 if node.node_type in {"section", "text_block"} else 0.25
    components.negative_penalty = negative_penalty(concepts, haystack, component_hits)
    components.status_context_penalty = status_context_penalty(target, haystack)
    components.cross_reference_penalty = cross_reference_penalty(target, haystack)
    raw_score = (
        components.exact_phrase
        + components.token_overlap
        + components.title_match
        + components.pattern_match
        + components.hierarchy_proximity
        + components.component_attribute_proximity
        + components.datatype_unit_pattern
        + components.certificate_date_pattern
        + components.domain_subdomain_match
        - components.negative_penalty
        - components.status_context_penalty
        - components.cross_reference_penalty
    )
    if not component_hits and components.title_match == 0:
        raw_score *= 0.55
    components.final_score = round(max(0.0, raw_score), 4)
    return components


def pattern_score(target: TargetSpecification, text: str) -> float:
    score = 0.0
    field = target.expected_field.lower()
    if target.expected_data_type == ExpectedDataType.DATE and re.search(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+\s+\d{4}\b", text
    ):
        score += 2.0
    if target.expected_data_type in {
        ExpectedDataType.INTEGER,
        ExpectedDataType.DECIMAL,
    } and re.search(r"\b\d+(?:\.\d+)?\b", text):
        score += 1.5
    if "certificate" in target.requirement_text.lower() and "certificate" in text:
        score += 2.0
    if "planning" in target.sub_domain.lower() and "planning" in text:
        score += 2.0
    if "commissioning" in field and "certificate" in text and "commissioning" in text:
        score += 3.0
    if "pv" in field and "unit 1" in text and any(term in text for term in ["pv commissioning form", "solar pv certificate"]):
        score += 4.0
    if "inverter" in field and any(term in text for term in ["inverter", "solis"]):
        score += 3.0
    if "module" in field and any(term in text for term in ["modules", "array module", "275wp"]):
        score += 3.0
    if "air_conditioning" in field and "unit 1" in text and any(
        term in text for term in ["model number", "unit serial number", "serial no"]
    ):
        score += 4.0
    if "bms" in field and any(term in text for term in ["trend iq", "points schedule", "outstation"]):
        score += 4.0
    if "fan" in field and any(term in text for term in ["measured volume", "design volume"]):
        score += 4.0
    if "fan" in field and "system title:" in text and any(
        term in text for term in ["measured volume", "fan total pressure", "commissioning engineer"]
    ):
        score += 5.0
    if "booster" in field and any(term in text for term in ["model no", "serial number"]):
        score += 6.0
    if "hot_water" in field and any(
        term in text for term in ["thermodynamic system", "domestic hot water", "water heater"]
    ):
        score += 4.0
    if "lift" in field and any(term in text for term in ["lift no", "owner documentation"]):
        score += 6.0
    if "distribution" in field and any(term in text for term in ["mccb", "switch panel", "distribution board"]):
        score += 5.0
    if "contractor" in field and any(
        term in text for term in ["installations completed by", "company", "contractor"]
    ):
        score += 8.0
    if "indoor" in field and "model" in field and any(
        term in text for term in ["first letters of model", "serial location", "indoor unit"]
    ):
        score += 8.0
    if "outdoor" in field and "model" in field and any(
        term in text for term in ["outdoor unit", "system model", "model number"]
    ):
        score += 6.0
    if "pv" in field and any(term in field for term in ["quantity", "count"]) and "number installed" in text:
        score += 14.0
    if "pv" in field and "model" in field and "system installed" in text and any(
        term in text for term in ["manufacturer", "model"]
    ):
        score += 6.0
    if "water" in field and any(term in text for term in ["certificate of conformity", "sample date", "test report"]):
        score += 4.0
    if "lifeline" in field and any(term in text for term in ["horizontal lifeline", "system length", "test loads"]):
        score += 4.0
    return score


def datatype_unit_score(
    target: TargetSpecification,
    concepts: QueryConcepts,
    text: str,
) -> float:
    score = 0.0
    if target.expected_data_type == ExpectedDataType.INTEGER and re.search(
        r"\b\d+\s*(?:no\.?|number|dock|leveller)", text
    ):
        if any(
            phrase in text[max(0, text.find(phrase) - 80) : text.find(phrase) + 80]
            and re.search(
                r"\b\d+\s*(?:no\.?|number|dock|leveller)",
                text[max(0, text.find(phrase) - 80) : text.find(phrase) + 80],
            )
            for phrase in concepts.component_phrases
            if text.find(phrase) >= 0
        ):
            score += 2.0
    if target.expected_data_type == ExpectedDataType.DECIMAL and re.search(
        r"\b\d+(?:\.\d+)?\s*(?:m2|m²|sqm|sq\.?\s*m|kwh?|kwp?)\b", text
    ):
        score += 2.0
    for hint in concepts.datatype_unit_hints:
        if hint and hint.lower() in text:
            score += 0.5
    return score


def certificate_date_reference_score(
    target: TargetSpecification,
    concepts: QueryConcepts,
    text: str,
) -> float:
    score = 0.0
    if set(concepts.certificate_date_reference_indicators) & set(tokenize(text)):
        score += 0.75
    if target.expected_data_type == ExpectedDataType.DATE and re.search(
        r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+\s+\d{4}\b",
        text,
    ):
        score += 2.0
    if "planning" in target.expected_field and re.search(r"\b(?:ref|reference|planning)\b", text):
        score += 1.5
    return score


def proximity_score(
    text: str,
    component_phrases: Sequence[str],
    attribute_terms: Sequence[str],
) -> float:
    if not component_phrases or not attribute_terms:
        return 0.0
    score = 0.0
    lines = [line.strip().lower() for line in re.split(r"[\r\n]+", text) if line.strip()]
    if not lines:
        lines = [text.lower()]
    for line in lines:
        component_present = any(phrase in line for phrase in component_phrases)
        attribute_present = any(term in line or term in tokenize(line) for term in attribute_terms)
        if component_present and attribute_present:
            score += 4.0
            break
    flat = normalize_space(text.lower())
    for phrase in component_phrases:
        start = flat.find(phrase)
        if start < 0:
            continue
        window = flat[max(0, start - 100) : start + len(phrase) + 100]
        if any(term in window for term in attribute_terms):
            score += 2.0
            break
    return score


def negative_penalty(
    concepts: QueryConcepts,
    text: str,
    component_hits: Sequence[str],
) -> float:
    if not concepts.negative_terms:
        return 0.0
    lower = text.lower()
    penalty = 1.5 * len([term for term in concepts.negative_terms if term in lower])
    if component_hits:
        penalty *= 0.45
    return penalty


def status_context_penalty(target: TargetSpecification, text: str) -> float:
    field = target.expected_field.lower()
    requirement = target.requirement_text.lower()
    lower = text.lower()
    planning_context = any(
        term in lower
        for term in [
            "planning granted",
            "local planning authority",
            "planning permission",
            "hereby approved",
            "shall be submitted",
            "shall be installed",
            "prior to occupation",
            "prior to superstructure",
        ]
    )
    certificate_context = any(
        term in lower
        for term in [
            "building control",
            "final certificate",
            "practical completion",
            "completion certificate",
            "commissioning certificate",
            "certificate number",
            "air permeability test certificate",
        ]
    )
    maintenance_context = any(
        term in lower
        for term in [
            "maintenance frequency",
            "planned maintenance",
            "planned cleaning",
            "maintenance recommendations",
            "periodically",
            "inspect",
            "clean annually",
            "cleaning procedures",
        ]
    )
    generic_reference_context = any(
        term in lower
        for term in [
            "product data sheet",
            "technical data sheet",
            "data sheet",
            "installation manual",
            "operation manual",
            "user manual",
            "safety data sheet",
            "material safety data",
            "coshh assessment",
            "limited warranty certificate",
            "ce declaration of conformity",
            "declaration of performance",
            "product conformity certification",
            "terms & conditions",
            "terms and conditions",
            "goods returns policy",
            "paving maintenance & repair guide",
        ]
    )
    residual_hazard_context = any(
        term in lower
        for term in [
            "remaining identified hazard",
            "proposed control measure",
            "risk assessment / method statement",
            "work permit to be issued",
        ]
    )
    emergency_contact_context = any(
        term in lower
        for term in [
            "emergency contacts",
            "emergency number",
            "gas leak",
            "uk power networks",
            "national grid",
            "thames water",
        ]
    )
    generic_material_list_context = (
        "hazardous materials used in construction" in lower
        and "common material types" in lower
    )
    fire_strategy_standard_context = any(
        term in lower
        for term in ["bs 5839", "refer to m&e", "refer to m & e", "fire alarm detection"]
    )
    certificate_index_context = (
        "certificates from the following companies are included" in lower
        or ("part 6 - index" in lower and "appendices" in lower)
    )
    work_permit_template_context = (
        "work permit" in lower
        and "valid for day of issue only" in lower
        and "nature of work" in lower
    )
    design_certificate_context = "design certificate" in lower or "certificate of design" in lower
    laboratory_report_context = any(
        term in lower for term in ["als environmental", "test report:", "date of issue"]
    )
    drawing_context = any(
        term in lower
        for term in [
            "drawing number",
            "sheet number",
            "scale drawn checked approved",
            "do not scale",
            "revision",
        ]
    )
    generic_guarantee_context = (
        ("guarantee" in lower or "guarantee ref" in lower)
        and any(term in lower for term in ["inspection", "maintenance", "appendix"])
    )
    product_guarantee_context = any(
        term in lower
        for term in [
            "guarantee ref",
            "this guarantee is given",
            "competent inspector",
            "u-value",
        ]
    )
    other_unit_context = bool(re.search(r"\bunit\s+[23]\b", lower)) and not bool(
        re.search(r"\bunit\s+1\b", lower)
    )
    unit1_context = bool(re.search(r"\bunit\s+1\b", lower))
    project_installation_context = unit1_context and any(
        term in lower
        for term in [
            "system installed",
            "supplied and installed",
            "commissioning certificate",
            "commissioning report",
            "installation certificate",
            "site:",
            "project:",
            "contract title:",
        ]
    )
    pv_string_reading_context = "pv commissioning form" in lower and any(
        term in lower for term in ["voc", "isc", "string", "array insulation"]
    )
    electrical_measurement_context = any(
        term in lower
        for term in [
            "voc",
            "isc",
            "test voltage",
            "array insulation",
            "irradiance",
            "meter reading",
            "earth continuity",
        ]
    )
    bms_point_context = "points schedule" in lower and any(
        term in lower for term in ["fault", "status", "input device", "output ref"]
    )
    report_or_drawing_date_context = any(
        term in lower
        for term in [
            "calcs date",
            "checked date",
            "approved date",
            "revision",
            "rev.",
            "drawn",
            "checked by",
            "issued for construction",
        ]
    )
    supplier_contact_context = any(
        term in lower
        for term in [
            "directory of suppliers",
            "supplier:",
            "telephone:",
            "fax no",
            "email",
            "company",
        ]
    ) and not any(
        term in lower
        for term in ["nature of installation", "product description", "work description"]
    )
    navigation_context = any(
        term in lower for term in ["part 4 - index", "part 5 - index", "building manual index", "contents"]
    )
    installed_identity_target = any(
        term in field for term in ["model", "manufacturer", "serial"]
    ) and any(term in field for term in ["installed", "installation", "equipment", "charger"])
    identity_target = any(term in field for term in ["model", "manufacturer", "serial"])
    installed_description_target = any(
        term in field for term in ["installed", "installation", "description", "type", "finish"]
    )
    installed_quantity_target = any(term in field for term in ["count", "quantity"]) or (
        "number" in field and "model_number" not in field and "serial_number" not in field
    )
    unit1_requested = "unit1" in field.replace("_", "") or "unit 1" in requirement
    installation_date_target = "installation_date" in field or (
        "equipment" in field and "date" in field
    )
    date_target = target.expected_data_type == ExpectedDataType.DATE
    direct_value_target = any(
        term in field
        for term in ["date", "model", "count", "quantity", "description", "type", "finish"]
    )
    if navigation_context and direct_value_target:
        return 20.0
    if certificate_index_context and direct_value_target:
        return 18.0
    if unit1_requested and other_unit_context:
        return 24.0
    if work_permit_template_context and direct_value_target:
        return 24.0
    if target.expected_data_type == ExpectedDataType.DATE and work_permit_template_context:
        return 32.0
    if date_target and "permit" in field and not work_permit_template_context:
        return 28.0
    if "commissioning_date" in field and design_certificate_context:
        return 24.0
    if "fire_alarm" in field and "commissioning_date" in field and "disabled refuge" in lower:
        return 30.0
    if "indoor" in field and "model" in field and "outdoor unit" in lower and not any(
        term in lower for term in ["first letters of model", "serial location", "indoor unit"]
    ):
        return 30.0
    if date_target and laboratory_report_context and "water" not in field:
        return 24.0
    if installation_date_target and laboratory_report_context:
        return 24.0
    if date_target and not re.search(
        r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+\s+\d{4}\b",
        lower,
    ):
        return 24.0
    if date_target and drawing_context and "commissioning" not in lower:
        return 24.0
    if installation_date_target and drawing_context:
        return 28.0
    if installed_description_target and (generic_guarantee_context or product_guarantee_context):
        return 24.0
    if "pv" in field and any(term in field for term in ["quantity", "count"]) and "number installed" not in lower:
        return 50.0
    if installed_quantity_target and maintenance_context:
        return 16.0
    if installed_identity_target and planning_context:
        return 8.0
    if installed_identity_target and residual_hazard_context:
        return 12.0
    if installation_date_target and (planning_context or certificate_context):
        return 8.0
    if installation_date_target and report_or_drawing_date_context:
        return 24.0
    if installation_date_target and not any(
        term in lower for term in ["installation date", "date installed", "installed on"]
    ):
        return 44.0
    if installation_date_target and maintenance_context:
        return 8.0
    if installation_date_target and not re.search(
        r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+\s+\d{4}\b",
        lower,
    ):
        return 16.0
    if identity_target and pv_string_reading_context and any(
        term in requirement for term in ["string", "voltage", "current", "measurement"]
    ):
        return 36.0
    if identity_target and "pv" in field and electrical_measurement_context and any(
        term in requirement for term in ["string", "voltage", "current", "measurement"]
    ):
        return 40.0
    if identity_target and "pv" in field and any(
        term in requirement for term in ["string", "voltage", "current", "measurement"]
    ):
        return 28.0
    if identity_target and electrical_measurement_context and not any(
        term in lower for term in ["model", "model no", "model number", "serial no"]
    ):
        return 28.0
    if identity_target and supplier_contact_context:
        return 12.0
    if identity_target and emergency_contact_context:
        return 20.0
    if "model" in field and certificate_context and not any(
        term in lower for term in ["model", "model no", "model number", "type no"]
    ):
        return 32.0
    if "model" in field and not any(
        term in lower
        for term in ["model", "model no", "model number", "type no", "serial", "reference", "lift no"]
    ):
        return 30.0
    if identity_target and generic_reference_context and not project_installation_context:
        return 32.0
    if installed_identity_target and generic_reference_context:
        return 24.0
    if installed_quantity_target and generic_reference_context and not project_installation_context:
        return 24.0 if any(term in lower for term in ["safety data sheet", "coshh"]) else 7.0
    if target.expected_data_type == ExpectedDataType.INTEGER and installed_quantity_target and not re.search(
        r"\b\d+\s*(?:no\.?|number|qty|quantity|installed|cycles|spaces|trees|bollards?)\b|\bnumber\s+installed\b",
        lower,
    ):
        return 24.0
    if installed_description_target and generic_reference_context and not project_installation_context:
        return 20.0
    if installed_description_target and generic_material_list_context:
        return 16.0
    if identity_target and fire_strategy_standard_context:
        return 18.0
    if identity_target and bms_point_context and not any(
        term in field for term in ["bms", "controller", "outstation"]
    ):
        return 18.0
    if "bms" in field and not any(
        term in lower for term in ["bms", "trend iq", "outstation", "points schedule"]
    ):
        return 18.0
    if "pv" in field and identity_target and not any(
        term in lower for term in ["pv", "photovoltaic", "inverter", "solar", "solis"]
    ):
        return 24.0
    if "cold_water_booster" in field and not any(
        term in lower for term in ["booster", "cwb", "pump"]
    ):
        return 30.0
    if "booster" in field and "capacity" in field and not any(
        term in lower for term in ["booster", "cwb", "pump", "design flow", "design pressure"]
    ):
        return 30.0
    if "electrical_distribution" in field and not (
        unit1_context
        and any(term in lower for term in ["mccb", "switch panel", "distribution board", "mp1"])
    ):
        return 24.0
    if "commissioning_date" in field and not any(
        term in lower for term in ["commissioning", "commissioned", "installation certificate"]
    ):
        return 18.0
    if identity_target and pv_string_reading_context and not any(
        term in lower for term in ["model", "model number", "solis"]
    ):
        return 18.0
    if "pv_inverter" in field and drawing_context:
        return 18.0
    if "air_conditioning" in field and identity_target and not (
        unit1_context
        and any(term in lower for term in ["model number", "unit serial number", "serial no"])
    ):
        return 18.0
    if any(term in field for term in ["equipment", "mechanical", "electrical"]) and (
        residual_hazard_context or fire_strategy_standard_context
    ):
        return 14.0
    if "office_floor_loading" in field and not (
        "office slab" in lower and ("7.5kn/m" in lower or "designed for imposed load" in lower)
    ):
        return 14.0
    if "warehouse_floor_loading" in field and not (
        "warehouse" in lower and ("50kn/m" in lower or "live warehouse" in lower)
    ):
        return 14.0
    if "foundation" in field and not any(
        term in lower
        for term in [
            "piled solution",
            "piled foundation",
            "foundation type",
            "adopted to support",
        ]
    ):
        return 10.0
    return 0.0


def cross_reference_penalty(target: TargetSpecification, text: str) -> float:
    lower = text.lower()
    field = target.expected_field.lower()
    certificate_or_date_target = (
        target.expected_data_type == ExpectedDataType.DATE
        or "certificate" in field
        or "reference" in field
    )
    if certificate_or_date_target and "refer to" in lower and "overleaf" in lower:
        return 10.0
    equipment_or_value_target = any(
        term in field
        for term in ["equipment", "mechanical", "electrical", "model", "manufacturer", "description"]
    )
    if equipment_or_value_target and "refer to" in lower and any(
        term in lower
        for term in [
            "part 3",
            "part 6",
            "m&e",
            "m & e",
            "manual",
            "drawings overleaf",
            "calculation report overleaf",
        ]
    ):
        return 10.0
    reference_target = any(term in field for term in ["drawing", "reference", "as_built"])
    if reference_target and "drawing" in field:
        has_drawing_identifier = any(
            term in lower for term in ["drawing number", "drawing no", "dwg no", "drg no"]
        )
        if "refer to" in lower and re.search(r"\bpart\s+6\s+appendix\s+[a-z]\b", lower):
            return 10.0
        if "as built drawings" in lower and not has_drawing_identifier:
            return 10.0
        if not has_drawing_identifier:
            return 8.0
    return 0.0


def deduplicate_overlapping_retrievals(
    ordered: Sequence[RetrievedEvidence],
) -> list[RetrievedEvidence]:
    retained: list[RetrievedEvidence] = []
    seen: set[tuple[str, int, int, str]] = set()
    for item in ordered:
        key = (
            item.source_id,
            item.page_start,
            item.page_end,
            normalize_space(item.excerpt.lower())[:500],
        )
        if key in seen:
            continue
        seen.add(key)
        retained.append(item)
    return retained


def build_evidence_bundle(
    target: TargetSpecification,
    retrieval: RetrievalResult,
    max_evidence_chars: int,
    pages_by_source: dict[str, list[ParsedPage]] | None = None,
) -> EvidenceBundleRecord:
    pieces: list[str] = []
    used = 0
    truncated = False
    page_text = {
        (page.source_id, page.page_number): page.text or ""
        for pages in (pages_by_source or {}).values()
        for page in pages
    }
    spans: list[EvidenceSpan] = []
    for item in retrieval.results:
        span = build_canonical_span(target, item, page_text)
        spans.append(span)
        prefix = (
            f"[span_id={span.span_id} rank={item.rank} source={item.source_file} "
            f"page={item.page_start} node={item.node_id}] "
        )
        chunk = prefix + span.text
        if used + len(chunk) > max_evidence_chars:
            truncated = True
            remaining = max_evidence_chars - used
            if remaining > len(prefix) + 50:
                pieces.append(chunk[:remaining])
            break
        pieces.append(chunk)
        used += len(chunk)
    combined = "\n\n".join(pieces)
    return EvidenceBundleRecord(
        target_row_id=target.target_row_id,
        evidence_items=retrieval.results,
        retrieval_status=retrieval.retrieval_status,
        evidence_spans=spans,
        combined_text=combined,
        character_count=len(combined),
        token_estimate=estimate_tokens(combined),
        truncated=truncated,
    )


def build_canonical_span(
    target: TargetSpecification,
    item: RetrievedEvidence,
    page_text: dict[tuple[str, int], str],
) -> EvidenceSpan:
    canonical = page_text.get((item.source_id, item.page_start), item.excerpt)
    concepts = query_concepts_for_target(target)
    terms = query_terms_for_concepts(concepts)
    start, end = best_span_offsets(canonical, terms, max_chars=700)
    text = canonical[start:end]
    identity = {
        "version": VERTICAL_SLICE_V2_VERSION,
        "target_row_id": target.target_row_id,
        "source_id": item.source_id,
        "page_number": item.page_start,
        "node_id": item.node_id,
        "start": start,
        "end": end,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return EvidenceSpan(
        span_id=f"span_{digest}",
        source_id=item.source_id,
        source_file=item.source_file,
        page_number=item.page_start,
        hierarchy_node_id=item.node_id,
        text=text,
        start_char=start,
        end_char=end,
        retrieval_rank=item.rank,
        score=item.score,
    )


def best_span_offsets(text: str, query_terms: Sequence[str], *, max_chars: int) -> tuple[int, int]:
    if not text:
        return (0, 0)
    lowered = text.lower()
    candidates: list[tuple[int, int, int]] = []
    line_offsets: list[tuple[int, int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        line_start = cursor + line.find(stripped) if stripped else cursor
        line_end = line_start + len(stripped)
        line_offsets.append((line_start, line_end, stripped))
        lower_line = stripped.lower()
        hits = sum(1 for term in query_terms if term and term.lower() in lower_line)
        if hits:
            candidates.append((hits, line_start, line_end))
        cursor += len(line)
    if candidates:
        _, start, end = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
        line_index = next(
            (
                index
                for index, (line_start, line_end, _) in enumerate(line_offsets)
                if line_start == start and line_end == end
            ),
            0,
        )
        end_index = line_index
        while end_index + 1 < len(line_offsets) and end - start < max_chars:
            next_start, next_end, next_text = line_offsets[end_index + 1]
            if not next_text and end_index > line_index:
                break
            if next_end - start > max_chars:
                break
            end = next_end
            end_index += 1
            if next_start > end:
                break
        return (max(0, start), min(len(text), max(end, start + 1)))
    positions = [
        lowered.find(term.lower())
        for term in query_terms
        if lowered.find(term.lower()) >= 0
    ]
    if positions:
        center = min(positions)
        start = max(0, center - max_chars // 3)
        return (start, min(len(text), start + max_chars))
    return (0, min(len(text), max_chars))


def build_retrieval_expectations(
    selected_targets: list[SelectedTarget],
) -> list[RetrievalExpectation]:
    expected: dict[str, tuple[str | None, list[int], str | None, bool, str]] = {}
    for selected in selected_targets:
        field = selected.target.expected_field
        if field in {
            "frame_primary_construction_type",
            "frame_construction_description",
            "roof_construction_description",
            "external_cladding_description",
            "wall_construction_description",
        }:
            expected[field] = (None, [5, 9, 11], None, False, "Part 1 description/plans.")
        elif field == "dock_leveller_count":
            expected[field] = (None, [8], None, False, "Part 1 ground floor plan text.")
        elif field == "pv_panel_component_description":
            expected[field] = (None, [9], None, False, "Part 1 roof plan PV text.")
        elif field == "floor_construction_description":
            expected[field] = (None, [10, 11, 12], None, False, "Part 4 concrete slab O&M.")
        elif field == "fire_alarm_system_description":
            expected[field] = (
                None,
                [10],
                None,
                False,
                "Part 6 health and safety responsibilities text.",
            )
        elif field in {"approved_use_classes", "landlord_planning_consent_obligations"}:
            expected[field] = (
                None,
                [15, 17, 18, 19, 20, 22],
                "planning",
                False,
                "Part 1 planning approval.",
            )
        elif field == "pv_certification_component_description":
            expected[field] = (None, [4, 9], None, False, "Part 6 certificate/PV references.")
        elif field == "construction_date":
            expected[field] = (
                None,
                [42, 45, 46],
                None,
                False,
                "Building control and practical completion certificates.",
            )
        elif field == "office_area_value":
            expected[field] = (
                None,
                [],
                None,
                True,
                "No supported office-area evidence expected in selected bounded pages.",
            )
        else:
            expected[field] = (None, [], None, True, "No manual expectation defined.")
    return [
        RetrievalExpectation(
            target_row_id=selected.target.target_row_id,
            expected_source_id=item[0],
            expected_pages=item[1],
            expected_section_contains=item[2],
            no_supported_evidence_expected=item[3],
            notes=item[4],
        )
        for selected in selected_targets
        for item in [expected[selected.target.expected_field]]
    ]


def evaluate_retrieval(
    expectations: list[RetrievalExpectation],
    retrieval_results: list[RetrievalResult],
) -> list[RetrievalEvaluationItem]:
    by_target = {result.target_row_id: result for result in retrieval_results}
    evaluations: list[RetrievalEvaluationItem] = []
    for expectation in expectations:
        result = by_target[expectation.target_row_id]
        if expectation.no_supported_evidence_expected:
            evaluations.append(
                RetrievalEvaluationItem(
                    target_row_id=expectation.target_row_id,
                    absent_correct=result.retrieval_status
                    in {"weak_evidence", "no_relevant_evidence"},
                    irrelevant_retrieval=result.retrieval_status == "evidence_found",
                    notes=expectation.notes,
                )
            )
            continue
        top1 = bool(result.results and _matches_expectation(result.results[0], expectation))
        top3 = any(_matches_expectation(item, expectation) for item in result.results[:3])
        evaluations.append(
            RetrievalEvaluationItem(
                target_row_id=expectation.target_row_id,
                top1_hit=top1,
                top3_hit=top3,
                irrelevant_retrieval=not top3,
                notes=expectation.notes,
            )
        )
    return evaluations


def parse_extraction_response(
    *,
    target: TargetSpecification,
    bundle: EvidenceBundleRecord,
    provider: str,
    model_name: str,
    response_text: str,
) -> ExtractionResult:
    try:
        payload = json.loads(_extract_json_object(response_text))
    except ValueError:
        return ExtractionResult(
            target_row_id=target.target_row_id,
            requirement_id=target.requirement_id,
            status="invalid_format",
            confidence=0,
            model_provider=provider,
            model_name=model_name,
            ambiguity_or_caveat="Model response was not valid JSON.",
        )
    status = payload.get("status")
    allowed = {
        "extracted",
        "multiple_candidates",
        "insufficient_evidence",
        "conflicting_evidence",
        "invalid_format",
        "model_error",
    }
    if status not in allowed:
        status = "invalid_format"
    span_ids = payload.get("selected_supporting_span_ids") or payload.get(
        "supporting_span_ids"
    ) or payload.get("selected_span_ids") or []
    if isinstance(span_ids, str):
        span_ids = [span_ids]
    if not isinstance(span_ids, list):
        span_ids = []
    evidence_item = _evidence_item_for_payload(bundle, payload)
    source_id = payload.get("source_id") or (evidence_item.source_id if evidence_item else None)
    page_number = payload.get("page_number") or (
        evidence_item.page_start if evidence_item else None
    )
    node_id = payload.get("hierarchy_node_id") or (evidence_item.node_id if evidence_item else None)
    source_file = evidence_item.source_file if evidence_item else None
    return ExtractionResult(
        target_row_id=target.target_row_id,
        requirement_id=target.requirement_id,
        raw_model_value=payload.get("raw_value", payload.get("extracted_value")),
        extracted_value=payload.get("extracted_value"),
        normalized_value=payload.get("normalized_value"),
        proposed_value_shape=payload.get("proposed_value_shape"),
        value_bearing_quote=payload.get("value_bearing_quote"),
        unit=payload.get("unit") or target.unit,
        status=status,
        confidence=float(payload.get("confidence", 0) or 0),
        supporting_span_ids=[str(item) for item in span_ids],
        supporting_evidence_excerpt=payload.get("supporting_evidence_excerpt"),
        source_id=source_id,
        source_file=source_file,
        page_number=int(page_number)
        if isinstance(page_number, int | float | str) and str(page_number).isdigit()
        else None,
        page_range=payload.get("page_range"),
        hierarchy_node_id=node_id,
        reasoning_summary=payload.get("reasoning_summary"),
        ambiguity_or_caveat=payload.get("ambiguity_or_caveat"),
        model_provider=provider,
        model_name=model_name,
    )


def materialize_selected_spans(
    extraction: ExtractionResult,
    bundle: EvidenceBundleRecord,
) -> ExtractionResult:
    if extraction.status != "extracted" or not extraction.supporting_span_ids:
        return extraction
    span_by_id = {span.span_id: span for span in bundle.evidence_spans}
    selected = [
        span_by_id[span_id]
        for span_id in extraction.supporting_span_ids
        if span_id in span_by_id
    ]
    if not selected:
        return extraction.model_copy(
            update={
                "supporting_evidence_excerpt": None,
                "source_id": None,
                "source_file": None,
                "page_number": None,
                "hierarchy_node_id": None,
            }
        )
    first = selected[0]
    return extraction.model_copy(
        update={
            "supporting_evidence_excerpt": first.text,
            "source_id": first.source_id,
            "source_file": first.source_file,
            "page_number": first.page_number,
            "hierarchy_node_id": first.hierarchy_node_id,
        }
    )


def infer_value_shape_assignments(
    selected_targets: list[SelectedTarget],
) -> list[ValueShapeAssignment]:
    return [infer_value_shape_assignment(selected.target) for selected in selected_targets]


def infer_value_shape_assignment(target: TargetSpecification) -> ValueShapeAssignment:
    field = target.expected_field.lower()
    definition = target.requirement_text.lower()
    basis: list[str] = [
        f"datatype={target.expected_data_type}",
        f"unit={target.unit}" if target.unit else "unit=<none>",
    ]
    warning: str | None = None
    confidence = 0.74
    if target.accepted_values:
        shape: ValueShapeFamily = "categorical"
        basis.append("reference-list-present")
        confidence = 0.82
    elif any(token in field for token in ["count", "number", "quantity"]):
        shape = "integer_count"
        basis.append("count-like-field-name")
        confidence = 0.9
    elif target.expected_data_type == ExpectedDataType.DATE or "date" in field:
        shape = "date"
        basis.append("date-datatype-or-field")
        confidence = 0.92
    elif "use_class" in field or "use_classes" in field or "classes" in field:
        shape = "ordered_or_unordered_list"
        basis.append("class-list-field-name")
        confidence = 0.88
    elif "reference" in field or "ref" in field or "consent" in field:
        shape = "identifier_or_reference"
        basis.append("reference-or-consent-field-name")
        confidence = 0.78
    elif target.expected_data_type == ExpectedDataType.DECIMAL and target.unit:
        if any(word in field for word in ["description", "type", "construction"]):
            shape = "descriptive_text"
            warning = "Dictionary declares decimal/unit but field semantics are descriptive."
            confidence = 0.7
        else:
            shape = "decimal_measurement"
            basis.append("decimal-datatype-with-unit")
            confidence = 0.82
    elif target.expected_data_type == ExpectedDataType.INTEGER:
        shape = "integer_count"
        basis.append("integer-datatype")
        confidence = 0.78
    elif target.expected_data_type == ExpectedDataType.DECIMAL:
        shape = "decimal_measurement"
        basis.append("decimal-datatype")
        confidence = 0.7
    elif any(word in field for word in ["description", "obligations"]):
        shape = "descriptive_text"
        basis.append("descriptive-field-name")
        confidence = 0.84
    elif any(word in definition for word in ["yes/no", "whether"]):
        shape = "boolean_or_presence"
        basis.append("boolean-like-definition")
    elif target.expected_data_type == ExpectedDataType.ENUM:
        shape = "categorical"
        basis.append("enum-datatype")
    else:
        shape = "short_text"
        basis.append("default-short-text")
    return ValueShapeAssignment(
        target_row_id=target.target_row_id,
        field_name=target.expected_field,
        dictionary_declared_datatype=str(target.expected_data_type),
        dictionary_declared_unit=target.unit,
        value_shape_family=shape,
        inference_basis=basis,
        confidence=confidence,
        ambiguity_or_mismatch_warning=warning,
    )


def calibrate_extraction_value(
    *,
    extraction: ExtractionResult,
    target: TargetSpecification,
    assignment: ValueShapeAssignment,
    bundle: EvidenceBundleRecord,
) -> ExtractionResult:
    raw = extraction.raw_model_value
    if raw is None:
        raw = extraction.extracted_value
    if extraction.status != "extracted":
        return extraction.model_copy(
            update={
                "raw_model_value": raw,
                "display_value": _display_value(extraction.normalized_value or raw),
                "proposed_value_shape": extraction.proposed_value_shape
                or assignment.value_shape_family,
            }
        )
    span_text = "\n".join(
        span.text
        for span in bundle.evidence_spans
        if span.span_id in extraction.supporting_span_ids
    )
    derived = derive_evidence_value_and_normalization(
        target=target,
        assignment=assignment,
        raw_model_value=raw,
        value_bearing_quote=extraction.value_bearing_quote,
        span_text=span_text,
    )
    return extraction.model_copy(
        update={
            "raw_model_value": raw,
            "evidence_value": derived.evidence_value,
            "normalized_value": derived.normalized_value,
            "display_value": derived.display_value,
            "unit": derived.unit if derived.unit is not None else extraction.unit,
            "proposed_value_shape": extraction.proposed_value_shape
            or assignment.value_shape_family,
            "extracted_value": raw,
        }
    )


def derive_evidence_value_and_normalization(
    *,
    target: TargetSpecification,
    assignment: ValueShapeAssignment,
    raw_model_value: object,
    value_bearing_quote: str | None,
    span_text: str,
) -> NormalizedValueRecord:
    shape = assignment.value_shape_family
    raw_text = "" if raw_model_value is None else str(raw_model_value)
    quote = (
        value_bearing_quote
        if value_bearing_quote and value_bearing_quote in span_text
        else None
    )
    issues: list[str] = []
    evidence_value: object | None = quote
    normalized: object | None = None
    unit: str | None = target.unit
    status: Literal["normalized", "already_normalized", "ambiguous", "not_applicable", "failed"]
    status = "normalized"
    if shape == "integer_count":
        count = _derive_count(span_text, target)
        if count is None:
            count = _word_or_int_count(raw_text)
        if count is None:
            status = "ambiguous" if _numeric_count(span_text) > 1 else "failed"
            issues.append("integer count could not be determined unambiguously")
        else:
            normalized = count
            evidence_value = _count_evidence_phrase(span_text, count) or quote or raw_text
    elif shape == "date":
        parsed = _derive_date(span_text) or _derive_date(raw_text)
        if parsed is None:
            status = "failed"
            issues.append("date could not be parsed")
        else:
            evidence_value = parsed[0]
            normalized = parsed[1]
    elif shape == "ordered_or_unordered_list":
        values = _derive_use_classes(span_text) or _derive_use_classes(raw_text)
        if values:
            evidence_value = _use_class_evidence_phrase(span_text, values) or quote or raw_text
            normalized = values
        else:
            status = "failed"
            issues.append("list values could not be parsed")
    elif shape == "decimal_measurement":
        measurement = _derive_measurement(span_text, target.unit)
        if measurement is None:
            measurement = _derive_measurement(raw_text, target.unit)
        if measurement is None:
            status = "ambiguous" if _numeric_count(span_text) > 1 else "failed"
            issues.append("measurement could not be determined unambiguously")
        else:
            evidence_value, normalized, unit = measurement
    elif shape in {"descriptive_text", "short_text", "categorical", "identifier_or_reference"}:
        evidence_value = quote or _best_supported_phrase(raw_text, span_text) or raw_text
        normalized = raw_model_value
        status = "already_normalized"
    else:
        status = "not_applicable"
    display = _display_value(normalized if normalized is not None else evidence_value)
    return NormalizedValueRecord(
        target_row_id=target.target_row_id,
        value_shape_family=shape,
        raw_model_value=raw_model_value,  # type: ignore[arg-type]
        evidence_value=evidence_value,  # type: ignore[arg-type]
        normalized_value=normalized,  # type: ignore[arg-type]
        display_value=display,
        unit=unit,
        normalization_status=status,
        issues=issues,
    )


def validate_evidence_layer(
    extraction_results: list[ExtractionResult],
    selected_targets: list[SelectedTarget],
    pages_by_source: dict[str, list[ParsedPage]],
    hierarchy: LightweightHierarchy,
    bundles: list[EvidenceBundleRecord],
) -> list[EvidenceValidationResult]:
    _ = selected_targets
    page_text = {
        (page.source_id, page.page_number): page.text or ""
        for pages in pages_by_source.values()
        for page in pages
    }
    node_ids = {node.node_id for node in hierarchy.nodes}
    span_by_id = {
        span.span_id: span
        for bundle in bundles
        for span in bundle.evidence_spans
    }
    results: list[EvidenceValidationResult] = []
    for extraction in extraction_results:
        issues: list[str] = []
        if extraction.status != "extracted":
            results.append(
                EvidenceValidationResult(
                    target_row_id=extraction.target_row_id,
                    status="valid",
                    selected_span_ids=extraction.supporting_span_ids,
                )
            )
            continue
        selected_spans = [span_by_id.get(span_id) for span_id in extraction.supporting_span_ids]
        missing = [
            span_id
            for span_id, span in zip(
                extraction.supporting_span_ids, selected_spans, strict=False
            )
            if span is None
        ]
        if missing:
            issues.append(f"unknown selected span ids: {', '.join(missing)}")
        present_spans = [span for span in selected_spans if span is not None]
        canonical_available = bool(present_spans)
        span_text = "\n".join(span.text for span in present_spans)
        evidence_value_present = bool(
            extraction.evidence_value is not None
            and str(extraction.evidence_value) in span_text
        )
        if not evidence_value_present:
            issues.append("evidence value is not present in selected canonical spans")
        provenance_valid = all(
            span.hierarchy_node_id in node_ids
            and span.text in page_text.get((span.source_id, span.page_number), "")
            for span in present_spans
        )
        if not provenance_valid:
            issues.append("span source/page/node provenance is invalid")
        results.append(
            EvidenceValidationResult(
                target_row_id=extraction.target_row_id,
                status="invalid" if issues else "valid",
                selected_span_ids=extraction.supporting_span_ids,
                canonical_evidence_available=canonical_available,
                evidence_value_present=evidence_value_present,
                source_page_node_valid=provenance_valid,
                issues=issues,
            )
        )
    return results


def validate_shape_layer(
    extraction_results: list[ExtractionResult],
    assignments: list[ValueShapeAssignment],
) -> list[ShapeValidationResult]:
    assignment_by_id = {item.target_row_id: item for item in assignments}
    results: list[ShapeValidationResult] = []
    for extraction in extraction_results:
        assignment = assignment_by_id[extraction.target_row_id]
        issues: list[str] = []
        status: LayerStatus = "valid"
        if extraction.status in {"model_error", "invalid_format"}:
            status = "invalid"
            issues.append(f"extraction status is {extraction.status}")
        elif extraction.status != "extracted":
            status = "valid"
        elif assignment.value_shape_family == "integer_count":
            if not isinstance(extraction.normalized_value, int):
                status = "review_required"
                issues.append("count did not normalize to an integer")
        elif assignment.value_shape_family == "date":
            if not (
                isinstance(extraction.normalized_value, str)
                and re.fullmatch(r"\d{4}-\d{2}-\d{2}", extraction.normalized_value)
            ):
                status = "invalid"
                issues.append("date did not normalize to ISO format")
            elif extraction.raw_model_value != extraction.normalized_value:
                status = "valid_after_normalization"
        elif assignment.value_shape_family == "ordered_or_unordered_list":
            if not isinstance(extraction.normalized_value, list) or not extraction.normalized_value:
                status = "invalid"
                issues.append("list did not normalize to distinct values")
            elif len(extraction.normalized_value) != len(
                {str(v) for v in extraction.normalized_value}
            ):
                status = "invalid"
                issues.append("list contains duplicate values")
            else:
                status = "valid_after_normalization"
        elif assignment.value_shape_family == "decimal_measurement":
            if not isinstance(extraction.normalized_value, int | float):
                status = "review_required"
                issues.append("measurement did not normalize to a number")
        elif assignment.value_shape_family in {
            "descriptive_text",
            "short_text",
            "categorical",
            "identifier_or_reference",
        }:
            if extraction.normalized_value in {None, ""}:
                status = "review_required"
                issues.append("text value is empty")
            elif (
                assignment.value_shape_family == "short_text"
                and len(str(extraction.normalized_value)) > 160
            ):
                status = "review_required"
                issues.append("short text value is too long")
        results.append(
            ShapeValidationResult(
                target_row_id=extraction.target_row_id,
                value_shape_family=assignment.value_shape_family,
                status=status,
                issues=issues,
            )
        )
    return results


def assess_schema_compatibility_v3(
    extraction_results: list[ExtractionResult],
    selected_targets: list[SelectedTarget],
    assignments: list[ValueShapeAssignment],
    evidence_results: list[EvidenceValidationResult],
    shape_results: list[ShapeValidationResult],
) -> list[SchemaCompatibilityResult]:
    target_by_id = {selected.target.target_row_id: selected.target for selected in selected_targets}
    assignment_by_id = {item.target_row_id: item for item in assignments}
    evidence_by_id = {item.target_row_id: item for item in evidence_results}
    shape_by_id = {item.target_row_id: item for item in shape_results}
    results: list[SchemaCompatibilityResult] = []
    for extraction in extraction_results:
        target = target_by_id[extraction.target_row_id]
        assignment = assignment_by_id[extraction.target_row_id]
        evidence = evidence_by_id[extraction.target_row_id]
        shape = shape_by_id[extraction.target_row_id]
        issues: list[str] = []
        if extraction.status in {"model_error", "invalid_format"}:
            compatibility = "not_evaluated"
            overall = "model_error"
        elif extraction.status != "extracted":
            compatibility = "not_evaluated"
            overall = "insufficient_evidence"
        elif evidence.status == "invalid":
            compatibility = "not_evaluated"
            overall = "invalid_evidence"
        elif shape.status in {"invalid", "review_required"}:
            compatibility = "incompatible"
            overall = "invalid_shape" if shape.status == "invalid" else "review_required"
            issues.extend(shape.issues)
        else:
            compatibility, overall, caveats = _dictionary_compatibility(
                target, assignment, extraction
            )
            issues.extend(caveats)
        results.append(
            SchemaCompatibilityResult(
                target_row_id=extraction.target_row_id,
                dictionary_datatype=str(target.expected_data_type),
                dictionary_unit=target.unit,
                observed_value=extraction.normalized_value,
                observed_unit=extraction.unit,
                compatibility=compatibility,  # type: ignore[arg-type]
                evidence_validity="valid" if evidence.status != "invalid" else "invalid",
                value_format_validity="valid" if shape.status != "invalid" else "invalid",
                overall_review_status=overall,  # type: ignore[arg-type]
                issues=issues,
            )
        )
    return results


def build_overall_validation_results(
    extraction_results: list[ExtractionResult],
    selected_targets: list[SelectedTarget],
    evidence_results: list[EvidenceValidationResult],
    shape_results: list[ShapeValidationResult],
    schema_results: list[SchemaCompatibilityResult],
) -> list[ValidationResult]:
    _ = selected_targets
    evidence_by_id = {item.target_row_id: item for item in evidence_results}
    shape_by_id = {item.target_row_id: item for item in shape_results}
    schema_by_id = {item.target_row_id: item for item in schema_results}
    results: list[ValidationResult] = []
    for extraction in extraction_results:
        evidence = evidence_by_id[extraction.target_row_id]
        shape = shape_by_id[extraction.target_row_id]
        schema = schema_by_id[extraction.target_row_id]
        issues = [*evidence.issues, *shape.issues, *schema.issues]
        results.append(
            ValidationResult(
                target_row_id=extraction.target_row_id,
                status=schema.overall_review_status,
                evidence_valid=evidence.status != "invalid",
                value_format_valid=shape.status != "invalid",
                issues=issues,
            )
        )
    return results


def _insufficient_evidence_result(
    target: TargetSpecification,
    provider: str,
    model_name: str,
    caveat: str,
) -> ExtractionResult:
    return ExtractionResult(
        target_row_id=target.target_row_id,
        requirement_id=target.requirement_id,
        status="insufficient_evidence",
        confidence=0,
        model_provider=provider,
        model_name=model_name,
        ambiguity_or_caveat=caveat,
    )


def validate_extractions(
    extraction_results: list[ExtractionResult],
    selected_targets: list[SelectedTarget],
    pages_by_source: dict[str, list[ParsedPage]],
    hierarchy: LightweightHierarchy,
    bundles: list[EvidenceBundleRecord] | None = None,
) -> list[ValidationResult]:
    target_by_id = {selected.target.target_row_id: selected.target for selected in selected_targets}
    page_text = {
        (page.source_id, page.page_number): page.text or ""
        for pages in pages_by_source.values()
        for page in pages
    }
    node_ids = {node.node_id for node in hierarchy.nodes}
    span_by_id = {
        span.span_id: span
        for bundle in (bundles or [])
        for span in bundle.evidence_spans
    }
    results: list[ValidationResult] = []
    for extraction in extraction_results:
        target = target_by_id[extraction.target_row_id]
        issues: list[str] = []
        evidence_issues: list[str] = []
        value_issues: list[str] = []
        if extraction.status in {"model_error", "invalid_format"}:
            value_issues.append(f"extraction status is {extraction.status}")
        if extraction.status == "extracted":
            if extraction.extracted_value in {None, ""}:
                value_issues.append("extracted status requires a value")
            if (
                not extraction.source_id
                or not extraction.page_number
                or not extraction.hierarchy_node_id
            ):
                evidence_issues.append("extracted status requires source/page/node evidence")
            if span_by_id and not extraction.supporting_span_ids:
                evidence_issues.append("extracted status requires selected canonical span id")
            for span_id in extraction.supporting_span_ids:
                if span_id not in span_by_id:
                    evidence_issues.append(f"selected span id is not in supplied bundle: {span_id}")
            if extraction.hierarchy_node_id and extraction.hierarchy_node_id not in node_ids:
                evidence_issues.append("hierarchy node is not present")
            if not extraction.supporting_evidence_excerpt:
                evidence_issues.append("supporting evidence excerpt is required")
            elif extraction.source_id and extraction.page_number:
                source_text = page_text.get((extraction.source_id, extraction.page_number), "")
                if normalize_space(extraction.supporting_evidence_excerpt) not in normalize_space(
                    source_text
                ):
                    evidence_issues.append(
                        "supporting evidence excerpt is not present in canonical page text"
                    )
            value_issues.extend(_validate_datatype(target, extraction))
            if target.unit and extraction.unit and target.unit.lower() != extraction.unit.lower():
                value_issues.append("unit does not match target unit")
            if target.accepted_values and extraction.normalized_value is not None:
                normalized = str(extraction.normalized_value).lower()
                allowed = {value.lower() for value in target.accepted_values}
                if normalized not in allowed:
                    value_issues.append("normalized value is not in accepted reference list")
            if extraction.extracted_value and extraction.supporting_evidence_excerpt:
                value = str(extraction.extracted_value).strip()
                if (
                    target.expected_data_type in {ExpectedDataType.STRING, ExpectedDataType.ENUM}
                    and value
                    and value.lower() not in extraction.supporting_evidence_excerpt.lower()
                ):
                    value_issues.append("direct text value does not appear in supporting evidence")
        if not 0 <= extraction.confidence <= 1:
            value_issues.append("confidence must be within 0..1")
        issues.extend(evidence_issues)
        issues.extend(value_issues)
        results.append(
            ValidationResult(
                target_row_id=extraction.target_row_id,
                status="invalid" if issues else "valid",
                evidence_valid=not evidence_issues,
                value_format_valid=not value_issues,
                issues=issues,
            )
        )
    return results


def assess_schema_compatibility(
    extraction_results: list[ExtractionResult],
    selected_targets: list[SelectedTarget],
    validation_results: list[ValidationResult],
) -> list[SchemaCompatibilityResult]:
    target_by_id = {selected.target.target_row_id: selected.target for selected in selected_targets}
    validation_by_id = {item.target_row_id: item for item in validation_results}
    results: list[SchemaCompatibilityResult] = []
    for extraction in extraction_results:
        target = target_by_id[extraction.target_row_id]
        validation = validation_by_id[extraction.target_row_id]
        issues: list[str] = []
        if extraction.status in {"insufficient_evidence", "conflicting_evidence"}:
            compatibility: str = "not_evaluated"
            overall: str = "insufficient_evidence"
        elif extraction.status in {"model_error", "invalid_format"}:
            compatibility = "not_evaluated"
            overall = "model_error"
        elif not validation.evidence_valid:
            compatibility = "not_evaluated"
            overall = "invalid_evidence"
        elif not validation.value_format_valid:
            datatype_issues = " ".join(validation.issues).lower()
            if (
                target.expected_data_type in {ExpectedDataType.DECIMAL, ExpectedDataType.INTEGER}
                and extraction.extracted_value is not None
                and not _looks_numeric(extraction.extracted_value)
            ):
                compatibility = "narrative_value_against_structured_constraint"
                overall = "valid_with_dictionary_caveat"
                issues.append(
                    "Supported narrative value conflicts with dictionary numeric metadata."
                )
            elif "unit does not match" in datatype_issues and extraction.unit:
                compatibility = "unit_not_applicable_to_observed_value"
                overall = "valid_with_dictionary_caveat"
                issues.append("Observed unit/value does not align with dictionary unit metadata.")
            else:
                compatibility = "incompatible_value"
                overall = "invalid_value"
        else:
            compatibility = "compatible"
            overall = "valid"
        results.append(
            SchemaCompatibilityResult(
                target_row_id=extraction.target_row_id,
                dictionary_datatype=str(target.expected_data_type),
                dictionary_unit=target.unit,
                observed_value=extraction.extracted_value,
                observed_unit=extraction.unit,
                compatibility=compatibility,  # type: ignore[arg-type]
                evidence_validity="valid" if validation.evidence_valid else "invalid",
                value_format_validity="valid" if validation.value_format_valid else "invalid",
                overall_review_status=overall,  # type: ignore[arg-type]
                issues=issues,
            )
        )
    return results


def write_vertical_slice_artifacts(result: VerticalSliceRunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, object] = {
        "selected_targets.json": [item.model_dump(mode="json") for item in result.selected_targets],
        "source_ranges.json": [item.model_dump(mode="json") for item in result.source_ranges],
        "hierarchy.json": result.hierarchy.model_dump(mode="json"),
        "retrieval_expectations.json": [
            item.model_dump(mode="json") for item in result.retrieval_expectations
        ],
        "retrieval_results.json": [
            item.model_dump(mode="json") for item in result.retrieval_results
        ],
        "retrieval_evaluation.json": [
            item.model_dump(mode="json") for item in result.retrieval_evaluation
        ],
        "value_shape_assignments.json": [
            item.model_dump(mode="json") for item in result.value_shape_assignments
        ],
        "evidence_spans.json": [item.model_dump(mode="json") for item in result.evidence_spans],
        "evidence_bundles.json": [item.model_dump(mode="json") for item in result.evidence_bundles],
        "extraction_results.json": [
            item.model_dump(mode="json") for item in result.extraction_results
        ],
        "normalized_values.json": [
            item.model_dump(mode="json") for item in result.normalized_values
        ],
        "evidence_validation.json": [
            item.model_dump(mode="json") for item in result.evidence_validation
        ],
        "shape_validation.json": [
            item.model_dump(mode="json") for item in result.shape_validation
        ],
        "validation_results.json": [
            item.model_dump(mode="json") for item in result.validation_results
        ],
        "schema_compatibility_results.json": [
            item.model_dump(mode="json") for item in result.schema_compatibility_results
        ],
        "telemetry.json": result.telemetry.model_dump(mode="json"),
    }
    if result.v1_v2_comparison is not None:
        artifacts["v1_v2_comparison.json"] = result.v1_v2_comparison.model_dump(mode="json")
    if result.v2_v3_comparison is not None:
        artifacts["v2_v3_comparison.json"] = result.v2_v3_comparison.model_dump(mode="json")
    for filename, payload in artifacts.items():
        _atomic_write_json(output_dir / filename, payload)
    _write_review_csv(result, output_dir / "extraction_review.csv")
    (output_dir / "extraction_summary.md").write_text(_summary_markdown(result), encoding="utf-8")
    if result.v1_v2_comparison is not None:
        (output_dir / "v1_v2_comparison.md").write_text(
            _comparison_markdown(result.v1_v2_comparison), encoding="utf-8"
        )
    if result.v2_v3_comparison is not None:
        (output_dir / "v2_v3_comparison.md").write_text(
            _v2_v3_comparison_markdown(result.v2_v3_comparison), encoding="utf-8"
        )


def run_vertical_slice(
    *,
    source_manifest: Path = Path("output/sprint3_source_ingestion/source_pack_manifest.json"),
    dictionary_path: Path = Path("data/input/data_dictionary/SEGRO_Extraction_Template.xlsx"),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    settings: Settings | None = None,
    extraction_client: ExtractionClient | None = None,
    baseline_dir: Path | None = None,
) -> VerticalSliceRunResult:
    return EvidenceFirstVerticalSliceService(
        source_manifest=source_manifest,
        dictionary_path=dictionary_path,
        output_dir=output_dir,
        cache_root=cache_root,
        settings=settings,
        extraction_client=extraction_client,
        baseline_dir=baseline_dir,
    ).run()


def run_vertical_slice_v2(
    *,
    source_manifest: Path = Path("output/sprint3_source_ingestion/source_pack_manifest.json"),
    dictionary_path: Path = Path("data/input/data_dictionary/SEGRO_Extraction_Template.xlsx"),
    output_dir: Path = DEFAULT_V2_OUTPUT_DIR,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    settings: Settings | None = None,
    extraction_client: ExtractionClient | None = None,
    baseline_dir: Path = DEFAULT_OUTPUT_DIR,
) -> VerticalSliceRunResult:
    return run_vertical_slice(
        source_manifest=source_manifest,
        dictionary_path=dictionary_path,
        output_dir=output_dir,
        cache_root=cache_root,
        settings=settings,
        extraction_client=extraction_client,
        baseline_dir=baseline_dir,
    )


def run_vertical_slice_v3(
    *,
    source_manifest: Path = Path("output/sprint3_source_ingestion/source_pack_manifest.json"),
    dictionary_path: Path = Path("data/input/data_dictionary/SEGRO_Extraction_Template.xlsx"),
    output_dir: Path = DEFAULT_V3_OUTPUT_DIR,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    settings: Settings | None = None,
    extraction_client: ExtractionClient | None = None,
    v2_baseline_dir: Path = DEFAULT_V2_OUTPUT_DIR,
) -> VerticalSliceRunResult:
    return EvidenceFirstVerticalSliceService(
        source_manifest=source_manifest,
        dictionary_path=dictionary_path,
        output_dir=output_dir,
        cache_root=cache_root,
        settings=settings,
        extraction_client=extraction_client,
        v2_baseline_dir=v2_baseline_dir,
    ).run()


def describe_planned_vertical_slice(
    *,
    source_manifest: Path,
    dictionary_path: Path,
    output_dir: Path,
    settings: Settings,
) -> dict[str, object]:
    targets = select_vertical_slice_targets(dictionary_path, output_dir / "planning_preview")
    registry = {source.source_id: source for source in load_source_registry(source_manifest)}
    ranges = select_source_ranges(registry)
    model = settings.text_model_name or "gpt-4o-mini"
    max_input_tokens = len(targets) * estimate_tokens("x" * MAX_EVIDENCE_CHARS)
    max_output_tokens = len(targets) * 600
    return {
        "selected_targets": [
            {
                "target_row_id": item.target.target_row_id,
                "requirement_id": item.target.requirement_id,
                "sub_domain": item.target.sub_domain,
                "expected_field": item.target.expected_field,
                "expected_data_type": item.target.expected_data_type,
                "unit": item.target.unit,
            }
            for item in targets
        ],
        "source_ranges": [item.model_dump(mode="json") for item in ranges],
        "expected_llm_calls": len(targets),
        "configured_model": model,
        "estimated_upper_tokens": {
            "input": max_input_tokens,
            "output": max_output_tokens,
        },
        "estimated_upper_cost_usd": estimate_cost_usd(model, max_input_tokens, max_output_tokens),
    }


def describe_planned_vertical_slice_v2(
    *,
    source_manifest: Path,
    dictionary_path: Path,
    output_dir: Path,
    cache_root: Path,
    settings: Settings,
) -> dict[str, object]:
    plan = describe_planned_vertical_slice(
        source_manifest=source_manifest,
        dictionary_path=dictionary_path,
        output_dir=output_dir,
        settings=settings,
    )
    registry = {source.source_id: source for source in load_source_registry(source_manifest)}
    ranges = select_source_ranges(registry)
    cache = CanonicalParsedPageCache(cache_root)
    cache_service = CachedBatchParsingService(cache=cache)
    cached_pages = 0
    total_pages = 0
    for source_range in ranges:
        source = registry[source_range.source_id]
        page_numbers = list(range(source_range.page_start, source_range.page_end + 1))
        total_pages += len(page_numbers)
        lookup = cache.read_range(
            source=source,
            source_path=source.original_path,
            page_numbers=page_numbers,
            parser_name=cache_service.parser_name,
            parser_version=cache_service.parser_version,
            parser_config_fingerprint=cache_service.parser_config_fingerprint,
        )
        cached_pages += lookup.hits
    plan["frozen_source_ranges"] = [item.model_dump(mode="json") for item in ranges]
    plan["all_required_pages_cached"] = cached_pages == total_pages
    plan["cached_page_count"] = cached_pages
    plan["required_page_count"] = total_pages
    plan["v1_baseline_dir"] = str(DEFAULT_OUTPUT_DIR)
    return plan


def build_v1_v2_comparison(
    baseline_dir: Path,
    v2_dir: Path,
    selected_targets: list[SelectedTarget],
    v2_retrieval_evaluation: list[RetrievalEvaluationItem],
    v2_extractions: list[ExtractionResult],
    v2_validations: list[ValidationResult],
    schema_results: list[SchemaCompatibilityResult],
) -> V1V2Comparison:
    v1_eval = _load_json_list(baseline_dir / "retrieval_evaluation.json")
    v1_extract = _load_json_list(baseline_dir / "extraction_results.json")
    v1_validation = _load_json_list(baseline_dir / "validation_results.json")
    v1_eval_by_id = {str(item["target_row_id"]): item for item in v1_eval}
    v1_extract_by_id = {str(item["target_row_id"]): item for item in v1_extract}
    v1_validation_by_id = {str(item["target_row_id"]): item for item in v1_validation}
    v2_eval_by_id = {item.target_row_id: item for item in v2_retrieval_evaluation}
    v2_extract_by_id = {item.target_row_id: item for item in v2_extractions}
    v2_validation_by_id = {item.target_row_id: item for item in v2_validations}
    schema_by_id = {item.target_row_id: item for item in schema_results}
    items: list[V1V2ComparisonItem] = []
    for selected in selected_targets:
        target_id = selected.target.target_row_id
        v1_item = v1_eval_by_id.get(target_id, {})
        v2_item = v2_eval_by_id[target_id]
        v1_extraction = v1_extract_by_id.get(target_id, {})
        v2_extraction = v2_extract_by_id[target_id]
        v1_valid = v1_validation_by_id.get(target_id, {})
        v2_valid = v2_validation_by_id[target_id]
        schema = schema_by_id[target_id]
        outcome = _comparison_outcome(
            bool(v1_item.get("top1_hit")),
            bool(v1_item.get("top3_hit")),
            v2_item.top1_hit,
            v2_item.top3_hit,
            str(v1_valid.get("status") or ""),
            v2_valid.status,
        )
        items.append(
            V1V2ComparisonItem(
                target_row_id=target_id,
                field_name=selected.target.expected_field,
                v1_top1_hit=bool(v1_item.get("top1_hit")),
                v1_top3_hit=bool(v1_item.get("top3_hit")),
                v2_top1_hit=v2_item.top1_hit,
                v2_top3_hit=v2_item.top3_hit,
                v1_extraction_status=v1_extraction.get("status"),
                v1_extraction_value=v1_extraction.get("extracted_value"),
                v2_extraction_status=v2_extraction.status,
                v2_extraction_value=v2_extraction.extracted_value,
                v1_validation_status=v1_valid.get("status"),
                v2_evidence_validation="valid" if v2_valid.evidence_valid else "invalid",
                v2_schema_compatibility=schema.compatibility,
                outcome=outcome,
            )
        )
    return V1V2Comparison(
        baseline_dir=str(baseline_dir),
        v2_dir=str(v2_dir),
        v1_top1_hits=sum(1 for item in items if item.v1_top1_hit),
        v1_top3_hits=sum(1 for item in items if item.v1_top3_hit),
        v2_top1_hits=sum(1 for item in items if item.v2_top1_hit),
        v2_top3_hits=sum(1 for item in items if item.v2_top3_hit),
        items=items,
    )


def _comparison_outcome(
    v1_top1: bool,
    v1_top3: bool,
    v2_top1: bool,
    v2_top3: bool,
    v1_validation: str,
    v2_validation: str,
) -> Literal["improved", "regressed", "unchanged"]:
    old_score = int(v1_top1) * 2 + int(v1_top3) + int(v1_validation == "valid")
    new_score = int(v2_top1) * 2 + int(v2_top3) + int(v2_validation == "valid")
    if new_score > old_score:
        return "improved"
    if new_score < old_score:
        return "regressed"
    return "unchanged"


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def build_v2_v3_comparison(
    baseline_dir: Path,
    v3_dir: Path,
    selected_targets: list[SelectedTarget],
    assignments: list[ValueShapeAssignment],
    extractions: list[ExtractionResult],
    validations: list[ValidationResult],
    schema_results: list[SchemaCompatibilityResult],
) -> V2V3Comparison:
    v2_validation = _load_json_list(baseline_dir / "validation_results.json")
    v2_schema = _load_json_list(baseline_dir / "schema_compatibility_results.json")
    v2_validation_by_id = {str(item["target_row_id"]): item for item in v2_validation}
    v2_schema_by_id = {str(item["target_row_id"]): item for item in v2_schema}
    assignment_by_id = {item.target_row_id: item for item in assignments}
    extraction_by_id = {item.target_row_id: item for item in extractions}
    validation_by_id = {item.target_row_id: item for item in validations}
    schema_by_id = {item.target_row_id: item for item in schema_results}
    items: list[V2V3ComparisonItem] = []
    for selected in selected_targets:
        target_id = selected.target.target_row_id
        validation = validation_by_id[target_id]
        schema = schema_by_id[target_id]
        extraction = extraction_by_id[target_id]
        assignment = assignment_by_id[target_id]
        v2_valid = v2_validation_by_id.get(target_id, {})
        v2_schema_item = v2_schema_by_id.get(target_id, {})
        diagnosis = diagnose_v2_invalidity(v2_valid, v2_schema_item)
        change = describe_v3_change(validation, schema, assignment)
        items.append(
            V2V3ComparisonItem(
                target_row_id=target_id,
                field_name=selected.target.expected_field,
                v2_validation_status=v2_valid.get("status"),
                v2_schema_compatibility=v2_schema_item.get("compatibility"),
                v3_overall_review_status=validation.status,
                v3_schema_compatibility=schema.compatibility,
                v3_value_shape=assignment.value_shape_family,
                v3_normalized_value=extraction.normalized_value,
                diagnosis=diagnosis,
                v3_change=change,
            )
        )
    return V2V3Comparison(
        baseline_dir=str(baseline_dir),
        v3_dir=str(v3_dir),
        v2_status_counts=dict(Counter(str(item.get("status")) for item in v2_validation)),
        v3_status_counts=dict(Counter(item.status for item in validations)),
        items=items,
    )


def diagnose_v2_invalidity(
    v2_validation: dict[str, Any],
    v2_schema: dict[str, Any],
) -> str:
    status = v2_validation.get("status")
    if status == "valid":
        return "V2 valid."
    overall = v2_schema.get("overall_review_status")
    if overall == "insufficient_evidence":
        return "V2 insufficient evidence."
    issues = " ".join(str(issue) for issue in v2_validation.get("issues", []))
    if "direct text value" in issues:
        return "V2 rejected evidence-supported narrative because full value was not verbatim."
    if "reference list" in issues:
        return "V2 rejected normalized value against dictionary reference-list metadata."
    if overall == "invalid_value":
        return "V2 labelled value-shape or dictionary mismatch as invalid value."
    return "V2 invalidity required calibration."


def describe_v3_change(
    validation: ValidationResult,
    schema: SchemaCompatibilityResult,
    assignment: ValueShapeAssignment,
) -> str:
    if validation.status in {"valid", "valid_after_normalization"}:
        return f"V3 validated with {assignment.value_shape_family} shape."
    if validation.status == "valid_with_dictionary_caveat":
        return f"V3 separated dictionary caveat: {schema.compatibility}."
    if validation.status == "insufficient_evidence":
        return "No change: unsupported target remains insufficient evidence."
    if validation.status == "review_required":
        return "V3 marks ambiguous normalization for review."
    return "V3 still requires correction or evidence review."


def _looks_numeric(value: object) -> bool:
    try:
        float(str(value).replace(",", ""))
    except ValueError:
        return False
    return True


def bounded_text(text: str, limit: int) -> str:
    return normalize_space(text)[:limit]


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", text.lower())


def best_excerpt(text: str, query_terms: Sequence[str], *, max_chars: int) -> str:
    normalized = normalize_space(text)
    lower = normalized.lower()
    positions = [lower.find(term.lower()) for term in query_terms if lower.find(term.lower()) >= 0]
    if not positions:
        return normalized[:max_chars]
    center = min(positions)
    start = max(0, center - max_chars // 3)
    end = min(len(normalized), start + max_chars)
    return normalized[start:end]


def hierarchy_path(node_by_id: dict[str, HierarchyNode], node_id: str) -> list[str]:
    path: list[str] = []
    current = node_by_id[node_id]
    while True:
        path.append(current.title)
        if current.parent_node_id is None:
            return list(reversed(path))
        current = node_by_id[current.parent_node_id]


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-5-mini": (0.25, 2.00),
        "gpt-5": (1.25, 10.00),
    }
    input_rate, output_rate = pricing.get(model_name, (0.0, 0.0))
    return round(
        (input_tokens / 1_000_000 * input_rate) + (output_tokens / 1_000_000 * output_rate), 6
    )


def _extraction_prompt(target: TargetSpecification, bundle: EvidenceBundleRecord) -> str:
    return json.dumps(
        {
            "target": {
                "target_row_id": target.target_row_id,
                "requirement_id": target.requirement_id,
                "field_name": target.expected_field,
                "description": target.requirement_text,
                "domain": target.sub_domain,
                "datatype": target.expected_data_type,
                "unit": target.unit,
                "accepted_values": target.accepted_values,
            },
            "evidence": [
                {
                    "span_id": span.span_id,
                    "rank": span.retrieval_rank,
                    "source_id": span.source_id,
                    "source_file": span.source_file,
                    "page_number": span.page_number,
                    "node_id": span.hierarchy_node_id,
                    "canonical_text": span.text,
                }
                for span in bundle.evidence_spans
            ],
            "retrieval_status": bundle.retrieval_status,
            "instructions": (
                "Select supporting span IDs from the supplied evidence only. "
                "Do not provide free-form evidence as the citation; span text is authoritative. "
                "Use insufficient_evidence when no supplied span supports the answer."
            ),
            "required_json_shape": {
                "raw_value": "exact value as you would state it, or null",
                "extracted_value": "same as raw_value for backward compatibility, or null",
                "proposed_value_shape": (
                    "one of categorical, integer_count, decimal_measurement, date, "
                    "identifier_or_reference, boolean_or_presence, short_text, "
                    "descriptive_text, ordered_or_unordered_list, unsupported_or_unknown"
                ),
                "normalized_value": "proposed normalized value or null",
                "unit": "unit or null",
                "status": (
                    "one of extracted, multiple_candidates, insufficient_evidence, "
                    "conflicting_evidence, invalid_format, model_error"
                ),
                "confidence": "number 0..1",
                "selected_supporting_span_ids": "array of supplied span_id values or empty array",
                "value_bearing_quote": "exact value-bearing quote or token from selected span",
                "reasoning_summary": "short evidence interpretation only",
                "ambiguity_or_caveat": "short caveat or null",
            },
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    raise ValueError("No JSON object found.")


def _evidence_item_for_payload(
    bundle: EvidenceBundleRecord,
    payload: dict[str, Any],
) -> RetrievedEvidence | None:
    node_id = payload.get("hierarchy_node_id")
    source_id = payload.get("source_id")
    page_number = payload.get("page_number")
    for item in bundle.evidence_items:
        if node_id and item.node_id == node_id:
            return item
        if (
            source_id
            and page_number
            and item.source_id == source_id
            and item.page_start <= int(page_number) <= item.page_end
        ):
            return item
    return bundle.evidence_items[0] if bundle.evidence_items else None


def _model_error(
    target: TargetSpecification,
    provider: str,
    model_name: str,
    message: str,
) -> ExtractionResult:
    return ExtractionResult(
        target_row_id=target.target_row_id,
        requirement_id=target.requirement_id,
        status="model_error",
        confidence=0,
        model_provider=provider,
        model_name=model_name,
        ambiguity_or_caveat=message[:500],
    )


def _validate_datatype(target: TargetSpecification, extraction: ExtractionResult) -> list[str]:
    value = extraction.extracted_value
    issues: list[str] = []
    if value is None:
        return issues
    text = str(value)
    if target.expected_data_type == ExpectedDataType.DATE and not (
        re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
        or
        re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
        or re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b", text)
    ):
        issues.append("date value is not parseable")
    if target.expected_data_type == ExpectedDataType.INTEGER:
        try:
            int(text.replace(",", ""))
        except ValueError:
            issues.append("integer value is not parseable")
    if target.expected_data_type == ExpectedDataType.DECIMAL:
        try:
            float(text.replace(",", ""))
        except ValueError:
            issues.append("decimal value is not parseable")
    return issues


def _derive_count(span_text: str, target: TargetSpecification) -> int | None:
    field_words = set(tokenize(target.expected_field.replace("_", " ")))
    component_terms = {
        token
        for token in field_words
        if token not in {"count", "number", "quantity", "value"}
    }
    matches: list[int] = []
    for match in re.finditer(r"\b(\d+)\s*(?:no\.?|number)?\s*([A-Za-z][A-Za-z ]{0,60})", span_text):
        value = int(match.group(1))
        window = match.group(0).lower()
        if component_terms and not (set(tokenize(window)) & component_terms):
            continue
        matches.append(value)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return None


def _word_or_int_count(text: str) -> int | None:
    digit = re.fullmatch(r"\s*(\d+)\s*", text)
    if digit:
        return int(digit.group(1))
    words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    lowered = text.lower()
    found = [value for word, value in words.items() if re.search(rf"\b{word}\b", lowered)]
    return found[0] if len(found) == 1 else None


def _numeric_count(text: str) -> int:
    return len(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def _count_evidence_phrase(text: str, value: int) -> str | None:
    pattern = rf"\b{value}\s*(?:No\.?)?\s*[A-Za-z][A-Za-z /&-]{{0,80}}"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return normalize_space(match.group(0)) if match else None


def _derive_date(text: str) -> tuple[str, str] | None:
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if iso:
        value = iso.group(0)
        return value, value
    numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", text)
    if numeric:
        day, month, year = numeric.groups()
        year = f"20{year}" if len(year) == 2 else year
        return numeric.group(0), f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    month_lookup = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    human = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+("
        + "|".join(month_lookup)
        + r")\s+(\d{4})\b",
        text,
        flags=re.IGNORECASE,
    )
    if human:
        day, month_name, year = human.groups()
        month = month_lookup[month_name.lower()]
        return human.group(0), f"{int(year):04d}-{month:02d}-{int(day):02d}"
    return None


def _derive_use_classes(text: str) -> list[str]:
    values = re.findall(r"\bB\d[a-z]?\b", text, flags=re.IGNORECASE)
    ordered: list[str] = []
    for value in values:
        canonical = value[0].upper() + value[1:]
        if canonical not in ordered:
            ordered.append(canonical)
    return ordered


def _use_class_evidence_phrase(text: str, values: Sequence[str]) -> str | None:
    if not values:
        return None
    first = text.lower().find(values[0].lower())
    last = text.lower().find(values[-1].lower())
    if first < 0 or last < 0:
        return None
    end = last + len(values[-1])
    return normalize_space(text[first:end])


def _derive_measurement(
    text: str,
    preferred_unit: str | None,
) -> tuple[str, float, str | None] | None:
    unit_pattern = preferred_unit or r"m2|m²|sqm|sq\.?\s*m|kwp?|kwh?|m"
    matches = re.findall(
        rf"\b(\d+(?:\.\d+)?)\s*({unit_pattern})\b",
        text,
        flags=re.IGNORECASE,
    )
    if len(matches) != 1:
        return None
    value, unit = matches[0]
    numeric = float(value)
    normalized: float | int = int(numeric) if numeric.is_integer() else numeric
    phrase = f"{value} {unit}"
    return phrase, normalized, unit


def _best_supported_phrase(raw_text: str, span_text: str) -> str | None:
    cleaned = normalize_space(raw_text)
    if cleaned and cleaned.lower() in normalize_space(span_text).lower():
        return cleaned
    words = [word for word in re.split(r"\s+", cleaned) if len(word) > 2]
    if not words:
        return None
    lower_span = span_text.lower()
    positions = [
        lower_span.find(word.lower())
        for word in words
        if lower_span.find(word.lower()) >= 0
    ]
    if not positions:
        return None
    start = max(0, min(positions) - 80)
    end = min(len(span_text), min(positions) + 220)
    return normalize_space(span_text[start:end])


def _display_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def _normalization_status(
    extraction: ExtractionResult,
) -> Literal["normalized", "already_normalized", "ambiguous", "not_applicable", "failed"]:
    if extraction.status != "extracted":
        return "not_applicable"
    if extraction.normalized_value is None:
        return "failed"
    if extraction.raw_model_value != extraction.normalized_value:
        return "normalized"
    return "already_normalized"


def _dictionary_compatibility(
    target: TargetSpecification,
    assignment: ValueShapeAssignment,
    extraction: ExtractionResult,
) -> tuple[str, str, list[str]]:
    caveats: list[str] = []
    if assignment.ambiguity_or_mismatch_warning:
        caveats.append(assignment.ambiguity_or_mismatch_warning)
        return "suspected_dictionary_metadata_mismatch", "valid_with_dictionary_caveat", caveats
    if target.unit and assignment.value_shape_family in {
        "descriptive_text",
        "short_text",
        "categorical",
        "ordered_or_unordered_list",
    }:
        caveats.append("Dictionary unit appears not applicable to observed value shape.")
        return "unit_not_applicable", "valid_with_dictionary_caveat", caveats
    if (
        assignment.value_shape_family == "date"
        and extraction.raw_model_value != extraction.normalized_value
    ):
        return "compatible_after_normalization", "valid_after_normalization", caveats
    if assignment.value_shape_family == "ordered_or_unordered_list":
        return "compatible_after_normalization", "valid_after_normalization", caveats
    return "compatible", "valid", caveats


def _matches_expectation(item: RetrievedEvidence, expectation: RetrievalExpectation) -> bool:
    source_ok = (
        expectation.expected_source_id is None or item.source_id == expectation.expected_source_id
    )
    page_ok = not expectation.expected_pages or any(
        item.page_start <= page <= item.page_end for page in expectation.expected_pages
    )
    section_ok = (
        expectation.expected_section_contains is None
        or expectation.expected_section_contains.lower() in " ".join(item.hierarchy_path).lower()
    )
    return source_ok and page_ok and section_ok


def _build_telemetry(
    *,
    source_ranges: list[SourceRange],
    cache_results: list[CachedBatchParseResult],
    hierarchy: LightweightHierarchy,
    retrieval_results: list[RetrievalResult],
    bundles: list[EvidenceBundleRecord],
    extraction_results: list[ExtractionResult],
    validation_results: list[ValidationResult],
    extraction_time_ms_by_target: dict[str, float],
    provider: str,
    model: str,
    total_wall_time_ms: float,
) -> VerticalSliceTelemetry:
    usage = ModelUsage(
        input_tokens=sum(item.model_usage.input_tokens for item in extraction_results),
        output_tokens=sum(item.model_usage.output_tokens for item in extraction_results),
        estimated_cost_usd=round(
            sum(item.model_usage.estimated_cost_usd for item in extraction_results), 6
        ),
    )
    return VerticalSliceTelemetry(
        source_ranges_requested=source_ranges,
        parser_cache_hits=sum(result.cache_hits for result in cache_results),
        parser_cache_misses=sum(result.cache_misses for result in cache_results),
        parser_worker_invocations=sum(result.worker_invocation_count for result in cache_results),
        parser_active_children_after_cleanup=len(multiprocessing.active_children()),
        hierarchy_node_counts_by_type=dict(Counter(node.node_type for node in hierarchy.nodes)),
        retrieval_time_ms_by_target={
            result.target_row_id: result.retrieval_time_ms for result in retrieval_results
        },
        top_retrieval_scores={
            result.target_row_id: result.top_score for result in retrieval_results
        },
        evidence_bundle_character_count={
            bundle.target_row_id: bundle.character_count for bundle in bundles
        },
        evidence_bundle_token_estimate={
            bundle.target_row_id: bundle.token_estimate for bundle in bundles
        },
        llm_provider=provider,
        llm_model=model,
        llm_calls=sum(1 for result in extraction_results if _counted_llm_call(result)),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        estimated_cost_usd=usage.estimated_cost_usd,
        extraction_time_ms_by_target=extraction_time_ms_by_target,
        validation_status_by_target={
            result.target_row_id: result.status for result in validation_results
        },
        total_wall_time_ms=total_wall_time_ms,
    )


def _counted_llm_call(result: ExtractionResult) -> bool:
    if result.model_usage.input_tokens or result.model_usage.output_tokens:
        return True
    skipped_by_retriever = bool(
        result.ambiguity_or_caveat
        and result.ambiguity_or_caveat.startswith("Retriever returned ")
    )
    return not (
        result.status == "insufficient_evidence" and skipped_by_retriever
    )


def _write_review_csv(result: VerticalSliceRunResult, path: Path) -> None:
    target_by_id = {item.target.target_row_id: item.target for item in result.selected_targets}
    retrieval_by_id = {item.target_row_id: item for item in result.retrieval_results}
    validation_by_id = {item.target_row_id: item for item in result.validation_results}
    schema_by_id = {item.target_row_id: item for item in result.schema_compatibility_results}
    assignment_by_id = {item.target_row_id: item for item in result.value_shape_assignments}
    evidence_validation_by_id = {item.target_row_id: item for item in result.evidence_validation}
    shape_validation_by_id = {item.target_row_id: item for item in result.shape_validation}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_id",
                "domain",
                "sub_domain",
                "field_name",
                "value_shape_family",
                "raw_model_value",
                "evidence_value",
                "normalized_value",
                "display_value",
                "extracted_value",
                "unit",
                "status",
                "confidence",
                "source_file",
                "page",
                "evidence_excerpt",
                "canonical_evidence",
                "retrieval_rank",
                "validation_status",
                "evidence_validation",
                "shape_validation",
                "schema_compatibility",
                "overall_review_status",
                "caveat",
            ],
        )
        writer.writeheader()
        for extraction in result.extraction_results:
            target = target_by_id[extraction.target_row_id]
            retrieval = retrieval_by_id[extraction.target_row_id]
            rank = next(
                (
                    item.rank
                    for item in retrieval.results
                    if item.node_id == extraction.hierarchy_node_id
                ),
                retrieval.results[0].rank if retrieval.results else None,
            )
            schema = schema_by_id.get(extraction.target_row_id)
            validation = validation_by_id[extraction.target_row_id]
            assignment = assignment_by_id.get(extraction.target_row_id)
            evidence_validation = evidence_validation_by_id.get(extraction.target_row_id)
            shape_validation = shape_validation_by_id.get(extraction.target_row_id)
            writer.writerow(
                {
                    "target_id": extraction.target_row_id,
                    "domain": target.metadata.get("domain") or "",
                    "sub_domain": target.sub_domain,
                    "field_name": target.expected_field,
                    "value_shape_family": assignment.value_shape_family if assignment else "",
                    "raw_model_value": extraction.raw_model_value,
                    "evidence_value": extraction.evidence_value,
                    "normalized_value": extraction.normalized_value,
                    "display_value": extraction.display_value,
                    "extracted_value": extraction.extracted_value,
                    "unit": extraction.unit,
                    "status": extraction.status,
                    "confidence": extraction.confidence,
                    "source_file": extraction.source_file,
                    "page": extraction.page_number or extraction.page_range,
                    "evidence_excerpt": extraction.supporting_evidence_excerpt,
                    "canonical_evidence": extraction.supporting_evidence_excerpt,
                    "retrieval_rank": rank,
                    "validation_status": validation.status,
                    "evidence_validation": evidence_validation.status
                    if evidence_validation
                    else ("valid" if validation.evidence_valid else "invalid"),
                    "shape_validation": shape_validation.status if shape_validation else "",
                    "schema_compatibility": schema.compatibility if schema else "",
                    "overall_review_status": schema.overall_review_status if schema else "",
                    "caveat": extraction.ambiguity_or_caveat,
                }
            )


def _summary_markdown(result: VerticalSliceRunResult) -> str:
    status_counts = Counter(item.status for item in result.extraction_results)
    retrieval_top1 = sum(1 for item in result.retrieval_evaluation if item.top1_hit)
    retrieval_top3 = sum(1 for item in result.retrieval_evaluation if item.top3_hit)
    passing_statuses = {"valid", "valid_after_normalization", "valid_with_dictionary_caveat"}
    valid = sum(1 for item in result.validation_results if item.status in passing_statuses)
    if result.v2_v3_comparison:
        title = "Evidence-First Vertical Slice V3"
    elif result.v1_v2_comparison:
        title = "Evidence-First Vertical Slice V2"
    else:
        title = "Evidence-First Vertical Slice V1"
    schema_counts = Counter(item.compatibility for item in result.schema_compatibility_results)
    review_counts = Counter(item.status for item in result.validation_results)
    lines = [
        f"# {title}",
        "",
        f"- Targets: {len(result.selected_targets)}",
        f"- Source ranges: {len(result.source_ranges)}",
        f"- Hierarchy nodes: {len(result.hierarchy.nodes)}",
        f"- Retrieval top-1 hits: {retrieval_top1}",
        f"- Retrieval top-3 hits: {retrieval_top3}",
        f"- Extraction status counts: {dict(status_counts)}",
        f"- Passing review statuses: {valid}",
        f"- Review status counts: {dict(review_counts)}",
        f"- Schema compatibility counts: {dict(schema_counts)}",
        f"- LLM calls: {result.telemetry.llm_calls}",
        f"- Tokens: input {result.telemetry.input_tokens}, output {result.telemetry.output_tokens}",
        f"- Estimated cost USD: {result.telemetry.estimated_cost_usd}",
        "",
        "## Target Results",
        "",
    ]
    target_by_id = {item.target.target_row_id: item.target for item in result.selected_targets}
    validation_by_id = {item.target_row_id: item for item in result.validation_results}
    for extraction in result.extraction_results:
        target = target_by_id[extraction.target_row_id]
        validation = validation_by_id[extraction.target_row_id]
        caveat = extraction.ambiguity_or_caveat or ""
        lines.append(
            f"- `{target.expected_field}`: {extraction.status}, "
            f"value={extraction.display_value or extraction.extracted_value!r}, "
            f"validation={validation.status}, "
            f"page={extraction.page_number}, caveat={caveat}"
        )
    return "\n".join(lines) + "\n"


def _comparison_markdown(comparison: V1V2Comparison) -> str:
    lines = [
        "# V1 versus V2 Comparison",
        "",
        f"- V1 top-1 hits: {comparison.v1_top1_hits}/14",
        f"- V1 top-3 hits: {comparison.v1_top3_hits}/14",
        f"- V2 top-1 hits: {comparison.v2_top1_hits}/14",
        f"- V2 top-3 hits: {comparison.v2_top3_hits}/14",
        "",
        "| Target | V1 top1/top3 | V2 top1/top3 | V1 extraction | "
        "V2 extraction | V2 schema | Outcome |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for item in comparison.items:
        lines.append(
            "| "
            f"{item.field_name} | "
            f"{item.v1_top1_hit}/{item.v1_top3_hit} | "
            f"{item.v2_top1_hit}/{item.v2_top3_hit} | "
            f"{item.v1_extraction_status}: {item.v1_extraction_value} | "
            f"{item.v2_extraction_status}: {item.v2_extraction_value} | "
            f"{item.v2_schema_compatibility} | "
            f"{item.outcome} |"
        )
    return "\n".join(lines) + "\n"


def _v2_v3_comparison_markdown(comparison: V2V3Comparison) -> str:
    lines = [
        "# V2 versus V3 Comparison",
        "",
        f"- V2 review status counts: {comparison.v2_status_counts}",
        f"- V3 review status counts: {comparison.v3_status_counts}",
        "",
        "| Target | Shape | V2 validation/schema | V3 status/schema | Change |",
        "|---|---|---|---|---|",
    ]
    for item in comparison.items:
        lines.append(
            "| "
            f"{item.field_name} | "
            f"{item.v3_value_shape} | "
            f"{item.v2_validation_status}/{item.v2_schema_compatibility} | "
            f"{item.v3_overall_review_status}/{item.v3_schema_compatibility} | "
            f"{item.v3_change} |"
        )
    return "\n".join(lines) + "\n"


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(
        f".{path.name}.{hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:12]}.tmp"
    )
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)
