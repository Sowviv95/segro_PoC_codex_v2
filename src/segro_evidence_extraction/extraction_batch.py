"""Bounded evidence-first extraction batch runner."""

from __future__ import annotations

import csv
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field

from segro_evidence_extraction.config import Settings, load_settings
from segro_evidence_extraction.dictionary.service import ingest_dictionary
from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.common import ExpectedDataType
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
from segro_evidence_extraction.vertical_slice import (
    DEFAULT_CACHE_ROOT,
    MAX_BATCH_PAGES,
    MAX_EVIDENCE_CHARS,
    AbstainingExtractionClient,
    EvidenceBundleRecord,
    EvidenceSpan,
    EvidenceValidationResult,
    ExtractionClient,
    ExtractionResult,
    JsonExtractionClient,
    LightweightHierarchy,
    NormalizedValueRecord,
    RetrievalEvaluationItem,
    RetrievalExpectation,
    RetrievalResult,
    SchemaCompatibilityResult,
    SelectedTarget,
    ShapeValidationResult,
    SourceRange,
    ValidationResult,
    ValueShapeAssignment,
    ValueShapeFamily,
    VerticalSliceTelemetry,
    _atomic_write_json,
    _build_telemetry,
    _display_value,
    _insufficient_evidence_result,
    _normalization_status,
    assess_schema_compatibility_v3,
    build_evidence_bundle,
    build_lightweight_hierarchy,
    build_overall_validation_results,
    calibrate_extraction_value,
    estimate_cost_usd,
    estimate_tokens,
    infer_value_shape_assignment,
    materialize_selected_spans,
    retrieve_evidence,
    split_range,
    validate_evidence_layer,
    validate_shape_layer,
)

DEFAULT_BATCH_OUTPUT_DIR = Path("output/enfield_unit1_evidence_first_batch_v1")
BATCH_TARGET_COUNT = 75
BATCH_REVIEW_SAMPLE_COUNT = 20
DEFAULT_COST_CEILING_USD = 0.10

SupportStatus = Literal["likely_supported", "uncertain", "expected_unsupported"]


class BatchSelectedTarget(StrictBaseModel):
    target: TargetSpecification
    value_shape: ValueShapeAssignment
    selection_reason: str
    expected_evidence_source_category: str
    expected_support_status: SupportStatus


class SourceRangePlan(StrictBaseModel):
    ranges: list[SourceRange]
    unique_pages_requested: int
    warnings: list[str] = Field(default_factory=list)


class BatchPreflightReport(StrictBaseModel):
    selected_target_count: int
    target_count_by_domain: dict[str, int]
    target_count_by_value_shape_family: dict[str, int]
    support_status_counts: dict[str, int]
    source_ranges: list[SourceRange]
    total_unique_pages_requested: int
    cache_hits_expected: int
    cache_misses_expected: int
    expected_parser_worker_invocations: int
    expected_llm_calls: int
    estimated_upper_input_tokens: int
    estimated_upper_output_tokens: int
    estimated_upper_cost_usd: float
    configured_model: str
    cost_ceiling_usd: float
    safety_gate_passed: bool
    safety_gate_errors: list[str] = Field(default_factory=list)


class HierarchySummary(StrictBaseModel):
    node_counts_by_type: dict[str, int]
    node_counts_by_source_and_type: dict[str, dict[str, int]]
    warnings: list[str] = Field(default_factory=list)


class FailureAnalysisItem(StrictBaseModel):
    target_row_id: str
    field_name: str
    domain: str
    value_shape_family: ValueShapeFamily
    extraction_status: str
    overall_review_status: str
    failure_category: str
    issues: list[str] = Field(default_factory=list)


class FailureAnalysis(StrictBaseModel):
    counts_by_failure_category: dict[str, int]
    items: list[FailureAnalysisItem]


class BatchRunResult(StrictBaseModel):
    output_dir: str
    selected_targets: list[BatchSelectedTarget]
    source_range_plan: SourceRangePlan
    preflight_report: BatchPreflightReport
    hierarchy: LightweightHierarchy
    hierarchy_summary: HierarchySummary
    retrieval_results: list[RetrievalResult]
    retrieval_sample_expectations: list[RetrievalExpectation]
    retrieval_sample_evaluation: list[RetrievalEvaluationItem]
    evidence_bundles: list[EvidenceBundleRecord]
    evidence_spans: list[EvidenceSpan]
    extraction_results: list[ExtractionResult]
    normalized_values: list[NormalizedValueRecord]
    evidence_validation: list[EvidenceValidationResult]
    shape_validation: list[ShapeValidationResult]
    schema_compatibility_results: list[SchemaCompatibilityResult]
    validation_results: list[ValidationResult]
    telemetry: VerticalSliceTelemetry
    failure_analysis: FailureAnalysis


class EvidenceFirstBatchService:
    def __init__(
        self,
        *,
        source_manifest: Path,
        dictionary_path: Path,
        output_dir: Path = DEFAULT_BATCH_OUTPUT_DIR,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        settings: Settings | None = None,
        extraction_client: ExtractionClient | None = None,
        batch_config: BatchWorkerConfig | None = None,
        max_evidence_chars: int = MAX_EVIDENCE_CHARS,
        cost_ceiling_usd: float = DEFAULT_COST_CEILING_USD,
    ) -> None:
        self.source_manifest = source_manifest
        self.dictionary_path = dictionary_path
        self.output_dir = output_dir
        self.cache_root = cache_root
        self.settings = settings or load_settings(Path("configs/default.yaml"))
        self.extraction_client = extraction_client or self._build_extraction_client()
        self.batch_config = batch_config or BatchWorkerConfig()
        self.max_evidence_chars = max_evidence_chars
        self.cost_ceiling_usd = cost_ceiling_usd

    def run(self) -> BatchRunResult:
        started = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        source_registry = {
            source.source_id: source for source in load_source_registry(self.source_manifest)
        }
        selected_targets = select_batch_targets(self.dictionary_path, self.output_dir)
        source_plan = plan_batch_source_ranges(source_registry)
        preflight = build_preflight_report(
            selected_targets=selected_targets,
            source_plan=source_plan,
            source_registry=source_registry,
            output_dir=self.output_dir,
            cache_root=self.cache_root,
            model_name=self.extraction_client.model_name,
            cost_ceiling_usd=self.cost_ceiling_usd,
        )
        write_preflight_artifacts(selected_targets, source_plan, preflight, self.output_dir)
        if not preflight.safety_gate_passed:
            msg = "; ".join(preflight.safety_gate_errors)
            raise ValueError(f"Batch preflight failed: {msg}")

        pages_by_source, cache_results = self._load_pages(source_registry, source_plan.ranges)
        hierarchy = build_lightweight_hierarchy(pages_by_source, source_registry)
        hierarchy_summary = summarize_hierarchy(hierarchy, source_registry)

        retrieval_results: list[RetrievalResult] = []
        evidence_bundles: list[EvidenceBundleRecord] = []
        evidence_spans: list[EvidenceSpan] = []
        selected_wrapped = [
            SelectedTarget(target=item.target, selection_reason=item.selection_reason)
            for item in selected_targets
        ]
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
            evidence_bundles.append(bundle)
            evidence_spans.extend(bundle.evidence_spans)

        sample_expectations = build_retrieval_sample_expectations(
            selected_targets,
            retrieval_results,
        )
        sample_evaluation = evaluate_batch_retrieval_sample(
            sample_expectations,
            retrieval_results,
        )

        extraction_results: list[ExtractionResult] = []
        extraction_time_ms_by_target: dict[str, float] = {}
        for selected, bundle in zip(selected_targets, evidence_bundles, strict=True):
            extract_start = time.perf_counter()
            if should_skip_extraction(selected, bundle):
                extraction = _insufficient_evidence_result(
                    selected.target,
                    self.extraction_client.provider,
                    self.extraction_client.model_name,
                    f"Retriever returned {bundle.retrieval_status}; batch skipped extraction.",
                )
            else:
                extraction = self.extraction_client.extract(
                    target=selected.target,
                    bundle=bundle,
                    max_output_tokens=600,
                )
                extraction = materialize_selected_spans(extraction, bundle)
            extraction = calibrate_extraction_value(
                extraction=extraction,
                target=selected.target,
                assignment=selected.value_shape,
                bundle=bundle,
            )
            extraction_results.append(extraction)
            extraction_time_ms_by_target[selected.target.target_row_id] = (
                time.perf_counter() - extract_start
            ) * 1000

        normalized_values = [
            NormalizedValueRecord(
                target_row_id=item.target_row_id,
                value_shape_family=selected.value_shape.value_shape_family,
                raw_model_value=item.raw_model_value,
                evidence_value=item.evidence_value,
                normalized_value=item.normalized_value,
                display_value=item.display_value or _display_value(item.normalized_value),
                unit=item.unit,
                normalization_status=_normalization_status(item),
                issues=[],
            )
            for item, selected in zip(extraction_results, selected_targets, strict=True)
        ]
        evidence_validation = validate_evidence_layer(
            extraction_results,
            selected_wrapped,
            pages_by_source,
            hierarchy,
            evidence_bundles,
        )
        shape_validation = validate_shape_layer(
            extraction_results,
            [item.value_shape for item in selected_targets],
        )
        schema_results = assess_schema_compatibility_v3(
            extraction_results,
            selected_wrapped,
            [item.value_shape for item in selected_targets],
            evidence_validation,
            shape_validation,
        )
        validation_results = build_overall_validation_results(
            extraction_results,
            selected_wrapped,
            evidence_validation,
            shape_validation,
            schema_results,
        )
        telemetry = _build_telemetry(
            source_ranges=source_plan.ranges,
            cache_results=cache_results,
            hierarchy=hierarchy,
            retrieval_results=retrieval_results,
            bundles=evidence_bundles,
            extraction_results=extraction_results,
            validation_results=validation_results,
            extraction_time_ms_by_target=extraction_time_ms_by_target,
            provider=self.extraction_client.provider,
            model=self.extraction_client.model_name,
            total_wall_time_ms=(time.perf_counter() - started) * 1000,
        )
        failure_analysis = build_failure_analysis(
            selected_targets,
            extraction_results,
            validation_results,
            evidence_validation,
            shape_validation,
            schema_results,
        )
        result = BatchRunResult(
            output_dir=str(self.output_dir),
            selected_targets=selected_targets,
            source_range_plan=source_plan,
            preflight_report=preflight,
            hierarchy=hierarchy,
            hierarchy_summary=hierarchy_summary,
            retrieval_results=retrieval_results,
            retrieval_sample_expectations=sample_expectations,
            retrieval_sample_evaluation=sample_evaluation,
            evidence_bundles=evidence_bundles,
            evidence_spans=evidence_spans,
            extraction_results=extraction_results,
            normalized_values=normalized_values,
            evidence_validation=evidence_validation,
            shape_validation=shape_validation,
            schema_compatibility_results=schema_results,
            validation_results=validation_results,
            telemetry=telemetry,
            failure_analysis=failure_analysis,
        )
        write_batch_artifacts(result, self.output_dir)
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


def select_batch_targets(dictionary_path: Path, output_dir: Path) -> list[BatchSelectedTarget]:
    dictionary_result = ingest_dictionary(
        dictionary_path,
        sheet_name="Extraction Template",
        mapping_config=Path("configs/dictionaries/segro_extraction_template_v1.yaml"),
        output_dir=output_dir / "dictionary_selection_ingestion",
    )
    selected: list[BatchSelectedTarget] = []
    used: set[str] = set()
    quotas: dict[ValueShapeFamily, int] = {
        "descriptive_text": 15,
        "categorical": 10,
        "integer_count": 10,
        "decimal_measurement": 10,
        "date": 8,
        "identifier_or_reference": 7,
        "ordered_or_unordered_list": 5,
        "boolean_or_presence": 5,
        "unsupported_or_unknown": 5,
    }
    targets_by_shape: dict[str, list[TargetSpecification]] = defaultdict(list)
    for target in dictionary_result.normalized_targets:
        assignment = infer_batch_value_shape(target)
        targets_by_shape[assignment.value_shape_family].append(target)
    for shape, quota in quotas.items():
        candidates = sorted(
            targets_by_shape.get(shape, []),
            key=lambda target: (
                -support_score(target),
                domain_priority(target.sub_domain),
                target.source_dictionary_provenance.row_number
                if target.source_dictionary_provenance
                else 99999,
                target.expected_field,
            ),
        )
        for target in candidates:
            shape_count = sum(
                1 for item in selected if item.value_shape.value_shape_family == shape
            )
            if shape_count >= quota:
                break
            if target.target_row_id in used:
                continue
            selected.append(batch_selected_target(target, shape))
            used.add(target.target_row_id)
    unsupported_candidates = sorted(
        (
            target
            for target in dictionary_result.normalized_targets
            if target.target_row_id not in used
            and expected_support_status(target) == "expected_unsupported"
        ),
        key=lambda target: (
            domain_priority(target.sub_domain),
            target.source_dictionary_provenance.row_number
            if target.source_dictionary_provenance
            else 99999,
            target.expected_field,
        ),
    )
    while (
        sum(1 for item in selected if item.expected_support_status == "expected_unsupported")
        < quotas["unsupported_or_unknown"]
        and unsupported_candidates
    ):
        target = unsupported_candidates.pop(0)
        selected.append(
            batch_selected_target(
                target,
                infer_batch_value_shape(target).value_shape_family,
            )
        )
        used.add(target.target_row_id)
    if len(selected) < BATCH_TARGET_COUNT:
        remaining = [
            target
            for target in dictionary_result.normalized_targets
            if target.target_row_id not in used
        ]
        for target in sorted(
            remaining,
            key=lambda item: (
                -support_score(item),
                domain_priority(item.sub_domain),
                item.source_dictionary_provenance.row_number
                if item.source_dictionary_provenance
                else 99999,
            ),
        ):
            selected.append(
                batch_selected_target(
                    target,
                    infer_batch_value_shape(target).value_shape_family,
                )
            )
            used.add(target.target_row_id)
            if len(selected) == BATCH_TARGET_COUNT:
                break
    selected = sorted(
        selected[:BATCH_TARGET_COUNT],
        key=lambda item: (
            item.value_shape.value_shape_family,
            item.target.sub_domain,
            item.target.source_dictionary_provenance.row_number
            if item.target.source_dictionary_provenance
            else 99999,
            item.target.expected_field,
        ),
    )
    if len(selected) != BATCH_TARGET_COUNT:
        msg = f"Selected {len(selected)} targets; expected {BATCH_TARGET_COUNT}."
        raise ValueError(msg)
    return selected


def infer_batch_value_shape(target: TargetSpecification) -> ValueShapeAssignment:
    field = target.expected_field.lower()
    if any(token in field for token in ["use_classes", "classes", "use_class"]):
        base = infer_value_shape_assignment(target)
        return base.model_copy(update={"value_shape_family": "ordered_or_unordered_list"})
    if target.expected_data_type == ExpectedDataType.BOOLEAN or field.startswith("has_"):
        base = infer_value_shape_assignment(target)
        return base.model_copy(
            update={
                "value_shape_family": "boolean_or_presence",
                "inference_basis": [*base.inference_basis, "batch-boolean-datatype-or-field"],
            }
        )
    if target.expected_data_type == ExpectedDataType.DECIMAL and any(
        token in field for token in ["reference", "model", "manufacturer", "name"]
    ):
        base = infer_value_shape_assignment(target)
        return base.model_copy(
            update={
                "value_shape_family": "identifier_or_reference",
                "ambiguity_or_mismatch_warning": (
                    "Dictionary declares decimal but field semantics indicate identifier text."
                ),
            }
        )
    return infer_value_shape_assignment(target)


def batch_selected_target(
    target: TargetSpecification,
    intended_shape: ValueShapeFamily,
) -> BatchSelectedTarget:
    assignment = infer_batch_value_shape(target)
    if assignment.value_shape_family != intended_shape:
        assignment = assignment.model_copy(update={"value_shape_family": intended_shape})
    support_status = expected_support_status(target)
    return BatchSelectedTarget(
        target=target,
        value_shape=assignment,
        selection_reason=selection_reason(target, assignment, support_status),
        expected_evidence_source_category=expected_source_category(target),
        expected_support_status=support_status,
    )


def support_score(target: TargetSpecification) -> int:
    field = target.expected_field.lower()
    text = f"{field} {target.requirement_text} {target.source_guidance or ''}".lower()
    score = 0
    for term in [
        "construction",
        "planning",
        "certificate",
        "completion",
        "dock",
        "door",
        "roof",
        "wall",
        "cladding",
        "floor",
        "fire",
        "pv",
        "photovoltaic",
        "fencing",
        "gate",
        "concrete",
        "area",
        "date",
    ]:
        if term in text:
            score += 2
    if target.sub_domain in {
        "Technical Specification",
        "Component",
        "Statutory Compliance",
        "Planning Consent",
        "Alterations",
        "Size",
        "Energy",
        "External Areas",
        "Structural Analysis",
    }:
        score += 3
    return score


def domain_priority(domain: str) -> int:
    order = [
        "Technical Specification",
        "Component",
        "Statutory Compliance",
        "Planning Consent",
        "Alterations",
        "Size",
        "Energy",
        "External Areas",
        "Structural Analysis",
        "Property",
    ]
    return order.index(domain) if domain in order else len(order)


def expected_support_status(target: TargetSpecification) -> SupportStatus:
    score = support_score(target)
    if target.sub_domain in {"Legal", "Location", "Property"} and score < 4:
        return "expected_unsupported"
    if score >= 6:
        return "likely_supported"
    return "uncertain"


def expected_source_category(target: TargetSpecification) -> str:
    field = target.expected_field.lower()
    if "planning" in field or target.sub_domain == "Planning Consent":
        return "planning approval"
    if "certificate" in field or "date" in field:
        return "certificates"
    if any(term in field for term in ["dock", "door", "roof", "wall", "floor", "cladding"]):
        return "building manuals"
    if any(term in field for term in ["pv", "photovoltaic", "energy"]):
        return "energy and appendices"
    if "fire" in field:
        return "health and safety"
    return "bounded manuals"


def selection_reason(
    target: TargetSpecification,
    assignment: ValueShapeAssignment,
    support_status: SupportStatus,
) -> str:
    return (
        f"Deterministic 75-target batch selection for {assignment.value_shape_family}; "
        f"domain={target.sub_domain}; support={support_status}."
    )


def plan_batch_source_ranges(
    source_registry: dict[str, SourceRegistryEntry],
) -> SourceRangePlan:
    by_path = {source.logical_path: source for source in source_registry.values()}
    desired = [
        ("Building Manual - Part 1 General.pdf", 4, 11, "V3 description/plans/PV/dock text."),
        ("Building Manual - Part 1 General.pdf", 14, 22, "Planning approvals and use classes."),
        (
            "Building Manual - Part 1 General.pdf",
            41,
            46,
            "Completion/building control certificates.",
        ),
        (
            "Building Manual - Part 2 Building Fabric.pdf",
            1,
            60,
            "Bounded building fabric index/specification pages.",
        ),
        (
            "Building Manual - Part 3 Building Services.pdf",
            1,
            30,
            "Bounded services index/fire/electrical pages.",
        ),
        (
            "Building Manual - Part 4 External Works.pdf",
            4,
            14,
            "External works/slab/fencing pages.",
        ),
        (
            "Building Manual - Part 5 The Health & Safety File.pdf",
            1,
            10,
            "Health and safety/fire/access pages.",
        ),
        (
            "Building Manual - Part 6 Appendices.pdf",
            4,
            10,
            "Appendices/certificates/maintenance/PV pages.",
        ),
        (
            "Rolec EV Charger Pre-Commissioning Information Sheet - Enfield U1.pdf",
            1,
            1,
            "Small bounded commissioning sample for EV/component targets.",
        ),
    ]
    ranges: list[SourceRange] = []
    for logical_path, start, end, reason in desired:
        if logical_path not in by_path:
            continue
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
    unique_pages = {
        (item.source_id, page)
        for item in ranges
        for page in range(item.page_start, item.page_end + 1)
    }
    return SourceRangePlan(ranges=ranges, unique_pages_requested=len(unique_pages))


def build_preflight_report(
    *,
    selected_targets: list[BatchSelectedTarget],
    source_plan: SourceRangePlan,
    source_registry: dict[str, SourceRegistryEntry],
    output_dir: Path,
    cache_root: Path,
    model_name: str,
    cost_ceiling_usd: float,
) -> BatchPreflightReport:
    cache_service = CachedBatchParsingService(cache=CanonicalParsedPageCache(cache_root))
    hits = 0
    misses = 0
    worker_invocations = 0
    for source_range in source_plan.ranges:
        source = source_registry[source_range.source_id]
        page_numbers = list(range(source_range.page_start, source_range.page_end + 1))
        lookup = cache_service.cache.read_range(
            source=source,
            source_path=source.original_path,
            page_numbers=page_numbers,
            parser_name=cache_service.parser_name,
            parser_version=cache_service.parser_version,
            parser_config_fingerprint=cache_service.parser_config_fingerprint,
        )
        hits += lookup.hits
        misses += lookup.misses
        worker_invocations += len(split_range_for_pages(lookup.missing_pages))
    eligible_calls = sum(
        1 for item in selected_targets if item.expected_support_status != "expected_unsupported"
    )
    max_input = eligible_calls * estimate_tokens("x" * MAX_EVIDENCE_CHARS)
    max_output = eligible_calls * 600
    cost = estimate_cost_usd(model_name, max_input, max_output)
    errors = preflight_errors(
        selected_targets,
        source_plan,
        eligible_calls,
        cost,
        cost_ceiling_usd,
        source_registry,
    )
    return BatchPreflightReport(
        selected_target_count=len(selected_targets),
        target_count_by_domain=dict(Counter(item.target.sub_domain for item in selected_targets)),
        target_count_by_value_shape_family=dict(
            Counter(item.value_shape.value_shape_family for item in selected_targets)
        ),
        support_status_counts=dict(
            Counter(item.expected_support_status for item in selected_targets)
        ),
        source_ranges=source_plan.ranges,
        total_unique_pages_requested=source_plan.unique_pages_requested,
        cache_hits_expected=hits,
        cache_misses_expected=misses,
        expected_parser_worker_invocations=worker_invocations,
        expected_llm_calls=eligible_calls,
        estimated_upper_input_tokens=max_input,
        estimated_upper_output_tokens=max_output,
        estimated_upper_cost_usd=cost,
        configured_model=model_name,
        cost_ceiling_usd=cost_ceiling_usd,
        safety_gate_passed=not errors,
        safety_gate_errors=errors,
    )


def split_range_for_pages(pages: list[int]) -> list[tuple[int, int]]:
    if not pages:
        return []
    ranges: list[tuple[int, int]] = []
    sorted_pages = sorted(pages)
    start = previous = sorted_pages[0]
    for page in sorted_pages[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.extend(split_range(start, previous, MAX_BATCH_PAGES))
        start = previous = page
    ranges.extend(split_range(start, previous, MAX_BATCH_PAGES))
    return ranges


def preflight_errors(
    selected_targets: list[BatchSelectedTarget],
    source_plan: SourceRangePlan,
    expected_calls: int,
    cost: float,
    ceiling: float,
    source_registry: dict[str, SourceRegistryEntry],
) -> list[str]:
    errors: list[str] = []
    if len(selected_targets) != BATCH_TARGET_COUNT:
        errors.append("target count is not exactly 75")
    for source_range in source_plan.ranges:
        if source_range.page_start < 1 or source_range.page_end < source_range.page_start:
            errors.append(f"unbounded or invalid source range: {source_range.logical_path}")
        source = source_registry[source_range.source_id]
        if source.page_count and (
            source_range.page_start == 1 and source_range.page_end >= source.page_count
        ):
            errors.append(f"whole-manual parsing requested: {source_range.logical_path}")
    if source_plan.unique_pages_requested > 200:
        errors.append(
            f"source plan requests {source_plan.unique_pages_requested} unique pages; "
            "bounded batch limit is 200"
        )
    if expected_calls > BATCH_TARGET_COUNT:
        errors.append("expected LLM calls exceed 75")
    if cost > ceiling:
        errors.append(f"estimated cost {cost} exceeds ceiling {ceiling}")
    return errors


def write_preflight_artifacts(
    selected_targets: list[BatchSelectedTarget],
    source_plan: SourceRangePlan,
    preflight: BatchPreflightReport,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        output_dir / "selected_targets.json",
        [item.model_dump(mode="json") for item in selected_targets],
    )
    _atomic_write_json(output_dir / "source_range_plan.json", source_plan.model_dump(mode="json"))
    _atomic_write_json(output_dir / "preflight_report.json", preflight.model_dump(mode="json"))
    (output_dir / "target_selection_summary.md").write_text(
        target_selection_summary_markdown(selected_targets, preflight),
        encoding="utf-8",
    )


def build_retrieval_sample_expectations(
    selected_targets: list[BatchSelectedTarget],
    retrieval_results: list[RetrievalResult],
) -> list[RetrievalExpectation]:
    by_id = {item.target_row_id: item for item in retrieval_results}
    chosen: list[BatchSelectedTarget] = []
    statuses = ["likely_supported", "uncertain", "expected_unsupported"]
    shapes = sorted({item.value_shape.value_shape_family for item in selected_targets})
    for status in statuses:
        for shape in shapes:
            candidate = next(
                (
                    item
                    for item in selected_targets
                    if item.expected_support_status == status
                    and item.value_shape.value_shape_family == shape
                    and item not in chosen
                ),
                None,
            )
            if candidate is not None:
                chosen.append(candidate)
            if len(chosen) == BATCH_REVIEW_SAMPLE_COUNT:
                break
        if len(chosen) == BATCH_REVIEW_SAMPLE_COUNT:
            break
    for item in selected_targets:
        if len(chosen) == BATCH_REVIEW_SAMPLE_COUNT:
            break
        if item not in chosen:
            chosen.append(item)
    expectations: list[RetrievalExpectation] = []
    for item in chosen[:BATCH_REVIEW_SAMPLE_COUNT]:
        retrieval = by_id[item.target.target_row_id]
        expected_pages: list[int] = []
        expected_source_id: str | None = None
        if item.expected_support_status != "expected_unsupported" and retrieval.results:
            expected_pages = [retrieval.results[0].page_start]
            expected_source_id = retrieval.results[0].source_id
        expectations.append(
            RetrievalExpectation(
                target_row_id=item.target.target_row_id,
                expected_source_id=expected_source_id,
                expected_pages=expected_pages,
                no_supported_evidence_expected=item.expected_support_status
                == "expected_unsupported",
                notes=(
                    "Batch V1 stratified retrieval review sample; expectation seeded from "
                    "deterministic source planning and top retrieved bounded page."
                ),
            )
        )
    return expectations


def evaluate_batch_retrieval_sample(
    expectations: list[RetrievalExpectation],
    retrieval_results: list[RetrievalResult],
) -> list[RetrievalEvaluationItem]:
    by_id = {item.target_row_id: item for item in retrieval_results}
    evaluations: list[RetrievalEvaluationItem] = []
    for expectation in expectations:
        result = by_id[expectation.target_row_id]
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
        top1 = bool(
            result.results
            and result.results[0].source_id == expectation.expected_source_id
            and result.results[0].page_start in expectation.expected_pages
        )
        top3 = any(
            item.source_id == expectation.expected_source_id
            and item.page_start in expectation.expected_pages
            for item in result.results[:3]
        )
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


def should_skip_extraction(
    selected: BatchSelectedTarget,
    bundle: EvidenceBundleRecord,
) -> bool:
    if bundle.retrieval_status in {"no_relevant_evidence", "weak_evidence"}:
        return True
    return (
        selected.expected_support_status == "expected_unsupported"
        and not bundle.evidence_spans
    )


def summarize_hierarchy(
    hierarchy: LightweightHierarchy,
    source_registry: dict[str, SourceRegistryEntry],
) -> HierarchySummary:
    counts_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for node in hierarchy.nodes:
        source = source_registry[node.source_id]
        counts_by_source[source.logical_path][node.node_type] += 1
    return HierarchySummary(
        node_counts_by_type=dict(Counter(node.node_type for node in hierarchy.nodes)),
        node_counts_by_source_and_type={
            source: dict(counts) for source, counts in sorted(counts_by_source.items())
        },
        warnings=hierarchy.warnings,
    )


def build_failure_analysis(
    selected_targets: list[BatchSelectedTarget],
    extraction_results: list[ExtractionResult],
    validation_results: list[ValidationResult],
    evidence_results: list[EvidenceValidationResult],
    shape_results: list[ShapeValidationResult],
    schema_results: list[SchemaCompatibilityResult],
) -> FailureAnalysis:
    extraction_by_id = {item.target_row_id: item for item in extraction_results}
    validation_by_id = {item.target_row_id: item for item in validation_results}
    evidence_by_id = {item.target_row_id: item for item in evidence_results}
    shape_by_id = {item.target_row_id: item for item in shape_results}
    schema_by_id = {item.target_row_id: item for item in schema_results}
    items: list[FailureAnalysisItem] = []
    for selected in selected_targets:
        target_id = selected.target.target_row_id
        validation = validation_by_id[target_id]
        passing_statuses = {
            "valid",
            "valid_after_normalization",
            "valid_with_dictionary_caveat",
        }
        if validation.status in passing_statuses:
            continue
        extraction = extraction_by_id[target_id]
        category = failure_category(
            extraction,
            validation,
            evidence_by_id[target_id],
            shape_by_id[target_id],
            schema_by_id[target_id],
        )
        items.append(
            FailureAnalysisItem(
                target_row_id=target_id,
                field_name=selected.target.expected_field,
                domain=selected.target.sub_domain,
                value_shape_family=selected.value_shape.value_shape_family,
                extraction_status=extraction.status,
                overall_review_status=validation.status,
                failure_category=category,
                issues=validation.issues,
            )
        )
    return FailureAnalysis(
        counts_by_failure_category=dict(Counter(item.failure_category for item in items)),
        items=items,
    )


def failure_category(
    extraction: ExtractionResult,
    validation: ValidationResult,
    evidence: EvidenceValidationResult,
    shape: ShapeValidationResult,
    schema: SchemaCompatibilityResult,
) -> str:
    if extraction.status in {"model_error", "invalid_format"}:
        return "model_error"
    if validation.status == "insufficient_evidence":
        return "insufficient_evidence"
    if getattr(evidence, "status", "") == "invalid":
        return "invalid_evidence"
    if shape.status in {"invalid", "review_required"}:
        return "normalization_or_shape"
    if schema.compatibility in {"incompatible", "incompatible_value"}:
        return "dictionary_compatibility"
    return "review_required"


def write_batch_artifacts(result: BatchRunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, object] = {
        "selected_targets.json": [
            item.model_dump(mode="json") for item in result.selected_targets
        ],
        "source_range_plan.json": result.source_range_plan.model_dump(mode="json"),
        "preflight_report.json": result.preflight_report.model_dump(mode="json"),
        "hierarchy.json": result.hierarchy.model_dump(mode="json"),
        "hierarchy_summary.json": result.hierarchy_summary.model_dump(mode="json"),
        "retrieval_results.json": [
            item.model_dump(mode="json") for item in result.retrieval_results
        ],
        "retrieval_sample_expectations.json": [
            item.model_dump(mode="json") for item in result.retrieval_sample_expectations
        ],
        "retrieval_sample_evaluation.json": [
            item.model_dump(mode="json") for item in result.retrieval_sample_evaluation
        ],
        "evidence_bundles.json": [
            item.model_dump(mode="json") for item in result.evidence_bundles
        ],
        "evidence_spans.json": [
            item.model_dump(mode="json") for item in result.evidence_spans
        ],
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
        "schema_compatibility_results.json": [
            item.model_dump(mode="json")
            for item in result.schema_compatibility_results
        ],
        "validation_results.json": [
            item.model_dump(mode="json") for item in result.validation_results
        ],
        "telemetry.json": result.telemetry.model_dump(mode="json"),
        "failure_analysis.json": result.failure_analysis.model_dump(mode="json"),
    }
    for filename, payload in artifacts.items():
        _atomic_write_json(output_dir / filename, payload)
    (output_dir / "target_selection_summary.md").write_text(
        target_selection_summary_markdown(result.selected_targets, result.preflight_report),
        encoding="utf-8",
    )
    (output_dir / "extraction_summary.md").write_text(
        batch_summary_markdown(result), encoding="utf-8"
    )
    (output_dir / "failure_analysis.md").write_text(
        failure_analysis_markdown(result.failure_analysis), encoding="utf-8"
    )
    write_batch_review_csv(result, output_dir / "extraction_review.csv")
    write_manual_review_sample(result, output_dir / "manual_review_sample.csv")


def target_selection_summary_markdown(
    selected_targets: list[BatchSelectedTarget],
    preflight: BatchPreflightReport,
) -> str:
    lines = [
        "# Evidence-First Batch V1 Target Selection",
        "",
        f"- Targets: {len(selected_targets)}",
        f"- Domains: {preflight.target_count_by_domain}",
        f"- Value shapes: {preflight.target_count_by_value_shape_family}",
        f"- Support statuses: {preflight.support_status_counts}",
        "",
        "## Targets",
        "",
    ]
    for item in selected_targets:
        provenance = item.target.source_dictionary_provenance
        row = provenance.row_number if provenance else None
        lines.append(
            f"- `{item.target.expected_field}` row={row} "
            f"domain={item.target.sub_domain} shape={item.value_shape.value_shape_family} "
            f"support={item.expected_support_status}"
        )
    return "\n".join(lines) + "\n"


def batch_summary_markdown(result: BatchRunResult) -> str:
    retrieval = result.retrieval_sample_evaluation
    passing = {"valid", "valid_after_normalization", "valid_with_dictionary_caveat"}
    lines = [
        "# Evidence-First Extraction Batch V1",
        "",
        f"- Targets: {len(result.selected_targets)}",
        f"- Unique pages requested: {result.source_range_plan.unique_pages_requested}",
        f"- Parser cache hits: {result.telemetry.parser_cache_hits}",
        f"- Parser cache misses: {result.telemetry.parser_cache_misses}",
        f"- Parser worker invocations: {result.telemetry.parser_worker_invocations}",
        f"- Retrieval sample top-1: {sum(item.top1_hit for item in retrieval)}/20",
        f"- Retrieval sample top-3: {sum(item.top3_hit for item in retrieval)}/20",
        f"- Retrieval sample correct absence: {sum(item.absent_correct for item in retrieval)}/20",
        f"- Retrieval sample irrelevant: {sum(item.irrelevant_retrieval for item in retrieval)}/20",
        f"- LLM calls: {result.telemetry.llm_calls}",
        f"- Tokens: input {result.telemetry.input_tokens}, output {result.telemetry.output_tokens}",
        f"- Estimated cost USD: {result.telemetry.estimated_cost_usd}",
        f"- Passing review statuses: "
        f"{sum(item.status in passing for item in result.validation_results)}/75",
        "- Review status counts: "
        f"{dict(Counter(item.status for item in result.validation_results))}",
        "",
    ]
    return "\n".join(lines)


def failure_analysis_markdown(analysis: FailureAnalysis) -> str:
    lines = [
        "# Batch V1 Failure Analysis",
        "",
        f"- Counts by category: {analysis.counts_by_failure_category}",
        "",
    ]
    for item in analysis.items:
        lines.append(
            f"- `{item.field_name}`: {item.failure_category}, "
            f"status={item.overall_review_status}, issues={'; '.join(item.issues)}"
        )
    return "\n".join(lines) + "\n"


def write_batch_review_csv(result: BatchRunResult, path: Path) -> None:
    target_by_id = {item.target.target_row_id: item for item in result.selected_targets}
    retrieval_by_id = {item.target_row_id: item for item in result.retrieval_results}
    evidence_by_id = {item.target_row_id: item for item in result.evidence_validation}
    shape_by_id = {item.target_row_id: item for item in result.shape_validation}
    schema_by_id = {
        item.target_row_id: item for item in result.schema_compatibility_results
    }
    validation_by_id = {item.target_row_id: item for item in result.validation_results}
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "target_id",
            "dictionary_row",
            "domain",
            "sub_domain",
            "field_name",
            "definition",
            "declared_datatype",
            "declared_unit",
            "value_shape_family",
            "raw_model_value",
            "evidence_value",
            "normalized_value",
            "display_value",
            "extraction_status",
            "confidence",
            "retrieval_status",
            "retrieval_rank",
            "source_file",
            "page",
            "hierarchy_node",
            "canonical_evidence_excerpt",
            "evidence_validation",
            "shape_validation",
            "dictionary_compatibility",
            "overall_review_status",
            "caveat",
            "model",
            "input_tokens",
            "output_tokens",
            "estimated_cost",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for extraction in result.extraction_results:
            selected = target_by_id[extraction.target_row_id]
            provenance = selected.target.source_dictionary_provenance
            retrieval = retrieval_by_id[extraction.target_row_id]
            first = retrieval.results[0] if retrieval.results else None
            writer.writerow(
                {
                    "target_id": extraction.target_row_id,
                    "dictionary_row": provenance.row_number if provenance else "",
                    "domain": selected.target.metadata.get("domain") or "",
                    "sub_domain": selected.target.sub_domain,
                    "field_name": selected.target.expected_field,
                    "definition": selected.target.requirement_text,
                    "declared_datatype": selected.target.expected_data_type,
                    "declared_unit": selected.target.unit,
                    "value_shape_family": selected.value_shape.value_shape_family,
                    "raw_model_value": extraction.raw_model_value,
                    "evidence_value": extraction.evidence_value,
                    "normalized_value": extraction.normalized_value,
                    "display_value": extraction.display_value,
                    "extraction_status": extraction.status,
                    "confidence": extraction.confidence,
                    "retrieval_status": retrieval.retrieval_status,
                    "retrieval_rank": first.rank if first else "",
                    "source_file": extraction.source_file or (first.source_file if first else ""),
                    "page": extraction.page_number or (first.page_start if first else ""),
                    "hierarchy_node": extraction.hierarchy_node_id
                    or (first.node_id if first else ""),
                    "canonical_evidence_excerpt": extraction.supporting_evidence_excerpt,
                    "evidence_validation": evidence_by_id[extraction.target_row_id].status,
                    "shape_validation": shape_by_id[extraction.target_row_id].status,
                    "dictionary_compatibility": schema_by_id[
                        extraction.target_row_id
                    ].compatibility,
                    "overall_review_status": validation_by_id[
                        extraction.target_row_id
                    ].status,
                    "caveat": extraction.ambiguity_or_caveat,
                    "model": extraction.model_name,
                    "input_tokens": extraction.model_usage.input_tokens,
                    "output_tokens": extraction.model_usage.output_tokens,
                    "estimated_cost": extraction.model_usage.estimated_cost_usd,
                }
            )


def write_manual_review_sample(result: BatchRunResult, path: Path) -> None:
    selected = select_manual_review_sample(result)
    target_by_id = {item.target.target_row_id: item for item in result.selected_targets}
    validation_by_id = {item.target_row_id: item for item in result.validation_results}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_id",
                "field_name",
                "definition",
                "value_shape_family",
                "raw_model_value",
                "normalized_value",
                "source_page",
                "canonical_evidence",
                "validation_status",
                "caveat",
            ],
        )
        writer.writeheader()
        for extraction in selected:
            target = target_by_id[extraction.target_row_id]
            writer.writerow(
                {
                    "target_id": extraction.target_row_id,
                    "field_name": target.target.expected_field,
                    "definition": target.target.requirement_text,
                    "value_shape_family": target.value_shape.value_shape_family,
                    "raw_model_value": extraction.raw_model_value,
                    "normalized_value": extraction.normalized_value,
                    "source_page": extraction.page_number,
                    "canonical_evidence": extraction.supporting_evidence_excerpt,
                    "validation_status": validation_by_id[extraction.target_row_id].status,
                    "caveat": extraction.ambiguity_or_caveat,
                }
            )


def select_manual_review_sample(result: BatchRunResult) -> list[ExtractionResult]:
    validation_by_id = {item.target_row_id: item for item in result.validation_results}
    target_by_id = {item.target.target_row_id: item for item in result.selected_targets}
    statuses = [
        "valid",
        "valid_after_normalization",
        "valid_with_dictionary_caveat",
        "review_required",
        "insufficient_evidence",
        "invalid_evidence",
        "invalid_shape",
        "model_error",
    ]
    chosen: list[ExtractionResult] = []
    for status in statuses:
        for extraction in result.extraction_results:
            if len(chosen) == BATCH_REVIEW_SAMPLE_COUNT:
                return chosen
            if extraction in chosen:
                continue
            if validation_by_id[extraction.target_row_id].status != status:
                continue
            chosen.append(extraction)
            break
    for extraction in sorted(
        result.extraction_results,
        key=lambda item: (
            target_by_id[item.target_row_id].value_shape.value_shape_family,
            target_by_id[item.target_row_id].target.sub_domain,
            target_by_id[item.target_row_id].target.expected_field,
        ),
    ):
        if len(chosen) == BATCH_REVIEW_SAMPLE_COUNT:
            break
        if extraction not in chosen:
            chosen.append(extraction)
    return chosen


def run_extraction_batch_v1(
    *,
    source_manifest: Path = Path("output/sprint3_source_ingestion/source_pack_manifest.json"),
    dictionary_path: Path = Path("data/input/data_dictionary/SEGRO_Extraction_Template.xlsx"),
    output_dir: Path = DEFAULT_BATCH_OUTPUT_DIR,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    settings: Settings | None = None,
    extraction_client: ExtractionClient | None = None,
    cost_ceiling_usd: float = DEFAULT_COST_CEILING_USD,
) -> BatchRunResult:
    return EvidenceFirstBatchService(
        source_manifest=source_manifest,
        dictionary_path=dictionary_path,
        output_dir=output_dir,
        cache_root=cache_root,
        settings=settings,
        extraction_client=extraction_client,
        cost_ceiling_usd=cost_ceiling_usd,
    ).run()


def describe_planned_extraction_batch_v1(
    *,
    source_manifest: Path,
    dictionary_path: Path,
    output_dir: Path,
    cache_root: Path,
    settings: Settings,
    cost_ceiling_usd: float = DEFAULT_COST_CEILING_USD,
) -> BatchPreflightReport:
    source_registry = {
        source.source_id: source for source in load_source_registry(source_manifest)
    }
    targets = select_batch_targets(dictionary_path, output_dir / "planning_preview")
    plan = plan_batch_source_ranges(source_registry)
    model = settings.text_model_name or "gpt-4o-mini"
    return build_preflight_report(
        selected_targets=targets,
        source_plan=plan,
        source_registry=source_registry,
        output_dir=output_dir,
        cache_root=cache_root,
        model_name=model,
        cost_ceiling_usd=cost_ceiling_usd,
    )
