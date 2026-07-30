"""Retrieval coverage diagnostics for the 75-target evidence-first batch."""

from __future__ import annotations

import csv
import json
import multiprocessing
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field

from segro_evidence_extraction.config import Settings, load_settings
from segro_evidence_extraction.models.base import StrictBaseModel
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
from segro_evidence_extraction.target_semantics import (
    SupportClassification,
    TargetIntent,
    derive_target_intent,
)
from segro_evidence_extraction.vertical_slice import (
    DEFAULT_CACHE_ROOT,
    MAX_BATCH_PAGES,
    EvidenceBundleRecord,
    EvidenceSpan,
    ExtractionResult,
    HierarchyNode,
    LightweightHierarchy,
    RetrievalEvaluationItem,
    RetrievalResult,
    RetrievalScoreBreakdown,
    RetrievedEvidence,
    SourceRange,
    ValidationResult,
    _atomic_write_json,
    build_evidence_bundle,
    build_lightweight_hierarchy,
    split_range,
)

DEFAULT_DIAGNOSTIC_OUTPUT_DIR = Path(
    "output/enfield_unit1_evidence_first_batch_v1_retrieval_diagnostic"
)
DEFAULT_BATCH_V1_DIR = Path("output/enfield_unit1_evidence_first_batch_v1")
MAX_PROBE_RANGES_PER_TARGET = 3
MAX_TOTAL_PROBE_PAGES = 150

FailureCategory = Literal[
    "retrieval_hit_but_attribute_unsupported",
    "retrieval_miss_within_current_corpus",
    "relevant_evidence_found_below_top3",
    "weak_query_semantics",
    "evidence_conflict",
    "no_relevant_evidence_in_current_corpus",
    "dictionary_target_ambiguous",
    "validation_or_extraction_issue_not_retrieval",
]

EligibilityStatus = Literal[
    "eligible_for_extraction",
    "eligible_with_ambiguity",
    "unsupported_in_available_sources",
    "unresolved_source_location",
    "dictionary_target_ambiguous",
    "conflicting_evidence",
    "component_present_attribute_absent",
]


class SupportScore(StrictBaseModel):
    component_score: float = 0
    attribute_score: float = 0
    cooccurrence_score: float = 0
    value_shape_indicator_score: float = 0
    source_title_relevance: float = 0
    completeness_score: float = 0
    contradiction_score: float = 0
    generic_only_penalty: float = 0
    final_score: float = 0


class DiagnosticCandidate(StrictBaseModel):
    node_id: str
    source_id: str
    source_file: str
    page_start: int
    page_end: int
    title: str
    excerpt: str
    score: SupportScore
    support_classification: SupportClassification


class CurrentCorpusDiagnosticItem(StrictBaseModel):
    target_row_id: str
    field_name: str
    prior_review_status: str
    prior_retrieval_status: str
    best_component_nodes: list[DiagnosticCandidate]
    best_attribute_nodes: list[DiagnosticCandidate]
    best_component_plus_attribute_nodes: list[DiagnosticCandidate]
    current_corpus_evidence_found: bool
    prior_retrieval_rank: int | None = None
    revised_diagnostic_rank: int | None = None
    failure_classification: FailureCategory
    explanation: str


class ProbeRange(StrictBaseModel):
    source_id: str
    logical_path: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    target_row_ids: list[str]
    reason: str


class PageProbePlan(StrictBaseModel):
    ranges: list[ProbeRange]
    unique_new_pages_requested: int
    warnings: list[str] = Field(default_factory=list)


class PageProbeResults(StrictBaseModel):
    ranges: list[ProbeRange]
    cache_hits: int
    cache_misses: int
    parser_worker_invocations: int
    parser_wall_time_ms: float
    active_child_count_after_cleanup: int
    pages_loaded: dict[str, list[int]]


class CorrectedSourceRangePlan(StrictBaseModel):
    retained_ranges: list[SourceRange]
    added_ranges: list[ProbeRange]
    irrelevant_range_notes: list[str]
    target_to_range_mappings: dict[str, list[str]]
    unique_page_count: int
    expected_cache_hits: int
    expected_cache_misses: int
    expected_worker_invocations: int
    warnings: list[str] = Field(default_factory=list)


class RetrievalEligibility(StrictBaseModel):
    target_row_id: str
    status: EligibilityStatus
    support_classification: SupportClassification
    source_id: str | None = None
    page_number: int | None = None
    node_id: str | None = None
    reason: str


class ExtractedResultAuditItem(StrictBaseModel):
    target_row_id: str
    field_name: str
    audit_classification: Literal[
        "correctly_supported",
        "supported_with_caveat",
        "component_only_overgeneralization",
        "wrong_attribute",
        "insufficient_evidence",
        "dictionary_target_ambiguous",
    ]
    support_classification: SupportClassification
    explanation: str


class BatchRetrievalDiagnosticTelemetry(StrictBaseModel):
    targets_diagnosed: int
    control_targets_audited: int
    current_corpus_searches: int
    candidate_nodes_inspected: int
    total_wall_time_ms: float
    new_page_probes: int
    unique_new_pages_requested: int
    cache_hits: int
    cache_misses: int
    parser_worker_invocations: int
    parser_wall_time_ms: float
    active_child_count_after_cleanup: int
    target_semantics_inference_counts: dict[str, int]
    support_classification_counts: dict[str, int]
    source_range_changes: dict[str, int]
    revised_retrieval_time_ms: float
    revised_top1: int
    revised_top3: int
    correct_absence: int
    component_only_false_positive_count: int
    irrelevant_retrieval_count: int
    extraction_eligibility_counts: dict[str, int]
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0


class BatchRetrievalDiagnosticResult(StrictBaseModel):
    diagnostic_targets: list[TargetSpecification]
    control_targets: list[TargetSpecification]
    target_semantics: list[TargetIntent]
    current_corpus_diagnostic: list[CurrentCorpusDiagnosticItem]
    extracted_result_attribute_audit: list[ExtractedResultAuditItem]
    page_probe_plan: PageProbePlan
    page_probe_results: PageProbeResults
    corrected_source_range_plan: CorrectedSourceRangePlan
    revised_hierarchy: LightweightHierarchy
    revised_retrieval_results: list[RetrievalResult]
    revised_evidence_spans: list[EvidenceSpan]
    retrieval_eligibility: list[RetrievalEligibility]
    full_diagnostic_expectations: list[dict[str, object]]
    full_diagnostic_evaluation: list[RetrievalEvaluationItem]
    target_failure_classification: list[CurrentCorpusDiagnosticItem]
    reextract_candidate_list: list[RetrievalEligibility]
    unsupported_target_list: list[RetrievalEligibility]
    unresolved_target_list: list[RetrievalEligibility]
    telemetry: BatchRetrievalDiagnosticTelemetry


def run_batch_retrieval_diagnostic(
    *,
    batch_v1_dir: Path = DEFAULT_BATCH_V1_DIR,
    output_dir: Path = DEFAULT_DIAGNOSTIC_OUTPUT_DIR,
    source_manifest: Path = Path("output/sprint3_source_ingestion/source_pack_manifest.json"),
    cache_root: Path = DEFAULT_CACHE_ROOT,
    settings: Settings | None = None,
) -> BatchRetrievalDiagnosticResult:
    _ = settings or load_settings(Path("configs/default.yaml"))
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = BatchArtifacts.load(batch_v1_dir)
    source_registry = {
        source.source_id: source for source in load_source_registry(source_manifest)
    }
    diagnostics, controls = diagnostic_and_control_targets(artifacts)
    targets = [*diagnostics, *controls]
    intents = [derive_target_intent(target) for target in targets]
    intent_by_id = {item.target_row_id: item for item in intents}

    original_ranges = [
        SourceRange.model_validate(item) for item in artifacts.source_range_plan["ranges"]
    ]
    original_pages, original_cache_results = load_pages_for_ranges(
        source_registry=source_registry,
        ranges=original_ranges,
        cache_root=cache_root,
        output_dir=output_dir / "current_corpus_page_loads",
    )
    current_hierarchy = build_lightweight_hierarchy(original_pages, source_registry)
    current_diagnostics = diagnose_current_corpus(
        diagnostics,
        intent_by_id,
        artifacts,
        current_hierarchy,
        source_registry,
    )
    audit = audit_extracted_results(
        artifacts,
        intent_by_id,
        current_hierarchy,
        source_registry,
    )
    probe_plan = build_page_probe_plan(
        current_diagnostics,
        intent_by_id,
        source_registry,
        original_ranges,
    )
    probe_results, probe_pages = execute_probe_plan(
        probe_plan=probe_plan,
        source_registry=source_registry,
        cache_root=cache_root,
        output_dir=output_dir / "page_probe_runs",
    )
    combined_pages = merge_pages(original_pages, probe_pages)
    corrected_plan = build_corrected_source_range_plan(
        original_ranges=original_ranges,
        probe_plan=probe_plan,
        probe_results=probe_results,
        diagnostics=current_diagnostics,
    )
    revised_hierarchy = build_lightweight_hierarchy(combined_pages, source_registry)
    revised_retrieval_start = time.perf_counter()
    revised_retrieval = [
        retrieve_attribute_aware(
            target=target,
            intent=derive_target_intent(target),
            hierarchy=revised_hierarchy,
            source_registry=source_registry,
        )
        for target in artifacts.selected_targets
    ]
    revised_retrieval_time_ms = (time.perf_counter() - revised_retrieval_start) * 1000
    bundles = [
        build_evidence_bundle(
            artifacts.target_by_id[result.target_row_id],
            result,
            2400,
            combined_pages,
        )
        for result in revised_retrieval
    ]
    spans = [span for bundle in bundles for span in bundle.evidence_spans]
    eligibility = [
        classify_eligibility(
            target=artifacts.target_by_id[result.target_row_id],
            intent=derive_target_intent(artifacts.target_by_id[result.target_row_id]),
            retrieval=result,
        )
        for result in revised_retrieval
    ]
    expectations, evaluation = evaluate_full_diagnostic(
        diagnostics=diagnostics,
        controls=controls,
        retrieval_results=revised_retrieval,
        eligibility=eligibility,
        current_diagnostics=current_diagnostics,
    )
    reextract = [
        item
        for item in eligibility
        if item.status in {"eligible_for_extraction", "eligible_with_ambiguity"}
        and artifacts.validation_by_id[item.target_row_id].status
        in {
            "insufficient_evidence",
            "invalid_evidence",
            "review_required",
            "invalid_shape",
            "conflicting_evidence",
            "model_error",
        }
    ]
    unsupported = [
        item for item in eligibility if item.status == "unsupported_in_available_sources"
    ]
    unresolved = [item for item in eligibility if item.status == "unresolved_source_location"]
    telemetry = build_telemetry(
        diagnostics=diagnostics,
        controls=controls,
        intents=intents,
        current_diagnostics=current_diagnostics,
        original_cache_results=original_cache_results,
        probe_results=probe_results,
        corrected_plan=corrected_plan,
        revised_retrieval_time_ms=revised_retrieval_time_ms,
        evaluation=evaluation,
        eligibility=eligibility,
        revised_retrieval=revised_retrieval,
        total_started=started,
    )
    result = BatchRetrievalDiagnosticResult(
        diagnostic_targets=diagnostics,
        control_targets=controls,
        target_semantics=intents,
        current_corpus_diagnostic=current_diagnostics,
        extracted_result_attribute_audit=audit,
        page_probe_plan=probe_plan,
        page_probe_results=probe_results,
        corrected_source_range_plan=corrected_plan,
        revised_hierarchy=revised_hierarchy,
        revised_retrieval_results=revised_retrieval,
        revised_evidence_spans=spans,
        retrieval_eligibility=eligibility,
        full_diagnostic_expectations=expectations,
        full_diagnostic_evaluation=evaluation,
        target_failure_classification=current_diagnostics,
        reextract_candidate_list=reextract,
        unsupported_target_list=unsupported,
        unresolved_target_list=unresolved,
        telemetry=telemetry,
    )
    write_diagnostic_artifacts(result, artifacts, output_dir)
    return result


class BatchArtifacts:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.selected_targets_raw = cast(
            list[dict[str, Any]],
            _read_json(root / "selected_targets.json"),
        )
        self.selected_targets = [
            TargetSpecification.model_validate(item["target"])
            for item in self.selected_targets_raw
        ]
        self.target_by_id = {item.target_row_id: item for item in self.selected_targets}
        self.source_range_plan = cast(
            dict[str, Any],
            _read_json(root / "source_range_plan.json"),
        )
        self.retrieval_results = [
            RetrievalResult.model_validate(item)
            for item in cast(list[dict[str, Any]], _read_json(root / "retrieval_results.json"))
        ]
        self.retrieval_by_id = {item.target_row_id: item for item in self.retrieval_results}
        self.extraction_results = [
            ExtractionResult.model_validate(item)
            for item in cast(list[dict[str, Any]], _read_json(root / "extraction_results.json"))
        ]
        self.extraction_by_id = {item.target_row_id: item for item in self.extraction_results}
        self.validation_results = [
            ValidationResult.model_validate(item)
            for item in cast(list[dict[str, Any]], _read_json(root / "validation_results.json"))
        ]
        self.validation_by_id = {item.target_row_id: item for item in self.validation_results}
        self.evidence_bundles = [
            EvidenceBundleRecord.model_validate(item)
            for item in cast(list[dict[str, Any]], _read_json(root / "evidence_bundles.json"))
        ]
        self.bundle_by_id = {item.target_row_id: item for item in self.evidence_bundles}
        self.extraction_review_rows = list(
            csv.DictReader((root / "extraction_review.csv").open(encoding="utf-8"))
        )

    @classmethod
    def load(cls, root: Path) -> BatchArtifacts:
        missing = [
            name
            for name in [
                "selected_targets.json",
                "source_range_plan.json",
                "retrieval_results.json",
                "evidence_bundles.json",
                "extraction_results.json",
                "validation_results.json",
                "extraction_review.csv",
            ]
            if not (root / name).exists()
        ]
        if missing:
            msg = f"Batch V1 artifacts missing: {missing}"
            raise FileNotFoundError(msg)
        return cls(root)


def diagnostic_and_control_targets(
    artifacts: BatchArtifacts,
) -> tuple[list[TargetSpecification], list[TargetSpecification]]:
    diagnostic_statuses = {
        "insufficient_evidence",
        "invalid_evidence",
        "review_required",
        "invalid_shape",
        "conflicting_evidence",
        "model_error",
        "invalid",
    }
    diagnostics = [
        artifacts.target_by_id[item.target_row_id]
        for item in artifacts.validation_results
        if item.status in diagnostic_statuses
    ]
    passing = [
        artifacts.target_by_id[item.target_row_id]
        for item in artifacts.validation_results
        if item.status in {"valid", "valid_after_normalization", "valid_with_dictionary_caveat"}
    ]
    controls = sorted(
        passing,
        key=lambda target: (
            str(target.metadata.get("domain") or target.sub_domain),
            target.sub_domain,
            target.expected_field,
        ),
    )[:10]
    return diagnostics, controls


def load_pages_for_ranges(
    *,
    source_registry: dict[str, SourceRegistryEntry],
    ranges: list[SourceRange],
    cache_root: Path,
    output_dir: Path,
) -> tuple[dict[str, list[ParsedPage]], list[CachedBatchParseResult]]:
    cache_service = CachedBatchParsingService(
        cache=CanonicalParsedPageCache(cache_root),
        batch_config=BatchWorkerConfig(),
    )
    pages_by_source: dict[str, dict[int, ParsedPage]] = defaultdict(dict)
    results: list[CachedBatchParseResult] = []
    for source_range in ranges:
        source = source_registry[source_range.source_id]
        for start, end in split_range(
            source_range.page_start,
            source_range.page_end,
            MAX_BATCH_PAGES,
        ):
            result = cache_service.parse(
                BatchRequest(
                    source=source,
                    source_path=source.original_path,
                    output_dir=str(
                        output_dir / f"{source.source_id}_{start:04d}_{end:04d}"
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


def diagnose_current_corpus(
    diagnostics: list[TargetSpecification],
    intent_by_id: dict[str, TargetIntent],
    artifacts: BatchArtifacts,
    hierarchy: LightweightHierarchy,
    source_registry: dict[str, SourceRegistryEntry],
) -> list[CurrentCorpusDiagnosticItem]:
    items: list[CurrentCorpusDiagnosticItem] = []
    for target in diagnostics:
        intent = intent_by_id[target.target_row_id]
        candidates = score_nodes(intent, hierarchy.nodes, source_registry)
        component = sorted(
            candidates,
            key=lambda item: item.score.component_score,
            reverse=True,
        )[:3]
        attribute = sorted(
            candidates,
            key=lambda item: item.score.attribute_score,
            reverse=True,
        )[:3]
        complete = sorted(candidates, key=lambda item: item.score.final_score, reverse=True)[:5]
        prior = artifacts.retrieval_by_id[target.target_row_id]
        prior_rank = prior_support_rank(prior, intent)
        revised_rank = next(
            (
                index
                for index, item in enumerate(complete, start=1)
                if item.support_classification == "supports_requested_attribute"
            ),
            None,
        )
        failure = current_failure_classification(
            target=target,
            intent=intent,
            prior=prior,
            complete=complete,
            prior_rank=prior_rank,
            revised_rank=revised_rank,
            validation_status=artifacts.validation_by_id[target.target_row_id].status,
        )
        items.append(
            CurrentCorpusDiagnosticItem(
                target_row_id=target.target_row_id,
                field_name=target.expected_field,
                prior_review_status=artifacts.validation_by_id[target.target_row_id].status,
                prior_retrieval_status=prior.retrieval_status,
                best_component_nodes=component,
                best_attribute_nodes=attribute,
                best_component_plus_attribute_nodes=complete[:3],
                current_corpus_evidence_found=bool(revised_rank),
                prior_retrieval_rank=prior_rank,
                revised_diagnostic_rank=revised_rank,
                failure_classification=failure,
                explanation=classification_explanation(failure, complete),
            )
        )
    return items


def score_nodes(
    intent: TargetIntent,
    nodes: list[HierarchyNode],
    source_registry: dict[str, SourceRegistryEntry],
) -> list[DiagnosticCandidate]:
    candidates: list[DiagnosticCandidate] = []
    for node in nodes:
        if node.node_type == "document":
            continue
        source = source_registry[node.source_id]
        text = f"{node.title}\n{node.text_summary}"
        score = score_support(intent, text, source.logical_path)
        support = classify_support(score)
        candidates.append(
            DiagnosticCandidate(
                node_id=node.node_id,
                source_id=node.source_id,
                source_file=source.logical_path,
                page_start=node.page_start,
                page_end=node.page_end,
                title=node.title,
                excerpt=(node.text_summary or node.title)[:900],
                score=score,
                support_classification=support,
            )
        )
    return candidates


def score_support(intent: TargetIntent, text: str, source_title: str = "") -> SupportScore:
    lower = text.lower()
    component_hits = sum(1 for term in intent.component_terms if term and term in lower)
    attribute_hits = sum(1 for term in intent.attribute_terms if term and term in lower)
    indicator_hits = sum(
        1 for term in intent.expected_value_indicators if term and _term_or_pattern(term, lower)
    )
    cooccurrence = component_attribute_cooccurrence(intent, lower)
    title_relevance = sum(1 for term in intent.likely_source_types if term in source_title.lower())
    contradiction = 1.0 if re.search(r"\b(no|none|not provided|not applicable)\b", lower) else 0.0
    generic_penalty = 1.5 if component_hits == 0 and attribute_hits > 0 else 0.0
    completeness = 0.0
    if component_hits and attribute_hits and (cooccurrence or indicator_hits):
        completeness = 4.0
    elif component_hits and intent.requested_attribute in {"description", "component_type"}:
        completeness = 2.0
    final = (
        component_hits * 2.5
        + attribute_hits * 2.0
        + cooccurrence * 3.0
        + indicator_hits * 1.0
        + title_relevance * 0.5
        + completeness
        - contradiction * 2.0
        - generic_penalty
    )
    return SupportScore(
        component_score=float(component_hits),
        attribute_score=float(attribute_hits),
        cooccurrence_score=float(cooccurrence),
        value_shape_indicator_score=float(indicator_hits),
        source_title_relevance=float(title_relevance),
        completeness_score=completeness,
        contradiction_score=contradiction,
        generic_only_penalty=generic_penalty,
        final_score=round(max(0.0, final), 4),
    )


def _term_or_pattern(term: str, lower: str) -> bool:
    if term in {"number", "quantity", "count", "no"}:
        return bool(re.search(r"\b\d+\s*(?:no\.?|number|quantity)?\b", lower))
    if term in {"date", "dated"}:
        return bool(
            re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", lower)
        )
    return term.lower() in lower


def component_attribute_cooccurrence(intent: TargetIntent, lower: str) -> int:
    lines = [line for line in re.split(r"[\r\n.;]+", lower) if line.strip()]
    for line in lines:
        if any(term in line for term in intent.component_terms) and any(
            term in line for term in intent.attribute_terms
        ):
            return 1
    for component in intent.component_terms:
        index = lower.find(component)
        if index < 0:
            continue
        window = lower[max(0, index - 90) : index + len(component) + 90]
        if any(term in window for term in intent.attribute_terms):
            return 1
    return 0


def classify_support(score: SupportScore) -> SupportClassification:
    if score.contradiction_score:
        return "conflicting"
    if score.component_score and score.attribute_score and (
        score.cooccurrence_score or score.completeness_score >= 4
    ):
        return "supports_requested_attribute"
    if score.component_score and not score.attribute_score:
        return "component_only"
    if score.attribute_score and not score.component_score:
        return "attribute_only"
    if score.final_score >= 3:
        return "weak_context"
    return "irrelevant"


def prior_support_rank(retrieval: RetrievalResult, intent: TargetIntent) -> int | None:
    for result in retrieval.results:
        score = score_support(intent, result.excerpt, result.source_file)
        if classify_support(score) == "supports_requested_attribute":
            return result.rank
    return None


def current_failure_classification(
    *,
    target: TargetSpecification,
    intent: TargetIntent,
    prior: RetrievalResult,
    complete: list[DiagnosticCandidate],
    prior_rank: int | None,
    revised_rank: int | None,
    validation_status: str,
) -> FailureCategory:
    if intent.ambiguity_notes and validation_status == "review_required":
        return "dictionary_target_ambiguous"
    if validation_status in {"invalid_evidence", "review_required", "invalid_shape"} and prior_rank:
        return "validation_or_extraction_issue_not_retrieval"
    if prior.results and all(
        score_support(intent, item.excerpt, item.source_file).component_score
        and not score_support(intent, item.excerpt, item.source_file).attribute_score
        for item in prior.results[:3]
    ):
        return "retrieval_hit_but_attribute_unsupported"
    if revised_rank and revised_rank > 3:
        return "relevant_evidence_found_below_top3"
    if revised_rank and prior_rank is None:
        return "retrieval_miss_within_current_corpus"
    if complete and complete[0].support_classification in {"component_only", "attribute_only"}:
        return "weak_query_semantics"
    if complete and complete[0].support_classification == "conflicting":
        return "evidence_conflict"
    if not target.requirement_text or intent.ambiguity_notes:
        return "dictionary_target_ambiguous"
    return "no_relevant_evidence_in_current_corpus"


def classification_explanation(
    failure: FailureCategory,
    candidates: list[DiagnosticCandidate],
) -> str:
    if not candidates:
        return "No candidate nodes were available in the bounded corpus."
    best = candidates[0]
    return (
        f"{failure}: best current-corpus candidate is {best.source_file} "
        f"page {best.page_start} with support={best.support_classification}, "
        f"component={best.score.component_score}, attribute={best.score.attribute_score}."
    )


def audit_extracted_results(
    artifacts: BatchArtifacts,
    intent_by_id: dict[str, TargetIntent],
    hierarchy: LightweightHierarchy,
    source_registry: dict[str, SourceRegistryEntry],
) -> list[ExtractedResultAuditItem]:
    node_by_id = {node.node_id: node for node in hierarchy.nodes}
    audit: list[ExtractedResultAuditItem] = []
    for extraction in artifacts.extraction_results:
        if extraction.status != "extracted":
            continue
        target = artifacts.target_by_id[extraction.target_row_id]
        intent = intent_by_id.get(extraction.target_row_id) or derive_target_intent(target)
        evidence_text = extraction.supporting_evidence_excerpt or ""
        node = node_by_id.get(extraction.hierarchy_node_id or "")
        source_file = source_registry[node.source_id].logical_path if node else ""
        score = score_support(intent, evidence_text, source_file)
        support = classify_support(score)
        audit_class: Literal[
            "correctly_supported",
            "supported_with_caveat",
            "component_only_overgeneralization",
            "wrong_attribute",
            "insufficient_evidence",
            "dictionary_target_ambiguous",
        ]
        if support == "supports_requested_attribute" and intent.ambiguity_notes:
            audit_class = "supported_with_caveat"
        elif support == "supports_requested_attribute":
            audit_class = "correctly_supported"
        elif support == "component_only":
            audit_class = "component_only_overgeneralization"
        elif support == "attribute_only":
            audit_class = "wrong_attribute"
        elif intent.ambiguity_notes:
            audit_class = "dictionary_target_ambiguous"
        else:
            audit_class = "insufficient_evidence"
        audit.append(
            ExtractedResultAuditItem(
                target_row_id=target.target_row_id,
                field_name=target.expected_field,
                audit_classification=audit_class,
                support_classification=support,
                explanation=(
                    f"Selected evidence support={support}; component={score.component_score}, "
                    f"attribute={score.attribute_score}, cooccurrence={score.cooccurrence_score}."
                ),
            )
        )
    return audit


def build_page_probe_plan(
    diagnostics: list[CurrentCorpusDiagnosticItem],
    intent_by_id: dict[str, TargetIntent],
    source_registry: dict[str, SourceRegistryEntry],
    original_ranges: list[SourceRange],
) -> PageProbePlan:
    original_pages = {
        (item.source_id, page)
        for item in original_ranges
        for page in range(item.page_start, item.page_end + 1)
    }
    source_by_name = {source.logical_path: source for source in source_registry.values()}
    proposed: list[ProbeRange] = []
    for item in diagnostics:
        if item.failure_classification not in {
            "no_relevant_evidence_in_current_corpus",
            "weak_query_semantics",
            "retrieval_hit_but_attribute_unsupported",
        }:
            continue
        intent = intent_by_id[item.target_row_id]
        for logical_path, start, end, reason in candidate_probe_ranges(intent):
            source = source_by_name.get(logical_path)
            if source is None:
                continue
            pages = [
                page
                for page in range(start, end + 1)
                if (source.source_id, page) not in original_pages
            ]
            if not pages:
                continue
            add_probe_range(proposed, source, min(pages), max(pages), item.target_row_id, reason)
            target_probe_count = len(
                [probe for probe in proposed if item.target_row_id in probe.target_row_ids]
            )
            if target_probe_count >= MAX_PROBE_RANGES_PER_TARGET:
                break
    deduped = sorted(
        proposed,
        key=lambda item: (item.logical_path, item.page_start, item.page_end, item.reason),
    )
    retained: list[ProbeRange] = []
    unique_pages: set[tuple[str, int]] = set()
    warnings: list[str] = []
    for probe in deduped:
        probe_pages = {
            (probe.source_id, page) for page in range(probe.page_start, probe.page_end + 1)
        }
        if len(unique_pages | probe_pages) > MAX_TOTAL_PROBE_PAGES:
            warnings.append("Probe ceiling reached; lower-priority source locations unresolved.")
            continue
        retained.append(probe)
        unique_pages |= probe_pages
    return PageProbePlan(
        ranges=retained,
        unique_new_pages_requested=len(unique_pages),
        warnings=warnings,
    )


def candidate_probe_ranges(intent: TargetIntent) -> list[tuple[str, int, int, str]]:
    text = (
        f"{intent.primary_component} {intent.requested_attribute} "
        f"{' '.join(intent.likely_source_types)}"
    )
    ranges: list[tuple[str, int, int, str]] = []
    if any(term in text for term in ["manufacturer", "model", "roof", "wall", "floor", "cladding"]):
        ranges.extend(
            [
                (
                    "Building Manual - Part 2 Building Fabric.pdf",
                    61,
                    70,
                    "fabric specification continuation",
                ),
                (
                    "Building Manual - Part 2 Building Fabric.pdf",
                    71,
                    80,
                    "fabric product schedules",
                ),
                (
                    "Building Manual - Part 2 Building Fabric.pdf",
                    81,
                    90,
                    "fabric manufacturers/models",
                ),
            ]
        )
    if any(term in text for term in ["fire", "alarm", "service", "electrical"]):
        ranges.extend(
            [
                (
                    "Building Manual - Part 3 Building Services.pdf",
                    31,
                    40,
                    "services section continuation",
                ),
                (
                    "Building Manual - Part 3 Building Services.pdf",
                    41,
                    50,
                    "fire/electrical schedules",
                ),
            ]
        )
    if any(term in text for term in ["date", "certificate", "pv", "photovoltaic", "solar"]):
        ranges.extend(
            [
                ("Building Manual - Part 6 Appendices.pdf", 11, 20, "appendix certificates"),
                (
                    "Building Manual - Part 6 Appendices.pdf",
                    21,
                    30,
                    "appendix commissioning records",
                ),
            ]
        )
    if any(term in text for term in ["planning", "consent", "use"]):
        ranges.append(("Building Manual - Part 1 General.pdf", 23, 32, "planning continuation"))
    if any(term in text for term in ["external", "fence", "gate", "yard"]):
        ranges.append(
            (
                "Building Manual - Part 4 External Works.pdf",
                15,
                24,
                "external works continuation",
            )
        )
    if not ranges:
        ranges.append(
            ("Building Manual - Part 1 General.pdf", 23, 32, "general index continuation")
        )
    return ranges


def add_probe_range(
    ranges: list[ProbeRange],
    source: SourceRegistryEntry,
    start: int,
    end: int,
    target_id: str,
    reason: str,
) -> None:
    if end - start + 1 > 10:
        end = start + 9
    for item in ranges:
        if (
            item.source_id == source.source_id
            and item.page_start == start
            and item.page_end == end
        ):
            if target_id not in item.target_row_ids:
                item.target_row_ids.append(target_id)
                item.target_row_ids.sort()
            if reason not in item.reason:
                item.reason = f"{item.reason}; {reason}"
            return
    ranges.append(
        ProbeRange(
            source_id=source.source_id,
            logical_path=source.logical_path,
            page_start=start,
            page_end=end,
            target_row_ids=[target_id],
            reason=reason,
        )
    )


def execute_probe_plan(
    *,
    probe_plan: PageProbePlan,
    source_registry: dict[str, SourceRegistryEntry],
    cache_root: Path,
    output_dir: Path,
) -> tuple[PageProbeResults, dict[str, list[ParsedPage]]]:
    ranges = [
        SourceRange(
            source_id=item.source_id,
            logical_path=item.logical_path,
            page_start=item.page_start,
            page_end=item.page_end,
            reason=item.reason,
        )
        for item in probe_plan.ranges
    ]
    started = time.perf_counter()
    pages, cache_results = load_pages_for_ranges(
        source_registry=source_registry,
        ranges=ranges,
        cache_root=cache_root,
        output_dir=output_dir,
    )
    result = PageProbeResults(
        ranges=probe_plan.ranges,
        cache_hits=sum(item.cache_hits for item in cache_results),
        cache_misses=sum(item.cache_misses for item in cache_results),
        parser_worker_invocations=sum(item.worker_invocation_count for item in cache_results),
        parser_wall_time_ms=(time.perf_counter() - started) * 1000,
        active_child_count_after_cleanup=len(multiprocessing.active_children()),
        pages_loaded={
            source_id: [page.page_number for page in source_pages]
            for source_id, source_pages in pages.items()
        },
    )
    return result, pages


def merge_pages(
    first: dict[str, list[ParsedPage]],
    second: dict[str, list[ParsedPage]],
) -> dict[str, list[ParsedPage]]:
    merged: dict[str, dict[int, ParsedPage]] = defaultdict(dict)
    for collection in [first, second]:
        for source_id, pages in collection.items():
            for page in pages:
                merged[source_id][page.page_number] = page
    return {
        source_id: [pages[number] for number in sorted(pages)]
        for source_id, pages in sorted(merged.items())
    }


def build_corrected_source_range_plan(
    *,
    original_ranges: list[SourceRange],
    probe_plan: PageProbePlan,
    probe_results: PageProbeResults,
    diagnostics: list[CurrentCorpusDiagnosticItem],
) -> CorrectedSourceRangePlan:
    target_map: dict[str, list[str]] = defaultdict(list)
    for probe in probe_plan.ranges:
        label = f"{probe.logical_path}:{probe.page_start}-{probe.page_end}"
        for target_id in probe.target_row_ids:
            target_map[target_id].append(label)
    for item in diagnostics:
        if item.best_component_plus_attribute_nodes:
            best = item.best_component_plus_attribute_nodes[0]
            target_map[item.target_row_id].append(f"{best.source_file}:{best.page_start}-{best.page_end}")
    unique_pages = {
        (item.source_id, page)
        for item in original_ranges
        for page in range(item.page_start, item.page_end + 1)
    }
    unique_pages |= {
        (item.source_id, page)
        for item in probe_plan.ranges
        for page in range(item.page_start, item.page_end + 1)
    }
    irrelevant = [
        f"{item.field_name}: previous top retrieval did not support requested attribute"
        for item in diagnostics
        if item.failure_classification == "retrieval_hit_but_attribute_unsupported"
    ]
    return CorrectedSourceRangePlan(
        retained_ranges=original_ranges,
        added_ranges=probe_plan.ranges,
        irrelevant_range_notes=irrelevant,
        target_to_range_mappings={
            key: sorted(set(value)) for key, value in sorted(target_map.items())
        },
        unique_page_count=len(unique_pages),
        expected_cache_hits=probe_results.cache_hits,
        expected_cache_misses=probe_results.cache_misses,
        expected_worker_invocations=probe_results.parser_worker_invocations,
        warnings=probe_plan.warnings,
    )


def retrieve_attribute_aware(
    *,
    target: TargetSpecification,
    intent: TargetIntent,
    hierarchy: LightweightHierarchy,
    source_registry: dict[str, SourceRegistryEntry],
) -> RetrievalResult:
    started = time.perf_counter()
    candidates = score_nodes(intent, hierarchy.nodes, source_registry)
    ranked = sorted(candidates, key=lambda item: item.score.final_score, reverse=True)
    useful = [
        item
        for item in ranked
        if item.support_classification
        in {"supports_requested_attribute", "component_only", "attribute_only", "weak_context"}
    ][:3]
    if useful and useful[0].support_classification == "supports_requested_attribute":
        status: Literal["evidence_found", "weak_evidence", "no_relevant_evidence"]
        status = "evidence_found"
    elif useful and useful[0].score.final_score >= 5:
        status = "weak_evidence"
    else:
        status = "no_relevant_evidence"
        useful = []
    results = [
        RetrievedEvidence(
            target_row_id=target.target_row_id,
            rank=index,
            node_id=item.node_id,
            source_id=item.source_id,
            source_file=item.source_file,
            page_start=item.page_start,
            page_end=item.page_end,
            score=item.score.final_score,
            score_components=RetrievalScoreBreakdown(
                token_overlap=item.score.component_score + item.score.attribute_score,
                title_match=item.score.source_title_relevance,
                pattern_match=item.score.value_shape_indicator_score,
                component_attribute_proximity=item.score.cooccurrence_score,
                negative_penalty=item.score.generic_only_penalty + item.score.contradiction_score,
                final_score=item.score.final_score,
            ),
            matched_terms=[
                *[term for term in intent.component_terms if term in item.excerpt.lower()],
                *[term for term in intent.attribute_terms if term in item.excerpt.lower()],
            ],
            hierarchy_path=[item.source_file, item.title],
            excerpt=item.excerpt,
        )
        for index, item in enumerate(useful, start=1)
    ]
    return RetrievalResult(
        target_row_id=target.target_row_id,
        query=f"{intent.primary_component} {intent.requested_attribute}",
        retrieval_status=status,
        results=results,
        retrieval_time_ms=(time.perf_counter() - started) * 1000,
        top_score=results[0].score if results else 0,
    )


def classify_eligibility(
    *,
    target: TargetSpecification,
    intent: TargetIntent,
    retrieval: RetrievalResult,
) -> RetrievalEligibility:
    if intent.ambiguity_notes and not retrieval.results:
        return RetrievalEligibility(
            target_row_id=target.target_row_id,
            status="dictionary_target_ambiguous",
            support_classification="weak_context",
            reason=(
                "Dictionary metadata or field semantics are ambiguous and no supporting "
                "evidence was found."
            ),
        )
    if not retrieval.results:
        return RetrievalEligibility(
            target_row_id=target.target_row_id,
            status="unsupported_in_available_sources",
            support_classification="irrelevant",
            reason="No component-plus-attribute support found in available bounded sources.",
        )
    best = retrieval.results[0]
    support = classify_support(score_support(intent, best.excerpt, best.source_file))
    if support == "supports_requested_attribute" and intent.ambiguity_notes:
        status: EligibilityStatus = "eligible_with_ambiguity"
    elif support == "supports_requested_attribute":
        status = "eligible_for_extraction"
    elif support == "conflicting":
        status = "conflicting_evidence"
    elif support == "component_only":
        status = "component_present_attribute_absent"
    elif intent.ambiguity_notes:
        status = "dictionary_target_ambiguous"
    else:
        status = "unsupported_in_available_sources"
    return RetrievalEligibility(
        target_row_id=target.target_row_id,
        status=status,
        support_classification=support,
        source_id=best.source_id,
        page_number=best.page_start,
        node_id=best.node_id,
        reason=f"Best revised retrieval support={support}; score={best.score}.",
    )


def evaluate_full_diagnostic(
    *,
    diagnostics: list[TargetSpecification],
    controls: list[TargetSpecification],
    retrieval_results: list[RetrievalResult],
    eligibility: list[RetrievalEligibility],
    current_diagnostics: list[CurrentCorpusDiagnosticItem],
) -> tuple[list[dict[str, object]], list[RetrievalEvaluationItem]]:
    retrieval_by_id = {item.target_row_id: item for item in retrieval_results}
    eligibility_by_id = {item.target_row_id: item for item in eligibility}
    current_by_id = {item.target_row_id: item for item in current_diagnostics}
    expectations: list[dict[str, object]] = []
    evaluations: list[RetrievalEvaluationItem] = []
    for target in [*diagnostics, *controls]:
        result = retrieval_by_id[target.target_row_id]
        eligible = eligibility_by_id[target.target_row_id]
        current = current_by_id.get(target.target_row_id)
        expected_absent = eligible.status in {
            "unsupported_in_available_sources",
            "unresolved_source_location",
            "dictionary_target_ambiguous",
            "component_present_attribute_absent",
        }
        expected_page = eligible.page_number
        expectations.append(
            {
                "target_row_id": target.target_row_id,
                "expected_page": expected_page,
                "expected_absent": expected_absent,
                "basis": (
                    current.failure_classification if current else "passing control audit"
                ),
            }
        )
        if expected_absent:
            absent_correct = result.retrieval_status in {"weak_evidence", "no_relevant_evidence"}
            evaluations.append(
                RetrievalEvaluationItem(
                    target_row_id=target.target_row_id,
                    absent_correct=absent_correct,
                    irrelevant_retrieval=result.retrieval_status == "evidence_found",
                    notes=eligible.reason,
                )
            )
            continue
        top1 = bool(result.results and result.results[0].page_start == expected_page)
        top3 = any(item.page_start == expected_page for item in result.results[:3])
        evaluations.append(
            RetrievalEvaluationItem(
                target_row_id=target.target_row_id,
                top1_hit=top1,
                top3_hit=top3,
                irrelevant_retrieval=not top3,
                notes=eligible.reason,
            )
        )
    return expectations, evaluations


def build_telemetry(
    *,
    diagnostics: list[TargetSpecification],
    controls: list[TargetSpecification],
    intents: list[TargetIntent],
    current_diagnostics: list[CurrentCorpusDiagnosticItem],
    original_cache_results: list[CachedBatchParseResult],
    probe_results: PageProbeResults,
    corrected_plan: CorrectedSourceRangePlan,
    revised_retrieval_time_ms: float,
    evaluation: list[RetrievalEvaluationItem],
    eligibility: list[RetrievalEligibility],
    revised_retrieval: list[RetrievalResult],
    total_started: float,
) -> BatchRetrievalDiagnosticTelemetry:
    support_counts = Counter(item.support_classification for item in eligibility)
    return BatchRetrievalDiagnosticTelemetry(
        targets_diagnosed=len(diagnostics),
        control_targets_audited=len(controls),
        current_corpus_searches=len(diagnostics),
        candidate_nodes_inspected=len(diagnostics)
        * sum(len(result.results) or 1 for result in revised_retrieval),
        new_page_probes=len(probe_results.ranges),
        unique_new_pages_requested=(
            sum(len(pages) for pages in probe_results.pages_loaded.values())
            if probe_results.pages_loaded
            else 0
        ),
        cache_hits=(
            sum(item.cache_hits for item in original_cache_results) + probe_results.cache_hits
        ),
        cache_misses=(
            sum(item.cache_misses for item in original_cache_results) + probe_results.cache_misses
        ),
        parser_worker_invocations=(
            sum(item.worker_invocation_count for item in original_cache_results)
            + probe_results.parser_worker_invocations
        ),
        parser_wall_time_ms=sum(item.parse_time_ms for item in original_cache_results)
        + probe_results.parser_wall_time_ms,
        active_child_count_after_cleanup=len(multiprocessing.active_children()),
        total_wall_time_ms=(time.perf_counter() - total_started) * 1000,
        target_semantics_inference_counts=dict(
            Counter(item.requested_attribute for item in intents)
        ),
        support_classification_counts={str(key): value for key, value in support_counts.items()},
        source_range_changes={
            "retained": len(corrected_plan.retained_ranges),
            "added": len(corrected_plan.added_ranges),
        },
        revised_retrieval_time_ms=revised_retrieval_time_ms,
        revised_top1=sum(item.top1_hit for item in evaluation),
        revised_top3=sum(item.top3_hit for item in evaluation),
        correct_absence=sum(item.absent_correct for item in evaluation),
        component_only_false_positive_count=sum(
            1 for item in eligibility if item.status == "component_present_attribute_absent"
        ),
        irrelevant_retrieval_count=sum(item.irrelevant_retrieval for item in evaluation),
        extraction_eligibility_counts=dict(Counter(item.status for item in eligibility)),
    )


def write_diagnostic_artifacts(
    result: BatchRetrievalDiagnosticResult,
    artifacts: BatchArtifacts,
    output_dir: Path,
) -> None:
    payloads: dict[str, object] = {
        "diagnostic_targets.json": [
            item.model_dump(mode="json") for item in result.diagnostic_targets
        ],
        "control_targets.json": [item.model_dump(mode="json") for item in result.control_targets],
        "target_semantics.json": [item.model_dump(mode="json") for item in result.target_semantics],
        "current_corpus_diagnostic.json": [
            item.model_dump(mode="json") for item in result.current_corpus_diagnostic
        ],
        "extracted_result_attribute_audit.json": [
            item.model_dump(mode="json") for item in result.extracted_result_attribute_audit
        ],
        "page_probe_plan.json": result.page_probe_plan.model_dump(mode="json"),
        "page_probe_results.json": result.page_probe_results.model_dump(mode="json"),
        "corrected_source_range_plan.json": result.corrected_source_range_plan.model_dump(
            mode="json"
        ),
        "revised_hierarchy.json": result.revised_hierarchy.model_dump(mode="json"),
        "revised_retrieval_results.json": [
            item.model_dump(mode="json") for item in result.revised_retrieval_results
        ],
        "revised_evidence_spans.json": [
            item.model_dump(mode="json") for item in result.revised_evidence_spans
        ],
        "retrieval_eligibility.json": [
            item.model_dump(mode="json") for item in result.retrieval_eligibility
        ],
        "full_diagnostic_expectations.json": result.full_diagnostic_expectations,
        "full_diagnostic_evaluation.json": [
            item.model_dump(mode="json") for item in result.full_diagnostic_evaluation
        ],
        "target_failure_classification.json": [
            item.model_dump(mode="json") for item in result.target_failure_classification
        ],
        "reextract_candidate_list.json": [
            item.model_dump(mode="json") for item in result.reextract_candidate_list
        ],
        "unsupported_target_list.json": [
            item.model_dump(mode="json") for item in result.unsupported_target_list
        ],
        "unresolved_target_list.json": [
            item.model_dump(mode="json") for item in result.unresolved_target_list
        ],
        "telemetry.json": result.telemetry.model_dump(mode="json"),
    }
    for filename, payload in payloads.items():
        _atomic_write_json(output_dir / filename, payload)
    (output_dir / "current_corpus_diagnostic.md").write_text(
        current_corpus_markdown(result), encoding="utf-8"
    )
    (output_dir / "extracted_result_attribute_audit.md").write_text(
        audit_markdown(result), encoding="utf-8"
    )
    (output_dir / "corrected_source_range_plan.md").write_text(
        corrected_plan_markdown(result), encoding="utf-8"
    )
    (output_dir / "full_diagnostic_evaluation.md").write_text(
        evaluation_markdown(result), encoding="utf-8"
    )
    (output_dir / "diagnostic_summary.md").write_text(
        diagnostic_summary_markdown(result), encoding="utf-8"
    )
    write_diagnostic_review_csv(result, artifacts, output_dir / "diagnostic_review.csv")


def current_corpus_markdown(result: BatchRetrievalDiagnosticResult) -> str:
    counts = Counter(item.failure_classification for item in result.current_corpus_diagnostic)
    lines = ["# Current Corpus Diagnostic", "", f"- Failure classifications: {dict(counts)}", ""]
    for item in result.current_corpus_diagnostic:
        lines.append(f"- `{item.field_name}`: {item.failure_classification}. {item.explanation}")
    return "\n".join(lines) + "\n"


def audit_markdown(result: BatchRetrievalDiagnosticResult) -> str:
    counts = Counter(item.audit_classification for item in result.extracted_result_attribute_audit)
    lines = ["# Extracted Result Attribute Audit", "", f"- Counts: {dict(counts)}", ""]
    for item in result.extracted_result_attribute_audit:
        lines.append(f"- `{item.field_name}`: {item.audit_classification}. {item.explanation}")
    return "\n".join(lines) + "\n"


def corrected_plan_markdown(result: BatchRetrievalDiagnosticResult) -> str:
    plan = result.corrected_source_range_plan
    lines = [
        "# Corrected Bounded Source Range Plan",
        "",
        f"- Retained ranges: {len(plan.retained_ranges)}",
        f"- Added ranges: {len(plan.added_ranges)}",
        f"- Unique page count: {plan.unique_page_count}",
        f"- Expected cache hits: {plan.expected_cache_hits}",
        f"- Expected cache misses: {plan.expected_cache_misses}",
        f"- Expected worker invocations: {plan.expected_worker_invocations}",
        "",
        "## Added Ranges",
    ]
    for item in plan.added_ranges:
        lines.append(
            f"- {item.logical_path} pages {item.page_start}-{item.page_end}: "
            f"{item.reason} ({len(item.target_row_ids)} targets)"
        )
    return "\n".join(lines) + "\n"


def evaluation_markdown(result: BatchRetrievalDiagnosticResult) -> str:
    evaluation = result.full_diagnostic_evaluation
    lines = [
        "# Full Diagnostic Evaluation",
        "",
        f"- Top-1: {sum(item.top1_hit for item in evaluation)}",
        f"- Top-3: {sum(item.top3_hit for item in evaluation)}",
        f"- Correct absence: {sum(item.absent_correct for item in evaluation)}",
        f"- Irrelevant retrievals: {sum(item.irrelevant_retrieval for item in evaluation)}",
    ]
    return "\n".join(lines) + "\n"


def diagnostic_summary_markdown(result: BatchRetrievalDiagnosticResult) -> str:
    return "\n".join(
        [
            "# Batch V1 Retrieval Diagnostic",
            "",
            f"- Targets diagnosed: {result.telemetry.targets_diagnosed}",
            f"- Controls audited: {result.telemetry.control_targets_audited}",
            f"- New probe pages: {result.telemetry.unique_new_pages_requested}",
            f"- Cache hits: {result.telemetry.cache_hits}",
            f"- Cache misses: {result.telemetry.cache_misses}",
            f"- Parser workers: {result.telemetry.parser_worker_invocations}",
            f"- Revised top-1: {result.telemetry.revised_top1}",
            f"- Revised top-3: {result.telemetry.revised_top3}",
            f"- Correct absence: {result.telemetry.correct_absence}",
            f"- Eligibility: {result.telemetry.extraction_eligibility_counts}",
            "- LLM calls: 0",
        ]
    ) + "\n"


def write_diagnostic_review_csv(
    result: BatchRetrievalDiagnosticResult,
    artifacts: BatchArtifacts,
    path: Path,
) -> None:
    intent_by_id = {item.target_row_id: item for item in result.target_semantics}
    current_by_id = {item.target_row_id: item for item in result.current_corpus_diagnostic}
    eligibility_by_id = {item.target_row_id: item for item in result.retrieval_eligibility}
    revised_by_id = {item.target_row_id: item for item in result.revised_retrieval_results}
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "target_id",
            "dictionary_row",
            "domain",
            "sub_domain",
            "field_name",
            "value_shape",
            "component",
            "requested_attribute",
            "batch_v1_review_status",
            "batch_v1_retrieval_status",
            "batch_v1_source_page",
            "current_corpus_evidence_found",
            "current_corpus_support_classification",
            "new_pages_probed",
            "revised_retrieval_source_page",
            "revised_support_classification",
            "final_failure_category",
            "extraction_eligibility",
            "explanation",
            "recommended_next_action",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for target in artifacts.selected_targets:
            intent = intent_by_id.get(target.target_row_id) or derive_target_intent(target)
            current = current_by_id.get(target.target_row_id)
            eligibility = eligibility_by_id[target.target_row_id]
            revised = revised_by_id[target.target_row_id]
            previous = artifacts.retrieval_by_id[target.target_row_id]
            first_previous = previous.results[0] if previous.results else None
            first_revised = revised.results[0] if revised.results else None
            writer.writerow(
                {
                    "target_id": target.target_row_id,
                    "dictionary_row": target.source_dictionary_provenance.row_number
                    if target.source_dictionary_provenance
                    else "",
                    "domain": target.metadata.get("domain") or target.sub_domain,
                    "sub_domain": target.sub_domain,
                    "field_name": target.expected_field,
                    "value_shape": intent.value_shape_family,
                    "component": intent.primary_component,
                    "requested_attribute": intent.requested_attribute,
                    "batch_v1_review_status": artifacts.validation_by_id[
                        target.target_row_id
                    ].status,
                    "batch_v1_retrieval_status": previous.retrieval_status,
                    "batch_v1_source_page": (
                        f"{first_previous.source_file}:{first_previous.page_start}"
                    )
                    if first_previous
                    else "",
                    "current_corpus_evidence_found": current.current_corpus_evidence_found
                    if current
                    else "",
                    "current_corpus_support_classification": (
                        current.best_component_plus_attribute_nodes[0].support_classification
                    )
                    if current and current.best_component_plus_attribute_nodes
                    else "",
                    "new_pages_probed": "; ".join(
                        f"{item.logical_path}:{item.page_start}-{item.page_end}"
                        for item in result.page_probe_plan.ranges
                        if target.target_row_id in item.target_row_ids
                    ),
                    "revised_retrieval_source_page": (
                        f"{first_revised.source_file}:{first_revised.page_start}"
                    )
                    if first_revised
                    else "",
                    "revised_support_classification": eligibility.support_classification,
                    "final_failure_category": (
                        current.failure_classification if current else "control"
                    ),
                    "extraction_eligibility": eligibility.status,
                    "explanation": eligibility.reason,
                    "recommended_next_action": next_action(eligibility),
                }
            )


def next_action(eligibility: RetrievalEligibility) -> str:
    if eligibility.status in {"eligible_for_extraction", "eligible_with_ambiguity"}:
        return "candidate_for_bounded_reextraction"
    if eligibility.status == "component_present_attribute_absent":
        return "do_not_extract_until_attribute_source_found"
    if eligibility.status == "dictionary_target_ambiguous":
        return "clarify_dictionary_metadata"
    return "do_not_send_to_model"


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
