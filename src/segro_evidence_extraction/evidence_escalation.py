"""Evidence escalation diagnostics for unresolved attribute-grounded targets."""

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
from segro_evidence_extraction.reextract_pass import (
    DEFAULT_REEXTRACT_V1_OUTPUT_DIR,
    DEFAULT_REEXTRACT_V2_OUTPUT_DIR,
    EXPECTED_REEXTRACT_V2_TARGET_COUNT,
)
from segro_evidence_extraction.retrieval_diagnostics import DEFAULT_DIAGNOSTIC_OUTPUT_DIR
from segro_evidence_extraction.target_semantics import TargetIntent
from segro_evidence_extraction.vertical_slice import (
    DEFAULT_CACHE_ROOT,
    EvidenceSpan,
    LightweightHierarchy,
    RetrievalResult,
    _atomic_write_json,
    split_range,
)

DEFAULT_ESCALATION_OUTPUT_DIR = Path("output/enfield_unit1_evidence_escalation_v1")
DEFAULT_SOURCE_MANIFEST = Path("output/sprint3_source_ingestion/source_pack_manifest.json")
MAX_ADJACENT_NEW_PAGES_PER_TARGET = 10
MAX_TOTAL_ADJACENT_NEW_PAGES = 50

TextEvidenceStatus = Literal[
    "attribute_evidence_found",
    "component_only",
    "weak_text",
    "no_text_evidence",
    "ambiguous_text",
]
TableEvidenceStatus = Literal[
    "table_evidence_present",
    "table_evidence_possible",
    "no_table_evidence",
]
VisualNeedStatus = Literal[
    "no_visual_needed",
    "visual_review_helpful",
    "OCR_candidate",
    "VLM_candidate",
    "drawing_measurement_candidate",
    "scanned_certificate_candidate",
]
FinalEscalationStatus = Literal[
    "text_evidence_found",
    "table_evidence_found",
    "OCR_recovered",
    "OCR_candidate",
    "VLM_candidate",
    "manual_visual_review",
    "dictionary_target_ambiguous",
    "source_unavailable",
    "unresolved",
]
ExtractionReadiness = Literal[
    "ready_for_text_reextract",
    "ready_after_OCR",
    "ready_after_VLM",
    "manual_review_only",
    "do_not_reextract",
    "unresolved",
]


class TargetIntentReview(StrictBaseModel):
    target_row_id: str
    field_name: str
    definition: str
    component: str
    requested_attribute: str
    value_shape: str
    prior_retrieved_pages: list[str] = Field(default_factory=list)
    prior_supporting_spans: list[str] = Field(default_factory=list)
    prior_abstention_reason: str | None = None
    likely_evidence_form: str
    likely_source_manual_section: str
    likely_evidence_modality: Literal["text", "table", "drawing", "image", "mixed"]


class AdjacentPagePlanItem(StrictBaseModel):
    target_row_id: str
    source_id: str
    logical_path: str
    anchor_page: int
    page_start: int
    page_end: int
    adjacent_pages: list[int]
    new_pages: list[int]
    reason: str


class AdjacentTextSearchResult(StrictBaseModel):
    target_row_id: str
    searched_pages: list[str]
    component_matches: list[str] = Field(default_factory=list)
    attribute_matches: list[str] = Field(default_factory=list)
    pattern_matches: list[str] = Field(default_factory=list)
    best_excerpt: str | None = None
    status: TextEvidenceStatus
    explanation: str


class TableEvidenceDiagnostic(StrictBaseModel):
    target_row_id: str
    status: TableEvidenceStatus
    candidate_pages: list[str] = Field(default_factory=list)
    candidate_rows: list[str] = Field(default_factory=list)
    candidate_headers: list[str] = Field(default_factory=list)
    explanation: str


class VisualPageTriage(StrictBaseModel):
    target_row_id: str
    source_page: str | None = None
    text_character_count: int = 0
    page_type: str
    visual_need: VisualNeedStatus
    explanation: str


class OcrCandidatePlanItem(StrictBaseModel):
    target_row_id: str
    candidate_page: str | None = None
    ocr_status: Literal["not_selected", "candidate"]
    reason: str


class VlmCandidatePlanItem(StrictBaseModel):
    target_row_id: str
    candidate_page: str | None = None
    vlm_status: Literal["not_selected", "candidate"]
    expected_question: str | None = None
    expected_output_type: str | None = None
    estimated_image_count: int = 0
    estimated_call_count: int = 0
    reason: str


class SourceAvailabilityAssessment(StrictBaseModel):
    target_row_id: str
    current_corpus_support: bool
    adjacent_text_support: bool
    table_support: bool
    visual_candidate: bool
    source_unavailable: bool
    explanation: str


class EscalationDecision(StrictBaseModel):
    target_row_id: str
    field_name: str
    final_status: FinalEscalationStatus
    recommended_next_action: str
    candidate_page_or_range: str | None = None
    evidence_type: str
    retry_extraction: bool
    ocr_required: bool
    vlm_required: bool
    manual_review_required: bool
    confidence: float = Field(ge=0, le=1)
    notes: str


class ReextractReadinessItem(StrictBaseModel):
    target_row_id: str
    readiness: ExtractionReadiness
    reason: str


class EscalationTelemetry(StrictBaseModel):
    target_count: int
    adjacent_ranges: int
    unique_pages_requested: int
    new_unique_pages_requested: int
    cache_hits: int
    cache_misses: int
    parser_worker_invocations: int
    active_child_count_after_cleanup: int
    text_status_counts: dict[str, int]
    table_status_counts: dict[str, int]
    visual_status_counts: dict[str, int]
    final_status_counts: dict[str, int]
    readiness_counts: dict[str, int]
    ocr_pages_run: int
    ocr_evidence_recovered: int
    vlm_candidate_count: int
    extraction_model_calls: int
    wall_time_ms: float


class EscalationRunResult(StrictBaseModel):
    frozen_targets: list[TargetSpecification]
    target_intent_review: list[TargetIntentReview]
    adjacent_page_plan: list[AdjacentPagePlanItem]
    adjacent_text_search: list[AdjacentTextSearchResult]
    table_evidence_diagnostic: list[TableEvidenceDiagnostic]
    visual_page_triage: list[VisualPageTriage]
    ocr_candidate_plan: list[OcrCandidatePlanItem]
    vlm_candidate_plan: list[VlmCandidatePlanItem]
    source_availability_assessment: list[SourceAvailabilityAssessment]
    escalation_decisions: list[EscalationDecision]
    reextract_readiness: list[ReextractReadinessItem]
    telemetry: EscalationTelemetry


class EscalationArtifacts:
    def __init__(
        self,
        *,
        reextract_v2_dir: Path,
        diagnostic_dir: Path,
    ) -> None:
        self.reextract_v2_dir = reextract_v2_dir
        self.diagnostic_dir = diagnostic_dir
        self.frozen_targets = [
            TargetSpecification.model_validate(raw)
            for raw in cast(
                list[dict[str, Any]],
                _read_json(reextract_v2_dir / "frozen_targets.json"),
            )
        ]
        self.target_by_id = {target.target_row_id: target for target in self.frozen_targets}
        self.raw_response_by_id = {
            str(item["target_row_id"]): item
            for item in cast(
                list[dict[str, Any]],
                _read_json(reextract_v2_dir / "raw_response_diagnostics.json"),
            )
        }
        self.extraction_by_id = {
            str(item["target_row_id"]): item
            for item in cast(
                list[dict[str, Any]], _read_json(reextract_v2_dir / "extraction_results.json")
            )
        }
        self.intent_by_id = {
            item.target_row_id: item
            for item in [
                TargetIntent.model_validate(raw)
                for raw in cast(
                    list[dict[str, Any]], _read_json(diagnostic_dir / "target_semantics.json")
                )
            ]
        }
        self.hierarchy = LightweightHierarchy.model_validate(
            _read_json(diagnostic_dir / "revised_hierarchy.json")
        )
        self.node_by_id = {node.node_id: node for node in self.hierarchy.nodes}
        self.retrieval_by_id = {
            item.target_row_id: item
            for item in [
                RetrievalResult.model_validate(raw)
                for raw in cast(
                    list[dict[str, Any]],
                    _read_json(diagnostic_dir / "revised_retrieval_results.json"),
                )
            ]
        }
        self.spans = [
            EvidenceSpan.model_validate(raw)
            for raw in cast(
                list[dict[str, Any]], _read_json(diagnostic_dir / "revised_evidence_spans.json")
            )
        ]
        self.spans_by_id = {span.span_id: span for span in self.spans}
        self.spans_by_target = _group_spans_by_target(self.spans, self.retrieval_by_id)
        self.failure_by_id = {
            str(item["target_row_id"]): item
            for item in cast(
                list[dict[str, Any]],
                _read_json(diagnostic_dir / "target_failure_classification.json"),
            )
        }
        self.eligibility_by_id = {
            str(item["target_row_id"]): item
            for item in cast(
                list[dict[str, Any]], _read_json(diagnostic_dir / "retrieval_eligibility.json")
            )
        }
        self.corrected_source_plan = cast(
            dict[str, Any], _read_json(diagnostic_dir / "corrected_source_range_plan.json")
        )
        self.page_probe_results = cast(
            dict[str, Any], _read_json(diagnostic_dir / "page_probe_results.json")
        )


def run_evidence_escalation_v1(
    *,
    reextract_v2_dir: Path = DEFAULT_REEXTRACT_V2_OUTPUT_DIR,
    v1_reextract_dir: Path = DEFAULT_REEXTRACT_V1_OUTPUT_DIR,
    diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_OUTPUT_DIR,
    output_dir: Path = DEFAULT_ESCALATION_OUTPUT_DIR,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> EscalationRunResult:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = EscalationArtifacts(
        reextract_v2_dir=reextract_v2_dir,
        diagnostic_dir=diagnostic_dir,
    )
    if len(artifacts.frozen_targets) != EXPECTED_REEXTRACT_V2_TARGET_COUNT:
        raise ValueError("frozen target count differs from seven")
    blocked_ids = _blocked_v1_ids(v1_reextract_dir)
    if any(target.target_row_id in blocked_ids for target in artifacts.frozen_targets):
        raise ValueError("V1 preflight-rejected target entered escalation scope")
    source_registry = {source.source_id: source for source in load_source_registry(source_manifest)}
    corrected_pages = _corrected_page_set(artifacts.corrected_source_plan)
    intent_review = build_target_intent_review(artifacts)
    adjacent_plan = build_adjacent_page_plan(artifacts, source_registry, corrected_pages)
    pages_by_key, parse_results = load_adjacent_pages(
        adjacent_plan,
        source_registry,
        cache_root,
        output_dir,
    )
    adjacent_search = [
        search_adjacent_text(target, artifacts, pages_by_key) for target in artifacts.frozen_targets
    ]
    table_diagnostics = [
        diagnose_table_evidence(target, artifacts, pages_by_key)
        for target in artifacts.frozen_targets
    ]
    visual_triage = [
        triage_visual_need(target, artifacts, pages_by_key) for target in artifacts.frozen_targets
    ]
    ocr_plan = [
        decide_ocr_candidate(target, visual_triage, table_diagnostics)
        for target in artifacts.frozen_targets
    ]
    vlm_plan = [
        decide_vlm_candidate(target, artifacts, visual_triage, table_diagnostics)
        for target in artifacts.frozen_targets
    ]
    source_assessment = [
        assess_source_availability(
            target,
            adjacent_search,
            table_diagnostics,
            visual_triage,
            vlm_plan,
        )
        for target in artifacts.frozen_targets
    ]
    decisions = [
        decide_escalation(
            target,
            adjacent_search,
            table_diagnostics,
            visual_triage,
            ocr_plan,
            vlm_plan,
            source_assessment,
        )
        for target in artifacts.frozen_targets
    ]
    readiness = [readiness_for_decision(decision) for decision in decisions]
    telemetry = build_telemetry(
        adjacent_plan=adjacent_plan,
        parse_results=parse_results,
        adjacent_search=adjacent_search,
        table_diagnostics=table_diagnostics,
        visual_triage=visual_triage,
        decisions=decisions,
        readiness=readiness,
        wall_time_ms=(time.perf_counter() - started) * 1000,
    )
    result = EscalationRunResult(
        frozen_targets=artifacts.frozen_targets,
        target_intent_review=intent_review,
        adjacent_page_plan=adjacent_plan,
        adjacent_text_search=adjacent_search,
        table_evidence_diagnostic=table_diagnostics,
        visual_page_triage=visual_triage,
        ocr_candidate_plan=ocr_plan,
        vlm_candidate_plan=vlm_plan,
        source_availability_assessment=source_assessment,
        escalation_decisions=decisions,
        reextract_readiness=readiness,
        telemetry=telemetry,
    )
    write_escalation_artifacts(result, output_dir)
    return result


def build_target_intent_review(artifacts: EscalationArtifacts) -> list[TargetIntentReview]:
    reviews: list[TargetIntentReview] = []
    for target in artifacts.frozen_targets:
        intent = artifacts.intent_by_id[target.target_row_id]
        retrieval = artifacts.retrieval_by_id[target.target_row_id]
        spans = artifacts.spans_by_target.get(target.target_row_id, [])
        likely_form = likely_evidence_form(intent, target)
        reviews.append(
            TargetIntentReview(
                target_row_id=target.target_row_id,
                field_name=target.expected_field,
                definition=target.requirement_text,
                component=intent.primary_component,
                requested_attribute=intent.requested_attribute,
                value_shape=intent.value_shape_family,
                prior_retrieved_pages=[
                    f"{item.source_file}:{item.page_start}" for item in retrieval.results
                ],
                prior_supporting_spans=[span.span_id for span in spans],
                prior_abstention_reason=cast(
                    str | None,
                    artifacts.raw_response_by_id.get(target.target_row_id, {}).get(
                        "rejection_reason"
                    ),
                ),
                likely_evidence_form=likely_form,
                likely_source_manual_section=likely_source_section(target, spans),
                likely_evidence_modality=likely_modality(intent, likely_form),
            )
        )
    return reviews


def build_adjacent_page_plan(
    artifacts: EscalationArtifacts,
    source_registry: dict[str, SourceRegistryEntry],
    corrected_pages: set[tuple[str, int]],
) -> list[AdjacentPagePlanItem]:
    planned: list[AdjacentPagePlanItem] = []
    global_new_pages: set[tuple[str, int]] = set()
    for target in artifacts.frozen_targets:
        spans = artifacts.spans_by_target.get(target.target_row_id, [])
        anchor = spans[0] if spans else None
        if anchor is None:
            continue
        source = source_registry[anchor.source_id]
        page_count = source.page_count or anchor.page_number + 5
        page_start = max(1, anchor.page_number - 5)
        page_end = min(page_count, anchor.page_number + 5)
        adjacent_pages = list(range(page_start, page_end + 1))
        candidate_new = [
            page for page in adjacent_pages if (anchor.source_id, page) not in corrected_pages
        ]
        selected_new: list[int] = []
        for page in candidate_new:
            if len(selected_new) >= MAX_ADJACENT_NEW_PAGES_PER_TARGET:
                break
            if len(global_new_pages) >= MAX_TOTAL_ADJACENT_NEW_PAGES:
                break
            key = (anchor.source_id, page)
            if key not in global_new_pages:
                global_new_pages.add(key)
                selected_new.append(page)
        planned.append(
            AdjacentPagePlanItem(
                target_row_id=target.target_row_id,
                source_id=anchor.source_id,
                logical_path=source.logical_path,
                anchor_page=anchor.page_number,
                page_start=page_start,
                page_end=page_end,
                adjacent_pages=adjacent_pages,
                new_pages=selected_new,
                reason="five-page adjacent context around prior supporting span",
            )
        )
    return dedupe_adjacent_plans(planned)


def dedupe_adjacent_plans(plans: list[AdjacentPagePlanItem]) -> list[AdjacentPagePlanItem]:
    seen: set[tuple[str, int, int, str]] = set()
    deduped: list[AdjacentPagePlanItem] = []
    for plan in plans:
        key = (plan.source_id, plan.page_start, plan.page_end, plan.target_row_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(plan)
    return deduped


def load_adjacent_pages(
    plans: list[AdjacentPagePlanItem],
    source_registry: dict[str, SourceRegistryEntry],
    cache_root: Path,
    output_dir: Path,
) -> tuple[dict[tuple[str, int], ParsedPage], list[CachedBatchParseResult]]:
    cache_service = CachedBatchParsingService(
        cache=CanonicalParsedPageCache(cache_root),
        batch_config=BatchWorkerConfig(
            stall_threshold_seconds=2.0,
            startup_timeout_seconds=10.0,
            max_restarts=0,
        ),
    )
    pages: dict[tuple[str, int], ParsedPage] = {}
    results: list[CachedBatchParseResult] = []
    seen_ranges: set[tuple[str, int, int]] = set()
    for plan in plans:
        for start, end in split_range(plan.page_start, plan.page_end):
            key = (plan.source_id, start, end)
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            source = source_registry[plan.source_id]
            result = cache_service.parse(
                BatchRequest(
                    source=source,
                    source_path=source.original_path,
                    output_dir=str(
                        output_dir / "worker_runs" / f"{source.source_id}_{start:04d}_{end:04d}"
                    ),
                    page_start=start,
                    page_end=end,
                )
            )
            results.append(result)
            for page in result.pages:
                pages[(page.source_id, page.page_number)] = page
    return pages, results


def search_adjacent_text(
    target: TargetSpecification,
    artifacts: EscalationArtifacts,
    pages_by_key: dict[tuple[str, int], ParsedPage],
) -> AdjacentTextSearchResult:
    intent = artifacts.intent_by_id[target.target_row_id]
    spans = artifacts.spans_by_target.get(target.target_row_id, [])
    source_pages = _candidate_page_keys(spans)
    component_matches: list[str] = []
    attribute_matches: list[str] = []
    pattern_matches: list[str] = []
    best_excerpt: str | None = None
    component_terms = _meaningful_component_terms(intent)
    attribute_terms = _meaningful_attribute_terms(intent)
    for key in source_pages:
        page = pages_by_key.get(key)
        if page is None:
            continue
        text = page.text or ""
        source_page = f"{page.source_id}:{page.page_number}"
        if _contains_any(text, component_terms):
            component_matches.append(source_page)
        if _contains_any(text, attribute_terms):
            attribute_matches.append(source_page)
        if _value_pattern(intent.value_shape_family, text):
            pattern_matches.append(source_page)
        if best_excerpt is None and (
            _contains_any(text, component_terms) or _contains_any(text, attribute_terms)
        ):
            best_excerpt = _line_excerpt(text, component_terms + attribute_terms)
    if component_matches and (attribute_matches or pattern_matches):
        status: TextEvidenceStatus = "attribute_evidence_found"
        explanation = "Component, attribute and value-shape indicators occur in adjacent text."
    elif component_matches and attribute_matches:
        status = "ambiguous_text"
        explanation = (
            "Component and attribute terms occur but no reliable value indicator was found."
        )
    elif component_matches:
        status = "component_only"
        explanation = "Adjacent text mentions the component without the requested attribute."
    elif attribute_matches or pattern_matches:
        status = "weak_text"
        explanation = "Adjacent text has weak attribute/value cues without component support."
    else:
        status = "no_text_evidence"
        explanation = "No adjacent text evidence was found."
    return AdjacentTextSearchResult(
        target_row_id=target.target_row_id,
        searched_pages=[f"{source}:{page}" for source, page in source_pages],
        component_matches=sorted(set(component_matches)),
        attribute_matches=sorted(set(attribute_matches)),
        pattern_matches=sorted(set(pattern_matches)),
        best_excerpt=best_excerpt,
        status=status,
        explanation=explanation,
    )


def diagnose_table_evidence(
    target: TargetSpecification,
    artifacts: EscalationArtifacts,
    pages_by_key: dict[tuple[str, int], ParsedPage],
) -> TableEvidenceDiagnostic:
    intent = artifacts.intent_by_id[target.target_row_id]
    candidate_pages: list[str] = []
    rows: list[str] = []
    headers: list[str] = []
    for key in _candidate_page_keys(artifacts.spans_by_target.get(target.target_row_id, [])):
        page = pages_by_key.get(key)
        if page is None:
            continue
        for line in (page.text or "").splitlines():
            if table_like_line(line):
                lower = line.lower()
                if _contains_any(lower, intent.component_terms + intent.attribute_terms):
                    candidate_pages.append(f"{page.source_id}:{page.page_number}")
                    rows.append(line.strip()[:300])
                    if any(
                        token in lower
                        for token in ["component", "manufacturer", "model", "date"]
                    ):
                        headers.append(line.strip()[:200])
    if rows and any(_contains_any(row, intent.component_terms) for row in rows):
        status: TableEvidenceStatus = "table_evidence_present"
        explanation = "Table-like rows contain component or requested-attribute evidence."
    elif rows:
        status = "table_evidence_possible"
        explanation = "Table-like rows exist but requested attribute support is weak."
    else:
        status = "no_table_evidence"
        explanation = "No target-relevant table-like text was detected."
    return TableEvidenceDiagnostic(
        target_row_id=target.target_row_id,
        status=status,
        candidate_pages=sorted(set(candidate_pages)),
        candidate_rows=rows[:5],
        candidate_headers=headers[:3],
        explanation=explanation,
    )


def triage_visual_need(
    target: TargetSpecification,
    artifacts: EscalationArtifacts,
    pages_by_key: dict[tuple[str, int], ParsedPage],
) -> VisualPageTriage:
    spans = artifacts.spans_by_target.get(target.target_row_id, [])
    key = _candidate_page_keys(spans)[0] if spans else None
    text = ""
    source_page = None
    if key is not None and key in pages_by_key:
        page = pages_by_key[key]
        text = page.text or ""
        source_page = f"{page.source_id}:{page.page_number}"
    char_count = len(text)
    lower = text.lower()
    drawing_terms = ["drawing", "layout", "plan", "elevation", "scale", "north"]
    certificate_terms = ["certificate", "commissioning", "installation"]
    if char_count < 150:
        page_type = "weak_or_sparse_text"
        visual_need: VisualNeedStatus = "visual_review_helpful"
        explanation = "Parsed text is sparse; visual review may reveal labels or schedules."
    elif any(term in lower for term in drawing_terms):
        page_type = "drawing_or_schedule"
        visual_need = "VLM_candidate"
        explanation = "Drawing/schedule cues suggest layout-sensitive evidence."
    elif any(term in lower for term in certificate_terms):
        page_type = "certificate_or_form"
        visual_need = "scanned_certificate_candidate" if char_count < 500 else "no_visual_needed"
        explanation = "Certificate/form cues are present."
    else:
        page_type = "text_page"
        visual_need = "no_visual_needed"
        explanation = "Parsed text is available and not obviously visual-only."
    return VisualPageTriage(
        target_row_id=target.target_row_id,
        source_page=source_page,
        text_character_count=char_count,
        page_type=page_type,
        visual_need=visual_need,
        explanation=explanation,
    )


def decide_ocr_candidate(
    target: TargetSpecification,
    visual_triage: list[VisualPageTriage],
    table_diagnostics: list[TableEvidenceDiagnostic],
) -> OcrCandidatePlanItem:
    visual = _by_id(visual_triage, target.target_row_id)
    table = _by_id(table_diagnostics, target.target_row_id)
    if (
        visual.visual_need in {"OCR_candidate", "scanned_certificate_candidate"}
        and table.status != "table_evidence_present"
    ):
        return OcrCandidatePlanItem(
            target_row_id=target.target_row_id,
            candidate_page=visual.source_page,
            ocr_status="candidate",
            reason="Textual certificate/form evidence may be present but parsed text is weak.",
        )
    return OcrCandidatePlanItem(
        target_row_id=target.target_row_id,
        candidate_page=visual.source_page,
        ocr_status="not_selected",
        reason="OCR is not appropriate for parsed text or layout/spatial evidence.",
    )


def decide_vlm_candidate(
    target: TargetSpecification,
    artifacts: EscalationArtifacts,
    visual_triage: list[VisualPageTriage],
    table_diagnostics: list[TableEvidenceDiagnostic],
) -> VlmCandidatePlanItem:
    visual = _by_id(visual_triage, target.target_row_id)
    table = _by_id(table_diagnostics, target.target_row_id)
    intent = artifacts.intent_by_id[target.target_row_id]
    needs_layout = visual.visual_need in {"VLM_candidate", "drawing_measurement_candidate"}
    if needs_layout and table.status != "table_evidence_present":
        return VlmCandidatePlanItem(
            target_row_id=target.target_row_id,
            candidate_page=visual.source_page,
            vlm_status="candidate",
            expected_question=(
                f"Does the page visually identify {intent.primary_component} "
                f"and its {intent.requested_attribute}?"
            ),
            expected_output_type=intent.value_shape_family,
            estimated_image_count=1,
            estimated_call_count=1,
            reason="Visual layout or drawing/schedule evidence is likely needed.",
        )
    return VlmCandidatePlanItem(
        target_row_id=target.target_row_id,
        candidate_page=visual.source_page,
        vlm_status="not_selected",
        estimated_image_count=0,
        estimated_call_count=0,
        reason="No material layout/spatial visual need was detected.",
    )


def assess_source_availability(
    target: TargetSpecification,
    adjacent_search: list[AdjacentTextSearchResult],
    table_diagnostics: list[TableEvidenceDiagnostic],
    visual_triage: list[VisualPageTriage],
    vlm_plan: list[VlmCandidatePlanItem],
) -> SourceAvailabilityAssessment:
    text = _by_id(adjacent_search, target.target_row_id)
    table = _by_id(table_diagnostics, target.target_row_id)
    visual = _by_id(visual_triage, target.target_row_id)
    vlm = _by_id(vlm_plan, target.target_row_id)
    text_support = text.status == "attribute_evidence_found"
    table_support = table.status == "table_evidence_present"
    visual_candidate = visual.visual_need != "no_visual_needed" or vlm.vlm_status == "candidate"
    unavailable = (
        not text_support
        and table.status == "no_table_evidence"
        and not visual_candidate
        and text.status in {"no_text_evidence", "component_only", "weak_text"}
    )
    explanation = (
        "No text/table/visual avenue remains."
        if unavailable
        else "At least one evidence avenue remains plausible."
    )
    return SourceAvailabilityAssessment(
        target_row_id=target.target_row_id,
        current_corpus_support=False,
        adjacent_text_support=text_support,
        table_support=table_support,
        visual_candidate=visual_candidate,
        source_unavailable=unavailable,
        explanation=explanation,
    )


def decide_escalation(
    target: TargetSpecification,
    adjacent_search: list[AdjacentTextSearchResult],
    table_diagnostics: list[TableEvidenceDiagnostic],
    visual_triage: list[VisualPageTriage],
    ocr_plan: list[OcrCandidatePlanItem],
    vlm_plan: list[VlmCandidatePlanItem],
    source_assessment: list[SourceAvailabilityAssessment],
) -> EscalationDecision:
    text = _by_id(adjacent_search, target.target_row_id)
    table = _by_id(table_diagnostics, target.target_row_id)
    visual = _by_id(visual_triage, target.target_row_id)
    ocr = _by_id(ocr_plan, target.target_row_id)
    vlm = _by_id(vlm_plan, target.target_row_id)
    source = _by_id(source_assessment, target.target_row_id)
    if text.status == "attribute_evidence_found":
        status: FinalEscalationStatus = "text_evidence_found"
        action = "Retry text extraction with adjacent text evidence."
        readiness: ExtractionReadiness = "ready_for_text_reextract"
        evidence_type = "text"
        confidence = 0.75
    elif table.status == "table_evidence_present":
        status = "table_evidence_found"
        action = "Retry extraction with table-like rows included as canonical spans."
        readiness = "ready_for_text_reextract"
        evidence_type = "table"
        confidence = 0.7
    elif ocr.ocr_status == "candidate":
        status = "OCR_candidate"
        action = "Run bounded OCR before any extraction retry."
        readiness = "ready_after_OCR"
        evidence_type = "ocr"
        confidence = 0.65
    elif vlm.vlm_status == "candidate":
        status = "VLM_candidate"
        action = "Use bounded visual review or VLM; do not retry text extraction first."
        readiness = "ready_after_VLM"
        evidence_type = "visual"
        confidence = 0.65
    elif visual.visual_need == "visual_review_helpful":
        status = "manual_visual_review"
        action = "Perform manual visual review of the candidate page."
        readiness = "manual_review_only"
        evidence_type = "visual"
        confidence = 0.55
    elif source.source_unavailable:
        status = "source_unavailable"
        action = "Do not re-extract until additional source material is available."
        readiness = "do_not_reextract"
        evidence_type = "none"
        confidence = 0.7
    else:
        status = "unresolved"
        action = "Keep unresolved; refine source targeting with manual page review."
        readiness = "unresolved"
        evidence_type = "mixed"
        confidence = 0.45
    return EscalationDecision(
        target_row_id=target.target_row_id,
        field_name=target.expected_field,
        final_status=status,
        recommended_next_action=action,
        candidate_page_or_range=visual.source_page
        or (text.searched_pages[0] if text.searched_pages else None),
        evidence_type=evidence_type,
        retry_extraction=readiness == "ready_for_text_reextract",
        ocr_required=readiness == "ready_after_OCR",
        vlm_required=readiness == "ready_after_VLM",
        manual_review_required=readiness == "manual_review_only",
        confidence=confidence,
        notes=text.explanation,
    )


def readiness_for_decision(decision: EscalationDecision) -> ReextractReadinessItem:
    if decision.final_status in {"text_evidence_found", "table_evidence_found"}:
        readiness: ExtractionReadiness = "ready_for_text_reextract"
    elif decision.final_status == "OCR_candidate":
        readiness = "ready_after_OCR"
    elif decision.final_status == "VLM_candidate":
        readiness = "ready_after_VLM"
    elif decision.final_status in {"manual_visual_review", "dictionary_target_ambiguous"}:
        readiness = "manual_review_only"
    elif decision.final_status == "source_unavailable":
        readiness = "do_not_reextract"
    else:
        readiness = "unresolved"
    return ReextractReadinessItem(
        target_row_id=decision.target_row_id,
        readiness=readiness,
        reason=decision.recommended_next_action,
    )


def build_telemetry(
    *,
    adjacent_plan: list[AdjacentPagePlanItem],
    parse_results: list[CachedBatchParseResult],
    adjacent_search: list[AdjacentTextSearchResult],
    table_diagnostics: list[TableEvidenceDiagnostic],
    visual_triage: list[VisualPageTriage],
    decisions: list[EscalationDecision],
    readiness: list[ReextractReadinessItem],
    wall_time_ms: float,
) -> EscalationTelemetry:
    unique_pages = {
        (plan.source_id, page) for plan in adjacent_plan for page in plan.adjacent_pages
    }
    new_pages = {(plan.source_id, page) for plan in adjacent_plan for page in plan.new_pages}
    return EscalationTelemetry(
        target_count=len(decisions),
        adjacent_ranges=len(adjacent_plan),
        unique_pages_requested=len(unique_pages),
        new_unique_pages_requested=len(new_pages),
        cache_hits=sum(result.cache_hits for result in parse_results),
        cache_misses=sum(result.cache_misses for result in parse_results),
        parser_worker_invocations=sum(result.worker_invocation_count for result in parse_results),
        active_child_count_after_cleanup=len(multiprocessing.active_children()),
        text_status_counts=dict(Counter(item.status for item in adjacent_search)),
        table_status_counts=dict(Counter(item.status for item in table_diagnostics)),
        visual_status_counts=dict(Counter(item.visual_need for item in visual_triage)),
        final_status_counts=dict(Counter(item.final_status for item in decisions)),
        readiness_counts=dict(Counter(item.readiness for item in readiness)),
        ocr_pages_run=0,
        ocr_evidence_recovered=0,
        vlm_candidate_count=sum(1 for item in decisions if item.final_status == "VLM_candidate"),
        extraction_model_calls=0,
        wall_time_ms=wall_time_ms,
    )


def write_escalation_artifacts(result: EscalationRunResult, output_dir: Path) -> None:
    payloads: dict[str, object] = {
        "frozen_targets.json": [item.model_dump(mode="json") for item in result.frozen_targets],
        "target_intent_review.json": [
            item.model_dump(mode="json") for item in result.target_intent_review
        ],
        "adjacent_page_plan.json": [
            item.model_dump(mode="json") for item in result.adjacent_page_plan
        ],
        "adjacent_text_search.json": [
            item.model_dump(mode="json") for item in result.adjacent_text_search
        ],
        "table_evidence_diagnostic.json": [
            item.model_dump(mode="json") for item in result.table_evidence_diagnostic
        ],
        "visual_page_triage.json": [
            item.model_dump(mode="json") for item in result.visual_page_triage
        ],
        "ocr_candidate_plan.json": [
            item.model_dump(mode="json") for item in result.ocr_candidate_plan
        ],
        "vlm_candidate_plan.json": [
            item.model_dump(mode="json") for item in result.vlm_candidate_plan
        ],
        "source_availability_assessment.json": [
            item.model_dump(mode="json") for item in result.source_availability_assessment
        ],
        "escalation_decisions.json": [
            item.model_dump(mode="json") for item in result.escalation_decisions
        ],
        "reextract_readiness.json": [
            item.model_dump(mode="json") for item in result.reextract_readiness
        ],
        "telemetry.json": result.telemetry.model_dump(mode="json"),
    }
    for filename, payload in payloads.items():
        _atomic_write_json(output_dir / filename, payload)
    write_escalation_review_csv(result, output_dir / "escalation_review.csv")
    (output_dir / "escalation_summary.md").write_text(
        escalation_summary_markdown(result),
        encoding="utf-8",
    )


def write_escalation_review_csv(result: EscalationRunResult, path: Path) -> None:
    intent_by_id = {item.target_row_id: item for item in result.target_intent_review}
    adjacent_by_id = {item.target_row_id: item for item in result.adjacent_text_search}
    table_by_id = {item.target_row_id: item for item in result.table_evidence_diagnostic}
    visual_by_id = {item.target_row_id: item for item in result.visual_page_triage}
    ocr_by_id = {item.target_row_id: item for item in result.ocr_candidate_plan}
    vlm_by_id = {item.target_row_id: item for item in result.vlm_candidate_plan}
    source_by_id = {item.target_row_id: item for item in result.source_availability_assessment}
    readiness_by_id = {item.target_row_id: item for item in result.reextract_readiness}
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "target_id",
            "field_name",
            "component",
            "requested_attribute",
            "prior_source_page",
            "prior_abstention_reason",
            "adjacent_pages_searched",
            "text_evidence_status",
            "table_evidence_status",
            "visual_page_status",
            "ocr_status",
            "vlm_status",
            "source_availability",
            "final_escalation_status",
            "recommended_next_action",
            "candidate_page_range",
            "extraction_readiness",
            "confidence",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for decision in result.escalation_decisions:
            target_id = decision.target_row_id
            intent = intent_by_id[target_id]
            adjacent = adjacent_by_id[target_id]
            writer.writerow(
                {
                    "target_id": target_id,
                    "field_name": decision.field_name,
                    "component": intent.component,
                    "requested_attribute": intent.requested_attribute,
                    "prior_source_page": "; ".join(intent.prior_retrieved_pages[:3]),
                    "prior_abstention_reason": intent.prior_abstention_reason,
                    "adjacent_pages_searched": "; ".join(adjacent.searched_pages),
                    "text_evidence_status": adjacent.status,
                    "table_evidence_status": table_by_id[target_id].status,
                    "visual_page_status": visual_by_id[target_id].visual_need,
                    "ocr_status": ocr_by_id[target_id].ocr_status,
                    "vlm_status": vlm_by_id[target_id].vlm_status,
                    "source_availability": source_by_id[target_id].source_unavailable,
                    "final_escalation_status": decision.final_status,
                    "recommended_next_action": decision.recommended_next_action,
                    "candidate_page_range": decision.candidate_page_or_range,
                    "extraction_readiness": readiness_by_id[target_id].readiness,
                    "confidence": decision.confidence,
                    "notes": decision.notes,
                }
            )


def escalation_summary_markdown(result: EscalationRunResult) -> str:
    lines = [
        "# Evidence Escalation Diagnostic V1",
        "",
        f"- Frozen targets: {result.telemetry.target_count}",
        f"- Adjacent ranges: {result.telemetry.adjacent_ranges}",
        f"- Unique pages requested: {result.telemetry.unique_pages_requested}",
        f"- New unique pages requested: {result.telemetry.new_unique_pages_requested}",
        f"- Cache hits: {result.telemetry.cache_hits}",
        f"- Cache misses: {result.telemetry.cache_misses}",
        f"- Parser workers: {result.telemetry.parser_worker_invocations}",
        f"- Final statuses: {result.telemetry.final_status_counts}",
        f"- Readiness: {result.telemetry.readiness_counts}",
        "- OCR pages actually run: 0",
        "- Extraction model calls: 0",
        "",
        "## Decisions",
    ]
    for decision in result.escalation_decisions:
        lines.append(
            f"- `{decision.field_name}`: {decision.final_status}; "
            f"{decision.recommended_next_action}"
        )
    return "\n".join(lines) + "\n"


def likely_evidence_form(intent: TargetIntent, target: TargetSpecification) -> str:
    field = target.expected_field.lower()
    if intent.value_shape_family == "integer_count":
        return "quantity/count label near component"
    if "capacity" in field or intent.value_shape_family == "decimal_measurement":
        return "schedule/table row with numeric capacity and unit"
    if "description" in field:
        return "component schedule, drawing annotation, or manual description"
    return "textual source statement"


def likely_source_section(target: TargetSpecification, spans: list[EvidenceSpan]) -> str:
    if spans:
        return spans[0].source_file
    component = (target.component_type or "").lower()
    if "pv" in component or "solar" in component:
        return "Building Manual - Part 6 Appendices or roof drawings"
    return "Building manuals and drawings"


def likely_modality(
    intent: TargetIntent, likely_form: str
) -> Literal["text", "table", "drawing", "image", "mixed"]:
    if "schedule" in likely_form or "table" in likely_form:
        return "table"
    if intent.value_shape_family == "integer_count":
        return "drawing"
    if "annotation" in likely_form:
        return "mixed"
    return "text"


def table_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    pipe_count = stripped.count("|")
    colon_fields = len(re.findall(r"\b[A-Z][A-Za-z /-]{2,}:", stripped))
    spaced_columns = bool(re.search(r"\w\s{2,}\w\s{2,}\w", stripped))
    return pipe_count >= 3 or colon_fields >= 2 or spaced_columns


def _contains_any(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms if term)


def _value_pattern(value_shape: str, text: str) -> bool:
    lower = text.lower()
    if value_shape == "integer_count":
        return bool(re.search(r"\b\d+\s*(?:no\.?|number|quantity)?\b", lower))
    if value_shape == "decimal_measurement":
        return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:m2|sqm|kw|kwp|kn|m)\b", lower))
    if value_shape == "date":
        return bool(re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", lower))
    return False


def _meaningful_component_terms(intent: TargetIntent) -> list[str]:
    terms = [intent.primary_component, *intent.component_terms]
    generic = {"green", "other", "roof", "floor", "component", "construction"}
    return _dedupe_terms([term for term in terms if term.lower() not in generic])


def _meaningful_attribute_terms(intent: TargetIntent) -> list[str]:
    generic = {
        "description",
        "component",
        "construction",
        "type",
        "name",
        "information",
        "system",
    }
    return _dedupe_terms([term for term in intent.attribute_terms if term.lower() not in generic])


def _dedupe_terms(terms: list[str]) -> list[str]:
    deduped: list[str] = []
    for term in terms:
        normalized = term.strip().lower()
        if len(normalized) < 3 or normalized in deduped:
            continue
        deduped.append(normalized)
    return deduped


def _line_excerpt(text: str, terms: list[str]) -> str | None:
    for line in text.splitlines():
        if _contains_any(line, terms):
            return line.strip()[:500]
    return None


def _candidate_page_keys(spans: list[EvidenceSpan]) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    for span in spans:
        for page in range(max(1, span.page_number - 5), span.page_number + 6):
            key = (span.source_id, page)
            if key not in keys:
                keys.append(key)
    return keys


def _group_spans_by_target(
    spans: list[EvidenceSpan],
    retrieval_by_id: dict[str, RetrievalResult],
) -> dict[str, list[EvidenceSpan]]:
    span_by_key = {
        (span.source_id, span.page_number, span.hierarchy_node_id): span for span in spans
    }
    grouped: dict[str, list[EvidenceSpan]] = defaultdict(list)
    for target_id, retrieval in retrieval_by_id.items():
        for result in retrieval.results:
            span = span_by_key.get((result.source_id, result.page_start, result.node_id))
            if span is not None:
                grouped[target_id].append(span)
    return dict(grouped)


def _corrected_page_set(plan: dict[str, Any]) -> set[tuple[str, int]]:
    pages: set[tuple[str, int]] = set()
    for key in ["retained_ranges", "added_ranges"]:
        for item in cast(list[dict[str, Any]], plan.get(key, [])):
            for page in range(int(item["page_start"]), int(item["page_end"]) + 1):
                pages.add((str(item["source_id"]), page))
    return pages


def _blocked_v1_ids(v1_dir: Path) -> set[str]:
    return {
        str(item["target_row_id"])
        for item in cast(list[dict[str, Any]], _read_json(v1_dir / "preflight_eligibility.json"))
        if item.get("accepted") is not True
    }


def _by_id(items: list[Any], target_id: str) -> Any:
    for item in items:
        if item.target_row_id == target_id:
            return item
    raise KeyError(target_id)


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
