"""Text-only re-extraction for escalation-approved attribute targets."""

from __future__ import annotations

import csv
import json
import multiprocessing
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import Field

from segro_evidence_extraction.config import Settings, load_settings
from segro_evidence_extraction.evidence_escalation import (
    DEFAULT_ESCALATION_OUTPUT_DIR,
    DEFAULT_SOURCE_MANIFEST,
    ReextractReadinessItem,
)
from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.source import SourceRegistryEntry
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.parsing.models import ParsedPage
from segro_evidence_extraction.parsing.page_cache import (
    CachedBatchParseResult,
    CachedBatchParsingService,
    CanonicalParsedPageCache,
)
from segro_evidence_extraction.parsing.service import load_source_registry
from segro_evidence_extraction.reextract_pass import (
    AttributeEvidenceValidationResult,
    ExtractionRequestRecord,
    ParsedResponseDiagnostic,
    StructuredExtractionEnvelope,
    build_normalized_records,
    build_reextract_validation,
    parse_structured_attribute_response,
    structured_response_schema,
)
from segro_evidence_extraction.target_semantics import TargetIntent
from segro_evidence_extraction.vertical_slice import (
    DEFAULT_CACHE_ROOT,
    EvidenceBundleRecord,
    EvidenceSpan,
    EvidenceValidationResult,
    ExtractionResult,
    NormalizedValueRecord,
    SchemaCompatibilityResult,
    SelectedTarget,
    ShapeValidationResult,
    ValidationResult,
    ValueShapeAssignment,
    ValueShapeFamily,
    _atomic_write_json,
    _display_value,
    _insufficient_evidence_result,
    assess_schema_compatibility_v3,
    calibrate_extraction_value,
    estimate_cost_usd,
    estimate_tokens,
    infer_value_shape_assignment,
    materialize_selected_spans,
    split_range,
    validate_shape_layer,
)

DEFAULT_ESCALATION_TEXT_REEXTRACT_OUTPUT_DIR = Path(
    "output/enfield_unit1_evidence_escalation_text_reextract_v1"
)
DEFAULT_REEXTRACT_V2_OUTPUT_DIR = Path("output/enfield_unit1_evidence_first_batch_v1_reextract_v2")
DEFAULT_BATCH_V1_OUTPUT_DIR = Path("output/enfield_unit1_evidence_first_batch_v1")
DEFAULT_REEXTRACT_V1_OUTPUT_DIR = Path("output/enfield_unit1_evidence_first_batch_v1_reextract_v1")
EXPECTED_TEXT_REEXTRACT_IDS = {
    "trg_0f92e062c431082f",
    "trg_af5d4799dad79a68",
}
MANUAL_REVIEW_BLOCKER = "manual_review_only"
DEFAULT_COST_CEILING_USD = 0.01
MAX_OUTPUT_TOKENS = 450

TextReextractOutcome = Literal[
    "corrected",
    "improved",
    "unchanged",
    "still_insufficient",
    "multiple_candidates",
    "regressed",
]


class ApprovedEvidenceRecord(StrictBaseModel):
    target_row_id: str
    field_name: str
    source_id: str
    source_file: str
    page_number: int = Field(ge=1)
    span_id: str
    hierarchy_node_id: str
    text: str
    value_candidates: list[str] = Field(default_factory=list)
    support_status: Literal["supported", "component_only", "attribute_only", "insufficient"]
    issues: list[str] = Field(default_factory=list)


class TextPreflightEligibilityItem(StrictBaseModel):
    target_row_id: str
    field_name: str
    accepted: bool
    readiness: str
    approved_span_ids: list[str] = Field(default_factory=list)
    requested_pages: list[str] = Field(default_factory=list)
    component_support: bool
    attribute_support: bool
    value_shape_support: bool
    local_association: bool
    reason: str


class TextPreflightReport(StrictBaseModel):
    target_ids: list[str]
    field_names: dict[str, str]
    approved_source_pages: dict[str, list[str]]
    approved_span_ids: dict[str, list[str]]
    expected_calls: int
    configured_model: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_ceiling_usd: float
    estimated_cost_usd: float
    cache_hits: int
    cache_misses: int
    parser_worker_invocations_expected: int
    safety_gate_passed: bool
    safety_gate_errors: list[str] = Field(default_factory=list)


class CanonicalEvidenceResult(StrictBaseModel):
    target_row_id: str
    status: Literal["valid", "invalid", "review_required"]
    selected_span_ids: list[str] = Field(default_factory=list)
    value_bearing_phrase_present: bool
    local_numeric_association: bool
    unit_association_valid: bool
    source_page_node_valid: bool
    issues: list[str] = Field(default_factory=list)


class PriorAttemptComparisonItem(StrictBaseModel):
    target_row_id: str
    field_name: str
    batch_v1_status: str | None = None
    batch_v1_value: Any = None
    reextract_v1_status: str | None = None
    reextract_v1_value: Any = None
    reextract_v2_status: str | None = None
    reextract_v2_value: Any = None
    escalation_decision: str
    escalation_evidence: list[str] = Field(default_factory=list)
    new_status: str
    new_value: Any = None
    new_review_status: str
    outcome: TextReextractOutcome


class TextReextractTelemetry(StrictBaseModel):
    frozen_target_count: int
    eligible_target_count: int
    preflight_rejected_count: int
    model: str
    model_call_count: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    wall_time_ms: float
    extraction_status_counts: dict[str, int]
    evidence_validation_counts: dict[str, int]
    review_status_counts: dict[str, int]
    comparison_outcome_counts: dict[str, int]
    cache_hits: int
    cache_misses: int
    parser_worker_invocations: int
    active_child_count_after_cleanup: int


class TextReextractRunResult(StrictBaseModel):
    frozen_targets: list[TargetSpecification]
    preflight_report: TextPreflightReport
    approved_evidence: list[ApprovedEvidenceRecord]
    extraction_requests: list[ExtractionRequestRecord]
    raw_response_diagnostics: list[ParsedResponseDiagnostic]
    parsed_response_results: list[StructuredExtractionEnvelope | None]
    extraction_results: list[ExtractionResult]
    canonical_evidence_results: list[CanonicalEvidenceResult]
    normalized_values: list[NormalizedValueRecord]
    evidence_validation: list[AttributeEvidenceValidationResult]
    shape_validation: list[ShapeValidationResult]
    schema_compatibility_results: list[SchemaCompatibilityResult]
    validation_results: list[ValidationResult]
    prior_attempt_comparison: list[PriorAttemptComparisonItem]
    telemetry: TextReextractTelemetry


class TextStructuredExtractionClient(Protocol):
    provider: str
    model_name: str

    def extract_structured(
        self,
        *,
        target: TargetSpecification,
        intent: TargetIntent,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> tuple[ExtractionResult, ParsedResponseDiagnostic, StructuredExtractionEnvelope | None]: ...


class EscalationTextOpenAIClient:
    provider = "openai"

    def __init__(self, *, api_key: str, model_name: str) -> None:
        self.api_key = api_key
        self.model_name = model_name

    def extract_structured(
        self,
        *,
        target: TargetSpecification,
        intent: TargetIntent,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> tuple[ExtractionResult, ParsedResponseDiagnostic, StructuredExtractionEnvelope | None]:
        payload = escalation_text_request_payload(
            model_name=self.model_name,
            target=target,
            intent=intent,
            bundle=bundle,
            max_output_tokens=max_output_tokens,
        )
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
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                response_payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            diagnostic = ParsedResponseDiagnostic(
                target_row_id=target.target_row_id,
                provider_error=str(exc),
                final_status="model_error",
                invalid_format_reason="transport_or_provider_error",
            )
            return (
                ExtractionResult(
                    target_row_id=target.target_row_id,
                    requirement_id=target.requirement_id,
                    status="model_error",
                    confidence=0,
                    model_provider=self.provider,
                    model_name=self.model_name,
                    ambiguity_or_caveat=str(exc),
                ),
                diagnostic,
                None,
            )
        content = str(response_payload["choices"][0]["message"].get("content") or "")
        finish_reason = str(response_payload["choices"][0].get("finish_reason") or "")
        extraction, diagnostic, parsed = parse_structured_attribute_response(
            target=target,
            bundle=bundle,
            provider=self.provider,
            model_name=self.model_name,
            response_text=content,
            finish_reason=finish_reason,
        )
        usage = response_payload.get("usage", {})
        extraction.model_usage.input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        extraction.model_usage.output_tokens = int(usage.get("completion_tokens", 0) or 0)
        extraction.model_usage.estimated_cost_usd = estimate_cost_usd(
            self.model_name,
            extraction.model_usage.input_tokens,
            extraction.model_usage.output_tokens,
        )
        return extraction, diagnostic, parsed


class TextReextractArtifacts:
    def __init__(
        self,
        *,
        escalation_dir: Path,
        reextract_v2_dir: Path,
        batch_v1_dir: Path,
        reextract_v1_dir: Path,
    ) -> None:
        self.escalation_dir = escalation_dir
        self.reextract_v2_dir = reextract_v2_dir
        self.batch_v1_dir = batch_v1_dir
        self.reextract_v1_dir = reextract_v1_dir
        self.escalation_targets = [
            TargetSpecification.model_validate(raw)
            for raw in cast(
                list[dict[str, Any]], _read_json(escalation_dir / "frozen_targets.json")
            )
        ]
        self.target_by_id = {target.target_row_id: target for target in self.escalation_targets}
        self.intent_by_id = {
            str(item["target_row_id"]): TargetIntent(
                target_row_id=str(item["target_row_id"]),
                field_name=str(item["field_name"]),
                field_definition=str(item["definition"]),
                domain=str(item.get("domain") or item.get("sub_domain") or ""),
                sub_domain=str(item.get("sub_domain") or ""),
                value_shape_family=_stage_value_shape(str(item["field_name"])),
                primary_component=str(item["component"]),
                component_terms=_component_terms_for_field(str(item["field_name"])),
                requested_attribute=_stage_requested_attribute(str(item["field_name"])),
                attribute_terms=_attribute_terms_for_field(str(item["field_name"])),
                expected_value_indicators=_value_indicators_for_field(str(item["field_name"])),
            )
            for item in cast(
                list[dict[str, Any]],
                _read_json(escalation_dir / "target_intent_review.json"),
            )
        }
        self.readiness_by_id = {
            item.target_row_id: item
            for item in [
                ReextractReadinessItem.model_validate(raw)
                for raw in cast(
                    list[dict[str, Any]], _read_json(escalation_dir / "reextract_readiness.json")
                )
            ]
        }
        self.adjacent_by_id = {
            str(item["target_row_id"]): item
            for item in cast(
                list[dict[str, Any]], _read_json(escalation_dir / "adjacent_text_search.json")
            )
        }
        self.decisions_by_id = {
            str(item["target_row_id"]): item
            for item in cast(
                list[dict[str, Any]], _read_json(escalation_dir / "escalation_decisions.json")
            )
        }
        self.reextract_v2_frozen_ids = [
            str(item["target_row_id"])
            for item in cast(
                list[dict[str, Any]], _read_json(reextract_v2_dir / "frozen_targets.json")
            )
        ]
        self.prior_extractions = {
            "batch_v1": _load_extractions(batch_v1_dir / "extraction_results.json"),
            "reextract_v1": _load_extractions(reextract_v1_dir / "extraction_results.json"),
            "reextract_v2": _load_extractions(reextract_v2_dir / "extraction_results.json"),
        }
        self.prior_validations = {
            "batch_v1": _load_validations(batch_v1_dir / "validation_results.json"),
            "reextract_v1": _load_validations(reextract_v1_dir / "validation_results.json"),
            "reextract_v2": _load_validations(reextract_v2_dir / "validation_results.json"),
        }


def run_escalation_text_reextract_v1(
    *,
    escalation_dir: Path = DEFAULT_ESCALATION_OUTPUT_DIR,
    reextract_v2_dir: Path = DEFAULT_REEXTRACT_V2_OUTPUT_DIR,
    batch_v1_dir: Path = DEFAULT_BATCH_V1_OUTPUT_DIR,
    reextract_v1_dir: Path = DEFAULT_REEXTRACT_V1_OUTPUT_DIR,
    output_dir: Path = DEFAULT_ESCALATION_TEXT_REEXTRACT_OUTPUT_DIR,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    settings: Settings | None = None,
    extraction_client: TextStructuredExtractionClient | None = None,
    cost_ceiling_usd: float = DEFAULT_COST_CEILING_USD,
    print_preflight: bool = False,
) -> TextReextractRunResult:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = settings or load_settings(Path("configs/default.yaml"))
    artifacts = TextReextractArtifacts(
        escalation_dir=escalation_dir,
        reextract_v2_dir=reextract_v2_dir,
        batch_v1_dir=batch_v1_dir,
        reextract_v1_dir=reextract_v1_dir,
    )
    target_ids = load_ready_text_target_ids(artifacts)
    targets = [artifacts.target_by_id[target_id] for target_id in target_ids]
    registry = {source.source_id: source for source in load_source_registry(source_manifest)}
    pages, parse_results = load_approved_pages(
        artifacts, target_ids, registry, cache_root, output_dir
    )
    approved_evidence = build_approved_evidence(artifacts, targets, pages, registry)
    bundles = {
        target.target_row_id: build_text_reextract_bundle(
            target,
            approved_evidence,
        )
        for target in targets
    }
    model_name = extraction_client.model_name if extraction_client is not None else "gpt-4o-mini"
    preflight_items, preflight_report = build_text_preflight(
        artifacts=artifacts,
        targets=targets,
        evidence=approved_evidence,
        bundles=bundles,
        parse_results=parse_results,
        model_name=model_name,
        cost_ceiling_usd=cost_ceiling_usd,
    )
    write_preflight_artifacts(output_dir, targets, preflight_report, approved_evidence)
    if print_preflight:
        print(json.dumps(preflight_report.model_dump(mode="json"), indent=2))
    if preflight_report.safety_gate_errors:
        # Scope errors are fatal. Evidence-strength failures produce auditable abstentions below.
        fatal = [
            error
            for error in preflight_report.safety_gate_errors
            if not error.startswith("preflight evidence rejected:")
        ]
        if fatal:
            raise ValueError("; ".join(fatal))
    client = extraction_client or _default_text_client(settings, model_name)
    extraction_results: list[ExtractionResult] = []
    diagnostics: list[ParsedResponseDiagnostic] = []
    parsed_results: list[StructuredExtractionEnvelope | None] = []
    requests: list[ExtractionRequestRecord] = []
    for item in preflight_items:
        target = artifacts.target_by_id[item.target_row_id]
        intent = artifacts.intent_by_id[item.target_row_id]
        bundle = bundles[item.target_row_id]
        if item.accepted:
            requests.append(
                ExtractionRequestRecord(
                    target_row_id=target.target_row_id,
                    field_name=target.expected_field,
                    component=intent.primary_component,
                    requested_attribute=intent.requested_attribute,
                    value_shape_family=intent.value_shape_family,
                    supporting_span_ids=[span.span_id for span in bundle.evidence_spans],
                    prompt_token_estimate=estimate_tokens(
                        escalation_text_prompt(target, intent, bundle)
                    ),
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                )
            )
            extraction, diagnostic, parsed = client.extract_structured(
                target=target,
                intent=intent,
                bundle=bundle,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        else:
            extraction = _insufficient_evidence_result(
                target,
                client.provider,
                client.model_name,
                item.reason,
            )
            diagnostic = ParsedResponseDiagnostic(
                target_row_id=target.target_row_id,
                final_status="skipped_preflight",
                invalid_format_reason=None,
            )
            parsed = None
        extraction = materialize_selected_spans(extraction, bundle)
        extraction = calibrate_text_reextract_value(extraction, target, intent, bundle)
        extraction_results.append(extraction)
        diagnostics.append(diagnostic)
        parsed_results.append(parsed)
    assignments = [
        stage_value_shape_assignment(target, artifacts.intent_by_id[target.target_row_id])
        for target in targets
    ]
    canonical_evidence_results = [
        validate_text_canonical_evidence(
            extraction,
            artifacts.intent_by_id[extraction.target_row_id],
            bundles[extraction.target_row_id],
        )
        for extraction in extraction_results
    ]
    evidence_validation = [
        AttributeEvidenceValidationResult(
            target_row_id=item.target_row_id,
            status=item.status,
            selected_span_ids=item.selected_span_ids,
            span_ids_valid=item.status != "invalid"
            or "Unknown selected span ID." not in item.issues,
            canonical_evidence_available=bool(item.selected_span_ids),
            value_bearing_phrase_present=item.value_bearing_phrase_present,
            requested_attribute_supported=item.local_numeric_association,
            component_value_linked=item.local_numeric_association,
            source_page_node_valid=item.source_page_node_valid,
            issues=item.issues,
        )
        for item in canonical_evidence_results
    ]
    shape_validation = validate_shape_layer(extraction_results, assignments)
    schema_evidence = [
        EvidenceValidationResult(
            target_row_id=item.target_row_id,
            status="valid" if item.status != "invalid" else "invalid",
            selected_span_ids=item.selected_span_ids,
            canonical_evidence_available=bool(item.selected_span_ids),
            evidence_value_present=item.value_bearing_phrase_present,
            source_page_node_valid=item.source_page_node_valid,
            issues=item.issues,
        )
        for item in canonical_evidence_results
    ]
    selected_targets = [
        SelectedTarget(target=target, selection_reason="escalation_text_reextract")
        for target in targets
    ]
    schema_results = assess_schema_compatibility_v3(
        extraction_results,
        selected_targets,
        assignments,
        schema_evidence,
        shape_validation,
    )
    validation_results = build_reextract_validation(
        extraction_results,
        evidence_validation,
        shape_validation,
        schema_results,
    )
    normalized_values = build_normalized_records(extraction_results, assignments)
    comparison = build_prior_attempt_comparison(
        artifacts,
        extraction_results,
        validation_results,
        approved_evidence,
    )
    telemetry = build_text_reextract_telemetry(
        target_count=len(targets),
        eligible_count=sum(1 for item in preflight_items if item.accepted),
        parse_results=parse_results,
        extraction_results=extraction_results,
        canonical_evidence_results=canonical_evidence_results,
        validation_results=validation_results,
        comparison=comparison,
        wall_time_ms=(time.perf_counter() - started) * 1000,
        model_name=model_name,
    )
    result = TextReextractRunResult(
        frozen_targets=targets,
        preflight_report=preflight_report,
        approved_evidence=approved_evidence,
        extraction_requests=requests,
        raw_response_diagnostics=diagnostics,
        parsed_response_results=parsed_results,
        extraction_results=extraction_results,
        canonical_evidence_results=canonical_evidence_results,
        normalized_values=normalized_values,
        evidence_validation=evidence_validation,
        shape_validation=shape_validation,
        schema_compatibility_results=schema_results,
        validation_results=validation_results,
        prior_attempt_comparison=comparison,
        telemetry=telemetry,
    )
    write_text_reextract_artifacts(result, output_dir)
    return result


def load_ready_text_target_ids(artifacts: TextReextractArtifacts) -> list[str]:
    ready = [
        target_id
        for target_id, item in artifacts.readiness_by_id.items()
        if item.readiness == "ready_for_text_reextract"
    ]
    ready = sorted(ready)
    if set(ready) != EXPECTED_TEXT_REEXTRACT_IDS:
        raise ValueError(f"ready text target IDs differ from frozen set: {ready}")
    manual = [
        target_id
        for target_id, item in artifacts.readiness_by_id.items()
        if item.readiness == MANUAL_REVIEW_BLOCKER
    ]
    if any(target_id in ready for target_id in manual):
        raise ValueError("manual-review target entered text re-extraction scope")
    if any(target_id not in artifacts.reextract_v2_frozen_ids for target_id in ready):
        raise ValueError("target was not present in prior frozen unresolved set")
    return ready


def load_approved_pages(
    artifacts: TextReextractArtifacts,
    target_ids: list[str],
    source_registry: dict[str, SourceRegistryEntry],
    cache_root: Path,
    output_dir: Path,
) -> tuple[dict[tuple[str, int], ParsedPage], list[CachedBatchParseResult]]:
    requested: dict[str, set[int]] = {}
    for target_id in target_ids:
        adjacent = artifacts.adjacent_by_id[target_id]
        for raw_page in cast(list[str], adjacent.get("searched_pages", [])):
            source_id, page = _parse_source_page(raw_page)
            requested.setdefault(source_id, set()).add(page)
    cache_service = CachedBatchParsingService(cache=CanonicalParsedPageCache(cache_root))
    pages: dict[tuple[str, int], ParsedPage] = {}
    results: list[CachedBatchParseResult] = []
    for source_id in sorted(requested):
        for start, end in _contiguous_ranges(sorted(requested[source_id])):
            for batch_start, batch_end in split_range(start, end):
                _ = output_dir
                source = source_registry[source_id]
                page_numbers = list(range(batch_start, batch_end + 1))
                started = time.perf_counter()
                lookup = cache_service.cache.read_range(
                    source=source,
                    source_path=source.original_path,
                    page_numbers=page_numbers,
                    parser_name=cache_service.parser_name,
                    parser_version=cache_service.parser_version,
                    parser_config_fingerprint=cache_service.parser_config_fingerprint,
                )
                result = CachedBatchParseResult(
                    source_id=source.source_id,
                    source_hash=source.file_hash,
                    parser_name=cache_service.parser_name,
                    parser_version=cache_service.parser_version,
                    parser_config_fingerprint=cache_service.parser_config_fingerprint,
                    requested_page_start=batch_start,
                    requested_page_end=batch_end,
                    cache_root=str(cache_root),
                    cache_hits=lookup.hits,
                    cache_misses=lookup.misses,
                    invalid_cache_entries=lookup.invalid_entries,
                    pages_loaded_from_cache=sorted(lookup.pages),
                    pages_newly_parsed=[],
                    page_artifacts_written=0,
                    missing_ranges_sent_to_workers=[
                        f"{start}-{end}" for start, end in _contiguous_ranges(lookup.missing_pages)
                    ],
                    worker_invocation_count=0,
                    restart_count=0,
                    total_wall_time_ms=(time.perf_counter() - started) * 1000,
                    cache_read_time_ms=lookup.read_ms,
                    parse_time_ms=0,
                    cache_write_time_ms=0,
                    active_child_count_after_cleanup=len(multiprocessing.active_children()),
                    pages=[lookup.pages[page] for page in sorted(lookup.pages)],
                    cache_warnings=lookup.invalid_warnings,
                )
                results.append(result)
                for parsed_page in result.pages:
                    pages[(parsed_page.source_id, parsed_page.page_number)] = parsed_page
    return pages, results


def build_approved_evidence(
    artifacts: TextReextractArtifacts,
    targets: list[TargetSpecification],
    pages: dict[tuple[str, int], ParsedPage],
    source_registry: dict[str, SourceRegistryEntry],
) -> list[ApprovedEvidenceRecord]:
    records: list[ApprovedEvidenceRecord] = []
    for target in targets:
        candidate_pages = [
            _parse_source_page(raw)
            for raw in cast(
                list[str],
                artifacts.adjacent_by_id[target.target_row_id].get("searched_pages", []),
            )
        ]
        for source_id, page_number in candidate_pages:
            page = pages.get((source_id, page_number))
            if page is None:
                continue
            source = source_registry[source_id]
            records.extend(_spans_for_page(target, page, source.logical_path))
    return records


def _spans_for_page(
    target: TargetSpecification,
    page: ParsedPage,
    source_file: str,
) -> list[ApprovedEvidenceRecord]:
    text = page.text or ""
    records: list[ApprovedEvidenceRecord] = []
    for start, _end, block in _candidate_blocks(text):
        status, candidates, issues = support_for_target(target.expected_field, block)
        if status == "insufficient":
            continue
        digest = _stable_digest(
            [target.target_row_id, page.source_id, str(page.page_number), str(start), block]
        )
        records.append(
            ApprovedEvidenceRecord(
                target_row_id=target.target_row_id,
                field_name=target.expected_field,
                source_id=page.source_id,
                source_file=source_file,
                page_number=page.page_number,
                span_id=f"esc_text_{target.target_row_id}_{page.source_id}_p{page.page_number:04d}_{digest}",
                hierarchy_node_id=f"page:{page.source_id}:{page.page_number:06d}",
                text=block,
                value_candidates=candidates,
                support_status=status,
                issues=issues,
            )
        )
    return records


def support_for_target(
    field_name: str,
    text: str,
) -> tuple[
    Literal["supported", "component_only", "attribute_only", "insufficient"], list[str], list[str]
]:
    if field_name == "floor_construction_capacity":
        return floor_capacity_support(text)
    if field_name == "dock_count":
        return dock_count_support(text)
    return ("insufficient", [], ["unsupported field for text escalation pass"])


def floor_capacity_support(
    text: str,
) -> tuple[
    Literal["supported", "component_only", "attribute_only", "insufficient"], list[str], list[str]
]:
    lower = text.lower()
    has_component = any(token in lower for token in ["floor", "slab", "concrete slab", "warehouse"])
    has_attr = any(token in lower for token in ["load", "loading", "capacity", "duty", "udl"])
    measurements = _floor_capacity_candidates(text)
    issues: list[str] = []
    if "n/mm" in lower or "compressive" in lower or "strength" in lower:
        issues.append("material strength evidence is not floor loading capacity")
        measurements = [item for item in measurements if "n/mm" not in item.lower()]
    if has_component and has_attr and measurements:
        return ("supported", measurements, issues)
    if has_component:
        return (
            "component_only",
            measurements,
            [*issues, "floor/slab component present without capacity"],
        )
    if has_attr or measurements:
        return (
            "attribute_only",
            measurements,
            [*issues, "load/capacity value present without floor/slab"],
        )
    return ("insufficient", [], ["no floor capacity evidence"])


def dock_count_support(
    text: str,
) -> tuple[
    Literal["supported", "component_only", "attribute_only", "insufficient"], list[str], list[str]
]:
    lower = text.lower()
    if "fire exit door" in lower and "dock" not in lower:
        return ("attribute_only", [], ["fire-door numbers are not dock count evidence"])
    has_component = any(
        token in lower
        for token in [
            "dock",
            "dock door",
            "dock doors",
            "dock leveller",
            "loading dock",
            "loading door",
        ]
    )
    candidates = _dock_count_candidates(text)
    if has_component and candidates:
        return ("supported", candidates, [])
    if has_component:
        return ("component_only", [], ["dock component present without associated integer"])
    if re.search(r"\b\d+\b", text):
        return ("attribute_only", [], ["integer present without dock component"])
    return ("insufficient", [], ["no dock count evidence"])


def build_text_reextract_bundle(
    target: TargetSpecification,
    evidence: list[ApprovedEvidenceRecord],
) -> EvidenceBundleRecord:
    target_evidence = [
        item
        for item in evidence
        if item.target_row_id == target.target_row_id and item.support_status == "supported"
    ]
    spans = [
        EvidenceSpan(
            span_id=item.span_id,
            source_id=item.source_id,
            source_file=item.source_file,
            page_number=item.page_number,
            hierarchy_node_id=item.hierarchy_node_id,
            text=item.text,
            start_char=0,
            end_char=len(item.text),
            retrieval_rank=index + 1,
            score=10.0,
        )
        for index, item in enumerate(target_evidence)
    ]
    combined = "\n\n".join(
        f"[span_id={span.span_id} source={span.source_file} page={span.page_number}] {span.text}"
        for span in spans
    )
    return EvidenceBundleRecord(
        target_row_id=target.target_row_id,
        evidence_items=[],
        retrieval_status="evidence_found" if spans else "no_relevant_evidence",
        evidence_spans=spans,
        combined_text=combined,
        character_count=len(combined),
        token_estimate=estimate_tokens(combined),
    )


def build_text_preflight(
    *,
    artifacts: TextReextractArtifacts,
    targets: list[TargetSpecification],
    evidence: list[ApprovedEvidenceRecord],
    bundles: dict[str, EvidenceBundleRecord],
    parse_results: list[CachedBatchParseResult],
    model_name: str,
    cost_ceiling_usd: float,
) -> tuple[list[TextPreflightEligibilityItem], TextPreflightReport]:
    errors: list[str] = []
    if {target.target_row_id for target in targets} != EXPECTED_TEXT_REEXTRACT_IDS:
        errors.append("target count or IDs do not match the two escalation-approved targets")
    if any(
        artifacts.readiness_by_id[target.target_row_id].readiness != "ready_for_text_reextract"
        for target in targets
    ):
        errors.append("manual-review target entered scope")
    if any(result.cache_misses for result in parse_results):
        errors.append("a parser cache miss would require a worker")
    if any(result.worker_invocation_count for result in parse_results):
        errors.append("parser worker was invoked")
    evidence_by_target: dict[str, list[ApprovedEvidenceRecord]] = {
        target.target_row_id: [
            item for item in evidence if item.target_row_id == target.target_row_id
        ]
        for target in targets
    }
    items: list[TextPreflightEligibilityItem] = []
    for target in targets:
        target_evidence = evidence_by_target[target.target_row_id]
        supported = [item for item in target_evidence if item.support_status == "supported"]
        span_ids = [item.span_id for item in supported]
        component = any(
            item.support_status in {"supported", "component_only"} for item in target_evidence
        )
        attribute = any(
            item.support_status in {"supported", "attribute_only"} for item in target_evidence
        )
        value_shape = any(item.value_candidates for item in supported)
        accepted = bool(span_ids and component and attribute and value_shape)
        reason = "accepted"
        if not accepted:
            reason = (
                "no escalation-approved span contains local component, "
                "requested attribute and value-shape evidence"
            )
        if not accepted:
            errors.append(f"preflight evidence rejected: {target.target_row_id}: {reason}")
        items.append(
            TextPreflightEligibilityItem(
                target_row_id=target.target_row_id,
                field_name=target.expected_field,
                accepted=accepted,
                readiness=artifacts.readiness_by_id[target.target_row_id].readiness,
                approved_span_ids=span_ids,
                requested_pages=sorted(
                    {f"{item.source_id}:{item.page_number}" for item in target_evidence}
                ),
                component_support=component,
                attribute_support=attribute,
                value_shape_support=value_shape,
                local_association=accepted,
                reason=reason,
            )
        )
    accepted_items = [item for item in items if item.accepted]
    estimated_input = sum(
        estimate_tokens(
            escalation_text_prompt(
                artifacts.target_by_id[item.target_row_id],
                artifacts.intent_by_id[item.target_row_id],
                bundles[item.target_row_id],
            )
        )
        for item in accepted_items
    )
    estimated_output = len(accepted_items) * MAX_OUTPUT_TOKENS
    estimated_cost = estimate_cost_usd(model_name, estimated_input, estimated_output)
    if estimated_cost > cost_ceiling_usd:
        errors.append(f"estimated cost {estimated_cost} exceeds ceiling {cost_ceiling_usd}")
    report = TextPreflightReport(
        target_ids=[target.target_row_id for target in targets],
        field_names={target.target_row_id: target.expected_field for target in targets},
        approved_source_pages={
            target.target_row_id: sorted(
                {
                    f"{item.source_id}:{item.page_number}"
                    for item in evidence_by_target[target.target_row_id]
                }
            )
            for target in targets
        },
        approved_span_ids={item.target_row_id: item.approved_span_ids for item in items},
        expected_calls=len(accepted_items),
        configured_model=model_name,
        estimated_input_tokens=estimated_input,
        estimated_output_tokens=estimated_output,
        estimated_cost_ceiling_usd=cost_ceiling_usd,
        estimated_cost_usd=estimated_cost,
        cache_hits=sum(result.cache_hits for result in parse_results),
        cache_misses=sum(result.cache_misses for result in parse_results),
        parser_worker_invocations_expected=sum(
            result.worker_invocation_count for result in parse_results
        ),
        safety_gate_passed=not [
            error for error in errors if not error.startswith("preflight evidence rejected:")
        ],
        safety_gate_errors=errors,
    )
    return items, report


def escalation_text_prompt(
    target: TargetSpecification,
    intent: TargetIntent,
    bundle: EvidenceBundleRecord,
) -> str:
    if target.expected_field == "floor_construction_capacity":
        rules = (
            "Valid floor capacity evidence contains a numeric capacity/load/duty value, "
            "its unit, and a local association with the floor construction, slab, warehouse floor, "
            "or concrete floor. Reject material strength values such as N/mm2, unrelated "
            "dimensions, external-area values, and generic safety text. If multiple load cases "
            "are present, return multiple_candidates and preserve the grounded candidates."
        )
        expected_shape = "decimal_measurement"
    elif target.expected_field == "dock_count":
        rules = (
            "Valid dock count evidence contains an integer locally associated with dock doors, "
            "loading docks, dock levellers, or the exact dock component. Reject fire exit door "
            "counts, unrelated drawing/grid numbers, and component-only headings. If the text "
            "is ambiguous between dock doors and dock levellers, return multiple_candidates."
        )
        expected_shape = "integer_count"
    else:
        rules = "Use only evidence that supports the requested attribute."
        expected_shape = intent.value_shape_family
    return (
        "Target requirement:\n"
        f"- target_id: {target.target_row_id}\n"
        f"- field_name: {target.expected_field}\n"
        f"- definition: {target.requirement_text}\n"
        f"- component_or_subject: {intent.primary_component}\n"
        f"- requested_attribute: {intent.requested_attribute}\n"
        f"- expected_value_shape: {expected_shape}\n"
        f"- declared_datatype: {target.expected_data_type}\n"
        f"- declared_unit: {target.unit}\n\n"
        "Attribute-specific evidence rules:\n"
        f"{rules}\n\n"
        "Return exactly one JSON object matching the supplied schema. Use only listed "
        "supporting_span_ids. value_bearing_text must be an exact substring of a selected "
        "span for extracted or multiple_candidates. Do not guess.\n\n"
        "Canonical evidence spans:\n"
        f"{bundle.combined_text}"
    )


def escalation_text_request_payload(
    *,
    model_name: str,
    target: TargetSpecification,
    intent: TargetIntent,
    bundle: EvidenceBundleRecord,
    max_output_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract only the requested attribute from approved canonical text spans. "
                    "Return one strict JSON object. Abstain unless the span explicitly links "
                    "the requested component, attribute and value."
                ),
            },
            {"role": "user", "content": escalation_text_prompt(target, intent, bundle)},
        ],
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "escalation_text_attribute_reextract_v1",
                "strict": True,
                "schema": structured_response_schema(),
            },
        },
    }


def calibrate_text_reextract_value(
    extraction: ExtractionResult,
    target: TargetSpecification,
    intent: TargetIntent,
    bundle: EvidenceBundleRecord,
) -> ExtractionResult:
    if extraction.status == "multiple_candidates":
        return extraction.model_copy(
            update={
                "proposed_value_shape": _stage_value_shape(target.expected_field),
                "display_value": _display_value(extraction.raw_model_value),
            }
        )
    assignment = stage_value_shape_assignment(target, intent)
    calibrated = calibrate_extraction_value(
        extraction=extraction,
        target=target,
        assignment=assignment,
        bundle=bundle,
    )
    if target.expected_field == "floor_construction_capacity" and calibrated.status == "extracted":
        measurement = _best_floor_measurement(str(calibrated.evidence_value or ""), bundle)
        if measurement is not None:
            value, unit, load_type = measurement
            calibrated = calibrated.model_copy(
                update={
                    "evidence_value": f"{value:g} {unit}",
                    "normalized_value": value,
                    "display_value": f"{value:g} {unit} ({load_type})",
                    "unit": unit,
                }
            )
    return calibrated


def validate_text_canonical_evidence(
    extraction: ExtractionResult,
    intent: TargetIntent,
    bundle: EvidenceBundleRecord,
) -> CanonicalEvidenceResult:
    if extraction.status not in {"extracted", "multiple_candidates"}:
        return CanonicalEvidenceResult(
            target_row_id=extraction.target_row_id,
            status="review_required" if extraction.status == "multiple_candidates" else "invalid",
            selected_span_ids=extraction.supporting_span_ids,
            value_bearing_phrase_present=False,
            local_numeric_association=False,
            unit_association_valid=False,
            source_page_node_valid=False,
            issues=[f"Extraction status is {extraction.status}."],
        )
    span_by_id = {span.span_id: span for span in bundle.evidence_spans}
    selected = [span_by_id.get(span_id) for span_id in extraction.supporting_span_ids]
    present = [span for span in selected if span is not None]
    issues: list[str] = []
    if len(present) != len(extraction.supporting_span_ids):
        issues.append("Unknown selected span ID.")
    quote = str(extraction.value_bearing_quote or extraction.evidence_value or "").strip()
    span_text = "\n".join(span.text for span in present)
    quote_present = bool(quote and quote in span_text)
    if not quote_present:
        issues.append("Value-bearing phrase is not present in selected canonical spans.")
    if intent.field_name == "floor_construction_capacity":
        supported = any(floor_capacity_support(span.text)[0] == "supported" for span in present)
        unit_ok = any(_floor_capacity_candidates(span.text) for span in present)
    elif intent.field_name == "dock_count":
        supported = any(dock_count_support(span.text)[0] == "supported" for span in present)
        unit_ok = True
    else:
        supported = bool(present)
        unit_ok = True
    if not supported:
        issues.append("Selected spans do not locally support the requested attribute.")
    if not unit_ok:
        issues.append("Required numeric unit is not locally associated with the value.")
    provenance_valid = all(
        span.source_id and span.page_number and span.hierarchy_node_id for span in present
    )
    if not provenance_valid:
        issues.append("Source/page/node provenance is invalid.")
    status: Literal["valid", "invalid", "review_required"] = (
        "review_required" if extraction.status == "multiple_candidates" and not issues else "valid"
    )
    if issues:
        status = "invalid"
    return CanonicalEvidenceResult(
        target_row_id=extraction.target_row_id,
        status=status,
        selected_span_ids=extraction.supporting_span_ids,
        value_bearing_phrase_present=quote_present,
        local_numeric_association=supported,
        unit_association_valid=unit_ok,
        source_page_node_valid=provenance_valid,
        issues=issues,
    )


def stage_value_shape_assignment(
    target: TargetSpecification, intent: TargetIntent
) -> ValueShapeAssignment:
    base = infer_value_shape_assignment(target)
    shape = _stage_value_shape(target.expected_field)
    if shape == base.value_shape_family:
        return base
    return base.model_copy(
        update={
            "value_shape_family": shape,
            "inference_basis": [
                *base.inference_basis,
                "escalation-text-v1-target-specific-attribute-requirement",
            ],
            "ambiguity_or_mismatch_warning": (
                "Stage contract requires measurement/count shape while dictionary "
                "metadata is preserved."
            ),
        }
    )


def build_prior_attempt_comparison(
    artifacts: TextReextractArtifacts,
    extractions: list[ExtractionResult],
    validation_results: list[ValidationResult],
    evidence: list[ApprovedEvidenceRecord],
) -> list[PriorAttemptComparisonItem]:
    validation_by_id = {item.target_row_id: item for item in validation_results}
    evidence_by_id: dict[str, list[str]] = {}
    for item in evidence:
        if item.support_status == "supported":
            evidence_by_id.setdefault(item.target_row_id, []).append(
                f"{item.source_file}:{item.page_number}:{item.span_id}"
            )
    comparisons: list[PriorAttemptComparisonItem] = []
    for extraction in extractions:
        target_id = extraction.target_row_id
        field_name = artifacts.target_by_id[target_id].expected_field
        new_status = validation_by_id[target_id].status
        batch = artifacts.prior_extractions["batch_v1"].get(target_id)
        r1 = artifacts.prior_extractions["reextract_v1"].get(target_id)
        r2 = artifacts.prior_extractions["reextract_v2"].get(target_id)
        outcome = _comparison_outcome(extraction, new_status)
        comparisons.append(
            PriorAttemptComparisonItem(
                target_row_id=target_id,
                field_name=field_name,
                batch_v1_status=cast(str | None, batch.get("status") if batch else None),
                batch_v1_value=batch.get("normalized_value") if batch else None,
                reextract_v1_status=cast(str | None, r1.get("status") if r1 else None),
                reextract_v1_value=r1.get("normalized_value") if r1 else None,
                reextract_v2_status=cast(str | None, r2.get("status") if r2 else None),
                reextract_v2_value=r2.get("normalized_value") if r2 else None,
                escalation_decision=str(artifacts.decisions_by_id[target_id]["final_status"]),
                escalation_evidence=sorted(evidence_by_id.get(target_id, [])),
                new_status=extraction.status,
                new_value=extraction.normalized_value,
                new_review_status=new_status,
                outcome=outcome,
            )
        )
    return comparisons


def build_text_reextract_telemetry(
    *,
    target_count: int,
    eligible_count: int,
    parse_results: list[CachedBatchParseResult],
    extraction_results: list[ExtractionResult],
    canonical_evidence_results: list[CanonicalEvidenceResult],
    validation_results: list[ValidationResult],
    comparison: list[PriorAttemptComparisonItem],
    wall_time_ms: float,
    model_name: str,
) -> TextReextractTelemetry:
    return TextReextractTelemetry(
        frozen_target_count=target_count,
        eligible_target_count=eligible_count,
        preflight_rejected_count=target_count - eligible_count,
        model=model_name,
        model_call_count=sum(
            1
            for item in extraction_results
            if item.model_provider == "openai" and item.model_usage.input_tokens > 0
        ),
        input_tokens=sum(item.model_usage.input_tokens for item in extraction_results),
        output_tokens=sum(item.model_usage.output_tokens for item in extraction_results),
        estimated_cost_usd=sum(item.model_usage.estimated_cost_usd for item in extraction_results),
        wall_time_ms=wall_time_ms,
        extraction_status_counts=dict(Counter(item.status for item in extraction_results)),
        evidence_validation_counts=dict(
            Counter(item.status for item in canonical_evidence_results)
        ),
        review_status_counts=dict(Counter(item.status for item in validation_results)),
        comparison_outcome_counts=dict(Counter(item.outcome for item in comparison)),
        cache_hits=sum(result.cache_hits for result in parse_results),
        cache_misses=sum(result.cache_misses for result in parse_results),
        parser_worker_invocations=sum(result.worker_invocation_count for result in parse_results),
        active_child_count_after_cleanup=len(multiprocessing.active_children()),
    )


def write_text_reextract_artifacts(result: TextReextractRunResult, output_dir: Path) -> None:
    _atomic_write_json(
        output_dir / "frozen_targets.json",
        [item.model_dump(mode="json") for item in result.frozen_targets],
    )
    _atomic_write_json(
        output_dir / "preflight_report.json", result.preflight_report.model_dump(mode="json")
    )
    _atomic_write_json(
        output_dir / "approved_evidence.json",
        [item.model_dump(mode="json") for item in result.approved_evidence],
    )
    _atomic_write_json(
        output_dir / "extraction_requests.json",
        [item.model_dump(mode="json") for item in result.extraction_requests],
    )
    _atomic_write_json(
        output_dir / "raw_response_diagnostics.json",
        [item.model_dump(mode="json") for item in result.raw_response_diagnostics],
    )
    _atomic_write_json(
        output_dir / "parsed_response_results.json",
        [item.model_dump(mode="json") if item else None for item in result.parsed_response_results],
    )
    _atomic_write_json(
        output_dir / "extraction_results.json",
        [item.model_dump(mode="json") for item in result.extraction_results],
    )
    _atomic_write_json(
        output_dir / "canonical_evidence_results.json",
        [item.model_dump(mode="json") for item in result.canonical_evidence_results],
    )
    _atomic_write_json(
        output_dir / "normalized_values.json",
        [item.model_dump(mode="json") for item in result.normalized_values],
    )
    _atomic_write_json(
        output_dir / "evidence_validation.json",
        [item.model_dump(mode="json") for item in result.evidence_validation],
    )
    _atomic_write_json(
        output_dir / "shape_validation.json",
        [item.model_dump(mode="json") for item in result.shape_validation],
    )
    _atomic_write_json(
        output_dir / "schema_compatibility_results.json",
        [item.model_dump(mode="json") for item in result.schema_compatibility_results],
    )
    _atomic_write_json(
        output_dir / "validation_results.json",
        [item.model_dump(mode="json") for item in result.validation_results],
    )
    _atomic_write_json(
        output_dir / "prior_attempt_comparison.json",
        [item.model_dump(mode="json") for item in result.prior_attempt_comparison],
    )
    _atomic_write_json(output_dir / "telemetry.json", result.telemetry.model_dump(mode="json"))
    _write_comparison_md(result, output_dir / "prior_attempt_comparison.md")
    _write_review_csv(result, output_dir / "reextract_review.csv")
    _write_summary_md(result, output_dir / "reextract_summary.md")


def write_preflight_artifacts(
    output_dir: Path,
    targets: list[TargetSpecification],
    preflight: TextPreflightReport,
    evidence: list[ApprovedEvidenceRecord],
) -> None:
    _atomic_write_json(
        output_dir / "frozen_targets.json", [item.model_dump(mode="json") for item in targets]
    )
    _atomic_write_json(output_dir / "preflight_report.json", preflight.model_dump(mode="json"))
    _atomic_write_json(
        output_dir / "approved_evidence.json", [item.model_dump(mode="json") for item in evidence]
    )


def _write_comparison_md(result: TextReextractRunResult, path: Path) -> None:
    lines = ["# Prior Attempt Comparison", ""]
    for item in result.prior_attempt_comparison:
        lines.append(
            f"- {item.field_name}: Batch V1={item.batch_v1_status}, "
            f"Reextract V1={item.reextract_v1_status}, Reextract V2={item.reextract_v2_status}, "
            f"new={item.new_status}/{item.new_review_status}, outcome={item.outcome}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_md(result: TextReextractRunResult, path: Path) -> None:
    lines = [
        "# Evidence Escalation Text Re-extraction V1",
        "",
        f"Frozen targets: {len(result.frozen_targets)}",
        f"Expected calls: {result.preflight_report.expected_calls}",
        f"Model calls: {result.telemetry.model_call_count}",
        f"Estimated cost USD: {result.telemetry.estimated_cost_usd:.6f}",
        "Cache hits/misses/workers: "
        f"{result.telemetry.cache_hits}/{result.telemetry.cache_misses}/"
        f"{result.telemetry.parser_worker_invocations}",
        "",
        "## Results",
    ]
    validation_by_id = {item.target_row_id: item for item in result.validation_results}
    for extraction in result.extraction_results:
        lines.append(
            f"- {extraction.target_row_id}: {extraction.status}, "
            f"value={extraction.display_value or extraction.normalized_value}, "
            f"review={validation_by_id[extraction.target_row_id].status}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_review_csv(result: TextReextractRunResult, path: Path) -> None:
    target_by_id = {item.target_row_id: item for item in result.frozen_targets}
    evidence_by_id = {item.target_row_id: item for item in result.canonical_evidence_results}
    validation_by_id = {item.target_row_id: item for item in result.validation_results}
    comparison_by_id = {item.target_row_id: item for item in result.prior_attempt_comparison}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target ID",
                "domain",
                "sub-domain",
                "field name",
                "component",
                "requested attribute",
                "value shape",
                "raw model value",
                "evidence value",
                "normalized value",
                "display value",
                "extraction status",
                "confidence",
                "source file",
                "page",
                "hierarchy node",
                "canonical evidence",
                "attribute-support result",
                "evidence validation",
                "shape validation",
                "dictionary compatibility",
                "overall review status",
                "Batch V1 value",
                "Batch V1 review status",
                "comparison outcome",
                "caveat",
            ],
        )
        writer.writeheader()
        shape_by_id = {item.target_row_id: item for item in result.shape_validation}
        schema_by_id = {item.target_row_id: item for item in result.schema_compatibility_results}
        for extraction in result.extraction_results:
            target = target_by_id[extraction.target_row_id]
            evidence = evidence_by_id[extraction.target_row_id]
            comparison = comparison_by_id[extraction.target_row_id]
            writer.writerow(
                {
                    "target ID": extraction.target_row_id,
                    "domain": target.metadata.get("domain") or target.sub_domain,
                    "sub-domain": target.sub_domain,
                    "field name": target.expected_field,
                    "component": _component_terms_for_field(target.expected_field)[0],
                    "requested attribute": _stage_requested_attribute(target.expected_field),
                    "value shape": _stage_value_shape(target.expected_field),
                    "raw model value": _display_value(extraction.raw_model_value),
                    "evidence value": _display_value(extraction.evidence_value),
                    "normalized value": _display_value(extraction.normalized_value),
                    "display value": extraction.display_value,
                    "extraction status": extraction.status,
                    "confidence": extraction.confidence,
                    "source file": extraction.source_file,
                    "page": extraction.page_number,
                    "hierarchy node": extraction.hierarchy_node_id,
                    "canonical evidence": extraction.supporting_evidence_excerpt,
                    "attribute-support result": evidence.status,
                    "evidence validation": evidence.status,
                    "shape validation": shape_by_id[extraction.target_row_id].status,
                    "dictionary compatibility": schema_by_id[
                        extraction.target_row_id
                    ].compatibility,
                    "overall review status": validation_by_id[extraction.target_row_id].status,
                    "Batch V1 value": _display_value(comparison.batch_v1_value),
                    "Batch V1 review status": artifacts_status(comparison.batch_v1_status),
                    "comparison outcome": comparison.outcome,
                    "caveat": extraction.ambiguity_or_caveat,
                }
            )


def _parse_source_page(value: str) -> tuple[str, int]:
    source_id, raw_page = value.rsplit(":", 1)
    return source_id, int(raw_page)


def _contiguous_ranges(pages: list[int]) -> list[tuple[int, int]]:
    if not pages:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = pages[0]
    for page in pages[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append((start, previous))
        start = previous = page
    ranges.append((start, previous))
    return ranges


def _candidate_blocks(text: str) -> list[tuple[int, int, str]]:
    lines = [
        (match.start(), match.end(), match.group(0).strip())
        for match in re.finditer(r"(?m)^.+$", text)
    ]
    blocks: list[tuple[int, int, str]] = []
    for index, (_start, _end, line) in enumerate(lines):
        if not line:
            continue
        context = lines[max(0, index - 1) : min(len(lines), index + 2)]
        block_start = context[0][0]
        block_end = context[-1][1]
        block = " | ".join(item[2] for item in context if item[2])
        blocks.append((block_start, block_end, block))
    return blocks


def _floor_capacity_candidates(text: str) -> list[str]:
    pattern = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:kN/m(?:2|²)|kn/m(?:2|²)|kn\s*/\s*m(?:2|²)|kpa|kg/m(?:2|²)|psf)\b",
        flags=re.IGNORECASE,
    )
    return [match.group(0) for match in pattern.finditer(text)]


def _dock_count_candidates(text: str) -> list[str]:
    patterns = [
        (
            r"(?<![.\d])\b\d+\s*(?:no\.?|number|quantity)\s+"
            r"(?:dock\s+doors?|loading\s+docks?|dock\s+levellers?|loading\s+doors?)\b"
        ),
        (
            r"\b(?:dock\s+doors?|loading\s+docks?|dock\s+levellers?|loading\s+doors?)"
            r"\D{0,30}(?<![.\d])\b\d+\b(?!\.\d)"
        ),
    ]
    values: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate = match.group(0)
            if re.search(r"\b\d+\.\d+(?:\.\d+)?\b", candidate):
                continue
            values.append(candidate)
    return sorted(set(values))


def _best_floor_measurement(
    evidence_value: str,
    bundle: EvidenceBundleRecord,
) -> tuple[float, str, str] | None:
    text = evidence_value or "\n".join(span.text for span in bundle.evidence_spans)
    match = re.search(
        r"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>kN/m(?:2|²)|kn/m(?:2|²)|kn\s*/\s*m(?:2|²)|kpa|kg/m(?:2|²)|psf)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    unit = match.group("unit").replace(" ", "")
    unit = unit.replace("kn", "kN").replace("m2", "m²")
    return float(match.group("value")), unit, "floor loading capacity"


def _stable_digest(parts: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\u001f".join(parts).encode("utf-8")).hexdigest()[:16]


def _stage_value_shape(field_name: str) -> ValueShapeFamily:
    if field_name == "floor_construction_capacity":
        return "decimal_measurement"
    if field_name == "dock_count":
        return "integer_count"
    return "descriptive_text"


def _stage_requested_attribute(field_name: str) -> str:
    if field_name == "floor_construction_capacity":
        return "floor loading capacity"
    if field_name == "dock_count":
        return "dock count"
    return "attribute"


def _component_terms_for_field(field_name: str) -> list[str]:
    if field_name == "floor_construction_capacity":
        return ["floor construction", "floor", "slab", "concrete slab", "warehouse floor"]
    if field_name == "dock_count":
        return ["dock", "dock door", "loading dock", "dock leveller", "loading door"]
    return [field_name.replace("_", " ")]


def _attribute_terms_for_field(field_name: str) -> list[str]:
    if field_name == "floor_construction_capacity":
        return ["capacity", "load", "loading", "duty", "udl"]
    if field_name == "dock_count":
        return ["count", "number", "quantity", "no"]
    return ["value"]


def _value_indicators_for_field(field_name: str) -> list[str]:
    if field_name == "floor_construction_capacity":
        return ["kN/m²", "kN/m2", "kPa", "kg/m²", "psf"]
    if field_name == "dock_count":
        return ["integer", "no", "number"]
    return []


def _comparison_outcome(extraction: ExtractionResult, review_status: str) -> TextReextractOutcome:
    if extraction.status == "multiple_candidates":
        return "multiple_candidates"
    if review_status in {"valid", "valid_after_normalization", "valid_with_dictionary_caveat"}:
        return "corrected"
    if extraction.status == "insufficient_evidence":
        return "still_insufficient"
    if extraction.status in {"model_error", "invalid_format"}:
        return "regressed"
    return "unchanged"


def _default_text_client(settings: Settings, model_name: str) -> TextStructuredExtractionClient:
    if not settings.hosted_llm_enabled or settings.openai_api_key is None:
        return AbstainingTextClient(model_name="abstaining-text-reextract")
    return EscalationTextOpenAIClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model_name=model_name,
    )


class AbstainingTextClient:
    provider = "mock"

    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name

    def extract_structured(
        self,
        *,
        target: TargetSpecification,
        intent: TargetIntent,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> tuple[ExtractionResult, ParsedResponseDiagnostic, StructuredExtractionEnvelope | None]:
        _ = (intent, bundle, max_output_tokens)
        return (
            _insufficient_evidence_result(
                target,
                self.provider,
                self.model_name,
                "Hosted model calls are disabled.",
            ),
            ParsedResponseDiagnostic(
                target_row_id=target.target_row_id, final_status="mock_abstain"
            ),
            None,
        )


def _load_extractions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        str(item["target_row_id"]): item for item in cast(list[dict[str, Any]], _read_json(path))
    }


def _load_validations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        str(item["target_row_id"]): item for item in cast(list[dict[str, Any]], _read_json(path))
    }


def artifacts_status(status: str | None) -> str | None:
    return status


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "ApprovedEvidenceRecord",
    "CanonicalEvidenceResult",
    "DEFAULT_ESCALATION_TEXT_REEXTRACT_OUTPUT_DIR",
    "EscalationTextOpenAIClient",
    "TextPreflightReport",
    "TextReextractRunResult",
    "build_approved_evidence",
    "dock_count_support",
    "floor_capacity_support",
    "run_escalation_text_reextract_v1",
    "support_for_target",
]
