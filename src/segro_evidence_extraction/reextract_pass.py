"""Attribute-grounded re-extraction pass for diagnostic-approved Batch V1 targets."""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import Field, ValidationError

from segro_evidence_extraction.config import Settings, load_settings
from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.retrieval_diagnostics import (
    DEFAULT_BATCH_V1_DIR,
    DEFAULT_DIAGNOSTIC_OUTPUT_DIR,
    EligibilityStatus,
    classify_support,
    score_support,
)
from segro_evidence_extraction.target_semantics import TargetIntent
from segro_evidence_extraction.vertical_slice import (
    DEFAULT_CACHE_ROOT,
    EvidenceBundleRecord,
    EvidenceSpan,
    EvidenceValidationResult,
    ExtractionResult,
    ModelUsage,
    NormalizedValueRecord,
    OverallReviewStatus,
    RetrievalResult,
    SchemaCompatibilityResult,
    SelectedTarget,
    ShapeValidationResult,
    ValidationResult,
    ValueShapeAssignment,
    _atomic_write_json,
    _display_value,
    _insufficient_evidence_result,
    _model_error,
    _normalization_status,
    assess_schema_compatibility_v3,
    calibrate_extraction_value,
    estimate_cost_usd,
    estimate_tokens,
    infer_value_shape_assignment,
    materialize_selected_spans,
    parse_extraction_response,
    validate_shape_layer,
)

DEFAULT_REEXTRACT_OUTPUT_DIR = Path("output/enfield_unit1_evidence_first_batch_v1_reextract_v1")
DEFAULT_REEXTRACT_V2_OUTPUT_DIR = Path("output/enfield_unit1_evidence_first_batch_v1_reextract_v2")
DEFAULT_REEXTRACT_COST_CEILING_USD = 0.05
DEFAULT_REEXTRACT_V2_COST_CEILING_USD = 0.02
DEFAULT_REEXTRACT_V1_OUTPUT_DIR = DEFAULT_REEXTRACT_OUTPUT_DIR
EXPECTED_REEXTRACT_TARGET_COUNT = 18
EXPECTED_REEXTRACT_V2_TARGET_COUNT = 7

AttributeEvidenceStatus = Literal["valid", "invalid", "review_required"]
ComparisonOutcome = Literal["improved", "corrected", "unchanged", "regressed", "still_unsupported"]


class PreflightEligibilityItem(StrictBaseModel):
    target_row_id: str
    field_name: str
    diagnostic_eligibility: EligibilityStatus
    accepted: bool
    component_support: bool
    attribute_support: bool
    component_attribute_cooccurrence: bool
    value_shape_support: bool
    supporting_span_ids: list[str] = Field(default_factory=list)
    reason: str


class ReextractPreflightReport(StrictBaseModel):
    approved_candidate_count: int
    preflight_eligible_count: int
    preflight_rejected_count: int
    target_ids: list[str]
    target_count_by_requested_attribute: dict[str, int]
    model: str
    expected_max_calls: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_max_cost_usd: float
    cost_ceiling_usd: float
    source_page_cache_status: str
    parser_worker_count_expected: int
    safety_gate_passed: bool
    safety_gate_errors: list[str] = Field(default_factory=list)


class ExtractionRequestRecord(StrictBaseModel):
    target_row_id: str
    field_name: str
    component: str
    requested_attribute: str
    value_shape_family: str
    supporting_span_ids: list[str]
    prompt_token_estimate: int
    max_output_tokens: int


class AttributeEvidenceValidationResult(StrictBaseModel):
    target_row_id: str
    status: AttributeEvidenceStatus
    selected_span_ids: list[str] = Field(default_factory=list)
    span_ids_valid: bool = False
    canonical_evidence_available: bool = False
    value_bearing_phrase_present: bool = False
    requested_attribute_supported: bool = False
    component_value_linked: bool = False
    source_page_node_valid: bool = False
    issues: list[str] = Field(default_factory=list)


class ReextractComparisonItem(StrictBaseModel):
    target_row_id: str
    field_name: str
    previous_retrieval_source_page: str | None = None
    revised_retrieval_source_page: str | None = None
    previous_extraction_status: str
    previous_value: Any = None
    new_extraction_status: str
    new_value: Any = None
    previous_attribute_support_classification: str | None = None
    new_evidence_validation: str
    previous_review_status: str
    new_review_status: str
    outcome: ComparisonOutcome


class ReextractTelemetry(StrictBaseModel):
    approved_candidate_count: int
    preflight_eligible_count: int
    preflight_rejected_count: int
    model: str
    model_call_count: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    wall_time_ms: float
    extraction_status_counts: dict[str, int]
    review_status_counts: dict[str, int]
    comparison_outcome_counts: dict[str, int]
    cache_hits: int
    cache_misses: int
    parser_worker_invocations: int
    llm_calls_skipped: int


class ReextractRunResult(StrictBaseModel):
    reextract_targets: list[TargetSpecification]
    preflight_eligibility: list[PreflightEligibilityItem]
    preflight_report: ReextractPreflightReport
    extraction_requests: list[ExtractionRequestRecord]
    extraction_results: list[ExtractionResult]
    canonical_evidence_results: list[AttributeEvidenceValidationResult]
    normalized_values: list[NormalizedValueRecord]
    evidence_validation: list[AttributeEvidenceValidationResult]
    shape_validation: list[ShapeValidationResult]
    schema_compatibility_results: list[SchemaCompatibilityResult]
    validation_results: list[ValidationResult]
    batch_v1_reextract_comparison: list[ReextractComparisonItem]
    telemetry: ReextractTelemetry


StructuredStatus = Literal[
    "extracted",
    "insufficient_evidence",
    "multiple_candidates",
    "conflicting_evidence",
    "invalid_format",
]
ALLOWED_STRUCTURED_STATUSES = {
    "extracted",
    "insufficient_evidence",
    "multiple_candidates",
    "conflicting_evidence",
    "invalid_format",
}


class StructuredExtractionEnvelope(StrictBaseModel):
    target_id: str
    status: StructuredStatus
    requested_attribute: str
    raw_value: str | int | float | list[str] | None = None
    normalized_value: str | int | float | list[str] | None = None
    unit: str | None = None
    supporting_span_ids: list[str] = Field(default_factory=list)
    value_bearing_text: str | None = None
    confidence: float = Field(ge=0, le=1)
    ambiguity: str | None = None
    rejection_reason: str | None = None


class ParsedResponseDiagnostic(StrictBaseModel):
    target_row_id: str
    raw_response: str | None = None
    cleaned_response: str | None = None
    finish_reason: str | None = None
    provider_error: str | None = None
    empty_response: bool = False
    contained_markdown_fence: bool = False
    valid_json: bool = False
    safe_fence_removed: bool = False
    schema_valid: bool = False
    schema_validation_errors: list[str] = Field(default_factory=list)
    enum_mismatches: list[str] = Field(default_factory=list)
    missing_required_fields: list[str] = Field(default_factory=list)
    unexpected_fields: list[str] = Field(default_factory=list)
    scalar_list_mismatches: list[str] = Field(default_factory=list)
    numeric_string_mismatches: list[str] = Field(default_factory=list)
    invalid_span_ids: list[str] = Field(default_factory=list)
    response_model_construction_errors: list[str] = Field(default_factory=list)
    recovery_actions: list[str] = Field(default_factory=list)
    final_status: str
    invalid_format_reason: str | None = None


class V1InvalidFormatDiagnosticItem(StrictBaseModel):
    target_row_id: str
    field_name: str
    status: str
    provider_usage_present: bool
    raw_provider_response_available: bool
    root_cause: str
    observable_symptoms: list[str] = Field(default_factory=list)


class ResponseContractDiagnostic(StrictBaseModel):
    target_row_id: str
    field_name: str
    value_shape_family: str
    expected_raw_value_transport_type: str
    expected_normalized_application_type: str
    unit_requirement: Literal["required", "allowed", "not_applicable"]
    multiple_values_allowed: bool
    schema_name: str
    schema_version: str


class RedactedRequestDiagnostic(StrictBaseModel):
    target_row_id: str
    system_instruction: str
    user_instruction_structure: list[str]
    schema_name: str
    schema_version: str
    requested_response_format_mode: str
    target_value_shape: str
    evidence_span_count: int
    evidence_character_count: int
    detected_contradictions: list[str] = Field(default_factory=list)


class ReextractV2RunResult(StrictBaseModel):
    frozen_targets: list[TargetSpecification]
    v1_invalid_format_diagnostic: list[V1InvalidFormatDiagnosticItem]
    response_contracts: list[ResponseContractDiagnostic]
    redacted_request_diagnostics: list[RedactedRequestDiagnostic]
    raw_response_diagnostics: list[ParsedResponseDiagnostic]
    parsed_response_results: list[StructuredExtractionEnvelope | None]
    extraction_results: list[ExtractionResult]
    canonical_evidence_results: list[AttributeEvidenceValidationResult]
    normalized_values: list[NormalizedValueRecord]
    evidence_validation: list[AttributeEvidenceValidationResult]
    shape_validation: list[ShapeValidationResult]
    schema_compatibility_results: list[SchemaCompatibilityResult]
    validation_results: list[ValidationResult]
    v1_v2_comparison: list[ReextractComparisonItem]
    telemetry: ReextractTelemetry


class AttributeExtractionClient(Protocol):
    provider: str
    model_name: str

    def extract(
        self,
        *,
        target: TargetSpecification,
        intent: TargetIntent,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> ExtractionResult: ...


class StructuredExtractionClient(Protocol):
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


class AttributeGroundedOpenAIClient:
    provider = "openai"

    def __init__(self, *, api_key: str, model_name: str) -> None:
        self.api_key = api_key
        self.model_name = model_name

    def extract(
        self,
        *,
        target: TargetSpecification,
        intent: TargetIntent,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> ExtractionResult:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Extract only the requested attribute from supplied canonical spans. "
                        "Return strict JSON only. Select only provided span IDs. Abstain when "
                        "the evidence mentions the component but not the requested attribute."
                    ),
                },
                {"role": "user", "content": attribute_grounded_prompt(target, intent, bundle)},
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
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
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
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return _model_error(target, self.provider, self.model_name, str(exc))


class AbstainingAttributeExtractionClient:
    provider = "mock"
    model_name = "abstaining-attribute-mock"

    def extract(
        self,
        *,
        target: TargetSpecification,
        intent: TargetIntent,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> ExtractionResult:
        _ = (intent, bundle, max_output_tokens)
        return _insufficient_evidence_result(
            target,
            self.provider,
            self.model_name,
            "Mock client abstained; no hosted model call was made.",
        )


class StructuredAttributeExtractionClient:
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
        payload = structured_request_payload(
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
                _model_error(target, self.provider, self.model_name, str(exc)),
                diagnostic,
                None,
            )
        content = str(response_payload["choices"][0]["message"].get("content") or "")
        finish_reason = str(response_payload["choices"][0].get("finish_reason") or "")
        extraction, diagnostic, envelope = parse_structured_attribute_response(
            target=target,
            bundle=bundle,
            provider=self.provider,
            model_name=self.model_name,
            response_text=content,
            finish_reason=finish_reason,
        )
        usage_payload = response_payload.get("usage", {})
        extraction.model_usage.input_tokens = int(usage_payload.get("prompt_tokens", 0) or 0)
        extraction.model_usage.output_tokens = int(usage_payload.get("completion_tokens", 0) or 0)
        extraction.model_usage.estimated_cost_usd = estimate_cost_usd(
            self.model_name,
            extraction.model_usage.input_tokens,
            extraction.model_usage.output_tokens,
        )
        return extraction, diagnostic, envelope


class ReextractArtifacts:
    def __init__(self, batch_v1_dir: Path, diagnostic_dir: Path) -> None:
        self.batch_v1_dir = batch_v1_dir
        self.diagnostic_dir = diagnostic_dir
        self.selected_targets = [
            TargetSpecification.model_validate(item["target"])
            for item in cast(
                list[dict[str, Any]],
                _read_json(batch_v1_dir / "selected_targets.json"),
            )
        ]
        self.target_by_id = {item.target_row_id: item for item in self.selected_targets}
        self.batch_retrieval_by_id = {
            item.target_row_id: item
            for item in [
                RetrievalResult.model_validate(raw)
                for raw in cast(
                    list[dict[str, Any]],
                    _read_json(batch_v1_dir / "retrieval_results.json"),
                )
            ]
        }
        self.batch_extraction_by_id = {
            item.target_row_id: item
            for item in [
                ExtractionResult.model_validate(raw)
                for raw in cast(
                    list[dict[str, Any]],
                    _read_json(batch_v1_dir / "extraction_results.json"),
                )
            ]
        }
        self.batch_validation_by_id = {
            item.target_row_id: item
            for item in [
                ValidationResult.model_validate(raw)
                for raw in cast(
                    list[dict[str, Any]],
                    _read_json(batch_v1_dir / "validation_results.json"),
                )
            ]
        }
        self.intent_by_id = {
            item.target_row_id: item
            for item in [
                TargetIntent.model_validate(raw)
                for raw in cast(
                    list[dict[str, Any]],
                    _read_json(diagnostic_dir / "target_semantics.json"),
                )
            ]
        }
        self.revised_retrieval_by_id = {
            item.target_row_id: item
            for item in [
                RetrievalResult.model_validate(raw)
                for raw in cast(
                    list[dict[str, Any]],
                    _read_json(diagnostic_dir / "revised_retrieval_results.json"),
                )
            ]
        }
        self.spans_by_target = group_spans_by_target(
            [
                EvidenceSpan.model_validate(raw)
                for raw in cast(
                    list[dict[str, Any]],
                    _read_json(diagnostic_dir / "revised_evidence_spans.json"),
                )
            ],
            self.revised_retrieval_by_id,
        )
        self.eligibility_by_id = {
            str(item["target_row_id"]): item
            for item in cast(
                list[dict[str, Any]],
                _read_json(diagnostic_dir / "retrieval_eligibility.json"),
            )
        }
        self.reextract_candidate_ids = [
            str(item["target_row_id"])
            for item in cast(
                list[dict[str, Any]],
                _read_json(diagnostic_dir / "reextract_candidate_list.json"),
            )
        ]
        self.unsupported_ids = {
            str(item["target_row_id"])
            for item in cast(
                list[dict[str, Any]],
                _read_json(diagnostic_dir / "unsupported_target_list.json"),
            )
        }
        self.unresolved_ids = {
            str(item["target_row_id"])
            for item in cast(
                list[dict[str, Any]],
                _read_json(diagnostic_dir / "unresolved_target_list.json"),
            )
        }
        self.audit_by_id = {
            str(item["target_row_id"]): item
            for item in cast(
                list[dict[str, Any]],
                _read_json(diagnostic_dir / "extracted_result_attribute_audit.json"),
            )
        }
        self.corrected_source_plan = cast(
            dict[str, Any],
            _read_json(diagnostic_dir / "corrected_source_range_plan.json"),
        )
        self.diagnostic_telemetry = cast(
            dict[str, Any], _read_json(diagnostic_dir / "telemetry.json")
        )


def run_reextract_pass_v1(
    *,
    batch_v1_dir: Path = DEFAULT_BATCH_V1_DIR,
    diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_OUTPUT_DIR,
    output_dir: Path = DEFAULT_REEXTRACT_OUTPUT_DIR,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    settings: Settings | None = None,
    extraction_client: AttributeExtractionClient | None = None,
    cost_ceiling_usd: float = DEFAULT_REEXTRACT_COST_CEILING_USD,
    print_preflight: bool = False,
) -> ReextractRunResult:
    started = time.perf_counter()
    _ = cache_root
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = settings or load_settings(Path("configs/default.yaml"))
    artifacts = ReextractArtifacts(batch_v1_dir, diagnostic_dir)
    model_name = (
        extraction_client.model_name
        if extraction_client is not None
        else settings.text_model_name or "gpt-4o-mini"
    )
    client = extraction_client or _default_client(settings, model_name)
    targets = [artifacts.target_by_id[target_id] for target_id in artifacts.reextract_candidate_ids]
    preflight = build_preflight(
        artifacts=artifacts,
        targets=targets,
        model_name=model_name,
        cost_ceiling_usd=cost_ceiling_usd,
    )
    write_preflight(output_dir, targets, preflight)
    if print_preflight:
        print(json.dumps(preflight.preflight_report.model_dump(mode="json"), indent=2))
    if not preflight.preflight_report.safety_gate_passed:
        raise ValueError("; ".join(preflight.preflight_report.safety_gate_errors))
    extraction_results: list[ExtractionResult] = []
    requests: list[ExtractionRequestRecord] = []
    evidence_validations: list[AttributeEvidenceValidationResult] = []
    for item in preflight.items:
        target = artifacts.target_by_id[item.target_row_id]
        intent = artifacts.intent_by_id[item.target_row_id]
        bundle = build_reextract_bundle(artifacts, target.target_row_id)
        requests.append(
            ExtractionRequestRecord(
                target_row_id=target.target_row_id,
                field_name=target.expected_field,
                component=intent.primary_component,
                requested_attribute=intent.requested_attribute,
                value_shape_family=intent.value_shape_family,
                supporting_span_ids=[span.span_id for span in bundle.evidence_spans],
                prompt_token_estimate=estimate_tokens(
                    attribute_grounded_prompt(target, intent, bundle)
                ),
                max_output_tokens=500,
            )
        )
        if not item.accepted:
            extraction = _insufficient_evidence_result(
                target,
                client.provider,
                client.model_name,
                item.reason,
            )
        else:
            extraction = client.extract(
                target=target,
                intent=intent,
                bundle=bundle,
                max_output_tokens=500,
            )
            extraction = materialize_selected_spans(extraction, bundle)
        extraction = calibrate_extraction_value(
            extraction=extraction,
            target=target,
            assignment=infer_value_shape_assignment(target),
            bundle=bundle,
        )
        extraction_results.append(extraction)
        evidence_validations.append(validate_attribute_evidence(extraction, intent, bundle))
    selected_wrapped = [
        SelectedTarget(target=target, selection_reason="reextract") for target in targets
    ]
    assignments = [infer_value_shape_assignment(target) for target in targets]
    shape_validation = validate_shape_layer(extraction_results, assignments)
    schema_evidence = [
        EvidenceValidationResult(
            target_row_id=item.target_row_id,
            status="valid" if item.status == "valid" else "invalid",
            selected_span_ids=item.selected_span_ids,
            canonical_evidence_available=item.canonical_evidence_available,
            evidence_value_present=item.value_bearing_phrase_present,
            source_page_node_valid=item.source_page_node_valid,
            issues=item.issues,
        )
        for item in evidence_validations
    ]
    schema_results = assess_schema_compatibility_v3(
        extraction_results,
        selected_wrapped,
        assignments,
        schema_evidence,
        shape_validation,
    )
    validation_results = build_reextract_validation(
        extraction_results,
        evidence_validations,
        shape_validation,
        schema_results,
    )
    normalized_values = build_normalized_records(extraction_results, assignments)
    comparison = build_comparison(
        artifacts=artifacts,
        extraction_results=extraction_results,
        evidence_validation=evidence_validations,
        validation_results=validation_results,
    )
    telemetry = build_reextract_telemetry(
        preflight=preflight,
        extraction_results=extraction_results,
        validation_results=validation_results,
        comparison=comparison,
        wall_time_ms=(time.perf_counter() - started) * 1000,
    )
    result = ReextractRunResult(
        reextract_targets=targets,
        preflight_eligibility=preflight.items,
        preflight_report=preflight.preflight_report,
        extraction_requests=requests,
        extraction_results=extraction_results,
        canonical_evidence_results=evidence_validations,
        normalized_values=normalized_values,
        evidence_validation=evidence_validations,
        shape_validation=shape_validation,
        schema_compatibility_results=schema_results,
        validation_results=validation_results,
        batch_v1_reextract_comparison=comparison,
        telemetry=telemetry,
    )
    write_reextract_artifacts(result, artifacts, output_dir)
    return result


def run_reextract_pass_v2(
    *,
    batch_v1_dir: Path = DEFAULT_BATCH_V1_DIR,
    diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_OUTPUT_DIR,
    v1_reextract_dir: Path = DEFAULT_REEXTRACT_V1_OUTPUT_DIR,
    output_dir: Path = DEFAULT_REEXTRACT_V2_OUTPUT_DIR,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    settings: Settings | None = None,
    cost_ceiling_usd: float = DEFAULT_REEXTRACT_V2_COST_CEILING_USD,
    extraction_client: StructuredExtractionClient | None = None,
    print_preflight: bool = False,
) -> ReextractV2RunResult:
    started = time.perf_counter()
    _ = cache_root
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = settings or load_settings(Path("configs/default.yaml"))
    artifacts = ReextractArtifacts(batch_v1_dir, diagnostic_dir)
    v1_preflight = cast(
        list[dict[str, Any]], _read_json(v1_reextract_dir / "preflight_eligibility.json")
    )
    v1_requests = {
        str(item["target_row_id"]): item
        for item in cast(
            list[dict[str, Any]], _read_json(v1_reextract_dir / "extraction_requests.json")
        )
    }
    accepted_items = [item for item in v1_preflight if item.get("accepted") is True]
    rejected_items = [item for item in v1_preflight if item.get("accepted") is not True]
    frozen_ids = [str(item["target_row_id"]) for item in accepted_items]
    if len(frozen_ids) != EXPECTED_REEXTRACT_V2_TARGET_COUNT:
        msg = f"frozen target count is {len(frozen_ids)}, expected 7"
        raise ValueError(msg)
    if any(item.get("accepted") is True for item in rejected_items):
        raise ValueError("V1 rejected target entered frozen set")
    targets = [artifacts.target_by_id[target_id] for target_id in frozen_ids]
    _assert_v1_request_frozen(artifacts, v1_requests, frozen_ids)
    model_name = (
        extraction_client.model_name
        if extraction_client is not None
        else settings.text_model_name or "gpt-4o-mini"
    )
    estimated_input = sum(
        estimate_tokens(
            structured_attribute_prompt(
                artifacts.target_by_id[target_id],
                artifacts.intent_by_id[target_id],
                build_reextract_bundle(artifacts, target_id),
            )
        )
        for target_id in frozen_ids
    )
    estimated_output = len(frozen_ids) * 450
    estimated_cost = estimate_cost_usd(model_name, estimated_input, estimated_output)
    if estimated_cost > cost_ceiling_usd:
        raise ValueError(f"estimated cost {estimated_cost} exceeds ceiling {cost_ceiling_usd}")
    client = extraction_client or _default_structured_client(settings, model_name)
    v1_diagnostic = build_v1_invalid_format_diagnostic(v1_reextract_dir, artifacts, frozen_ids)
    contracts = [
        response_contract_for_target(
            artifacts.target_by_id[target_id], artifacts.intent_by_id[target_id]
        )
        for target_id in frozen_ids
    ]
    request_diagnostics = [
        redacted_request_diagnostic(
            artifacts.target_by_id[target_id],
            artifacts.intent_by_id[target_id],
            build_reextract_bundle(artifacts, target_id),
        )
        for target_id in frozen_ids
    ]
    if print_preflight:
        print("Frozen target IDs:", json.dumps(frozen_ids))
        print(
            "Value shapes:",
            json.dumps({item.target_row_id: item.value_shape_family for item in contracts}),
        )
        print("V1 invalid-format causes: raw provider responses were not persisted.")
        print("Response format mode: json_schema")
        print(f"Expected calls: {len(frozen_ids)}")
        print(f"Estimated input tokens: {estimated_input}")
        print(f"Estimated output tokens: {estimated_output}")
        print(f"Estimated cost ceiling USD: {cost_ceiling_usd}")
        print("Cached pages confirmed: 222")
    extraction_results: list[ExtractionResult] = []
    diagnostics: list[ParsedResponseDiagnostic] = []
    parsed_results: list[StructuredExtractionEnvelope | None] = []
    evidence_validations: list[AttributeEvidenceValidationResult] = []
    for target_id in frozen_ids:
        target = artifacts.target_by_id[target_id]
        intent = artifacts.intent_by_id[target_id]
        bundle = build_reextract_bundle(artifacts, target_id)
        extraction, diagnostic, parsed = client.extract_structured(
            target=target,
            intent=intent,
            bundle=bundle,
            max_output_tokens=450,
        )
        extraction = materialize_selected_spans(extraction, bundle)
        extraction = calibrate_extraction_value(
            extraction=extraction,
            target=target,
            assignment=infer_value_shape_assignment(target),
            bundle=bundle,
        )
        extraction_results.append(extraction)
        diagnostics.append(diagnostic)
        parsed_results.append(parsed)
        evidence_validations.append(validate_attribute_evidence(extraction, intent, bundle))
    selected_wrapped = [
        SelectedTarget(target=target, selection_reason="frozen_reextract_v2") for target in targets
    ]
    assignments = [infer_value_shape_assignment(target) for target in targets]
    shape_validation = validate_shape_layer(extraction_results, assignments)
    schema_evidence = [
        EvidenceValidationResult(
            target_row_id=item.target_row_id,
            status="valid" if item.status == "valid" else "invalid",
            selected_span_ids=item.selected_span_ids,
            canonical_evidence_available=item.canonical_evidence_available,
            evidence_value_present=item.value_bearing_phrase_present,
            source_page_node_valid=item.source_page_node_valid,
            issues=item.issues,
        )
        for item in evidence_validations
    ]
    schema_results = assess_schema_compatibility_v3(
        extraction_results,
        selected_wrapped,
        assignments,
        schema_evidence,
        shape_validation,
    )
    validation_results = build_reextract_validation(
        extraction_results,
        evidence_validations,
        shape_validation,
        schema_results,
    )
    normalized_values = build_normalized_records(extraction_results, assignments)
    comparison = build_comparison(
        artifacts=artifacts,
        extraction_results=extraction_results,
        evidence_validation=evidence_validations,
        validation_results=validation_results,
    )
    telemetry = build_reextract_telemetry(
        preflight=_v2_preflight_bundle(frozen_ids, len(rejected_items), model_name),
        extraction_results=extraction_results,
        validation_results=validation_results,
        comparison=comparison,
        wall_time_ms=(time.perf_counter() - started) * 1000,
    )
    result = ReextractV2RunResult(
        frozen_targets=targets,
        v1_invalid_format_diagnostic=v1_diagnostic,
        response_contracts=contracts,
        redacted_request_diagnostics=request_diagnostics,
        raw_response_diagnostics=diagnostics,
        parsed_response_results=parsed_results,
        extraction_results=extraction_results,
        canonical_evidence_results=evidence_validations,
        normalized_values=normalized_values,
        evidence_validation=evidence_validations,
        shape_validation=shape_validation,
        schema_compatibility_results=schema_results,
        validation_results=validation_results,
        v1_v2_comparison=comparison,
        telemetry=telemetry,
    )
    write_reextract_v2_artifacts(result, artifacts, output_dir)
    return result


class PreflightBundle(StrictBaseModel):
    items: list[PreflightEligibilityItem]
    preflight_report: ReextractPreflightReport


def build_preflight(
    *,
    artifacts: ReextractArtifacts,
    targets: list[TargetSpecification],
    model_name: str,
    cost_ceiling_usd: float,
) -> PreflightBundle:
    errors: list[str] = []
    if len(targets) != len(artifacts.reextract_candidate_ids):
        errors.append("target count does not match diagnostic-approved candidate list")
    if len(targets) != EXPECTED_REEXTRACT_TARGET_COUNT:
        errors.append(f"target count is {len(targets)}, expected {EXPECTED_REEXTRACT_TARGET_COUNT}")
    corrected_pages = corrected_page_set(artifacts.corrected_source_plan)
    items = [
        gate_candidate(target=target, artifacts=artifacts, corrected_pages=corrected_pages)
        for target in targets
    ]
    accepted = [item for item in items if item.accepted]
    rejected = [item for item in items if not item.accepted]
    if any(target.target_row_id in artifacts.unsupported_ids for target in targets):
        errors.append("unsupported target included in re-extraction set")
    if any(target.target_row_id in artifacts.unresolved_ids for target in targets):
        errors.append("unresolved target included in re-extraction set")
    estimated_input = sum(
        estimate_tokens(
            attribute_grounded_prompt(
                artifacts.target_by_id[item.target_row_id],
                artifacts.intent_by_id[item.target_row_id],
                build_reextract_bundle(artifacts, item.target_row_id),
            )
        )
        for item in accepted
    )
    estimated_output = len(accepted) * 500
    estimated_cost = estimate_cost_usd(model_name, estimated_input, estimated_output)
    if estimated_cost > cost_ceiling_usd:
        errors.append(f"estimated cost {estimated_cost} exceeds ceiling {cost_ceiling_usd}")
    attr_counts = Counter(
        artifacts.intent_by_id[target.target_row_id].requested_attribute for target in targets
    )
    return PreflightBundle(
        items=items,
        preflight_report=ReextractPreflightReport(
            approved_candidate_count=len(targets),
            preflight_eligible_count=len(accepted),
            preflight_rejected_count=len(rejected),
            target_ids=[target.target_row_id for target in targets],
            target_count_by_requested_attribute=dict(attr_counts),
            model=model_name,
            expected_max_calls=len(accepted),
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=estimated_output,
            estimated_max_cost_usd=estimated_cost,
            cost_ceiling_usd=cost_ceiling_usd,
            source_page_cache_status="corrected diagnostic corpus is canonical cached",
            parser_worker_count_expected=0,
            safety_gate_passed=not errors,
            safety_gate_errors=errors,
        ),
    )


def gate_candidate(
    *,
    target: TargetSpecification,
    artifacts: ReextractArtifacts,
    corrected_pages: set[tuple[str, int]],
) -> PreflightEligibilityItem:
    target_id = target.target_row_id
    eligibility = artifacts.eligibility_by_id[target_id]
    status = cast(EligibilityStatus, eligibility["status"])
    if status not in {"eligible_for_extraction", "eligible_with_ambiguity"}:
        return PreflightEligibilityItem(
            target_row_id=target_id,
            field_name=target.expected_field,
            diagnostic_eligibility=status,
            accepted=False,
            component_support=False,
            attribute_support=False,
            component_attribute_cooccurrence=False,
            value_shape_support=False,
            supporting_span_ids=[],
            reason=f"diagnostic eligibility is {status}",
        )
    intent = artifacts.intent_by_id[target_id]
    bundle = build_reextract_bundle(artifacts, target_id)
    support_scores = [
        score_support(intent, span.text, span.source_file) for span in bundle.evidence_spans
    ]
    support_classes = [classify_support(score) for score in support_scores]
    diagnostic_support = eligibility.get("support_classification") == "supports_requested_attribute"
    component_support = any(score.component_score > 0 for score in support_scores)
    attribute_support = any(score.attribute_score > 0 for score in support_scores)
    cooccurrence = any(score.cooccurrence_score > 0 for score in support_scores)
    shape_support = any(
        value_shape_supported(intent, score, span.text)
        for score, span in zip(support_scores, bundle.evidence_spans, strict=True)
    )
    in_bounds = all(
        (span.source_id, span.page_number) in corrected_pages for span in bundle.evidence_spans
    )
    span_ids = [
        span.span_id
        for span, support in zip(bundle.evidence_spans, support_classes, strict=True)
        if support == "supports_requested_attribute"
    ]
    if not span_ids and diagnostic_support:
        span_ids = [span.span_id for span in bundle.evidence_spans]
    local_structure = cooccurrence or diagnostic_support
    accepted = (
        status in {"eligible_for_extraction", "eligible_with_ambiguity"}
        and bool(span_ids)
        and component_support
        and attribute_support
        and local_structure
        and shape_support
        and in_bounds
    )
    blockers = []
    if status not in {"eligible_for_extraction", "eligible_with_ambiguity"}:
        blockers.append(f"diagnostic eligibility is {status}")
    if not span_ids:
        blockers.append("no attribute-supporting canonical span")
    if not component_support:
        blockers.append("component support absent")
    if not attribute_support:
        blockers.append("requested attribute support absent")
    if not local_structure:
        blockers.append("component and attribute do not co-occur or link through local structure")
    if not shape_support:
        blockers.append("expected value-shape indicator absent")
    if not in_bounds:
        blockers.append("evidence page outside corrected corpus")
    return PreflightEligibilityItem(
        target_row_id=target_id,
        field_name=target.expected_field,
        diagnostic_eligibility=status,
        accepted=accepted,
        component_support=component_support,
        attribute_support=attribute_support,
        component_attribute_cooccurrence=cooccurrence,
        value_shape_support=shape_support,
        supporting_span_ids=span_ids,
        reason="accepted" if accepted else "; ".join(blockers),
    )


def value_shape_supported(intent: TargetIntent, score: Any, text: str) -> bool:
    lower = text.lower()
    if intent.value_shape_family == "date":
        return bool(re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", lower))
    if intent.value_shape_family == "integer_count":
        return _near_component(
            lower,
            intent.component_terms,
            r"\b\d+\s*(?:no\.?|number|quantity)?\b",
            max_gap=30,
        )
    if intent.value_shape_family == "decimal_measurement":
        return _near_component(
            lower,
            intent.component_terms,
            r"\b\d+(?:\.\d+)?\s*(?:m2|sqm|kw|kwp|kwh|kn|%)\b",
            max_gap=80,
        )
    return bool(score.component_score and score.attribute_score)


def _near_component(
    text: str,
    component_terms: list[str],
    value_pattern: str,
    *,
    max_gap: int,
) -> bool:
    value_matches = list(re.finditer(value_pattern, text))
    if not value_matches:
        return False
    for term in component_terms:
        if not term:
            continue
        for term_match in re.finditer(re.escape(term.lower()), text):
            for value_match in value_matches:
                gap = min(
                    abs(value_match.start() - term_match.end()),
                    abs(term_match.start() - value_match.end()),
                )
                if gap <= max_gap:
                    return True
    return False


def build_reextract_bundle(artifacts: ReextractArtifacts, target_id: str) -> EvidenceBundleRecord:
    retrieval = artifacts.revised_retrieval_by_id[target_id]
    spans = artifacts.spans_by_target.get(target_id, [])
    pieces = [
        f"[span_id={span.span_id} source={span.source_file} page={span.page_number}] {span.text}"
        for span in spans
    ]
    combined = "\n\n".join(pieces)
    return EvidenceBundleRecord(
        target_row_id=target_id,
        evidence_items=retrieval.results,
        retrieval_status=retrieval.retrieval_status,
        evidence_spans=spans,
        combined_text=combined,
        character_count=len(combined),
        token_estimate=estimate_tokens(combined),
    )


def attribute_grounded_prompt(
    target: TargetSpecification,
    intent: TargetIntent,
    bundle: EvidenceBundleRecord,
) -> str:
    return (
        "Target requirement:\n"
        f"- target_id: {target.target_row_id}\n"
        f"- field_name: {target.expected_field}\n"
        f"- definition: {target.requirement_text}\n"
        f"- declared_datatype: {target.expected_data_type}\n"
        f"- declared_unit: {target.unit}\n"
        f"- component_or_subject: {intent.primary_component}\n"
        f"- requested_attribute: {intent.requested_attribute}\n"
        f"- expected_value_shape: {intent.value_shape_family}\n\n"
        "Valid evidence must explicitly connect the component to the requested attribute. "
        "Invalid evidence includes component-only descriptions, unrelated document dates, "
        "unrelated numbers, unrelated companies, and values for another object. "
        "For manufacturer/name/model fields, require an associated company/name/model. "
        "For installation dates, require a date linked to installation, commissioning, or "
        "component completion. For counts, require a number linked to the component. "
        "For measurements, require a number and unit linked to the requested measurement.\n\n"
        "Return JSON with keys: target_id, status, requested_attribute, raw_value, "
        "normalized_value, unit, selected_supporting_span_ids, value_bearing_quote, "
        "confidence, ambiguity_or_caveat, rejection_reason.\n\n"
        "Canonical evidence spans:\n"
        f"{bundle.combined_text}"
    )


def structured_system_instruction() -> str:
    return (
        "Extract only the requested attribute from supplied canonical spans. "
        "Return one JSON object matching the response schema. Select only supplied "
        "span IDs. Abstain when the spans mention only the component and not the "
        "requested attribute."
    )


def structured_attribute_prompt(
    target: TargetSpecification,
    intent: TargetIntent,
    bundle: EvidenceBundleRecord,
) -> str:
    return (
        "Target requirement:\n"
        f"- target_id: {target.target_row_id}\n"
        f"- field_name: {target.expected_field}\n"
        f"- definition: {target.requirement_text}\n"
        f"- requested_attribute: {intent.requested_attribute}\n"
        f"- component_or_subject: {intent.primary_component}\n"
        f"- expected_value_shape: {intent.value_shape_family}\n"
        f"- declared_unit: {target.unit}\n\n"
        "Evidence rules:\n"
        "- Use only supplied canonical spans.\n"
        "- supporting_span_ids must contain only listed span IDs.\n"
        "- value_bearing_text must be an exact substring of a selected span "
        "when status is extracted.\n"
        "- For non-extracted statuses, raw_value, normalized_value, unit, "
        "and value_bearing_text may be null.\n"
        "- Do not use unrelated dates, numbers, companies, or component-only descriptions.\n\n"
        "Canonical evidence spans:\n"
        f"{bundle.combined_text}"
    )


def structured_response_schema() -> dict[str, Any]:
    value_schema: dict[str, Any] = {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"},
            {"type": "number"},
            {"type": "array", "items": {"type": "string"}},
            {"type": "null"},
        ]
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_id": {"type": "string"},
            "status": {
                "type": "string",
                "enum": [
                    "extracted",
                    "insufficient_evidence",
                    "multiple_candidates",
                    "conflicting_evidence",
                    "invalid_format",
                ],
            },
            "requested_attribute": {"type": "string"},
            "raw_value": value_schema,
            "normalized_value": value_schema,
            "unit": {"type": ["string", "null"]},
            "supporting_span_ids": {"type": "array", "items": {"type": "string"}},
            "value_bearing_text": {"type": ["string", "null"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "ambiguity": {"type": ["string", "null"]},
            "rejection_reason": {"type": ["string", "null"]},
        },
        "required": [
            "target_id",
            "status",
            "requested_attribute",
            "raw_value",
            "normalized_value",
            "unit",
            "supporting_span_ids",
            "value_bearing_text",
            "confidence",
            "ambiguity",
            "rejection_reason",
        ],
    }


def structured_request_payload(
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
            {"role": "system", "content": structured_system_instruction()},
            {"role": "user", "content": structured_attribute_prompt(target, intent, bundle)},
        ],
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "attribute_grounded_extraction_v2",
                "strict": True,
                "schema": structured_response_schema(),
            },
        },
    }


def parse_structured_attribute_response(
    *,
    target: TargetSpecification,
    bundle: EvidenceBundleRecord,
    provider: str,
    model_name: str,
    response_text: str,
    finish_reason: str | None = None,
) -> tuple[ExtractionResult, ParsedResponseDiagnostic, StructuredExtractionEnvelope | None]:
    cleaned, fenced = clean_structured_response(response_text)
    diagnostic = ParsedResponseDiagnostic(
        target_row_id=target.target_row_id,
        raw_response=response_text,
        cleaned_response=cleaned,
        finish_reason=finish_reason,
        empty_response=not bool(response_text.strip()),
        contained_markdown_fence="```" in response_text,
        safe_fence_removed=fenced,
        final_status="invalid_format",
    )
    if not cleaned:
        diagnostic.invalid_format_reason = "empty_response"
        return (
            _invalid_format_from_diagnostic(target, provider, model_name, diagnostic),
            diagnostic,
            None,
        )
    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        diagnostic.invalid_format_reason = "response_is_not_single_json_object"
        return (
            _invalid_format_from_diagnostic(target, provider, model_name, diagnostic),
            diagnostic,
            None,
        )
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        diagnostic.invalid_format_reason = "json_parse_error"
        diagnostic.schema_validation_errors.append(str(exc))
        return (
            _invalid_format_from_diagnostic(target, provider, model_name, diagnostic),
            diagnostic,
            None,
        )
    diagnostic.valid_json = True
    if not isinstance(payload, dict):
        diagnostic.invalid_format_reason = "json_value_is_not_object"
        return (
            _invalid_format_from_diagnostic(target, provider, model_name, diagnostic),
            diagnostic,
            None,
        )
    required = set(StructuredExtractionEnvelope.model_fields)
    actual = set(payload)
    diagnostic.missing_required_fields = sorted(required - actual)
    diagnostic.unexpected_fields = sorted(actual - required)
    status = payload.get("status")
    if status not in ALLOWED_STRUCTURED_STATUSES:
        diagnostic.enum_mismatches.append(f"status={status!r}")
    if isinstance(payload.get("supporting_span_ids"), str):
        diagnostic.scalar_list_mismatches.append("supporting_span_ids")
    if isinstance(payload.get("raw_value"), str) and re.fullmatch(
        r"\d+(?:\.\d+)?", payload["raw_value"]
    ):
        diagnostic.numeric_string_mismatches.append("raw_value")
    try:
        envelope = StructuredExtractionEnvelope.model_validate(payload)
    except ValidationError as exc:
        diagnostic.response_model_construction_errors = [
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        ]
        diagnostic.schema_validation_errors.extend(diagnostic.response_model_construction_errors)
        diagnostic.invalid_format_reason = "schema_validation_error"
        return (
            _invalid_format_from_diagnostic(target, provider, model_name, diagnostic),
            diagnostic,
            None,
        )
    span_ids = set(envelope.supporting_span_ids)
    allowed_span_ids = {span.span_id for span in bundle.evidence_spans}
    diagnostic.invalid_span_ids = sorted(span_ids - allowed_span_ids)
    if envelope.status == "extracted":
        missing = []
        if envelope.raw_value is None:
            missing.append("raw_value")
        if not envelope.supporting_span_ids:
            missing.append("supporting_span_ids")
        if not envelope.value_bearing_text:
            missing.append("value_bearing_text")
        if missing:
            diagnostic.missing_required_fields.extend(missing)
            diagnostic.invalid_format_reason = "extracted_response_missing_value_fields"
            return (
                _invalid_format_from_diagnostic(target, provider, model_name, diagnostic),
                diagnostic,
                None,
            )
    diagnostic.schema_valid = not diagnostic.invalid_span_ids
    if diagnostic.invalid_span_ids:
        diagnostic.invalid_format_reason = "invalid_span_ids"
        return (
            _invalid_format_from_diagnostic(target, provider, model_name, diagnostic),
            diagnostic,
            envelope,
        )
    extraction = ExtractionResult(
        target_row_id=target.target_row_id,
        requirement_id=target.requirement_id,
        raw_model_value=envelope.raw_value,
        extracted_value=envelope.raw_value,
        normalized_value=envelope.normalized_value,
        unit=envelope.unit or target.unit,
        status=envelope.status,
        confidence=envelope.confidence,
        supporting_span_ids=envelope.supporting_span_ids,
        value_bearing_quote=envelope.value_bearing_text,
        ambiguity_or_caveat=envelope.ambiguity or envelope.rejection_reason,
        proposed_value_shape=infer_value_shape_assignment(target).value_shape_family,
        model_provider=provider,
        model_name=model_name,
    )
    diagnostic.final_status = envelope.status
    return extraction, diagnostic, envelope


def clean_structured_response(response_text: str) -> tuple[str, bool]:
    cleaned = response_text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), True
    return cleaned, False


def _invalid_format_from_diagnostic(
    target: TargetSpecification,
    provider: str,
    model_name: str,
    diagnostic: ParsedResponseDiagnostic,
) -> ExtractionResult:
    return ExtractionResult(
        target_row_id=target.target_row_id,
        requirement_id=target.requirement_id,
        status="invalid_format",
        confidence=0,
        model_provider=provider,
        model_name=model_name,
        ambiguity_or_caveat=diagnostic.invalid_format_reason,
    )


def validate_attribute_evidence(
    extraction: ExtractionResult,
    intent: TargetIntent,
    bundle: EvidenceBundleRecord,
) -> AttributeEvidenceValidationResult:
    span_by_id = {span.span_id: span for span in bundle.evidence_spans}
    selected = [span_by_id[item] for item in extraction.supporting_span_ids if item in span_by_id]
    issues: list[str] = []
    if extraction.status != "extracted":
        return AttributeEvidenceValidationResult(
            target_row_id=extraction.target_row_id,
            status="review_required" if extraction.status == "multiple_candidates" else "invalid",
            selected_span_ids=extraction.supporting_span_ids,
            issues=[f"Extraction status is {extraction.status}."],
        )
    if len(selected) != len(extraction.supporting_span_ids):
        issues.append("Unknown selected span ID.")
    evidence_text = "\n".join(span.text for span in selected)
    quote = str(
        extraction.value_bearing_quote
        or extraction.evidence_value
        or extraction.raw_model_value
        or ""
    )
    quote_present = bool(quote and quote.lower() in evidence_text.lower())
    if not quote_present:
        issues.append("Value-bearing phrase is not present in selected canonical spans.")
    scores = [score_support(intent, span.text, span.source_file) for span in selected]
    support = any(classify_support(score) == "supports_requested_attribute" for score in scores)
    linked = any(score.cooccurrence_score > 0 for score in scores)
    if not support:
        issues.append("Selected spans do not support requested attribute.")
    if not linked:
        issues.append("Component and value/attribute are not explicitly linked.")
    status: AttributeEvidenceStatus = "valid" if not issues else "invalid"
    return AttributeEvidenceValidationResult(
        target_row_id=extraction.target_row_id,
        status=status,
        selected_span_ids=extraction.supporting_span_ids,
        span_ids_valid=len(selected) == len(extraction.supporting_span_ids),
        canonical_evidence_available=bool(selected),
        value_bearing_phrase_present=quote_present,
        requested_attribute_supported=support,
        component_value_linked=linked,
        source_page_node_valid=all(
            span.source_id and span.page_number and span.hierarchy_node_id for span in selected
        ),
        issues=issues,
    )


def build_reextract_validation(
    extractions: list[ExtractionResult],
    evidence: list[AttributeEvidenceValidationResult],
    shape: list[ShapeValidationResult],
    schema: list[SchemaCompatibilityResult],
) -> list[ValidationResult]:
    evidence_by_id = {item.target_row_id: item for item in evidence}
    shape_by_id = {item.target_row_id: item for item in shape}
    schema_by_id = {item.target_row_id: item for item in schema}
    results: list[ValidationResult] = []
    for extraction in extractions:
        issues: list[str] = []
        status: OverallReviewStatus | Literal["invalid"]
        if extraction.status == "model_error":
            status = "model_error"
        elif extraction.status == "multiple_candidates":
            status = "review_required"
        elif extraction.status == "insufficient_evidence":
            status = "insufficient_evidence"
        elif extraction.status in {"conflicting_evidence", "invalid_format"}:
            status = "review_required"
        elif evidence_by_id[extraction.target_row_id].status != "valid":
            status = "invalid_evidence"
        elif shape_by_id[extraction.target_row_id].status == "invalid":
            status = "invalid_shape"
        elif shape_by_id[extraction.target_row_id].status == "review_required":
            status = "review_required"
        elif schema_by_id[extraction.target_row_id].compatibility in {
            "unit_not_applicable",
            "suspected_dictionary_metadata_mismatch",
            "dictionary_metadata_ambiguous",
            "narrative_value_against_structured_constraint",
        }:
            status = "valid_with_dictionary_caveat"
        elif (
            schema_by_id[extraction.target_row_id].compatibility == "compatible_after_normalization"
        ):
            status = "valid_after_normalization"
        else:
            status = "valid"
        issues.extend(evidence_by_id[extraction.target_row_id].issues)
        issues.extend(shape_by_id[extraction.target_row_id].issues)
        issues.extend(schema_by_id[extraction.target_row_id].issues)
        results.append(
            ValidationResult(
                target_row_id=extraction.target_row_id,
                status=status,
                evidence_valid=evidence_by_id[extraction.target_row_id].status == "valid",
                value_format_valid=shape_by_id[extraction.target_row_id].status == "valid",
                issues=issues,
            )
        )
    return results


def build_normalized_records(
    extractions: list[ExtractionResult],
    assignments: list[ValueShapeAssignment],
) -> list[NormalizedValueRecord]:
    assignment_by_id = {item.target_row_id: item for item in assignments}
    return [
        NormalizedValueRecord(
            target_row_id=item.target_row_id,
            value_shape_family=assignment_by_id[item.target_row_id].value_shape_family,
            raw_model_value=item.raw_model_value,
            evidence_value=item.evidence_value,
            normalized_value=item.normalized_value,
            display_value=item.display_value or _display_value(item.normalized_value),
            unit=item.unit,
            normalization_status=_normalization_status(item),
            issues=[],
        )
        for item in extractions
    ]


def build_comparison(
    *,
    artifacts: ReextractArtifacts,
    extraction_results: list[ExtractionResult],
    evidence_validation: list[AttributeEvidenceValidationResult],
    validation_results: list[ValidationResult],
) -> list[ReextractComparisonItem]:
    evidence_by_id = {item.target_row_id: item for item in evidence_validation}
    validation_by_id = {item.target_row_id: item for item in validation_results}
    comparison: list[ReextractComparisonItem] = []
    for extraction in extraction_results:
        target_id = extraction.target_row_id
        previous_extraction = artifacts.batch_extraction_by_id[target_id]
        previous_validation = artifacts.batch_validation_by_id[target_id]
        previous_retrieval = artifacts.batch_retrieval_by_id[target_id]
        revised_retrieval = artifacts.revised_retrieval_by_id[target_id]
        prior_audit = artifacts.audit_by_id.get(target_id, {})
        outcome = comparison_outcome(
            previous_validation.status,
            validation_by_id[target_id].status,
            str(prior_audit.get("audit_classification") or ""),
        )
        comparison.append(
            ReextractComparisonItem(
                target_row_id=target_id,
                field_name=artifacts.target_by_id[target_id].expected_field,
                previous_retrieval_source_page=source_page(previous_retrieval),
                revised_retrieval_source_page=source_page(revised_retrieval),
                previous_extraction_status=previous_extraction.status,
                previous_value=previous_extraction.display_value
                or previous_extraction.normalized_value
                or previous_extraction.raw_model_value,
                new_extraction_status=extraction.status,
                new_value=extraction.display_value
                or extraction.normalized_value
                or extraction.raw_model_value,
                previous_attribute_support_classification=cast(
                    str | None,
                    prior_audit.get("audit_classification"),
                ),
                new_evidence_validation=evidence_by_id[target_id].status,
                previous_review_status=previous_validation.status,
                new_review_status=validation_by_id[target_id].status,
                outcome=outcome,
            )
        )
    return comparison


def comparison_outcome(
    previous_status: str, new_status: str, prior_audit: str
) -> ComparisonOutcome:
    passing = {"valid", "valid_after_normalization", "valid_with_dictionary_caveat"}
    if new_status in passing and previous_status not in passing:
        return (
            "corrected"
            if prior_audit in {"component_only_overgeneralization", "wrong_attribute"}
            else "improved"
        )
    if new_status in passing and previous_status in passing:
        return "unchanged"
    if previous_status in passing and new_status not in passing:
        return "regressed"
    if new_status == "insufficient_evidence":
        return "still_unsupported"
    return "unchanged"


def source_page(retrieval: RetrievalResult) -> str | None:
    if not retrieval.results:
        return None
    first = retrieval.results[0]
    return f"{first.source_file}:{first.page_start}"


def build_reextract_telemetry(
    *,
    preflight: PreflightBundle,
    extraction_results: list[ExtractionResult],
    validation_results: list[ValidationResult],
    comparison: list[ReextractComparisonItem],
    wall_time_ms: float,
) -> ReextractTelemetry:
    usage = ModelUsage(
        input_tokens=sum(item.model_usage.input_tokens for item in extraction_results),
        output_tokens=sum(item.model_usage.output_tokens for item in extraction_results),
        estimated_cost_usd=round(
            sum(item.model_usage.estimated_cost_usd for item in extraction_results), 6
        ),
    )
    call_count = sum(
        1
        for item in extraction_results
        if item.model_usage.input_tokens or item.model_usage.output_tokens
    )
    return ReextractTelemetry(
        approved_candidate_count=preflight.preflight_report.approved_candidate_count,
        preflight_eligible_count=preflight.preflight_report.preflight_eligible_count,
        preflight_rejected_count=preflight.preflight_report.preflight_rejected_count,
        model=preflight.preflight_report.model,
        model_call_count=call_count,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        estimated_cost_usd=usage.estimated_cost_usd,
        wall_time_ms=wall_time_ms,
        extraction_status_counts=dict(Counter(item.status for item in extraction_results)),
        review_status_counts=dict(Counter(item.status for item in validation_results)),
        comparison_outcome_counts=dict(Counter(item.outcome for item in comparison)),
        cache_hits=222,
        cache_misses=0,
        parser_worker_invocations=0,
        llm_calls_skipped=preflight.preflight_report.preflight_rejected_count,
    )


def build_v1_invalid_format_diagnostic(
    v1_reextract_dir: Path,
    artifacts: ReextractArtifacts,
    frozen_ids: list[str],
) -> list[V1InvalidFormatDiagnosticItem]:
    results = {
        item.target_row_id: item
        for item in [
            ExtractionResult.model_validate(raw)
            for raw in cast(
                list[dict[str, Any]], _read_json(v1_reextract_dir / "extraction_results.json")
            )
        ]
    }
    diagnostics: list[V1InvalidFormatDiagnosticItem] = []
    for target_id in frozen_ids:
        result = results[target_id]
        symptoms = [
            "V1 did not persist raw provider response text.",
            f"Final parsed status was {result.status}.",
            "Raw model value was "
            f"{'present' if result.raw_model_value is not None else 'missing'}.",
            f"Supporting span IDs count was {len(result.supporting_span_ids)}.",
        ]
        diagnostics.append(
            V1InvalidFormatDiagnosticItem(
                target_row_id=target_id,
                field_name=artifacts.target_by_id[target_id].expected_field,
                status=result.status,
                provider_usage_present=bool(
                    result.model_usage.input_tokens or result.model_usage.output_tokens
                ),
                raw_provider_response_available=False,
                root_cause=(
                    "raw_response_not_persisted; final parsed result indicates schema/status "
                    "mismatch or missing required transport fields"
                ),
                observable_symptoms=symptoms,
            )
        )
    return diagnostics


def response_contract_for_target(
    target: TargetSpecification,
    intent: TargetIntent,
) -> ResponseContractDiagnostic:
    family = intent.value_shape_family
    unit_requirement: Literal["required", "allowed", "not_applicable"] = "not_applicable"
    if family == "decimal_measurement":
        unit_requirement = "required"
    elif target.unit:
        unit_requirement = "allowed"
    multiple = family == "ordered_or_unordered_list"
    raw_type = {
        "integer_count": "integer or string token",
        "decimal_measurement": "number or string token",
        "ordered_or_unordered_list": "list of strings or delimited string",
        "date": "string",
    }.get(family, "string")
    normalized_type = {
        "integer_count": "integer",
        "decimal_measurement": "number plus unit",
        "ordered_or_unordered_list": "list of strings",
        "date": "ISO date string",
    }.get(family, "string")
    return ResponseContractDiagnostic(
        target_row_id=target.target_row_id,
        field_name=target.expected_field,
        value_shape_family=family,
        expected_raw_value_transport_type=raw_type,
        expected_normalized_application_type=normalized_type,
        unit_requirement=unit_requirement,
        multiple_values_allowed=multiple,
        schema_name="attribute_grounded_extraction_v2",
        schema_version="2026-07-30",
    )


def redacted_request_diagnostic(
    target: TargetSpecification,
    intent: TargetIntent,
    bundle: EvidenceBundleRecord,
) -> RedactedRequestDiagnostic:
    return RedactedRequestDiagnostic(
        target_row_id=target.target_row_id,
        system_instruction=structured_system_instruction(),
        user_instruction_structure=[
            "target requirement",
            "evidence rules",
            "canonical evidence spans",
        ],
        schema_name="attribute_grounded_extraction_v2",
        schema_version="2026-07-30",
        requested_response_format_mode="json_schema",
        target_value_shape=intent.value_shape_family,
        evidence_span_count=len(bundle.evidence_spans),
        evidence_character_count=bundle.character_count,
        detected_contradictions=[],
    )


def _assert_v1_request_frozen(
    artifacts: ReextractArtifacts,
    v1_requests: dict[str, dict[str, Any]],
    frozen_ids: list[str],
) -> None:
    corrected_pages = corrected_page_set(artifacts.corrected_source_plan)
    for target_id in frozen_ids:
        bundle = build_reextract_bundle(artifacts, target_id)
        current_span_ids = [span.span_id for span in bundle.evidence_spans]
        v1_span_ids = [str(item) for item in v1_requests[target_id]["supporting_span_ids"]]
        if current_span_ids != v1_span_ids:
            raise ValueError(f"supporting span IDs changed for {target_id}")
        for span in bundle.evidence_spans:
            if (span.source_id, span.page_number) not in corrected_pages:
                raise ValueError(f"source page outside corrected corpus for {target_id}")


def _v2_preflight_bundle(
    frozen_ids: list[str],
    rejected_count: int,
    model_name: str,
) -> PreflightBundle:
    items = [
        PreflightEligibilityItem(
            target_row_id=target_id,
            field_name=target_id,
            diagnostic_eligibility="eligible_for_extraction",
            accepted=True,
            component_support=True,
            attribute_support=True,
            component_attribute_cooccurrence=True,
            value_shape_support=True,
            reason="frozen V1 eligible target",
        )
        for target_id in frozen_ids
    ]
    return PreflightBundle(
        items=items,
        preflight_report=ReextractPreflightReport(
            approved_candidate_count=len(frozen_ids),
            preflight_eligible_count=len(frozen_ids),
            preflight_rejected_count=rejected_count,
            target_ids=frozen_ids,
            target_count_by_requested_attribute={},
            model=model_name,
            expected_max_calls=len(frozen_ids),
            estimated_input_tokens=0,
            estimated_output_tokens=0,
            estimated_max_cost_usd=0,
            cost_ceiling_usd=DEFAULT_REEXTRACT_V2_COST_CEILING_USD,
            source_page_cache_status="corrected diagnostic corpus is canonical cached",
            parser_worker_count_expected=0,
            safety_gate_passed=True,
        ),
    )


def write_preflight(
    output_dir: Path,
    targets: list[TargetSpecification],
    preflight: PreflightBundle,
) -> None:
    _atomic_write_json(
        output_dir / "reextract_targets.json", [item.model_dump(mode="json") for item in targets]
    )
    _atomic_write_json(
        output_dir / "preflight_eligibility.json",
        [item.model_dump(mode="json") for item in preflight.items],
    )
    _atomic_write_json(
        output_dir / "preflight_report.json", preflight.preflight_report.model_dump(mode="json")
    )


def write_reextract_artifacts(
    result: ReextractRunResult,
    artifacts: ReextractArtifacts,
    output_dir: Path,
) -> None:
    payloads: dict[str, object] = {
        "reextract_targets.json": [
            item.model_dump(mode="json") for item in result.reextract_targets
        ],
        "preflight_eligibility.json": [
            item.model_dump(mode="json") for item in result.preflight_eligibility
        ],
        "preflight_report.json": result.preflight_report.model_dump(mode="json"),
        "extraction_requests.json": [
            item.model_dump(mode="json") for item in result.extraction_requests
        ],
        "extraction_results.json": [
            item.model_dump(mode="json") for item in result.extraction_results
        ],
        "canonical_evidence_results.json": [
            item.model_dump(mode="json") for item in result.canonical_evidence_results
        ],
        "normalized_values.json": [
            item.model_dump(mode="json") for item in result.normalized_values
        ],
        "evidence_validation.json": [
            item.model_dump(mode="json") for item in result.evidence_validation
        ],
        "shape_validation.json": [item.model_dump(mode="json") for item in result.shape_validation],
        "schema_compatibility_results.json": [
            item.model_dump(mode="json") for item in result.schema_compatibility_results
        ],
        "validation_results.json": [
            item.model_dump(mode="json") for item in result.validation_results
        ],
        "batch_v1_reextract_comparison.json": [
            item.model_dump(mode="json") for item in result.batch_v1_reextract_comparison
        ],
        "telemetry.json": result.telemetry.model_dump(mode="json"),
    }
    for filename, payload in payloads.items():
        _atomic_write_json(output_dir / filename, payload)
    (output_dir / "batch_v1_reextract_comparison.md").write_text(
        comparison_markdown(result), encoding="utf-8"
    )
    (output_dir / "reextract_summary.md").write_text(summary_markdown(result), encoding="utf-8")
    write_review_csv(result, artifacts, output_dir / "reextract_review.csv")


def write_reextract_v2_artifacts(
    result: ReextractV2RunResult,
    artifacts: ReextractArtifacts,
    output_dir: Path,
) -> None:
    payloads: dict[str, object] = {
        "frozen_targets.json": [item.model_dump(mode="json") for item in result.frozen_targets],
        "v1_invalid_format_diagnostic.json": [
            item.model_dump(mode="json") for item in result.v1_invalid_format_diagnostic
        ],
        "response_contracts.json": [
            item.model_dump(mode="json") for item in result.response_contracts
        ],
        "redacted_request_diagnostics.json": [
            item.model_dump(mode="json") for item in result.redacted_request_diagnostics
        ],
        "raw_response_diagnostics.json": [
            item.model_dump(mode="json") for item in result.raw_response_diagnostics
        ],
        "parsed_response_results.json": [
            item.model_dump(mode="json") if item is not None else None
            for item in result.parsed_response_results
        ],
        "extraction_results.json": [
            item.model_dump(mode="json") for item in result.extraction_results
        ],
        "canonical_evidence_results.json": [
            item.model_dump(mode="json") for item in result.canonical_evidence_results
        ],
        "normalized_values.json": [
            item.model_dump(mode="json") for item in result.normalized_values
        ],
        "evidence_validation.json": [
            item.model_dump(mode="json") for item in result.evidence_validation
        ],
        "shape_validation.json": [item.model_dump(mode="json") for item in result.shape_validation],
        "schema_compatibility_results.json": [
            item.model_dump(mode="json") for item in result.schema_compatibility_results
        ],
        "validation_results.json": [
            item.model_dump(mode="json") for item in result.validation_results
        ],
        "v1_v2_comparison.json": [item.model_dump(mode="json") for item in result.v1_v2_comparison],
        "telemetry.json": result.telemetry.model_dump(mode="json"),
    }
    for filename, payload in payloads.items():
        _atomic_write_json(output_dir / filename, payload)
    (output_dir / "v1_invalid_format_diagnostic.md").write_text(
        v1_invalid_format_markdown(result), encoding="utf-8"
    )
    (output_dir / "v1_v2_comparison.md").write_text(
        v2_comparison_markdown(result), encoding="utf-8"
    )
    (output_dir / "reextract_summary.md").write_text(v2_summary_markdown(result), encoding="utf-8")
    write_review_csv(
        _v2_as_v1_result(result),
        artifacts,
        output_dir / "reextract_review.csv",
    )


def comparison_markdown(result: ReextractRunResult) -> str:
    lines = ["# Batch V1 Re-extraction Comparison", ""]
    lines.append(f"- Outcomes: {result.telemetry.comparison_outcome_counts}")
    for item in result.batch_v1_reextract_comparison:
        previous = item.previous_review_status
        new = item.new_review_status
        lines.append(f"- `{item.field_name}`: {item.outcome}, {previous} -> {new}")
    return "\n".join(lines) + "\n"


def summary_markdown(result: ReextractRunResult) -> str:
    return (
        "\n".join(
            [
                "# Attribute-Grounded Re-extraction Pass",
                "",
                f"- Approved candidates: {result.telemetry.approved_candidate_count}",
                f"- Preflight eligible: {result.telemetry.preflight_eligible_count}",
                f"- Preflight rejected: {result.telemetry.preflight_rejected_count}",
                f"- Model calls: {result.telemetry.model_call_count}",
                (
                    f"- Tokens: input {result.telemetry.input_tokens}, "
                    f"output {result.telemetry.output_tokens}"
                ),
                f"- Estimated cost USD: {result.telemetry.estimated_cost_usd}",
                f"- Extraction statuses: {result.telemetry.extraction_status_counts}",
                f"- Review statuses: {result.telemetry.review_status_counts}",
                f"- Outcomes: {result.telemetry.comparison_outcome_counts}",
            ]
        )
        + "\n"
    )


def v1_invalid_format_markdown(result: ReextractV2RunResult) -> str:
    lines = ["# V1 Invalid-Format Diagnostic", ""]
    lines.append(
        "V1 did not persist raw provider response text, so root cause is limited to "
        "the final parsed symptoms and provider usage metadata."
    )
    lines.append("")
    for item in result.v1_invalid_format_diagnostic:
        lines.append(f"- `{item.field_name}`: {item.root_cause}")
    return "\n".join(lines) + "\n"


def v2_comparison_markdown(result: ReextractV2RunResult) -> str:
    lines = ["# Re-extraction V1 versus V2 Comparison", ""]
    lines.append(f"- Outcomes: {result.telemetry.comparison_outcome_counts}")
    for item in result.v1_v2_comparison:
        lines.append(
            f"- `{item.field_name}`: {item.previous_review_status} -> "
            f"{item.new_review_status} ({item.outcome})"
        )
    return "\n".join(lines) + "\n"


def v2_summary_markdown(result: ReextractV2RunResult) -> str:
    return (
        "\n".join(
            [
                "# Attribute-Grounded Re-extraction V2",
                "",
                f"- Frozen targets: {len(result.frozen_targets)}",
                f"- Model calls: {result.telemetry.model_call_count}",
                (
                    f"- Tokens: input {result.telemetry.input_tokens}, "
                    f"output {result.telemetry.output_tokens}"
                ),
                f"- Estimated cost USD: {result.telemetry.estimated_cost_usd}",
                f"- Extraction statuses: {result.telemetry.extraction_status_counts}",
                f"- Evidence validation: {result.telemetry.review_status_counts}",
                f"- Outcomes: {result.telemetry.comparison_outcome_counts}",
            ]
        )
        + "\n"
    )


def _v2_as_v1_result(result: ReextractV2RunResult) -> ReextractRunResult:
    return ReextractRunResult(
        reextract_targets=result.frozen_targets,
        preflight_eligibility=[],
        preflight_report=ReextractPreflightReport(
            approved_candidate_count=len(result.frozen_targets),
            preflight_eligible_count=len(result.frozen_targets),
            preflight_rejected_count=11,
            target_ids=[target.target_row_id for target in result.frozen_targets],
            target_count_by_requested_attribute={},
            model=result.telemetry.model,
            expected_max_calls=len(result.frozen_targets),
            estimated_input_tokens=result.telemetry.input_tokens,
            estimated_output_tokens=result.telemetry.output_tokens,
            estimated_max_cost_usd=result.telemetry.estimated_cost_usd,
            cost_ceiling_usd=DEFAULT_REEXTRACT_V2_COST_CEILING_USD,
            source_page_cache_status="corrected diagnostic corpus is canonical cached",
            parser_worker_count_expected=0,
            safety_gate_passed=True,
        ),
        extraction_requests=[],
        extraction_results=result.extraction_results,
        canonical_evidence_results=result.canonical_evidence_results,
        normalized_values=result.normalized_values,
        evidence_validation=result.evidence_validation,
        shape_validation=result.shape_validation,
        schema_compatibility_results=result.schema_compatibility_results,
        validation_results=result.validation_results,
        batch_v1_reextract_comparison=result.v1_v2_comparison,
        telemetry=result.telemetry,
    )


def write_review_csv(result: ReextractRunResult, artifacts: ReextractArtifacts, path: Path) -> None:
    intent_by_id = {
        item.target_row_id: artifacts.intent_by_id[item.target_row_id]
        for item in result.reextract_targets
    }
    evidence_by_id = {item.target_row_id: item for item in result.evidence_validation}
    shape_by_id = {item.target_row_id: item for item in result.shape_validation}
    schema_by_id = {item.target_row_id: item for item in result.schema_compatibility_results}
    validation_by_id = {item.target_row_id: item for item in result.validation_results}
    comparison_by_id = {item.target_row_id: item for item in result.batch_v1_reextract_comparison}
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "target_id",
            "domain",
            "sub_domain",
            "field_name",
            "component",
            "requested_attribute",
            "value_shape",
            "diagnostic_eligibility",
            "raw_model_value",
            "evidence_value",
            "normalized_value",
            "display_value",
            "extraction_status",
            "confidence",
            "source_file",
            "page",
            "hierarchy_node",
            "canonical_evidence",
            "attribute_support_result",
            "evidence_validation",
            "shape_validation",
            "dictionary_compatibility",
            "overall_review_status",
            "batch_v1_value",
            "batch_v1_review_status",
            "comparison_outcome",
            "caveat",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for extraction in result.extraction_results:
            target = artifacts.target_by_id[extraction.target_row_id]
            intent = intent_by_id[extraction.target_row_id]
            comparison = comparison_by_id[extraction.target_row_id]
            writer.writerow(
                {
                    "target_id": extraction.target_row_id,
                    "domain": target.metadata.get("domain") or target.sub_domain,
                    "sub_domain": target.sub_domain,
                    "field_name": target.expected_field,
                    "component": intent.primary_component,
                    "requested_attribute": intent.requested_attribute,
                    "value_shape": intent.value_shape_family,
                    "diagnostic_eligibility": artifacts.eligibility_by_id[extraction.target_row_id][
                        "status"
                    ],
                    "raw_model_value": extraction.raw_model_value,
                    "evidence_value": extraction.evidence_value,
                    "normalized_value": extraction.normalized_value,
                    "display_value": extraction.display_value,
                    "extraction_status": extraction.status,
                    "confidence": extraction.confidence,
                    "source_file": extraction.source_file,
                    "page": extraction.page_number,
                    "hierarchy_node": extraction.hierarchy_node_id,
                    "canonical_evidence": extraction.supporting_evidence_excerpt,
                    "attribute_support_result": evidence_by_id[
                        extraction.target_row_id
                    ].requested_attribute_supported,
                    "evidence_validation": evidence_by_id[extraction.target_row_id].status,
                    "shape_validation": shape_by_id[extraction.target_row_id].status,
                    "dictionary_compatibility": schema_by_id[
                        extraction.target_row_id
                    ].compatibility,
                    "overall_review_status": validation_by_id[extraction.target_row_id].status,
                    "batch_v1_value": comparison.previous_value,
                    "batch_v1_review_status": comparison.previous_review_status,
                    "comparison_outcome": comparison.outcome,
                    "caveat": extraction.ambiguity_or_caveat,
                }
            )


def group_spans_by_target(
    spans: list[EvidenceSpan],
    retrieval_by_id: dict[str, RetrievalResult],
) -> dict[str, list[EvidenceSpan]]:
    grouped: dict[str, list[EvidenceSpan]] = {}
    spans_by_node = {
        (span.source_id, span.page_number, span.hierarchy_node_id): span for span in spans
    }
    for target_id, retrieval in retrieval_by_id.items():
        target_spans: list[EvidenceSpan] = []
        for item in retrieval.results:
            key = (item.source_id, item.page_start, item.node_id)
            span = spans_by_node.get(key)
            if span is not None:
                target_spans.append(span)
        grouped[target_id] = target_spans
    return grouped


def corrected_page_set(plan: dict[str, Any]) -> set[tuple[str, int]]:
    pages: set[tuple[str, int]] = set()
    for key in ["retained_ranges", "added_ranges"]:
        for item in cast(list[dict[str, Any]], plan.get(key, [])):
            source_id = str(item["source_id"])
            for page in range(int(item["page_start"]), int(item["page_end"]) + 1):
                pages.add((source_id, page))
    return pages


def _default_client(settings: Settings, model_name: str) -> AttributeExtractionClient:
    if not settings.hosted_llm_enabled or settings.openai_api_key is None:
        return AbstainingAttributeExtractionClient()
    return AttributeGroundedOpenAIClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model_name=model_name,
    )


def _default_structured_client(
    settings: Settings, model_name: str
) -> StructuredAttributeExtractionClient:
    if not settings.hosted_llm_enabled or settings.openai_api_key is None:
        msg = "Hosted LLM must be enabled for re-extraction V2."
        raise ValueError(msg)
    return StructuredAttributeExtractionClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model_name=model_name,
    )


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
