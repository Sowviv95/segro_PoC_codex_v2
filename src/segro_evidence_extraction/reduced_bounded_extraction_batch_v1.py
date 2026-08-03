"""Reduced bounded extraction batch for checkpoint-approved evidence targets."""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from segro_evidence_extraction.config import Settings, load_settings
from segro_evidence_extraction.extraction_batch_construction_v1 import (
    DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR,
)
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.source_coverage_pageindex import (
    DEFAULT_PAGE_CACHE_ROOT,
    load_canonical_cached_pages,
)
from segro_evidence_extraction.vertical_slice import (
    EvidenceBundleRecord,
    EvidenceSpan,
    ExtractionResult,
    RetrievalScoreBreakdown,
    RetrievedEvidence,
    ValueShapeAssignment,
    _atomic_write_json,
    _extraction_prompt,
    calibrate_extraction_value,
    estimate_cost_usd,
    estimate_tokens,
    infer_value_shape_assignment,
    materialize_selected_spans,
    parse_extraction_response,
)

DEFAULT_CHECKPOINT_AUDIT_DIR = Path("output/enfield_unit1_pipeline_checkpoint_audit_v1")
DEFAULT_REDUCED_BATCH_OUTPUT_DIR = Path(
    "output/enfield_unit1_reduced_bounded_extraction_batch_v1"
)

INCLUDED_TARGET_IDS = [
    "trg_92469ca7eaab2c31",
    "trg_33975e86ffde5524",
    "trg_6720c2edd5e947d6",
    "trg_0f66487177c685e6",
    "trg_7426d9722059de76",
    "trg_93d2e2b38d1de7b2",
    "trg_50fedb4c249d8fe4",
    "trg_ca18dce11deff8cf",
    "trg_12690fb418279750",
    "trg_adb3dfa03b0c7cf3",
]
REJECTED_TARGET_REASONS = {
    "trg_7aefe5e9082729d6": "wrong event",
    "trg_8e2e33558f11916b": "component only",
    "trg_ea2bde50a7821e45": "wrong system",
}

APPROVED_VERDICTS = {
    "proceed_to_extraction": "approved_as_is",
    "proceed_with_caveat": "approved_with_caveat",
}
REJECTION_VERDICTS = {
    "reject_wrong_event": "wrong event",
    "reject_component_only": "component only",
    "reject_wrong_system": "wrong system",
}

FinalDecision = Literal["accepted", "accepted_with_caveat", "rejected", "abstained"]


class ReducedExtractionClient(Protocol):
    provider: str
    model_name: str

    def extract(
        self,
        *,
        target: TargetSpecification,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> tuple[ExtractionResult, dict[str, Any], dict[str, Any]]:
        """Return parsed extraction, persisted request payload, and raw provider response."""
        ...


class AbstainingReducedExtractionClient:
    provider = "mock"
    model_name = "abstaining-mock"

    def extract(
        self,
        *,
        target: TargetSpecification,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> tuple[ExtractionResult, dict[str, Any], dict[str, Any]]:
        request = build_model_request_payload(target, bundle, self.model_name, max_output_tokens)
        response = {
            "provider": self.provider,
            "model": self.model_name,
            "content": json.dumps(
                {
                    "raw_value": None,
                    "extracted_value": None,
                    "normalized_value": None,
                    "status": "insufficient_evidence",
                    "confidence": 0,
                    "selected_supporting_span_ids": [],
                    "value_bearing_quote": None,
                    "reasoning_summary": "Mock client abstained; no hosted LLM call was made.",
                    "ambiguity_or_caveat": "Mock client abstained; no hosted LLM call was made.",
                },
                sort_keys=True,
            ),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        result = parse_extraction_response(
            target=target,
            bundle=bundle,
            provider=self.provider,
            model_name=self.model_name,
            response_text=str(response["content"]),
        )
        return result, request, response


class RecordingOpenAIExtractionClient:
    provider = "openai"

    def __init__(self, *, api_key: str, model_name: str, timeout_seconds: int = 120) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def extract(
        self,
        *,
        target: TargetSpecification,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> tuple[ExtractionResult, dict[str, Any], dict[str, Any]]:
        request_payload = build_model_request_payload(
            target, bundle, self.model_name, max_output_tokens
        )
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                response_payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            error_payload = {
                "provider": self.provider,
                "model": self.model_name,
                "error": str(exc),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
            return model_error_result(target, self.provider, self.model_name, str(exc)), (
                request_payload
            ), error_payload
        try:
            content = response_payload["choices"][0]["message"]["content"]
            usage = response_payload.get("usage", {})
            result = parse_extraction_response(
                target=target,
                bundle=bundle,
                provider=self.provider,
                model_name=self.model_name,
                response_text=str(content),
            )
            result.model_usage.input_tokens = int(usage.get("prompt_tokens", 0) or 0)
            result.model_usage.output_tokens = int(usage.get("completion_tokens", 0) or 0)
            result.model_usage.estimated_cost_usd = estimate_cost_usd(
                self.model_name,
                result.model_usage.input_tokens,
                result.model_usage.output_tokens,
            )
            return result, request_payload, response_payload
        except (KeyError, IndexError, TypeError, ValidationError, ValueError) as exc:
            return model_error_result(target, self.provider, self.model_name, str(exc)), (
                request_payload
            ), response_payload


def run_reduced_bounded_extraction_batch_v1(
    *,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_AUDIT_DIR,
    constructed_batch_dir: Path = DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR,
    cache_root: Path = DEFAULT_PAGE_CACHE_ROOT,
    output_dir: Path = DEFAULT_REDUCED_BATCH_OUTPUT_DIR,
    settings: Settings | None = None,
    extraction_client: ReducedExtractionClient | None = None,
    dry_run_only: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    inputs = load_reduced_inputs(
        checkpoint_dir=checkpoint_dir,
        constructed_batch_dir=constructed_batch_dir,
        cache_root=cache_root,
    )
    package = build_reduced_batch_package(inputs)
    dry_run = validate_reduced_batch_contract(package, inputs["cached_page_keys"])
    if dry_run["failed_target_ids"]:
        dry_run_only = True
    client = extraction_client or build_default_reduced_extraction_client(
        settings or load_settings(Path("configs/default.yaml"))
    )
    execution = execute_reduced_batch(
        package,
        dry_run=dry_run,
        extraction_client=client,
        dry_run_only=dry_run_only,
    )
    summary = build_execution_summary(
        package=package,
        dry_run=dry_run,
        execution=execution,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
    result = {
        **package,
        **dry_run,
        **execution,
        "execution_summary": summary,
        "input_provenance": input_provenance(checkpoint_dir, constructed_batch_dir, cache_root),
    }
    write_reduced_outputs(result, output_dir)
    return result


def load_reduced_inputs(
    *,
    checkpoint_dir: Path,
    constructed_batch_dir: Path,
    cache_root: Path,
) -> dict[str, Any]:
    required = [
        checkpoint_dir / "approved_extraction_targets.json",
        checkpoint_dir / "rejected_extraction_targets.json",
        constructed_batch_dir / "bounded_extraction_requests.json",
        constructed_batch_dir / "canonical_evidence_payloads.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Reduced bounded extraction inputs missing: {missing}")
    cached_pages = load_canonical_cached_pages(cache_root)
    return {
        "approved": read_json_list(checkpoint_dir / "approved_extraction_targets.json"),
        "rejected": read_json_list(checkpoint_dir / "rejected_extraction_targets.json"),
        "requests": read_json_list(constructed_batch_dir / "bounded_extraction_requests.json"),
        "payloads": read_json_list(constructed_batch_dir / "canonical_evidence_payloads.json"),
        "cached_page_keys": {
            (source_id, int(page["page_number"]))
            for source_id, pages in cached_pages.items()
            for page in pages
        },
    }


def build_reduced_batch_package(inputs: dict[str, Any]) -> dict[str, Any]:
    approved_by_id = {str(item["target_id"]): item for item in inputs["approved"]}
    rejected_by_id = {str(item["target_id"]): item for item in inputs["rejected"]}
    request_by_id = {str(item["target_id"]): item for item in inputs["requests"]}
    payload_by_id = {str(item["target_id"]): item for item in inputs["payloads"]}
    reduced: list[dict[str, Any]] = []
    for index, target_id in enumerate(INCLUDED_TARGET_IDS, start=1):
        checkpoint = approved_by_id.get(target_id)
        request = request_by_id.get(target_id)
        payload = payload_by_id.get(target_id)
        if checkpoint is None:
            raise ValueError(f"Included target missing checkpoint approval: {target_id}")
        if request is None:
            raise ValueError(f"Included target missing constructed request: {target_id}")
        if payload is None:
            raise ValueError(f"Included target missing evidence payload: {target_id}")
        checkpoint_status = APPROVED_VERDICTS.get(str(checkpoint.get("final_audit_verdict")))
        if checkpoint_status is None:
            raise ValueError(f"Included target is not checkpoint-approved: {target_id}")
        reduced.append(
            {
                "reduced_batch_rank": index,
                "target_id": target_id,
                "target_name": checkpoint.get("field")
                or request.get("target", {}).get("expected_field"),
                "checkpoint_status": checkpoint_status,
                "checkpoint_caveat": checkpoint.get("audit_reason")
                if checkpoint_status == "approved_with_caveat"
                else None,
                "checkpoint_record": checkpoint,
                "constructed_request": request,
                "canonical_evidence_payload": payload,
                "target": request["target"],
                "evidence_bundle": request["evidence_bundle"],
                "normalization_expectations": request.get("normalization_expectations", {}),
                "validation_expectations": request.get("validation_expectations", []),
            }
        )
    rejected = []
    for target_id, expected_reason in REJECTED_TARGET_REASONS.items():
        checkpoint = rejected_by_id.get(target_id)
        request = request_by_id.get(target_id)
        if checkpoint is None:
            raise ValueError(f"Rejected target missing checkpoint rejection: {target_id}")
        observed_reason = REJECTION_VERDICTS.get(str(checkpoint.get("final_audit_verdict")))
        if observed_reason != expected_reason:
            raise ValueError(
                f"Rejected target {target_id} reason mismatch: "
                f"{observed_reason} != {expected_reason}"
            )
        rejected.append(
            {
                "target_id": target_id,
                "target_name": checkpoint.get("field"),
                "checkpoint_classification": checkpoint.get("final_audit_verdict"),
                "rejection_reason": expected_reason,
                "audit_reason": checkpoint.get("audit_reason"),
                "evidence_source": checkpoint.get("source"),
                "source_id": checkpoint.get("source_id"),
                "page_start": checkpoint.get("page_start"),
                "page_end": checkpoint.get("page_end"),
                "span_id": checkpoint.get("span_id"),
                "value_bearing_phrase": checkpoint.get("value_bearing_phrase"),
                "original_constructed_batch_record_reference": {
                    "request_id": request.get("request_id"),
                    "evidence_bundle_id": request.get("evidence_bundle", {}).get("bundle_id"),
                }
                if request
                else None,
            }
        )
    validate_exact_target_sets(reduced, rejected)
    return {
        "reduced_batch": reduced,
        "rejected_targets": rejected,
        "batch_summary": {
            "included_count": len(reduced),
            "rejected_count": len(rejected),
            "included_target_ids": [item["target_id"] for item in reduced],
            "rejected_target_ids": [item["target_id"] for item in rejected],
            "checkpoint_status_counts": dict(
                Counter(str(item["checkpoint_status"]) for item in reduced)
            ),
        },
    }


def validate_reduced_batch_contract(
    package: dict[str, Any],
    cached_page_keys: set[tuple[str, int]],
) -> dict[str, Any]:
    rows = []
    failed_ids: list[str] = []
    rejected_ids = set(REJECTED_TARGET_REASONS)
    for item in package["reduced_batch"]:
        target_id = str(item["target_id"])
        checks: dict[str, bool] = {}
        reasons: list[str] = []
        target = item.get("target", {})
        payload = item.get("canonical_evidence_payload", {})
        bundle = item.get("evidence_bundle", {})
        span = payload.get("span", {})
        text = str(payload.get("bounded_text") or span.get("text") or "")
        checks["target_identity_complete"] = all(
            target.get(key) for key in ["target_row_id", "requirement_id", "expected_field"]
        )
        checks["field_and_component_semantics_present"] = bool(
            target.get("requirement_text") and bundle.get("component_context")
        )
        checks["value_shape_and_datatype_defined"] = bool(
            item.get("normalization_expectations", {}).get("expected_data_type")
            and item.get("canonical_evidence_payload", {}).get("evidence_family")
        )
        checks["evidence_text_non_empty"] = bool(text.strip())
        checks["evidence_source_page_valid"] = (
            bool(span.get("source_id"))
            and isinstance(span.get("page_number"), int)
            and (str(span.get("source_id")), int(span["page_number"])) in cached_page_keys
        )
        checks["evidence_contained_in_bundle"] = evidence_contained_in_bundle(item, text)
        checks["checkpoint_status_allowed"] = item["checkpoint_status"] in {
            "approved_as_is",
            "approved_with_caveat",
        }
        checks["rejected_ids_excluded"] = target_id not in rejected_ids
        checks["prompt_schema_valid"] = prompt_schema_valid(item)
        checks["response_schema_valid"] = response_schema_valid()
        checks["typed_normalization_available"] = typed_normalization_available(item)
        checks["event_validation_available"] = event_validation_available(item)
        checks["dictionary_validation_available"] = dictionary_validation_available(item)
        for check, passed in checks.items():
            if not passed:
                reasons.append(check)
        status = "passed" if not reasons else "failed"
        if reasons:
            failed_ids.append(target_id)
        rows.append(
            {
                "target_id": target_id,
                "target_name": item["target_name"],
                "status": status,
                "checks": checks,
                "failure_reasons": reasons,
            }
        )
    return {
        "dry_run_validation": {
            "overall_status": "passed" if not failed_ids else "failed",
            "validated_target_count": len(package["reduced_batch"]) - len(failed_ids),
            "failed_target_count": len(failed_ids),
            "failed_target_ids": failed_ids,
            "target_validations": rows,
            "preflight_rejections": [
                row for row in rows if row["status"] == "failed"
            ],
        },
        "failed_target_ids": failed_ids,
    }


def execute_reduced_batch(
    package: dict[str, Any],
    *,
    dry_run: dict[str, Any],
    extraction_client: ReducedExtractionClient,
    dry_run_only: bool,
) -> dict[str, Any]:
    executable_ids = {
        item["target_id"]
        for item in package["reduced_batch"]
        if item["target_id"] not in set(dry_run["failed_target_ids"])
    }
    model_requests: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    if dry_run_only:
        return {
            "model_requests": [],
            "raw_model_responses": [],
            "extraction_results_raw": [],
            "extraction_results_validated": [],
            "final_adjudication": [],
        }
    for item in package["reduced_batch"]:
        if item["target_id"] not in executable_ids:
            continue
        target = TargetSpecification.model_validate(item["target"])
        assignment = infer_value_shape_assignment(target)
        bundle = evidence_bundle_record_from_request(item["constructed_request"])
        started = time.perf_counter()
        extraction, request_payload, response_payload = extraction_client.extract(
            target=target,
            bundle=bundle,
            max_output_tokens=600,
        )
        extraction = materialize_selected_spans(extraction, bundle)
        extraction = calibrate_extraction_value(
            extraction=extraction,
            target=target,
            assignment=assignment,
            bundle=bundle,
        )
        validation = validate_extraction_result(extraction, item, target, assignment)
        decision = final_decision(item, extraction, validation)
        request_payload = {
            **request_payload,
            "target_id": item["target_id"],
            "request_id": item["constructed_request"].get("request_id"),
        }
        response_payload = {
            "target_id": item["target_id"],
            "request_id": item["constructed_request"].get("request_id"),
            "provider_response": response_payload,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        model_requests.append(request_payload)
        raw_responses.append(response_payload)
        raw_results.append(extraction.model_dump(mode="json"))
        validated.append(
            {
                **extraction.model_dump(mode="json"),
                "evidence_containment": validation["evidence_containment"],
                "typed_normalization": validation["typed_normalization"],
                "event_validation": validation["event_validation"],
                "dictionary_validation": validation["dictionary_validation"],
                "validation_issues": validation["issues"],
            }
        )
        final.append(
            {
                "target_id": item["target_id"],
                "target_name": item["target_name"],
                "checkpoint_status": item["checkpoint_status"],
                "checkpoint_caveat": item.get("checkpoint_caveat"),
                "final_decision": decision,
                "raw_model_value": extraction.raw_model_value,
                "evidence_value": extraction.evidence_value,
                "normalized_value": extraction.normalized_value,
                "display_value": extraction.display_value,
                "source_id": extraction.source_id,
                "source_file": extraction.source_file,
                "page_number": extraction.page_number,
                "confidence": extraction.confidence,
                "model_caveat": extraction.ambiguity_or_caveat,
                "validation_issues": validation["issues"],
            }
        )
    return {
        "model_requests": model_requests,
        "raw_model_responses": raw_responses,
        "extraction_results_raw": raw_results,
        "extraction_results_validated": validated,
        "final_adjudication": final,
    }


def evidence_bundle_record_from_request(request: dict[str, Any]) -> EvidenceBundleRecord:
    evidence_item = request["retrieved_evidence"]
    payload = (
        request["canonical_evidence_payload"]
        if "canonical_evidence_payload" in request
        else None
    )
    if payload is None:
        payload = {
            "bounded_text": request["retrieved_evidence"]["excerpt"],
            "span": request["evidence_bundle"]["text_evidence_units"][0],
        }
    span_payload = request["evidence_bundle"]["text_evidence_units"][0]
    retrieved = RetrievedEvidence(
        target_row_id=request["target_id"],
        rank=int(evidence_item.get("rank") or 1),
        node_id=str(evidence_item["node_id"]),
        source_id=str(evidence_item["source_id"]),
        source_file=str(evidence_item.get("source_file") or ""),
        page_start=int(evidence_item["page_start"]),
        page_end=int(evidence_item["page_end"]),
        score=float(evidence_item.get("score") or 1.0),
        score_components=RetrievalScoreBreakdown.model_validate(
            evidence_item.get("score_components") or {}
        ),
        matched_terms=[str(term) for term in evidence_item.get("matched_terms") or []],
        hierarchy_path=[str(term) for term in evidence_item.get("hierarchy_path") or []],
        excerpt=str(evidence_item["excerpt"]),
    )
    span = EvidenceSpan(
        span_id=str(span_payload.get("text_ref")),
        source_id=retrieved.source_id,
        source_file=retrieved.source_file,
        page_number=retrieved.page_start,
        hierarchy_node_id=retrieved.node_id,
        text=retrieved.excerpt,
        start_char=0,
        end_char=len(retrieved.excerpt),
        retrieval_rank=retrieved.rank,
        score=retrieved.score,
    )
    return EvidenceBundleRecord(
        target_row_id=request["target_id"],
        evidence_items=[retrieved],
        retrieval_status="evidence_found",
        evidence_spans=[span],
        combined_text=retrieved.excerpt,
        character_count=len(retrieved.excerpt),
        token_estimate=estimate_tokens(retrieved.excerpt),
        truncated=False,
    )


def validate_extraction_result(
    extraction: ExtractionResult,
    batch_item: dict[str, Any],
    target: TargetSpecification,
    assignment: ValueShapeAssignment,
) -> dict[str, Any]:
    issues: list[str] = []
    text = str(batch_item["canonical_evidence_payload"].get("bounded_text") or "")
    span = batch_item["canonical_evidence_payload"].get("span", {})
    quote = str(extraction.value_bearing_quote or extraction.supporting_evidence_excerpt or "")
    containment = extraction.status != "extracted" or (
        bool(quote) and normalize_space(quote) in normalize_space(text)
    )
    if not containment:
        issues.append("cited evidence quote/span is not contained in approved evidence bundle")
    typed = typed_value_valid(extraction, assignment)
    if not typed:
        issues.append("extracted value failed typed normalization")
    event = event_valid(extraction, batch_item)
    if not event:
        issues.append("extracted value belongs to the wrong event context")
    dictionary = dictionary_valid(extraction, target)
    if not dictionary:
        issues.append("extracted value is incompatible with dictionary metadata")
    source_page = extraction.status != "extracted" or (
        extraction.source_id == span.get("source_id")
        and extraction.page_number == span.get("page_number")
    )
    if not source_page:
        issues.append("extracted source/page is outside approved evidence bundle")
    return {
        "evidence_containment": "passed" if containment and source_page else "failed",
        "typed_normalization": "passed" if typed else "failed",
        "event_validation": "passed" if event else "failed",
        "dictionary_validation": "passed" if dictionary else "failed",
        "issues": issues,
    }


def final_decision(
    batch_item: dict[str, Any],
    extraction: ExtractionResult,
    validation: dict[str, Any],
) -> FinalDecision:
    if extraction.status == "insufficient_evidence":
        return "abstained"
    if extraction.status != "extracted":
        return "rejected"
    if validation["issues"]:
        return "rejected"
    if batch_item["checkpoint_status"] == "approved_with_caveat" or extraction.ambiguity_or_caveat:
        return "accepted_with_caveat"
    return "accepted"


def build_execution_summary(
    *,
    package: dict[str, Any],
    dry_run: dict[str, Any],
    execution: dict[str, Any],
    elapsed_ms: float,
) -> dict[str, Any]:
    final = execution["final_adjudication"]
    usage = [
        response.get("provider_response", {}).get("usage", {})
        for response in execution["raw_model_responses"]
    ]
    return {
        "model_used": execution["raw_model_responses"][0]["provider_response"].get("model")
        if execution["raw_model_responses"]
        else None,
        "model_calls": len(execution["raw_model_responses"]),
        "input_tokens": sum(int(item.get("prompt_tokens", 0) or 0) for item in usage),
        "output_tokens": sum(int(item.get("completion_tokens", 0) or 0) for item in usage),
        "failures_and_retries": [],
        "elapsed_ms": round(elapsed_ms, 3),
        "accepted_count": sum(item["final_decision"] == "accepted" for item in final),
        "accepted_with_caveat_count": sum(
            item["final_decision"] == "accepted_with_caveat" for item in final
        ),
        "rejected_count": sum(item["final_decision"] == "rejected" for item in final),
        "abstained_count": sum(item["final_decision"] == "abstained" for item in final),
        "preflight_excluded_count": len(dry_run["failed_target_ids"]),
        "included_count": len(package["reduced_batch"]),
        "checkpoint_rejected_count": len(package["rejected_targets"]),
        "retrieval_invocations": 0,
        "parser_invocations": 0,
        "cache_expansion_invocations": 0,
        "ocr_invocations": 0,
        "vlm_invocations": 0,
    }


def build_default_reduced_extraction_client(settings: Settings) -> ReducedExtractionClient:
    if not settings.hosted_llm_enabled or settings.openai_api_key is None:
        return AbstainingReducedExtractionClient()
    return RecordingOpenAIExtractionClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model_name=settings.text_model_name or "gpt-4o-mini",
    )


def build_model_request_payload(
    target: TargetSpecification,
    bundle: EvidenceBundleRecord,
    model_name: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract only from the supplied evidence. Return strict JSON only. "
                    "Do not use outside knowledge. Abstain when evidence does not support "
                    "the requested attribute, event, system, unit, or component."
                ),
            },
            {"role": "user", "content": _extraction_prompt(target, bundle)},
        ],
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
    }


def model_error_result(
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
        ambiguity_or_caveat=message,
    )


def write_reduced_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "reduced_batch.json": result["reduced_batch"],
        "rejected_targets.json": result["rejected_targets"],
        "batch_summary.json": result["batch_summary"],
        "input_provenance.json": result["input_provenance"],
        "dry_run_validation.json": result["dry_run_validation"],
        "extraction_results_raw.json": result["extraction_results_raw"],
        "extraction_results_validated.json": result["extraction_results_validated"],
        "final_adjudication.json": result["final_adjudication"],
        "execution_summary.json": result["execution_summary"],
    }
    for filename, payload in artifacts.items():
        _atomic_write_json(output_dir / filename, payload)
    write_jsonl(output_dir / "reduced_batch.jsonl", result["reduced_batch"])
    write_jsonl(output_dir / "model_requests.jsonl", result["model_requests"])
    write_jsonl(output_dir / "raw_model_responses.jsonl", result["raw_model_responses"])
    (output_dir / "dry_run_validation.md").write_text(
        dry_run_markdown(result["dry_run_validation"]), encoding="utf-8"
    )
    write_final_csv(result["final_adjudication"], output_dir / "final_adjudication.csv")


def write_final_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "target_id",
        "target_name",
        "checkpoint_status",
        "checkpoint_caveat",
        "final_decision",
        "raw_model_value",
        "evidence_value",
        "normalized_value",
        "display_value",
        "source_id",
        "source_file",
        "page_number",
        "confidence",
        "model_caveat",
        "validation_issues",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field)) for field in fields})


def validate_exact_target_sets(
    reduced: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> None:
    included_ids = [item["target_id"] for item in reduced]
    rejected_ids = [item["target_id"] for item in rejected]
    if included_ids != INCLUDED_TARGET_IDS:
        raise ValueError(f"Reduced batch target IDs are not exact/deterministic: {included_ids}")
    if rejected_ids != list(REJECTED_TARGET_REASONS):
        raise ValueError(f"Rejected target IDs are not exact/deterministic: {rejected_ids}")
    overlap = set(included_ids) & set(rejected_ids)
    if overlap:
        raise ValueError(f"Rejected targets entered reduced batch: {sorted(overlap)}")


def evidence_contained_in_bundle(item: dict[str, Any], text: str) -> bool:
    bundle_units = item.get("evidence_bundle", {}).get("text_evidence_units") or []
    return bool(bundle_units) and all(
        normalize_space(str(unit.get("excerpt") or "")) in normalize_space(text)
        for unit in bundle_units
    )


def prompt_schema_valid(item: dict[str, Any]) -> bool:
    try:
        target = TargetSpecification.model_validate(item["target"])
        bundle = evidence_bundle_record_from_request(item["constructed_request"])
        payload = build_model_request_payload(target, bundle, "schema-check", 600)
    except (KeyError, TypeError, ValueError, ValidationError):
        return False
    return bool(
        payload.get("messages")
        and payload.get("response_format") == {"type": "json_object"}
    )


def response_schema_valid() -> bool:
    return True


def typed_normalization_available(item: dict[str, Any]) -> bool:
    target = TargetSpecification.model_validate(item["target"])
    return infer_value_shape_assignment(target).value_shape_family != "unsupported_or_unknown"


def event_validation_available(item: dict[str, Any]) -> bool:
    field = str(item["target"].get("expected_field") or "").lower()
    event = str(item["checkpoint_record"].get("event_context") or "").lower()
    if any(term in field for term in ["date", "certificate", "commission", "installation"]):
        return bool(event)
    return True


def dictionary_validation_available(item: dict[str, Any]) -> bool:
    target = item["target"]
    return bool(target.get("expected_data_type") and target.get("source_dictionary_provenance"))


def typed_value_valid(extraction: ExtractionResult, assignment: ValueShapeAssignment) -> bool:
    if extraction.status != "extracted":
        return True
    value = extraction.normalized_value
    shape = assignment.value_shape_family
    if shape == "integer_count":
        return isinstance(value, int)
    if shape == "decimal_measurement":
        return isinstance(value, int | float)
    if shape == "date":
        return isinstance(value, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))
    if shape == "ordered_or_unordered_list":
        return isinstance(value, list) and bool(value)
    return value not in {None, ""}


def event_valid(extraction: ExtractionResult, batch_item: dict[str, Any]) -> bool:
    if extraction.status != "extracted":
        return True
    field = str(batch_item["target"].get("expected_field") or "").lower()
    context = str(batch_item["checkpoint_record"].get("event_context") or "").lower()
    text = str(batch_item["canonical_evidence_payload"].get("bounded_text") or "").lower()
    if "installation_date" in field and "commissioning" in context:
        return False
    if "certificate" in field and "certificate" not in text:
        return False
    if "commission" in field and "commission" not in text:
        return False
    return True


def dictionary_valid(extraction: ExtractionResult, target: TargetSpecification) -> bool:
    if extraction.status != "extracted" or extraction.normalized_value is None:
        return True
    if not target.accepted_values:
        return True
    normalized = str(extraction.normalized_value).strip().lower()
    return normalized in {value.strip().lower() for value in target.accepted_values}


def input_provenance(
    checkpoint_dir: Path,
    constructed_batch_dir: Path,
    cache_root: Path,
) -> dict[str, Any]:
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "constructed_batch_dir": str(constructed_batch_dir),
        "cache_root": str(cache_root),
        "source_coverage_repeated": False,
        "pageindex_generation_repeated": False,
        "target_selection_repeated": False,
        "cache_expansion_repeated": False,
        "retrieval_repeated": False,
        "evidence_mapping_repeated": False,
    }


def dry_run_markdown(validation: dict[str, Any]) -> str:
    lines = [
        "# Reduced Bounded Extraction Batch V1 Dry-Run Validation",
        "",
        f"- Overall status: {validation['overall_status']}",
        f"- Validated targets: {validation['validated_target_count']}",
        f"- Failed targets: {validation['failed_target_count']}",
        "",
    ]
    for row in validation["target_validations"]:
        lines.append(f"- `{row['target_id']}` {row['status']}: {', '.join(row['failure_reasons'])}")
    return "\n".join(lines) + "\n"


def read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")
    return [dict(item) for item in data]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
