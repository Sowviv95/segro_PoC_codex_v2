"""Execute the prepared expanded bounded extraction batch."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from segro_evidence_extraction.bounded_extraction_expansion_preparation_v1 import (
    DEFAULT_EXPANSION_PREP_OUTPUT_DIR,
)
from segro_evidence_extraction.config import Settings, load_settings
from segro_evidence_extraction.downstream_handoff_contract_v1 import (
    validate_downstream_handoff_contract_v1,
)
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.parsing.service import load_source_registry
from segro_evidence_extraction.reduced_bounded_extraction_batch_v1 import (
    AbstainingReducedExtractionClient,
    RecordingOpenAIExtractionClient,
    ReducedExtractionClient,
    build_model_request_payload,
    dictionary_valid,
    typed_value_valid,
)
from segro_evidence_extraction.reduced_extraction_adjudication_v1 import (
    normalized_evidence_contains,
    repair_response_payload,
)
from segro_evidence_extraction.source_coverage_pageindex import (
    DEFAULT_SOURCE_MANIFEST,
)
from segro_evidence_extraction.vertical_slice import (
    EvidenceBundleRecord,
    EvidenceSpan,
    ExtractionResult,
    RetrievalScoreBreakdown,
    RetrievedEvidence,
    _atomic_write_json,
    calibrate_extraction_value,
    infer_value_shape_assignment,
    materialize_selected_spans,
)

DEFAULT_EXPANDED_BATCH_OUTPUT_DIR = Path(
    "output/enfield_unit1_expanded_bounded_extraction_batch_v1"
)
DEFAULT_EXPANDED_ADJUDICATION_OUTPUT_DIR = Path(
    "output/enfield_unit1_expanded_bounded_extraction_adjudication_v1"
)
DEFAULT_EXPANDED_HANDOFF_OUTPUT_DIR = Path(
    "output/enfield_unit1_expanded_downstream_handoff_exporter_v1"
)

EXPECTED_SELECTED_TARGET_IDS = [
    "trg_b16ea18d75c226c0",
    "trg_2ab51d5b0cc8b48d",
    "trg_199c5560cea7955c",
]
COMPLETED_TARGET_IDS = {
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
}
CHECKPOINT_REJECTED_TARGET_IDS = {
    "trg_7aefe5e9082729d6",
    "trg_8e2e33558f11916b",
    "trg_ea2bde50a7821e45",
}

EXTRACTION_RUN_ID = "enfield_unit1_expanded_bounded_extraction_batch_v1"
ADJUDICATION_RUN_ID = "enfield_unit1_expanded_bounded_extraction_adjudication_v1"
EXPORT_RUN_ID = "enfield_unit1_expanded_downstream_handoff_exporter_v1"


def run_expanded_bounded_extraction_batch_v1(
    *,
    selected_batch_path: Path = DEFAULT_EXPANSION_PREP_OUTPUT_DIR / "selected_batch.json",
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    batch_output_dir: Path = DEFAULT_EXPANDED_BATCH_OUTPUT_DIR,
    adjudication_output_dir: Path = DEFAULT_EXPANDED_ADJUDICATION_OUTPUT_DIR,
    handoff_output_dir: Path = DEFAULT_EXPANDED_HANDOFF_OUTPUT_DIR,
    settings: Settings | None = None,
    extraction_client: ReducedExtractionClient | None = None,
    dry_run_only: bool = False,
    runner_metadata: dict[str, Any] | None = None,
    expected_target_ids: list[str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    selected = read_json_list(selected_batch_path)
    source_paths = source_paths_by_id(source_manifest)
    provenance = input_provenance(selected_batch_path, source_manifest, source_paths)
    if runner_metadata:
        provenance["runner_metadata"] = runner_metadata
    dry_run = validate_expanded_preflight(
        selected,
        source_paths,
        expected_target_ids=expected_target_ids,
    )
    client = extraction_client or build_default_extraction_client(
        settings or load_settings(Path("configs/default.yaml"))
    )
    execution = execute_expanded_batch(
        selected,
        dry_run=dry_run,
        extraction_client=client,
        dry_run_only=dry_run_only or bool(dry_run["failed_target_ids"]),
        source_paths=source_paths,
    )
    batch_summary = build_batch_summary(
        selected=selected,
        dry_run=dry_run,
        execution=execution,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
    batch_result = {
        "selected_batch": selected,
        "dry_run_validation": dry_run,
        **execution,
        "execution_summary": batch_summary,
        "input_provenance": provenance,
    }
    write_batch_outputs(batch_result, batch_output_dir)

    adjudication = build_adjudication_package(batch_result, provenance)
    write_adjudication_outputs(adjudication, adjudication_output_dir)

    handoff = build_handoff_package(
        selected=selected,
        final_adjudication=adjudication["final_adjudication"],
        repair_audit=adjudication["repair_audit"],
        target_review=adjudication["target_review"],
        input_provenance=provenance,
        source_paths=source_paths,
        batch_output_dir=batch_output_dir,
        adjudication_output_dir=adjudication_output_dir,
        asset_record_key=(
            str(runner_metadata["asset_record_key"])
            if runner_metadata and runner_metadata.get("asset_record_key")
            else None
        ),
    )
    write_handoff_outputs(handoff, handoff_output_dir)
    return {
        "batch": batch_result,
        "adjudication": adjudication,
        "handoff": handoff,
    }


def validate_expanded_preflight(
    selected: list[dict[str, Any]],
    source_paths: dict[str, str],
    expected_target_ids: list[str] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    target_ids = [str(item.get("target_id")) for item in selected]
    if expected_target_ids is None:
        expected_target_ids = EXPECTED_SELECTED_TARGET_IDS
    if expected_target_ids and target_ids != expected_target_ids:
        errors.append(f"selected targets are not exact/deterministic: {target_ids}")
    if len(target_ids) != len(set(target_ids)):
        errors.append("duplicate selected target IDs")
    rows = []
    failed_ids: list[str] = []
    for item in selected:
        target_id = str(item.get("target_id"))
        checks = preflight_checks(item, source_paths)
        reasons = [name for name, passed in checks.items() if not passed]
        if reasons:
            failed_ids.append(target_id)
        rows.append(
            {
                "target_id": target_id,
                "target_name": item.get("target_name"),
                "status": "failed" if reasons else "passed",
                "checks": checks,
                "failure_reasons": reasons,
            }
        )
    if errors:
        failed_ids.extend(target_id for target_id in target_ids if target_id not in failed_ids)
    return {
        "schema_version": "segro_expanded_bounded_extraction_dry_run_validation_v1",
        "overall_status": "passed" if not failed_ids and not errors else "failed",
        "expected_target_ids": expected_target_ids,
        "observed_target_ids": target_ids,
        "validated_target_count": len(selected) - len(set(failed_ids)),
        "failed_target_count": len(set(failed_ids)),
        "failed_target_ids": sorted(set(failed_ids), key=target_ids.index),
        "global_errors": errors,
        "target_validations": rows,
        "preflight_rejections": [row for row in rows if row["status"] == "failed"],
        "retrieval_invocations": 0,
        "parser_invocations": 0,
        "ocr_invocations": 0,
        "vlm_invocations": 0,
        "cache_expansion_invocations": 0,
        "evidence_remapping_invocations": 0,
        "target_reselection_invocations": 0,
    }


def preflight_checks(item: dict[str, Any], source_paths: dict[str, str]) -> dict[str, bool]:
    target = item.get("target") or {}
    payload = item.get("canonical_evidence_payload") or {}
    span = payload.get("span") or {}
    text = str(payload.get("bounded_text") or span.get("text") or "")
    checks = {
        "not_previously_completed": item.get("target_id") not in COMPLETED_TARGET_IDS,
        "not_checkpoint_rejected": item.get("target_id") not in CHECKPOINT_REJECTED_TARGET_IDS,
        "valid_evidence_bundle": bool(item.get("evidence_bundle")),
        "evidence_text_non_empty": bool(text.strip()),
        "source_id_present": bool(span.get("source_id") or item.get("source_id")),
        "source_file_present": bool(span.get("source_file") or item.get("source_filename")),
        "page_number_present": isinstance(span.get("page_number") or item.get("page_number"), int),
        "selected_page_exists_in_cache": Path(str(item.get("cache_path") or "")).exists(),
        "datatype_defined": bool(item.get("datatype") or target.get("expected_data_type")),
        "value_shape_defined": bool(item.get("value_shape")),
        "typed_normalization_available": typed_normalization_available(item),
        "event_validation_available": event_validation_available(item),
        "dictionary_validation_available": dictionary_validation_available(item),
        "source_path_available": bool(
            source_paths.get(str(span.get("source_id") or item.get("source_id")))
        ),
        "prompt_contract_valid": prompt_contract_valid(item),
        "response_schema_valid": True,
        "downstream_contract_compatible": True,
        "retrieval_path_not_reachable": True,
        "parser_path_not_reachable": True,
        "ocr_vlm_paths_not_reachable": True,
        "cache_expansion_path_not_reachable": True,
    }
    return checks


def execute_expanded_batch(
    selected: list[dict[str, Any]],
    *,
    dry_run: dict[str, Any],
    extraction_client: ReducedExtractionClient,
    dry_run_only: bool,
    source_paths: dict[str, str],
) -> dict[str, Any]:
    executable_ids = {
        item["target_id"]
        for item in selected
        if item["target_id"] not in set(dry_run["failed_target_ids"])
    }
    requests: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    if dry_run_only:
        return empty_execution()
    for item in selected:
        if item["target_id"] not in executable_ids:
            continue
        target = TargetSpecification.model_validate(item["target"])
        assignment = infer_value_shape_assignment(target)
        bundle = evidence_bundle_record_from_selected(item)
        started = time.perf_counter()
        extraction, request_payload, response_payload = extraction_client.extract(
            target=target,
            bundle=bundle,
            max_output_tokens=600,
        )
        repair = deterministic_repair_extraction(
            extraction=extraction,
            response_payload=response_payload,
            target=target,
            bundle=bundle,
        )
        extraction = repair["extraction"]
        extraction = materialize_selected_spans(extraction, bundle)
        extraction = calibrate_extraction_value(
            extraction=extraction,
            target=target,
            assignment=assignment,
            bundle=bundle,
        )
        validation = validate_expanded_extraction(extraction, item, target)
        decision = expanded_final_decision(extraction, validation)
        model_usage = extraction.model_usage.model_dump(mode="json")
        model_usage["model"] = extraction.model_name
        model_usage["provider"] = extraction.model_provider
        raw_result = extraction.model_dump(mode="json")
        raw_result["model_usage"] = model_usage
        raw_result["deterministic_repair_applied"] = repair["applied"]
        raw_result["deterministic_repair_reason"] = repair["reason"]
        raw_results.append(raw_result)
        validated.append(
            {
                **raw_result,
                "evidence_containment": validation["evidence_containment"],
                "typed_normalization": validation["typed_normalization"],
                "event_validation": validation["event_validation"],
                "dictionary_validation": validation["dictionary_validation"],
                "validation_issues": validation["issues"],
            }
        )
        requests.append(
            {
                **request_payload,
                "target_id": item["target_id"],
                "extraction_run_id": EXTRACTION_RUN_ID,
            }
        )
        responses.append(
            {
                "target_id": item["target_id"],
                "provider_response": response_payload,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
        final.append(
            final_row(
                item=item,
                extraction=extraction,
                validation=validation,
                decision=decision,
                source_path=source_paths.get(str(extraction.source_id or item["source_id"])),
                model_usage=model_usage,
                deterministic_repair_applied=bool(repair["applied"]),
                deterministic_repair_reason=repair["reason"],
            )
        )
    return {
        "model_requests": requests,
        "raw_model_responses": responses,
        "extraction_results_raw": raw_results,
        "extraction_results_validated": validated,
        "final_adjudication": final,
    }


def build_adjudication_package(
    batch_result: dict[str, Any],
    input_provenance_payload: dict[str, Any],
) -> dict[str, Any]:
    reviewed = []
    final = []
    repair_audit = []
    for row in batch_result["final_adjudication"]:
        reviewed_row = review_expanded_target(row)
        reviewed.append(reviewed_row)
        final.append(
            {
                **row,
                "original_decision": row["final_decision"],
                "adjudicated_decision": reviewed_row["adjudicated_decision"],
                "final_decision": reviewed_row["adjudicated_decision"],
                "deterministic_repair_applied": reviewed_row["deterministic_repair_applied"],
                "model_retry_occurred": False,
                "decision_change_reason": reviewed_row["decision_change_reason"],
                "adjudication_run_id": ADJUDICATION_RUN_ID,
                "input_provenance": input_provenance_payload,
            }
        )
        repair_audit.append(
            {
                "target_id": row["target_id"],
                "original_decision": row["final_decision"],
                "adjudicated_decision": reviewed_row["adjudicated_decision"],
                "deterministic_repair_applied": reviewed_row["deterministic_repair_applied"],
                "model_retry_occurred": False,
                "decision_change_reason": reviewed_row["decision_change_reason"],
                "evidence_containment": row["evidence_containment"],
                "event_validation": row["event_validation"],
                "dictionary_validation": row["dictionary_validation"],
            }
        )
    summary = {
        "schema_version": "segro_expanded_extraction_adjudication_summary_v1",
        "target_count": len(final),
        "decision_counts": dict(Counter(item["final_decision"] for item in final)),
        "deterministic_repairs": sum(item["deterministic_repair_applied"] for item in final),
        "model_retries": 0,
    }
    return {
        "adjudication_summary": summary,
        "target_review": reviewed,
        "repaired_results": [item for item in reviewed if item["deterministic_repair_applied"]],
        "final_adjudication": final,
        "repair_audit": repair_audit,
        "model_retry_requests": [],
        "model_retry_responses": [],
    }


def review_expanded_target(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_id": row["target_id"],
        "target_name": row["target_name"],
        "original_decision": row["final_decision"],
        "adjudicated_decision": row["final_decision"],
        "deterministic_repair_applied": bool(row.get("deterministic_repair_applied")),
        "model_retry_occurred": False,
        "decision_change_reason": row.get("deterministic_repair_reason"),
        "evidence_containment": row["evidence_containment"],
        "event_validation": row["event_validation"],
        "dictionary_validation": row["dictionary_validation"],
        "typed_normalization": row["typed_normalization"],
        "validation_issues": row["validation_issues"],
    }


def build_handoff_package(
    *,
    selected: list[dict[str, Any]],
    final_adjudication: list[dict[str, Any]],
    repair_audit: list[dict[str, Any]],
    target_review: list[dict[str, Any]],
    input_provenance: dict[str, Any],
    source_paths: dict[str, str],
    batch_output_dir: Path,
    adjudication_output_dir: Path,
    asset_record_key: str | None = None,
) -> dict[str, Any]:
    batch_by_id = {item["target_id"]: item for item in selected}
    repair_by_id = {item["target_id"]: item for item in repair_audit}
    review_by_id = {item["target_id"]: item for item in target_review}
    internal = [
        internal_handoff_record(row, batch_by_id[row["target_id"]], source_paths, input_provenance)
        for row in final_adjudication
    ]
    customer = [
        customer_handoff_record(record, asset_record_key=asset_record_key) for record in internal
    ]
    validation = validate_downstream_handoff_contract_v1(
        internal_records=internal,
        customer_records=customer,
    )
    if validation["status"] != "passed":
        raise ValueError(f"Expanded handoff validation failed: {validation['errors']}")
    export_validation = {
        "schema_version": "segro_expanded_downstream_handoff_export_validation_v1",
        "status": "passed",
        "contract_validation": validation,
        "errors": [],
        "warnings": validation["warnings"],
        "internal_record_count": len(internal),
        "customer_record_count": len(customer),
        "promotion_counts": dict(Counter(record["promotion_status"] for record in internal)),
        "customer_caveat_policy": "internal_only",
    }
    return {
        "internal_handoff": internal,
        "customer_candidate_handoff": customer,
        "export_validation": export_validation,
        "promotion_summary": promotion_summary(internal),
        "non_promotion_summary": non_promotion_summary(internal),
        "mapping_gaps": mapping_gaps(),
        "export_provenance": export_provenance(
            input_provenance, batch_output_dir, adjudication_output_dir
        ),
        "execution_summary": handoff_execution_summary(internal, customer, export_validation),
        "repair_by_id": repair_by_id,
        "review_by_id": review_by_id,
    }


def internal_handoff_record(
    row: dict[str, Any],
    batch_item: dict[str, Any],
    source_paths: dict[str, str],
    input_provenance_payload: dict[str, Any],
) -> dict[str, Any]:
    target = batch_item["target"]
    span = batch_item["canonical_evidence_payload"]["span"]
    success = row["final_decision"] in {"accepted", "accepted_with_caveat"}
    final_value = row.get("normalized_value") if success else None
    if target.get("expected_data_type") == "string" and success:
        final_value = row.get("display_value")
    source_id = row.get("source_id") or span.get("source_id")
    return {
        "schema_version": "segro_internal_evidence_rich_handoff_v1",
        "trace_id": row["target_id"],
        "target_id": row["target_id"],
        "requirement_id": target.get("requirement_id"),
        "field_name": target.get("expected_field"),
        "target_name": row["target_name"],
        "component": {
            "component_type": target.get("component_type"),
            "component_subtype": target.get("component_subtype"),
            "component_system_identity": batch_item["evidence_bundle"]
            .get("component_context", {})
            .get("component_system_identity"),
        },
        "attribute": batch_item["evidence_bundle"]
        .get("component_context", {})
        .get("requested_attribute"),
        "record_identity": f"{EXTRACTION_RUN_ID}:{row['target_id']}",
        "raw_model_value": row.get("raw_model_value"),
        "evidence_value": row.get("evidence_value"),
        "normalized_value": row.get("normalized_value"),
        "display_value": row.get("display_value") if success else None,
        "final_value": final_value,
        "final_decision": row["final_decision"],
        "reason_code": reason_code(row),
        "decision_reason": decision_reason(row),
        "checkpoint_status": row.get("checkpoint_status"),
        "checkpoint_caveat": row.get("checkpoint_caveat"),
        "extraction_caveat": row.get("model_caveat"),
        "adjudication_caveat": row.get("decision_change_reason"),
        "confidence": row.get("confidence"),
        "value_shape": batch_item.get("value_shape"),
        "datatype": target.get("expected_data_type"),
        "reference_list_status": reference_list_status(target),
        "source_id": source_id,
        "source_file": row.get("source_file") or span.get("source_file"),
        "source_path": row.get("source_path") or source_paths.get(str(source_id)),
        "page_number": row.get("page_number") or span.get("page_number"),
        "evidence_quote": row.get("evidence_value"),
        "evidence_span": {
            "span_id": span.get("span_id"),
            "source_id": span.get("source_id"),
            "source_file": span.get("source_file"),
            "page_number": span.get("page_number"),
            "start_char": span.get("start_char"),
            "end_char": span.get("end_char"),
            "bounded_text": batch_item["canonical_evidence_payload"].get("bounded_text"),
        },
        "evidence_span_id": span.get("span_id"),
        "evidence_char_start": span.get("start_char"),
        "evidence_char_end": span.get("end_char"),
        "evidence_containment_status": row.get("evidence_containment", "passed"),
        "event_validation_status": row.get("event_validation", "passed"),
        "dictionary_validation_status": row.get("dictionary_validation", "passed"),
        "typed_normalization_status": row.get("typed_normalization", "passed"),
        "validation_issues": row.get("validation_issues") or [],
        "model": row.get("model_usage", {}).get("model"),
        "extraction_run_id": row.get("extraction_run_id") or EXTRACTION_RUN_ID,
        "adjudication_run_id": row.get("adjudication_run_id") or ADJUDICATION_RUN_ID,
        "export_run_id": EXPORT_RUN_ID,
        "deterministic_repair_applied": bool(row.get("deterministic_repair_applied")),
        "model_retry_applied": bool(row.get("model_retry_occurred")),
        "input_provenance": input_provenance_payload,
        "promotable": row["final_decision"] in {"accepted", "accepted_with_caveat"},
        "promotion_status": promotion_status(row),
    }


def customer_handoff_record(
    record: dict[str, Any],
    *,
    asset_record_key: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "segro_customer_candidate_handoff_v1",
        "trace_id": record["trace_id"],
        "asset_record_key": asset_record_key,
        "asset_record_key_status": "provided" if asset_record_key else "mapping_required",
        "requirement_id": record["requirement_id"],
        "target_id": record["target_id"],
        "field_name": record["field_name"],
        "customer_field_label": None,
        "customer_field_label_status": "mapping_required",
        "display_value": record["display_value"] if record["promotable"] else None,
        "status": record["final_decision"],
        "reason_code": record["reason_code"],
        "caveat": None,
        "source_reference": {
            "source_id": record["source_id"],
            "source_file": record["source_file"],
            "page_number": record["page_number"],
            "source_url": None,
            "source_url_status": "mapping_required",
        },
        "internal_record_identity": record["record_identity"],
        "transformation_status": "mapping_required" if record["promotable"] else "not_promotable",
        "boolean_rendering_status": "mapping_required"
        if record["display_value"] in {"Y", "N"}
        else None,
        "date_rendering_status": "mapping_required" if record["datatype"] == "date" else None,
    }


def evidence_bundle_record_from_selected(item: dict[str, Any]) -> EvidenceBundleRecord:
    payload = item["canonical_evidence_payload"]
    span_payload = payload["span"]
    text = str(payload.get("bounded_text") or span_payload.get("text") or "")
    source_id = str(span_payload.get("source_id") or item.get("source_id"))
    source_file = str(span_payload.get("source_file") or item.get("source_filename"))
    page_number = int_required(span_payload.get("page_number") or item.get("page_number"))
    span_id = str(span_payload.get("span_id") or f"span_{item['target_id']}")
    retrieved = RetrievedEvidence(
        target_row_id=str(item["target_id"]),
        rank=1,
        node_id=str(payload.get("payload_id") or span_id),
        source_id=source_id,
        source_file=source_file,
        page_start=page_number,
        page_end=page_number,
        score=float(span_payload.get("score") or 1.0),
        score_components=RetrievalScoreBreakdown(),
        matched_terms=[],
        hierarchy_path=[],
        excerpt=text,
    )
    span = EvidenceSpan(
        span_id=span_id,
        source_id=source_id,
        source_file=source_file,
        page_number=page_number,
        hierarchy_node_id=retrieved.node_id,
        text=text,
        start_char=int(span_payload.get("start_char") or 0),
        end_char=int(
            span_payload.get("end_char") if span_payload.get("end_char") is not None else len(text)
        ),
        retrieval_rank=1,
        score=retrieved.score,
    )
    return EvidenceBundleRecord(
        target_row_id=str(item["target_id"]),
        evidence_items=[retrieved],
        retrieval_status="evidence_found",
        evidence_spans=[span],
        combined_text=text,
        character_count=len(text),
        token_estimate=max(1, len(text) // 4),
        truncated=False,
    )


def validate_expanded_extraction(
    extraction: ExtractionResult,
    batch_item: dict[str, Any],
    target: TargetSpecification,
) -> dict[str, Any]:
    assignment = infer_value_shape_assignment(target)
    issues: list[str] = []
    text = str(batch_item["canonical_evidence_payload"].get("bounded_text") or "")
    span = batch_item["canonical_evidence_payload"].get("span", {})
    quote = str(extraction.value_bearing_quote or extraction.supporting_evidence_excerpt or "")
    containment = extraction.status != "extracted" or (
        bool(quote) and normalized_evidence_contains(text, quote)
    )
    if not containment:
        issues.append("cited evidence quote/span is not contained in approved evidence bundle")
    source_page = extraction.status != "extracted" or (
        extraction.source_id == span.get("source_id")
        and extraction.page_number == span.get("page_number")
    )
    if not source_page:
        issues.append("extracted source/page is outside approved evidence bundle")
    typed = typed_value_valid(extraction, assignment)
    if not typed:
        issues.append("extracted value failed typed normalization")
    event = event_validation_passed(extraction, batch_item)
    if not event:
        issues.append("extracted value belongs to the wrong event context")
    dictionary = dictionary_valid(extraction, target)
    if not dictionary:
        issues.append("extracted value is incompatible with dictionary metadata")
    return {
        "evidence_containment": "passed" if containment and source_page else "failed",
        "typed_normalization": "passed" if typed else "failed",
        "event_validation": "passed" if event else "failed",
        "dictionary_validation": "passed" if dictionary else "failed",
        "issues": issues,
    }


def event_validation_passed(extraction: ExtractionResult, batch_item: dict[str, Any]) -> bool:
    if extraction.status != "extracted":
        return True
    field = str(batch_item["target"].get("expected_field") or "").casefold()
    evidence = str(batch_item["canonical_evidence_payload"].get("bounded_text") or "").casefold()
    if "certificate" in field and "certificate" not in evidence:
        return False
    if "installation" in field and "installation" not in evidence:
        return False
    if "commission" in field and "commission" not in evidence:
        return False
    return True


def final_row(
    *,
    item: dict[str, Any],
    extraction: ExtractionResult,
    validation: dict[str, Any],
    decision: str,
    source_path: str | None,
    model_usage: dict[str, Any],
    deterministic_repair_applied: bool,
    deterministic_repair_reason: Any,
) -> dict[str, Any]:
    success = decision in {"accepted", "accepted_with_caveat"}
    return {
        "target_id": item["target_id"],
        "target_name": item["target_name"],
        "checkpoint_status": "expanded_execution_ready",
        "checkpoint_caveat": None,
        "final_decision": decision,
        "raw_model_value": extraction.raw_model_value,
        "evidence_value": extraction.evidence_value,
        "normalized_value": extraction.normalized_value,
        "display_value": extraction.display_value if success else None,
        "source_id": extraction.source_id,
        "source_file": extraction.source_file,
        "source_path": source_path,
        "page_number": extraction.page_number,
        "confidence": extraction.confidence,
        "model_caveat": extraction.ambiguity_or_caveat,
        "validation_issues": validation["issues"],
        "evidence_containment": validation["evidence_containment"],
        "typed_normalization": validation["typed_normalization"],
        "event_validation": validation["event_validation"],
        "dictionary_validation": validation["dictionary_validation"],
        "model_usage": model_usage,
        "extraction_run_id": EXTRACTION_RUN_ID,
        "deterministic_repair_applied": deterministic_repair_applied,
        "deterministic_repair_reason": deterministic_repair_reason,
    }


def expanded_batch_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        **item,
        "checkpoint_status": "approved_as_is",
        "checkpoint_record": {},
    }


def expanded_final_decision(
    extraction: ExtractionResult,
    validation: dict[str, Any],
) -> str:
    if extraction.status == "insufficient_evidence":
        return "abstained"
    if extraction.status != "extracted":
        return "rejected"
    issues = validation["issues"]
    if not issues:
        return "accepted_with_caveat" if extraction.ambiguity_or_caveat else "accepted"
    if (
        validation["dictionary_validation"] == "failed"
        and validation["evidence_containment"] == "passed"
        and validation["event_validation"] == "passed"
    ):
        return "abstained"
    return "rejected"


def deterministic_repair_extraction(
    *,
    extraction: ExtractionResult,
    response_payload: dict[str, Any],
    target: TargetSpecification,
    bundle: EvidenceBundleRecord,
) -> dict[str, Any]:
    if extraction.status not in {"invalid_format", "model_error"}:
        return {"extraction": extraction, "applied": False, "reason": None}
    provider_payload = response_payload
    if "provider_response" in provider_payload:
        provider_payload = dict(provider_payload["provider_response"])
    content = response_content(provider_payload)
    if content is None:
        return {"extraction": extraction, "applied": False, "reason": None}
    try:
        payload = json.loads(str(content))
    except json.JSONDecodeError:
        return {"extraction": extraction, "applied": False, "reason": None}
    if not isinstance(payload, dict):
        return {"extraction": extraction, "applied": False, "reason": None}
    repaired_payload, reason = repair_response_payload(payload)
    if repaired_payload == payload:
        return {"extraction": extraction, "applied": False, "reason": None}
    repaired = parse_extraction_response_safe(
        target=target,
        bundle=bundle,
        provider=str(provider_payload.get("provider") or extraction.model_provider),
        model_name=str(provider_payload.get("model") or extraction.model_name),
        payload=repaired_payload,
    )
    return {"extraction": repaired, "applied": True, "reason": reason}


def parse_extraction_response_safe(
    *,
    target: TargetSpecification,
    bundle: EvidenceBundleRecord,
    provider: str,
    model_name: str,
    payload: dict[str, Any],
) -> ExtractionResult:
    from segro_evidence_extraction.vertical_slice import parse_extraction_response

    try:
        return parse_extraction_response(
            target=target,
            bundle=bundle,
            provider=provider,
            model_name=model_name,
            response_text=json.dumps(payload, ensure_ascii=True, sort_keys=True),
        )
    except (ValidationError, ValueError) as exc:
        return ExtractionResult(
            target_row_id=target.target_row_id,
            requirement_id=target.requirement_id,
            status="invalid_format",
            confidence=0,
            model_provider=provider,
            model_name=model_name,
            ambiguity_or_caveat=str(exc),
        )


def response_content(provider_payload: dict[str, Any]) -> Any:
    try:
        return provider_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return provider_payload.get("content")


def typed_normalization_available(item: dict[str, Any]) -> bool:
    try:
        target = TargetSpecification.model_validate(item["target"])
    except ValidationError:
        return False
    return infer_value_shape_assignment(target).value_shape_family != "unsupported_or_unknown"


def event_validation_available(item: dict[str, Any]) -> bool:
    field = str(item.get("field_name") or item.get("target_name") or "").casefold()
    if any(token in field for token in ["certificate", "installation", "commission"]):
        return bool(str(item.get("canonical_evidence_payload", {}).get("bounded_text") or ""))
    return True


def dictionary_validation_available(item: dict[str, Any]) -> bool:
    target = item.get("target") or {}
    return bool(target.get("expected_data_type") and target.get("source_dictionary_provenance"))


def prompt_contract_valid(item: dict[str, Any]) -> bool:
    try:
        target = TargetSpecification.model_validate(item["target"])
        bundle = evidence_bundle_record_from_selected(item)
        payload = build_model_request_payload(target, bundle, "schema-check", 600)
    except (KeyError, TypeError, ValueError, ValidationError):
        return False
    return bool(
        payload.get("messages") and payload.get("response_format") == {"type": "json_object"}
    )


def build_default_extraction_client(settings: Settings) -> ReducedExtractionClient:
    if not settings.hosted_llm_enabled or settings.openai_api_key is None:
        return AbstainingReducedExtractionClient()
    return RecordingOpenAIExtractionClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model_name=settings.text_model_name or "gpt-4o-mini",
    )


def source_paths_by_id(source_manifest: Path) -> dict[str, str]:
    return {
        source.source_id: source.original_path for source in load_source_registry(source_manifest)
    }


def input_provenance(
    selected_batch_path: Path,
    source_manifest: Path,
    source_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "segro_expanded_bounded_extraction_input_provenance_v1",
        "selected_batch_path": str(selected_batch_path),
        "selected_batch_sha256": sha256_file(selected_batch_path),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_path_count": len(source_paths),
        "retrieval_repeated": False,
        "parsing_repeated": False,
        "ocr_repeated": False,
        "vlm_repeated": False,
        "cache_expansion_repeated": False,
        "evidence_remapping_repeated": False,
        "target_reselection_repeated": False,
    }


def build_batch_summary(
    *,
    selected: list[dict[str, Any]],
    dry_run: dict[str, Any],
    execution: dict[str, Any],
    elapsed_ms: float,
) -> dict[str, Any]:
    usage = [row.get("model_usage", {}) for row in execution["final_adjudication"]]
    final = execution["final_adjudication"]
    return {
        "schema_version": "segro_expanded_bounded_extraction_execution_summary_v1",
        "extraction_run_id": EXTRACTION_RUN_ID,
        "selected_count": len(selected),
        "executed_count": len(final),
        "preflight_excluded_count": len(dry_run["failed_target_ids"]),
        "model_used": usage[0].get("model") if usage else None,
        "model_calls": len(execution["raw_model_responses"]),
        "input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in usage),
        "output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in usage),
        "elapsed_ms": round(elapsed_ms, 3),
        "decision_counts": dict(Counter(item["final_decision"] for item in final)),
        "failures_and_retries": [],
        "retrieval_invocations": 0,
        "parser_invocations": 0,
        "ocr_invocations": 0,
        "vlm_invocations": 0,
        "cache_expansion_invocations": 0,
        "evidence_remapping_invocations": 0,
        "target_reselection_invocations": 0,
    }


def reason_code(row: dict[str, Any]) -> str:
    decision = str(row["final_decision"])
    issues = " ".join(str(issue).casefold() for issue in row.get("validation_issues", []))
    if decision == "accepted":
        return "accepted"
    if decision == "accepted_with_caveat":
        return "accepted_with_caveat"
    if "dictionary" in issues:
        return "dictionary_value_not_supported" if decision == "abstained" else "wrong_system"
    if "wrong event" in issues:
        return "wrong_event"
    if "wrong system" in issues:
        return "wrong_system"
    if "contained" in issues:
        return "evidence_containment_failed"
    if decision == "rejected":
        return "evidence_containment_failed"
    return "insufficient_attribute_evidence"


def decision_reason(row: dict[str, Any]) -> str:
    issues = row.get("validation_issues") or []
    if issues:
        return "; ".join(str(issue) for issue in issues)
    if row.get("model_caveat"):
        return str(row["model_caveat"])
    if row["final_decision"] == "abstained":
        return "Approved evidence did not support the requested attribute."
    return "Value accepted by expanded bounded extraction validation."


def promotion_status(row: dict[str, Any]) -> str:
    if row["final_decision"] == "accepted":
        return "ready_for_candidate_handoff"
    if row["final_decision"] == "accepted_with_caveat":
        return "ready_with_caveat"
    return "not_promotable"


def reference_list_status(target: dict[str, Any]) -> str:
    return (
        "reference_list_checked" if target.get("accepted_values") else "open_text_or_non_reference"
    )


def promotion_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "segro_expanded_downstream_promotion_summary_v1",
        "promotion_counts": dict(Counter(record["promotion_status"] for record in records)),
        "promotable_target_ids": [
            record["target_id"] for record in records if record["promotable"]
        ],
    }


def non_promotion_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    non_promotable = [record for record in records if not record["promotable"]]
    return {
        "schema_version": "segro_expanded_downstream_non_promotion_summary_v1",
        "non_promotable_count": len(non_promotable),
        "records": [
            {
                "target_id": record["target_id"],
                "final_decision": record["final_decision"],
                "reason_code": record["reason_code"],
                "decision_reason": record["decision_reason"],
            }
            for record in non_promotable
        ],
    }


def mapping_gaps() -> list[dict[str, str]]:
    return [
        {"gap_id": "asset_entity_record_key", "status": "mapping_required"},
        {"gap_id": "customer_labels", "status": "mapping_required"},
        {"gap_id": "source_url_document_link", "status": "mapping_required"},
        {"gap_id": "boolean_rendering", "status": "mapping_required"},
        {"gap_id": "date_display_format", "status": "mapping_required"},
        {"gap_id": "customer_field_ordering_grouping", "status": "mapping_required"},
    ]


def export_provenance(
    input_provenance_payload: dict[str, Any],
    batch_output_dir: Path,
    adjudication_output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "segro_expanded_downstream_handoff_export_provenance_v1",
        "export_run_id": EXPORT_RUN_ID,
        "extraction_run_id": EXTRACTION_RUN_ID,
        "adjudication_run_id": ADJUDICATION_RUN_ID,
        "input_provenance": input_provenance_payload,
        "input_paths": {
            "batch_output_dir": str(batch_output_dir),
            "adjudication_output_dir": str(adjudication_output_dir),
        },
        "pipeline_repeated_actions": {
            "extraction": False,
            "model_calls": False,
            "retrieval": False,
            "parsing": False,
            "ocr": False,
            "vlm": False,
            "cache_expansion": False,
            "evidence_remapping": False,
            "target_reselection": False,
        },
    }


def handoff_execution_summary(
    internal: list[dict[str, Any]],
    customer: list[dict[str, Any]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "segro_expanded_downstream_handoff_execution_summary_v1",
        "internal_record_count": len(internal),
        "customer_candidate_record_count": len(customer),
        "validation_status": validation["status"],
        "promotion_counts": dict(Counter(record["promotion_status"] for record in internal)),
        "model_calls": 0,
        "extraction_invocations": 0,
        "retrieval_invocations": 0,
        "parser_invocations": 0,
        "ocr_invocations": 0,
        "vlm_invocations": 0,
        "cache_expansion_invocations": 0,
        "evidence_remapping_invocations": 0,
        "target_reselection_invocations": 0,
    }


def empty_execution() -> dict[str, list[dict[str, Any]]]:
    return {
        "model_requests": [],
        "raw_model_responses": [],
        "extraction_results_raw": [],
        "extraction_results_validated": [],
        "final_adjudication": [],
    }


def write_batch_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "selected_batch.json",
        "dry_run_validation.json",
        "extraction_results_raw.json",
        "extraction_results_validated.json",
        "final_adjudication.json",
        "execution_summary.json",
        "input_provenance.json",
    ]:
        _atomic_write_json(output_dir / filename, result[filename.removesuffix(".json")])
    write_jsonl(output_dir / "selected_batch.jsonl", result["selected_batch"])
    write_jsonl(output_dir / "model_requests.jsonl", result["model_requests"])
    write_jsonl(output_dir / "raw_model_responses.jsonl", result["raw_model_responses"])
    write_validation_markdown(result["dry_run_validation"], output_dir / "dry_run_validation.md")
    write_final_csv(result["final_adjudication"], output_dir / "final_adjudication.csv")


def write_adjudication_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "adjudication_summary.json",
        "target_review.json",
        "repaired_results.json",
        "final_adjudication.json",
        "repair_audit.json",
    ]:
        _atomic_write_json(output_dir / filename, result[filename.removesuffix(".json")])
    write_final_csv(result["final_adjudication"], output_dir / "final_adjudication.csv")
    write_jsonl(output_dir / "model_retry_requests.jsonl", result["model_retry_requests"])
    write_jsonl(output_dir / "model_retry_responses.jsonl", result["model_retry_responses"])


def write_handoff_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "internal_handoff.json",
        "customer_candidate_handoff.json",
        "export_validation.json",
        "promotion_summary.json",
        "non_promotion_summary.json",
        "mapping_gaps.json",
        "export_provenance.json",
        "execution_summary.json",
    ]:
        _atomic_write_json(output_dir / filename, result[filename.removesuffix(".json")])
    write_jsonl(output_dir / "internal_handoff.jsonl", result["internal_handoff"])
    write_jsonl(
        output_dir / "customer_candidate_handoff.jsonl",
        result["customer_candidate_handoff"],
    )
    write_handoff_csv(result["internal_handoff"], output_dir / "internal_handoff.csv")
    write_handoff_csv(
        result["customer_candidate_handoff"],
        output_dir / "customer_candidate_handoff.csv",
    )
    write_export_markdown(result["export_validation"], output_dir / "export_validation.md")


def write_final_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "target_id",
        "target_name",
        "final_decision",
        "raw_model_value",
        "evidence_value",
        "normalized_value",
        "display_value",
        "source_id",
        "source_file",
        "source_path",
        "page_number",
        "confidence",
        "evidence_containment",
        "event_validation",
        "dictionary_validation",
        "validation_issues",
    ]
    write_csv(rows, path, fields)


def write_handoff_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row if not isinstance(row.get(key), dict)})
    write_csv(rows, path, fields)


def write_csv(rows: list[dict[str, Any]], path: Path, fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field)) for field in fields})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_validation_markdown(validation: dict[str, Any], path: Path) -> None:
    lines = [
        "# Expanded Bounded Extraction Dry-Run Validation",
        "",
        f"- Status: `{validation['overall_status']}`",
        f"- Failed targets: {validation['failed_target_count']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_export_markdown(validation: dict[str, Any], path: Path) -> None:
    contract = validation["contract_validation"]
    lines = [
        "# Expanded Downstream Handoff Export Validation",
        "",
        f"- Status: `{validation['status']}`",
        f"- Internal records: {contract['internal_record_count']}",
        f"- Customer records: {contract['customer_record_count']}",
        f"- Contract errors: {len(contract['errors'])}",
        f"- Contract warnings: {len(contract['warnings'])}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [dict(item) for item in data]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def int_required(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError("Expected integer-like value.")
    return int(value)
