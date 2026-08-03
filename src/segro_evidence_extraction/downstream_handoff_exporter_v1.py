"""Export adjudicated extraction results into downstream handoff records."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from segro_evidence_extraction.downstream_handoff_contract_v1 import (
    validate_downstream_handoff_contract_v1,
)
from segro_evidence_extraction.reduced_bounded_extraction_batch_v1 import (
    DEFAULT_REDUCED_BATCH_OUTPUT_DIR,
)
from segro_evidence_extraction.reduced_extraction_adjudication_v1 import (
    DEFAULT_REDUCED_ADJUDICATION_OUTPUT_DIR,
)
from segro_evidence_extraction.vertical_slice import _atomic_write_json

DEFAULT_HANDOFF_REVIEW_DIR = Path(
    "output/enfield_unit1_downstream_extraction_handoff_review_v1"
)
DEFAULT_HANDOFF_EXPORT_OUTPUT_DIR = Path(
    "output/enfield_unit1_downstream_handoff_exporter_v1"
)

INTERNAL_SCHEMA_VERSION = "segro_internal_evidence_rich_handoff_v1"
CUSTOMER_SCHEMA_VERSION = "segro_customer_candidate_handoff_v1"
EXTRACTION_RUN_ID = "enfield_unit1_reduced_bounded_extraction_batch_v1"
ADJUDICATION_RUN_ID = "enfield_unit1_reduced_extraction_adjudication_v1"
EXPORT_RUN_ID = "enfield_unit1_downstream_handoff_exporter_v1"

CustomerCaveatPolicy = Literal["internal_only", "include"]

MAPPING_GAPS = [
    {
        "gap_id": "customer_field_ordering_grouping",
        "description": "Customer field ordering and grouping are not defined by current artifacts.",
        "status": "mapping_required",
    },
    {
        "gap_id": "customer_labels",
        "description": (
            "Customer labels versus internal field names are not authoritatively mapped."
        ),
        "status": "mapping_required",
    },
    {
        "gap_id": "boolean_rendering",
        "description": "Boolean or Yes/No rendering, including Y versus Yes, is unresolved.",
        "status": "mapping_required",
    },
    {
        "gap_id": "date_display_format",
        "description": (
            "Customer date display format is unresolved; ISO normalized value is retained "
            "internally."
        ),
        "status": "mapping_required",
    },
    {
        "gap_id": "customer_caveat_visibility",
        "description": (
            "Customer visibility of caveats is controlled by exporter policy until final "
            "delivery rules exist."
        ),
        "status": "policy_controlled",
    },
    {
        "gap_id": "asset_entity_record_key",
        "description": "Asset or entity record key is not available in current artifacts.",
        "status": "mapping_required",
    },
    {
        "gap_id": "source_url_document_link",
        "description": (
            "Source URL or document-link mapping for source references is not available."
        ),
        "status": "mapping_required",
    },
]


def run_downstream_handoff_exporter_v1(
    *,
    adjudication_dir: Path = DEFAULT_REDUCED_ADJUDICATION_OUTPUT_DIR,
    reduced_batch_dir: Path = DEFAULT_REDUCED_BATCH_OUTPUT_DIR,
    review_dir: Path = DEFAULT_HANDOFF_REVIEW_DIR,
    output_dir: Path = DEFAULT_HANDOFF_EXPORT_OUTPUT_DIR,
    customer_caveat_policy: CustomerCaveatPolicy = "internal_only",
) -> dict[str, Any]:
    inputs = load_export_inputs(
        adjudication_dir=adjudication_dir,
        reduced_batch_dir=reduced_batch_dir,
        review_dir=review_dir,
    )
    internal_records = build_internal_handoff_records(inputs)
    customer_records = build_customer_candidate_records(
        internal_records,
        customer_caveat_policy=customer_caveat_policy,
    )
    validation = validate_export(
        internal_records=internal_records,
        customer_records=customer_records,
        inputs=inputs,
        customer_caveat_policy=customer_caveat_policy,
    )
    if validation["status"] != "passed":
        raise ValueError(f"Downstream handoff export validation failed: {validation['errors']}")
    result = {
        "internal_handoff": internal_records,
        "customer_candidate_handoff": customer_records,
        "export_validation": validation,
        "promotion_summary": build_promotion_summary(internal_records),
        "non_promotion_summary": build_non_promotion_summary(internal_records),
        "mapping_gaps": MAPPING_GAPS,
        "export_provenance": build_export_provenance(inputs),
        "execution_summary": build_execution_summary(
            internal_records=internal_records,
            customer_records=customer_records,
            validation=validation,
            customer_caveat_policy=customer_caveat_policy,
        ),
    }
    write_export_outputs(result, output_dir)
    return result


def load_export_inputs(
    *,
    adjudication_dir: Path,
    reduced_batch_dir: Path,
    review_dir: Path,
) -> dict[str, Any]:
    paths = {
        "final_adjudication": adjudication_dir / "final_adjudication.json",
        "adjudication_summary": adjudication_dir / "adjudication_summary.json",
        "target_review": adjudication_dir / "target_review.json",
        "repair_audit": adjudication_dir / "repair_audit.json",
        "reduced_batch": reduced_batch_dir / "reduced_batch.json",
        "input_provenance": reduced_batch_dir / "input_provenance.json",
        "proposed_internal_schema": review_dir / "proposed_internal_schema.json",
        "proposed_customer_schema": review_dir / "proposed_customer_schema.json",
        "promotion_manifest": review_dir / "promotion_manifest.json",
        "non_promotion_manifest": review_dir / "non_promotion_manifest.json",
        "reason_code_dictionary": review_dir / "reason_code_dictionary.json",
        "field_classification": review_dir / "field_classification.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Downstream handoff exporter inputs missing: {missing}")
    return {
        "paths": paths,
        "final_adjudication": read_json_list(paths["final_adjudication"]),
        "adjudication_summary": read_json_object(paths["adjudication_summary"]),
        "target_review": read_json_list(paths["target_review"]),
        "repair_audit": read_json_list(paths["repair_audit"]),
        "reduced_batch": read_json_list(paths["reduced_batch"]),
        "input_provenance": read_json_object(paths["input_provenance"]),
        "proposed_internal_schema": read_json_object(paths["proposed_internal_schema"]),
        "proposed_customer_schema": read_json_object(paths["proposed_customer_schema"]),
        "promotion_manifest": read_json_list(paths["promotion_manifest"]),
        "non_promotion_manifest": read_json_list(paths["non_promotion_manifest"]),
        "reason_code_dictionary": read_json_object(paths["reason_code_dictionary"]),
        "field_classification": read_json_object(paths["field_classification"]),
    }


def build_internal_handoff_records(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    batch_by_id = {item["target_id"]: item for item in inputs["reduced_batch"]}
    review_by_id = {item["target_id"]: item for item in inputs["target_review"]}
    repair_by_id = {item["target_id"]: item for item in inputs["repair_audit"]}
    manifest_by_id = {
        item["target_id"]: item
        for item in [*inputs["promotion_manifest"], *inputs["non_promotion_manifest"]]
    }
    records = []
    for row in inputs["final_adjudication"]:
        target_id = str(row["target_id"])
        batch_item = batch_by_id[target_id]
        target = batch_item["target"]
        bundle = batch_item["evidence_bundle"]
        payload = batch_item["canonical_evidence_payload"]
        span = payload.get("span") or {}
        manifest = manifest_by_id.get(target_id, {})
        reason_code = reason_code_for_record(row, manifest)
        decision_reason = decision_reason_for_record(row, manifest)
        final_value = final_value_for_record(row, target)
        validation = validation_statuses(row, review_by_id.get(target_id))
        record_identity = f"{EXTRACTION_RUN_ID}:{target_id}"
        component_context = bundle.get("component_context") or {}
        records.append(
            {
                "schema_version": INTERNAL_SCHEMA_VERSION,
                "trace_id": target_id,
                "target_id": target_id,
                "requirement_id": target.get("requirement_id"),
                "field_name": target.get("expected_field"),
                "target_name": row.get("target_name"),
                "component": {
                    "component_type": target.get("component_type"),
                    "component_subtype": target.get("component_subtype"),
                    "component_system_identity": component_context.get(
                        "component_system_identity"
                    ),
                },
                "attribute": component_context.get("requested_attribute"),
                "record_identity": record_identity,
                "raw_model_value": row.get("raw_model_value"),
                "evidence_value": row.get("evidence_value"),
                "normalized_value": row.get("normalized_value"),
                "display_value": row.get("display_value") if is_promotable(row) else None,
                "final_value": final_value,
                "final_decision": row.get("final_decision"),
                "reason_code": reason_code,
                "decision_reason": decision_reason,
                "checkpoint_status": row.get("checkpoint_status"),
                "checkpoint_caveat": row.get("checkpoint_caveat"),
                "extraction_caveat": row.get("model_caveat"),
                "adjudication_caveat": row.get("decision_change_reason"),
                "confidence": row.get("confidence"),
                "value_shape": payload.get("evidence_family"),
                "datatype": target.get("expected_data_type"),
                "reference_list_status": reference_list_status(target),
                "source_id": row.get("source_id") or span.get("source_id"),
                "source_file": row.get("source_file") or span.get("source_file"),
                "source_path": None,
                "page_number": row.get("page_number") or span.get("page_number"),
                "evidence_quote": row.get("evidence_value"),
                "evidence_span": {
                    "span_id": span.get("span_id"),
                    "source_id": span.get("source_id"),
                    "source_file": span.get("source_file"),
                    "page_number": span.get("page_number"),
                    "start_char": span.get("start_char"),
                    "end_char": span.get("end_char"),
                    "bounded_text": payload.get("bounded_text"),
                },
                "evidence_span_id": span.get("span_id"),
                "evidence_char_start": span.get("start_char"),
                "evidence_char_end": span.get("end_char"),
                "evidence_containment_status": validation["evidence_containment_status"],
                "event_validation_status": validation["event_validation_status"],
                "dictionary_validation_status": validation["dictionary_validation_status"],
                "typed_normalization_status": validation["typed_normalization_status"],
                "validation_issues": row.get("validation_issues") or [],
                "model": model_for_record(row),
                "extraction_run_id": EXTRACTION_RUN_ID,
                "adjudication_run_id": ADJUDICATION_RUN_ID,
                "export_run_id": EXPORT_RUN_ID,
                "deterministic_repair_applied": bool(row.get("deterministic_repair_applied")),
                "model_retry_applied": bool(row.get("model_retry_occurred")),
                "input_provenance": inputs["input_provenance"],
                "repair_audit": repair_by_id.get(target_id),
                "target_review": review_by_id.get(target_id),
                "promotable": is_promotable(row),
                "promotion_status": promotion_status_for_record(row),
            }
        )
    return records


def build_customer_candidate_records(
    internal_records: list[dict[str, Any]],
    *,
    customer_caveat_policy: CustomerCaveatPolicy,
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": CUSTOMER_SCHEMA_VERSION,
            "trace_id": record["trace_id"],
            "asset_record_key": None,
            "asset_record_key_status": "mapping_required",
            "requirement_id": record["requirement_id"],
            "target_id": record["target_id"],
            "field_name": record["field_name"],
            "customer_field_label": None,
            "customer_field_label_status": "mapping_required",
            "display_value": record["display_value"] if record["promotable"] else None,
            "status": record["final_decision"],
            "reason_code": record["reason_code"],
            "caveat": customer_caveat(record, customer_caveat_policy),
            "source_reference": {
                "source_id": record["source_id"],
                "source_file": record["source_file"],
                "page_number": record["page_number"],
                "source_url": None,
                "source_url_status": "mapping_required",
            },
            "internal_record_identity": record["record_identity"],
            "transformation_status": transformation_status(record),
            "boolean_rendering_status": boolean_rendering_status(record),
            "date_rendering_status": date_rendering_status(record),
        }
        for record in internal_records
    ]


def validate_export(
    *,
    internal_records: list[dict[str, Any]],
    customer_records: list[dict[str, Any]],
    inputs: dict[str, Any],
    customer_caveat_policy: CustomerCaveatPolicy,
) -> dict[str, Any]:
    errors: list[str] = []
    target_ids = [record["target_id"] for record in internal_records]
    final_ids = [record["target_id"] for record in inputs["final_adjudication"]]
    promotion_targets = [item["target_id"] for item in inputs["promotion_manifest"]]
    non_promotion_targets = [item["target_id"] for item in inputs["non_promotion_manifest"]]
    contract_validation = validate_downstream_handoff_contract_v1(
        internal_records=internal_records,
        customer_records=customer_records,
    )
    errors.extend(
        f"{error['trace_id']}: {error['code']}: {error['message']}"
        for error in contract_validation["errors"]
    )

    if target_ids != final_ids:
        errors.append("Internal handoff ordering does not match final adjudication ordering.")
    if len(internal_records) != len(final_ids):
        errors.append(
            "Internal handoff does not contain exactly one record per adjudicated target."
        )
    if len(set(target_ids)) != len(target_ids):
        errors.append("Duplicate target IDs found in internal handoff.")
    if [record["target_id"] for record in customer_records] != target_ids:
        errors.append("Customer handoff ordering does not match internal handoff ordering.")
    counts = Counter(record["promotion_status"] for record in internal_records)
    expected_counts = {
        "ready_for_candidate_handoff": len(
            [
                record
                for record in internal_records
                if record["final_decision"] == "accepted"
            ]
        ),
        "ready_with_caveat": len(
            [
                record
                for record in internal_records
                if record["final_decision"] == "accepted_with_caveat"
            ]
        ),
        "not_promotable": len(
            [
                record
                for record in internal_records
                if record["final_decision"] in {"abstained", "rejected"}
            ]
        ),
    }
    for key, expected in expected_counts.items():
        if counts.get(key, 0) != expected:
            errors.append(f"Promotion count mismatch for {key}: {counts.get(key, 0)} != {expected}")
    observed_promotion_targets = [
        record["target_id"] for record in internal_records if record["promotable"]
    ]
    if observed_promotion_targets != promotion_targets:
        errors.append("Promotable targets do not reconcile to promotion manifest.")
    if [
        record["target_id"]
        for record in internal_records
        if not record["promotable"]
    ] != non_promotion_targets:
        errors.append("Non-promotable targets do not reconcile to non-promotion manifest.")
    for key in ["asset_record_key", "customer_field_label"]:
        if any(record[key] is not None for record in customer_records):
            errors.append(f"Customer mapping was fabricated for {key}.")
    if any(record["source_reference"]["source_url"] is not None for record in customer_records):
        errors.append("Source URL was fabricated.")
    if customer_caveat_policy == "internal_only" and any(
        record["caveat"] is not None for record in customer_records
    ):
        errors.append("Customer caveat was exposed despite internal_only policy.")
    return {
        "schema_version": "segro_downstream_handoff_export_validation_v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": [
            f"{warning['trace_id']}: {warning['code']}: {warning['message']}"
            for warning in contract_validation["warnings"]
        ],
        "internal_record_count": len(internal_records),
        "customer_record_count": len(customer_records),
        "promotion_counts": dict(counts),
        "customer_caveat_policy": customer_caveat_policy,
        "input_artifact_hashes": artifact_hashes(inputs),
        "contract_validation": contract_validation,
    }


def reason_code_for_record(row: dict[str, Any], manifest: dict[str, Any]) -> str:
    decision = str(row.get("final_decision"))
    if decision == "accepted":
        return "accepted"
    if decision == "accepted_with_caveat":
        return "accepted_with_caveat"
    reason_code = manifest.get("reason_code")
    if reason_code:
        return str(reason_code)
    if decision == "rejected":
        return "evidence_containment_failed"
    return "insufficient_attribute_evidence"


def decision_reason_for_record(row: dict[str, Any], manifest: dict[str, Any]) -> str:
    if manifest.get("exact_reason"):
        return str(manifest["exact_reason"])
    if row.get("decision_change_reason"):
        return str(row["decision_change_reason"])
    if row.get("checkpoint_caveat"):
        return str(row["checkpoint_caveat"])
    if row.get("model_caveat"):
        return str(row["model_caveat"])
    return "Value accepted by final adjudication."


def final_value_for_record(row: dict[str, Any], target: dict[str, Any]) -> Any:
    if not is_promotable(row):
        return None
    if target.get("expected_data_type") == "string":
        return row.get("display_value")
    return row.get("normalized_value")


def validation_statuses(row: dict[str, Any], review: dict[str, Any] | None) -> dict[str, str]:
    issues = row.get("validation_issues") or []
    return {
        "evidence_containment_status": row.get("evidence_containment")
        or (review or {}).get("evidence_containment")
        or ("failed" if any("contained" in str(issue) for issue in issues) else "passed"),
        "event_validation_status": row.get("event_validation")
        or (review or {}).get("event_validation")
        or "passed",
        "dictionary_validation_status": row.get("dictionary_validation")
        or (review or {}).get("dictionary_validation")
        or "passed",
        "typed_normalization_status": row.get("typed_normalization")
        or (review or {}).get("typed_normalization")
        or "passed",
    }


def is_promotable(row: dict[str, Any]) -> bool:
    return row.get("final_decision") in {"accepted", "accepted_with_caveat"}


def promotion_status_for_record(row: dict[str, Any]) -> str:
    if row.get("final_decision") == "accepted":
        return "ready_for_candidate_handoff"
    if row.get("final_decision") == "accepted_with_caveat":
        return "ready_with_caveat"
    return "not_promotable"


def reference_list_status(target: dict[str, Any]) -> str:
    if target.get("accepted_values"):
        return "reference_list_checked"
    return "open_text_or_non_reference"


def model_for_record(row: dict[str, Any]) -> str | None:
    usage = row.get("model_usage")
    if isinstance(usage, dict) and usage.get("model"):
        return str(usage["model"])
    return None


def customer_caveat(
    record: dict[str, Any],
    customer_caveat_policy: CustomerCaveatPolicy,
) -> str | None:
    if customer_caveat_policy == "internal_only":
        return None
    return record.get("checkpoint_caveat") or record.get("extraction_caveat")


def transformation_status(record: dict[str, Any]) -> str:
    if not record["promotable"]:
        return "not_promotable"
    if record["datatype"] in {"date", "boolean"} or record["display_value"] == "Y":
        return "mapping_required"
    return "candidate_ready"


def boolean_rendering_status(record: dict[str, Any]) -> str | None:
    if record["display_value"] in {"Y", "N"}:
        return "mapping_required"
    return None


def date_rendering_status(record: dict[str, Any]) -> str | None:
    if record["datatype"] == "date":
        return "mapping_required"
    return None


def build_promotion_summary(internal_records: list[dict[str, Any]]) -> dict[str, Any]:
    promotable = [record for record in internal_records if record["promotable"]]
    return {
        "schema_version": "segro_downstream_promotion_summary_v1",
        "promotion_counts": dict(
            Counter(record["promotion_status"] for record in internal_records)
        ),
        "promotable_count": len(promotable),
        "promotable_target_ids": [record["target_id"] for record in promotable],
        "ready_for_candidate_handoff": [
            record["target_id"]
            for record in internal_records
            if record["promotion_status"] == "ready_for_candidate_handoff"
        ],
        "ready_with_caveat": [
            record["target_id"]
            for record in internal_records
            if record["promotion_status"] == "ready_with_caveat"
        ],
    }


def build_non_promotion_summary(internal_records: list[dict[str, Any]]) -> dict[str, Any]:
    records = [record for record in internal_records if not record["promotable"]]
    return {
        "schema_version": "segro_downstream_non_promotion_summary_v1",
        "non_promotable_count": len(records),
        "non_promotable_target_ids": [record["target_id"] for record in records],
        "reason_code_counts": dict(Counter(record["reason_code"] for record in records)),
        "records": [
            {
                "target_id": record["target_id"],
                "target_name": record["target_name"],
                "final_decision": record["final_decision"],
                "reason_code": record["reason_code"],
                "decision_reason": record["decision_reason"],
                "promotion_status": record["promotion_status"],
            }
            for record in records
        ],
    }


def build_export_provenance(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "segro_downstream_handoff_export_provenance_v1",
        "export_run_id": EXPORT_RUN_ID,
        "extraction_run_id": EXTRACTION_RUN_ID,
        "adjudication_run_id": ADJUDICATION_RUN_ID,
        "input_paths": {key: str(path) for key, path in inputs["paths"].items()},
        "input_artifact_hashes": artifact_hashes(inputs),
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


def build_execution_summary(
    *,
    internal_records: list[dict[str, Any]],
    customer_records: list[dict[str, Any]],
    validation: dict[str, Any],
    customer_caveat_policy: CustomerCaveatPolicy,
) -> dict[str, Any]:
    return {
        "schema_version": "segro_downstream_handoff_export_execution_summary_v1",
        "internal_record_count": len(internal_records),
        "customer_candidate_record_count": len(customer_records),
        "promotion_counts": dict(
            Counter(record["promotion_status"] for record in internal_records)
        ),
        "final_decision_counts": dict(
            Counter(record["final_decision"] for record in internal_records)
        ),
        "customer_caveat_policy": customer_caveat_policy,
        "validation_status": validation["status"],
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


def artifact_hashes(inputs: dict[str, Any]) -> dict[str, str]:
    return {key: sha256_file(path) for key, path in inputs["paths"].items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_export_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_outputs = {
        "internal_handoff.json": result["internal_handoff"],
        "customer_candidate_handoff.json": result["customer_candidate_handoff"],
        "export_validation.json": result["export_validation"],
        "promotion_summary.json": result["promotion_summary"],
        "non_promotion_summary.json": result["non_promotion_summary"],
        "mapping_gaps.json": result["mapping_gaps"],
        "export_provenance.json": result["export_provenance"],
        "execution_summary.json": result["execution_summary"],
    }
    for filename, payload in json_outputs.items():
        _atomic_write_json(output_dir / filename, payload)
    write_jsonl(output_dir / "internal_handoff.jsonl", result["internal_handoff"])
    write_jsonl(
        output_dir / "customer_candidate_handoff.jsonl",
        result["customer_candidate_handoff"],
    )
    write_internal_csv(result["internal_handoff"], output_dir / "internal_handoff.csv")
    write_customer_csv(
        result["customer_candidate_handoff"],
        output_dir / "customer_candidate_handoff.csv",
    )
    write_validation_markdown(result["export_validation"], output_dir / "export_validation.md")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_internal_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "schema_version",
        "trace_id",
        "target_id",
        "requirement_id",
        "field_name",
        "target_name",
        "attribute",
        "record_identity",
        "raw_model_value",
        "evidence_value",
        "normalized_value",
        "display_value",
        "final_value",
        "final_decision",
        "reason_code",
        "decision_reason",
        "checkpoint_status",
        "checkpoint_caveat",
        "extraction_caveat",
        "adjudication_caveat",
        "confidence",
        "datatype",
        "reference_list_status",
        "source_id",
        "source_file",
        "source_path",
        "page_number",
        "evidence_span_id",
        "evidence_char_start",
        "evidence_char_end",
        "evidence_containment_status",
        "event_validation_status",
        "dictionary_validation_status",
        "promotable",
        "promotion_status",
    ]
    write_csv(rows, path, fields)


def write_customer_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "schema_version",
        "trace_id",
        "asset_record_key",
        "asset_record_key_status",
        "requirement_id",
        "target_id",
        "field_name",
        "customer_field_label",
        "customer_field_label_status",
        "display_value",
        "status",
        "reason_code",
        "caveat",
        "internal_record_identity",
        "transformation_status",
        "boolean_rendering_status",
        "date_rendering_status",
    ]
    write_csv(rows, path, fields)


def write_csv(rows: list[dict[str, Any]], path: Path, fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field)) for field in fields})


def write_validation_markdown(validation: dict[str, Any], path: Path) -> None:
    lines = [
        "# Downstream Handoff Export Validation",
        "",
        f"- Status: `{validation['status']}`",
        f"- Internal records: {validation['internal_record_count']}",
        f"- Customer candidate records: {validation['customer_record_count']}",
        f"- Customer caveat policy: `{validation['customer_caveat_policy']}`",
        f"- Promotion counts: `{json.dumps(validation['promotion_counts'], sort_keys=True)}`",
        "",
        "## Errors",
    ]
    if validation["errors"]:
        lines.extend(f"- {error}" for error in validation["errors"])
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Warnings")
    if validation["warnings"]:
        lines.extend(f"- {warning}" for warning in validation["warnings"])
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(data)


def read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [dict(item) for item in data]


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)
