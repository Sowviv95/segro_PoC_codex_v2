"""Frozen downstream handoff contract validation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from segro_evidence_extraction.vertical_slice import _atomic_write_json

DEFAULT_CONTRACT_SCHEMA_DIR = Path("schemas/downstream_handoff")
DEFAULT_CONTRACT_VALIDATION_OUTPUT_DIR = Path("output/enfield_unit1_export_contract_freeze_v1")
INTERNAL_SCHEMA_VERSION = "segro_internal_evidence_rich_handoff_v1"
CUSTOMER_SCHEMA_VERSION = "segro_customer_candidate_handoff_v1"
CONTRACT_VERSION = "1.0.0"

INTERNAL_ALWAYS_REQUIRED = [
    "schema_version",
    "trace_id",
    "target_id",
    "requirement_id",
    "field_name",
    "final_decision",
    "reason_code",
    "promotable",
    "promotion_status",
    "evidence_containment_status",
    "event_validation_status",
    "dictionary_validation_status",
    "extraction_run_id",
    "adjudication_run_id",
    "input_provenance",
    "model",
    "source_id",
    "source_file",
    "source_path",
    "page_number",
]
INTERNAL_NON_NULL_REQUIRED = [
    "schema_version",
    "trace_id",
    "target_id",
    "requirement_id",
    "field_name",
    "final_decision",
    "reason_code",
    "promotable",
    "promotion_status",
    "evidence_containment_status",
    "event_validation_status",
    "dictionary_validation_status",
    "extraction_run_id",
    "adjudication_run_id",
    "input_provenance",
    "source_id",
    "source_file",
    "page_number",
]
CUSTOMER_ALWAYS_REQUIRED = [
    "schema_version",
    "trace_id",
    "asset_record_key",
    "requirement_id",
    "target_id",
    "field_name",
    "customer_field_label",
    "display_value",
    "status",
    "reason_code",
    "caveat",
    "source_reference",
    "internal_record_identity",
    "transformation_status",
]
CUSTOMER_NON_NULL_REQUIRED = [
    "schema_version",
    "trace_id",
    "requirement_id",
    "target_id",
    "field_name",
    "status",
    "reason_code",
    "source_reference",
    "internal_record_identity",
    "transformation_status",
]
SUPPORTED_DECISIONS = {"accepted", "accepted_with_caveat", "abstained", "rejected"}
SUCCESS_DECISIONS = {"accepted", "accepted_with_caveat"}
NON_PROMOTABLE_DECISIONS = {"abstained", "rejected"}
REJECTED_REASON_CODES = {
    "evidence_containment_failed",
    "response_schema_invalid",
    "wrong_event",
    "wrong_system",
    "component_only",
}
ABSTAINED_REASON_CODES = {
    "dictionary_value_not_supported",
    "insufficient_attribute_evidence",
    "wrong_event",
    "wrong_system",
    "component_only",
}


def run_downstream_handoff_contract_validation_v1(
    *,
    internal_path: Path,
    customer_path: Path,
    schema_dir: Path = DEFAULT_CONTRACT_SCHEMA_DIR,
    output_dir: Path = DEFAULT_CONTRACT_VALIDATION_OUTPUT_DIR,
) -> dict[str, Any]:
    schemas = load_contract_schemas(schema_dir)
    internal_records = read_json_records(internal_path)
    customer_records = read_json_records(customer_path)
    validation = validate_downstream_handoff_contract_v1(
        internal_records=internal_records,
        customer_records=customer_records,
        schemas=schemas,
    )
    result = {
        "contract_validation": validation,
        "contract_freeze_summary": contract_freeze_summary(validation, schemas),
        "compatibility_matrix": compatibility_matrix(schemas),
        "metadata_gap_policy": metadata_gap_policy(schemas),
        "artifact_versioning_policy": artifact_versioning_policy(schemas),
    }
    write_contract_validation_outputs(result, output_dir)
    return result


def load_contract_schemas(schema_dir: Path = DEFAULT_CONTRACT_SCHEMA_DIR) -> dict[str, Any]:
    return {
        "internal_schema": read_json_object(
            schema_dir / "segro_internal_evidence_rich_handoff_v1.schema.json"
        ),
        "customer_schema": read_json_object(
            schema_dir / "segro_customer_candidate_handoff_v1.schema.json"
        ),
        "reason_codes": read_json_object(schema_dir / "reason_codes_v1.json"),
        "manifest": read_json_object(schema_dir / "contract_manifest_v1.json"),
    }


def validate_downstream_handoff_contract_v1(
    *,
    internal_records: list[dict[str, Any]],
    customer_records: list[dict[str, Any]],
    schemas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schemas = schemas or load_contract_schemas()
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    reason_codes = schemas["reason_codes"]["reason_codes"]
    for record in internal_records:
        validate_internal_record(record, reason_codes, errors, warnings)
    for record in customer_records:
        validate_customer_record(record, reason_codes, errors, warnings)
    validate_uniqueness(internal_records, "target_id", "internal", errors)
    validate_uniqueness(internal_records, "trace_id", "internal", errors)
    validate_uniqueness(customer_records, "target_id", "customer", errors)
    validate_uniqueness(customer_records, "trace_id", "customer", errors)
    validate_customer_traceability(internal_records, customer_records, errors)
    return {
        "schema_version": "segro_downstream_handoff_contract_validation_v1",
        "contract_version": CONTRACT_VERSION,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "internal_record_count": len(internal_records),
        "customer_record_count": len(customer_records),
        "decision_counts": dict(Counter(r.get("final_decision") for r in internal_records)),
        "promotion_counts": dict(Counter(r.get("promotion_status") for r in internal_records)),
    }


def validate_internal_record(
    record: dict[str, Any],
    reason_codes: dict[str, Any],
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> None:
    trace = str(record.get("trace_id") or record.get("target_id") or "<unknown>")
    require_fields(record, INTERNAL_ALWAYS_REQUIRED, INTERNAL_NON_NULL_REQUIRED, "internal", errors)
    validate_schema_version(record, INTERNAL_SCHEMA_VERSION, "internal", errors)
    decision = record.get("final_decision")
    reason = record.get("reason_code")
    if decision not in SUPPORTED_DECISIONS:
        add_error(errors, "unsupported_decision", trace, f"Unsupported decision: {decision}")
    validate_reason_compatibility(decision, reason, reason_codes, trace, "internal", errors)
    if decision == "accepted":
        require_success_value(record, trace, errors)
        if (
            record.get("promotable") is not True
            or record.get("promotion_status") != "ready_for_candidate_handoff"
        ):
            add_error(
                errors,
                "invalid_promotion_status",
                trace,
                "Accepted record has invalid promotion status.",
            )
    if decision == "accepted_with_caveat":
        require_success_value(record, trace, errors)
        if (
            record.get("promotable") is not True
            or record.get("promotion_status") != "ready_with_caveat"
        ):
            add_error(
                errors,
                "invalid_promotion_status",
                trace,
                "Caveated record has invalid promotion status.",
            )
        if not (record.get("checkpoint_caveat") or record.get("extraction_caveat")):
            add_error(
                errors,
                "missing_caveat",
                trace,
                "Accepted-with-caveat record lacks an internal caveat.",
            )
    if decision in NON_PROMOTABLE_DECISIONS:
        if (
            record.get("promotable") is not False
            or record.get("promotion_status") != "not_promotable"
        ):
            add_error(
                errors,
                "invalid_promotion_status",
                trace,
                "Non-promotable record has invalid promotion status.",
            )
        if record.get("display_value") is not None or record.get("final_value") is not None:
            add_error(
                errors, "non_promotable_value", trace, "Non-promotable record contains a value."
            )
    if record.get("model") is None:
        add_warning(
            warnings,
            "missing_model_metadata",
            trace,
            "Model metadata unavailable from authoritative upstream artifacts.",
        )
    if record.get("source_path") is None:
        if record.get("source_id") and record.get("source_file"):
            add_warning(
                warnings,
                "missing_source_path",
                trace,
                "Source path unavailable; source_id and source_file retained.",
            )
        else:
            add_error(
                errors,
                "missing_source_reference",
                trace,
                "source_path is null without source_id/source_file.",
            )


def validate_customer_record(
    record: dict[str, Any],
    reason_codes: dict[str, Any],
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> None:
    trace = str(record.get("trace_id") or record.get("target_id") or "<unknown>")
    require_fields(record, CUSTOMER_ALWAYS_REQUIRED, CUSTOMER_NON_NULL_REQUIRED, "customer", errors)
    validate_schema_version(record, CUSTOMER_SCHEMA_VERSION, "customer", errors)
    validate_reason_compatibility(
        record.get("status"), record.get("reason_code"), reason_codes, trace, "customer", errors
    )
    if record.get("status") in SUCCESS_DECISIONS and not record.get("display_value"):
        add_error(
            errors,
            "missing_success_value",
            trace,
            "Successful customer candidate lacks display_value.",
        )
    if record.get("status") in NON_PROMOTABLE_DECISIONS and record.get("display_value") is not None:
        add_error(
            errors,
            "non_promotable_value",
            trace,
            "Non-promotable customer candidate contains a value.",
        )
    if (
        record.get("asset_record_key") is not None
        and record.get("asset_record_key_status") != "provided"
    ):
        add_error(
            errors,
            "fabricated_mapping",
            trace,
            "asset_record_key must be null unless supplied by authoritative config.",
        )
    if record.get("customer_field_label") is not None:
        add_error(
            errors,
            "fabricated_mapping",
            trace,
            "customer_field_label must remain null until mapped.",
        )
    source_reference = record.get("source_reference") or {}
    for field in ["source_id", "source_file", "page_number"]:
        if source_reference.get(field) in (None, ""):
            add_error(
                errors,
                "missing_source_reference",
                trace,
                f"Customer source_reference missing {field}.",
            )
    if source_reference.get("source_url") is not None:
        add_error(errors, "fabricated_mapping", trace, "source_url must remain null until mapped.")
    if source_reference.get("source_url_status") != "mapping_required":
        add_error(
            errors, "missing_mapping_status", trace, "source_url_status must be mapping_required."
        )
    if record.get("asset_record_key") is None:
        add_warning(warnings, "missing_asset_record_key", trace, "Asset key mapping is unresolved.")
    if record.get("customer_field_label") is None:
        add_warning(
            warnings,
            "missing_customer_field_label",
            trace,
            "Customer field label mapping is unresolved.",
        )


def validate_schema_version(
    record: dict[str, Any],
    expected: str,
    record_type: str,
    errors: list[dict[str, str]],
) -> None:
    trace = str(record.get("trace_id") or record.get("target_id") or "<unknown>")
    observed = record.get("schema_version")
    if observed == expected:
        return
    if isinstance(observed, str) and observed.startswith(expected + "."):
        return
    if isinstance(observed, str) and observed.endswith("_v2"):
        add_error(
            errors,
            "unsupported_major_version",
            trace,
            f"{record_type} schema version {observed} is unsupported.",
        )
        return
    add_error(
        errors,
        "schema_version_mismatch",
        trace,
        f"{record_type} schema version {observed} does not match {expected}.",
    )


def validate_reason_compatibility(
    decision: Any,
    reason: Any,
    reason_codes: dict[str, Any],
    trace: str,
    record_type: str,
    errors: list[dict[str, str]],
) -> None:
    if reason not in reason_codes:
        add_error(
            errors,
            "unknown_reason_code",
            trace,
            f"{record_type} reason code {reason!r} is not supported.",
        )
        return
    allowed = set(reason_codes[str(reason)]["allowed_decisions"])
    if decision not in allowed:
        add_error(
            errors,
            "incompatible_decision_reason",
            trace,
            f"{decision!r} is not compatible with reason code {reason!r}.",
        )
    if decision == "abstained" and reason not in ABSTAINED_REASON_CODES:
        add_error(
            errors,
            "incompatible_decision_reason",
            trace,
            "Abstained records require an absence/insufficiency reason.",
        )
    if decision == "rejected" and reason not in REJECTED_REASON_CODES:
        add_error(
            errors,
            "incompatible_decision_reason",
            trace,
            "Rejected records require a validation or extraction-failure reason.",
        )


def require_fields(
    record: dict[str, Any],
    present_fields: list[str],
    non_null_fields: list[str],
    record_type: str,
    errors: list[dict[str, str]],
) -> None:
    trace = str(record.get("trace_id") or record.get("target_id") or "<unknown>")
    for field in present_fields:
        if field not in record:
            add_error(
                errors,
                "missing_required_field",
                trace,
                f"{record_type} record missing field {field}.",
            )
    for field in non_null_fields:
        if record.get(field) in (None, "", {}):
            add_error(
                errors,
                "missing_required_value",
                trace,
                f"{record_type} record field {field} is null or empty.",
            )


def require_success_value(record: dict[str, Any], trace: str, errors: list[dict[str, str]]) -> None:
    if record.get("display_value") in (None, "") or record.get("final_value") is None:
        add_error(
            errors,
            "missing_success_value",
            trace,
            "Successful internal record lacks display/final value.",
        )


def validate_uniqueness(
    records: list[dict[str, Any]],
    field: str,
    record_type: str,
    errors: list[dict[str, str]],
) -> None:
    values = [record.get(field) for record in records]
    for value, count in Counter(values).items():
        if value is not None and count > 1:
            add_error(
                errors,
                f"duplicate_{field}",
                str(value),
                f"Duplicate {field} in {record_type} records.",
            )


def validate_customer_traceability(
    internal_records: list[dict[str, Any]],
    customer_records: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> None:
    identities = {record.get("record_identity") for record in internal_records}
    traces = {record.get("trace_id") for record in internal_records}
    targets = {record.get("target_id") for record in internal_records}
    for record in customer_records:
        trace = str(record.get("trace_id") or "<unknown>")
        if record.get("internal_record_identity") not in identities:
            add_error(
                errors,
                "missing_internal_link",
                trace,
                "Customer record does not link to an internal record.",
            )
        if record.get("trace_id") not in traces:
            add_error(
                errors,
                "missing_trace_link",
                trace,
                "Customer trace_id is absent from internal records.",
            )
        if record.get("target_id") not in targets:
            add_error(
                errors,
                "missing_target_link",
                trace,
                "Customer target_id is absent from internal records.",
            )


def contract_freeze_summary(validation: dict[str, Any], schemas: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "segro_export_contract_freeze_summary_v1",
        "contract_version": CONTRACT_VERSION,
        "validation_status": validation["status"],
        "internal_schema": schemas["manifest"]["internal_schema"],
        "customer_schema": schemas["manifest"]["customer_schema"],
        "reason_code_count": len(schemas["reason_codes"]["reason_codes"]),
        "errors": len(validation["errors"]),
        "warnings": len(validation["warnings"]),
    }


def compatibility_matrix(schemas: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "segro_downstream_handoff_compatibility_matrix_v1",
        "v1_additive_nullable_metadata": "compatible",
        "unknown_reason_codes": "error",
        "unsupported_major_versions": "error",
        "field_removal_or_rename": "requires_new_major_version",
        "manifest": schemas["manifest"],
    }


def metadata_gap_policy(schemas: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "segro_downstream_metadata_gap_policy_v1",
        "model": {
            **schemas["manifest"]["missing_metadata_policy"]["model"],
            "future_source": (
                "future extraction workflows should propagate provider/model metadata "
                "into final adjudication"
            ),
            "backfill_policy": "do not guess or backfill old artifacts",
        },
        "source_path": {
            **schemas["manifest"]["missing_metadata_policy"]["source_path"],
            "future_source": (
                "source registration should propagate registered source path through "
                "adjudication and export"
            ),
            "backfill_policy": "do not invent paths or document URLs",
        },
    }


def artifact_versioning_policy(schemas: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "segro_downstream_artifact_versioning_policy_v1",
        **schemas["manifest"]["artifact_versioning_policy"],
        "runtime_exports_are_reproducible": True,
        "exports_record_required_metadata": [
            "schema_version",
            "contract_version",
            "exporter_version",
            "input_hashes",
            "run_identity",
        ],
    }


def write_contract_validation_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in result.items():
        _atomic_write_json(output_dir / f"{name}.json", payload)
    write_validation_markdown(result["contract_validation"], output_dir / "contract_validation.md")


def write_validation_markdown(validation: dict[str, Any], path: Path) -> None:
    lines = [
        "# Export Contract Freeze V1 Validation",
        "",
        f"- Status: `{validation['status']}`",
        f"- Internal records: {validation['internal_record_count']}",
        f"- Customer records: {validation['customer_record_count']}",
        f"- Errors: {len(validation['errors'])}",
        f"- Warnings: {len(validation['warnings'])}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [dict(item) for item in data]


def read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(data)


def add_error(errors: list[dict[str, str]], code: str, trace: str, message: str) -> None:
    errors.append({"code": code, "trace_id": trace, "message": message})


def add_warning(warnings: list[dict[str, str]], code: str, trace: str, message: str) -> None:
    warnings.append({"code": code, "trace_id": trace, "message": message})
