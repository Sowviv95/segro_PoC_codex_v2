from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from segro_evidence_extraction.downstream_handoff_contract_v1 import (
    CUSTOMER_SCHEMA_VERSION,
    INTERNAL_SCHEMA_VERSION,
    load_contract_schemas,
    run_downstream_handoff_contract_validation_v1,
    validate_downstream_handoff_contract_v1,
)

FIXTURE_DIR = Path("tests/fixtures/downstream_handoff_contract_v1")


def test_valid_internal_and_customer_schema_fixtures() -> None:
    internal = [
        load_fixture("accepted_internal"),
        load_fixture("accepted_with_caveat_internal"),
        load_fixture("abstained_internal"),
        load_fixture("rejected_internal"),
    ]
    customer = [customer_from_internal(record) for record in internal]

    validation = validate_downstream_handoff_contract_v1(
        internal_records=internal,
        customer_records=customer,
    )

    assert validation["status"] == "passed"
    assert validation["internal_record_count"] == 4
    assert validation["customer_record_count"] == 4


def test_successful_records_require_values_and_caveated_records_require_caveat() -> None:
    accepted = load_fixture("accepted_internal")
    accepted["display_value"] = None
    validation = validate_single(accepted)
    assert has_error(validation, "missing_success_value")

    caveated = load_fixture("accepted_with_caveat_internal")
    caveated["checkpoint_caveat"] = None
    caveated["extraction_caveat"] = None
    validation = validate_single(caveated)
    assert has_error(validation, "missing_caveat")


def test_abstained_and_rejected_records_cannot_promote_values() -> None:
    abstained = load_fixture("abstained_internal")
    abstained["display_value"] = "unsupported"
    validation = validate_single(abstained)
    assert has_error(validation, "non_promotable_value")

    rejected = load_fixture("rejected_internal")
    rejected["final_value"] = "unsupported"
    validation = validate_single(rejected)
    assert has_error(validation, "non_promotable_value")


def test_decision_reason_unknown_reason_and_unsupported_major_version_fail() -> None:
    invalid_reason = load_fixture("invalid_decision_reason_internal")
    validation = validate_single(invalid_reason)
    assert has_error(validation, "incompatible_decision_reason")

    unknown = load_fixture("accepted_internal")
    unknown["reason_code"] = "unknown"
    validation = validate_single(unknown)
    assert has_error(validation, "unknown_reason_code")

    unsupported = load_fixture("unsupported_schema_version_internal")
    validation = validate_single(unsupported)
    assert has_error(validation, "unsupported_major_version")


def test_compatible_v1_additive_metadata_is_allowed() -> None:
    internal = load_fixture("accepted_internal")
    internal["schema_version"] = f"{INTERNAL_SCHEMA_VERSION}.1"
    internal["new_nullable_metadata"] = None
    customer = customer_from_internal(internal)
    customer["schema_version"] = f"{CUSTOMER_SCHEMA_VERSION}.1"
    customer["new_nullable_metadata"] = None

    validation = validate_downstream_handoff_contract_v1(
        internal_records=[internal],
        customer_records=[customer],
    )

    assert validation["status"] == "passed"


def test_missing_model_and_source_path_emit_warnings_only() -> None:
    internal = [
        load_fixture("missing_model_warning_internal"),
        load_fixture("missing_source_path_warning_internal"),
    ]
    customer = [customer_from_internal(record) for record in internal]

    validation = validate_downstream_handoff_contract_v1(
        internal_records=internal,
        customer_records=customer,
    )

    assert validation["status"] == "passed"
    warning_codes = {warning["code"] for warning in validation["warnings"]}
    assert "missing_model_metadata" in warning_codes
    assert "missing_source_path" in warning_codes


def test_duplicate_target_and_trace_fail() -> None:
    internal = [load_fixture("accepted_internal"), load_fixture("accepted_internal")]
    customer = [customer_from_internal(record) for record in internal]

    validation = validate_downstream_handoff_contract_v1(
        internal_records=internal,
        customer_records=customer,
    )

    assert has_error(validation, "duplicate_target_id")
    assert has_error(validation, "duplicate_trace_id")


def test_customer_to_internal_traceability_failure() -> None:
    internal = [load_fixture("accepted_internal")]
    customer = [customer_from_internal(internal[0])]
    customer[0]["internal_record_identity"] = "missing"

    validation = validate_downstream_handoff_contract_v1(
        internal_records=internal,
        customer_records=customer,
    )

    assert has_error(validation, "missing_internal_link")


def test_current_enfield_outputs_validate() -> None:
    output_dir = Path("output/test_downstream_handoff_contract_validation")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    try:
        result = run_downstream_handoff_contract_validation_v1(
            internal_path=Path(
                "output/enfield_unit1_downstream_handoff_exporter_v1/internal_handoff.json"
            ),
            customer_path=Path(
                "output/enfield_unit1_downstream_handoff_exporter_v1/"
                "customer_candidate_handoff.json"
            ),
            output_dir=output_dir,
        )

        assert result["contract_validation"]["status"] == "passed"
        assert result["contract_validation"]["internal_record_count"] == 10
        assert (output_dir / "contract_validation.json").exists()
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


def test_schema_artifacts_are_loadable() -> None:
    schemas = load_contract_schemas()

    assert schemas["internal_schema"]["schema_name"] == INTERNAL_SCHEMA_VERSION
    assert schemas["customer_schema"]["schema_name"] == CUSTOMER_SCHEMA_VERSION
    assert "accepted" in schemas["reason_codes"]["reason_codes"]
    assert schemas["manifest"]["contract_version"] == "1.0.0"


def validate_single(record: dict[str, Any]) -> dict[str, Any]:
    return validate_downstream_handoff_contract_v1(
        internal_records=[record],
        customer_records=[customer_from_internal(record)],
    )


def customer_from_internal(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CUSTOMER_SCHEMA_VERSION,
        "trace_id": record["trace_id"],
        "asset_record_key": None,
        "requirement_id": record["requirement_id"],
        "target_id": record["target_id"],
        "field_name": record["field_name"],
        "customer_field_label": None,
        "display_value": record["display_value"] if record["promotable"] else None,
        "status": record["final_decision"],
        "reason_code": record["reason_code"],
        "caveat": record.get("checkpoint_caveat")
        if record["final_decision"].endswith("caveat")
        else None,
        "source_reference": {
            "source_id": record["source_id"],
            "source_file": record["source_file"],
            "page_number": record["page_number"],
            "source_url": None,
            "source_url_status": "mapping_required",
        },
        "internal_record_identity": record["record_identity"],
        "transformation_status": (
            "not_promotable" if not record["promotable"] else "mapping_required"
        ),
    }


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8-sig"))


def has_error(validation: dict[str, Any], code: str) -> bool:
    return any(error["code"] == code for error in validation["errors"])
