from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import cast

from segro_evidence_extraction.downstream_handoff_exporter_v1 import (
    DEFAULT_HANDOFF_REVIEW_DIR,
    DEFAULT_REDUCED_ADJUDICATION_OUTPUT_DIR,
    DEFAULT_REDUCED_BATCH_OUTPUT_DIR,
    CustomerCaveatPolicy,
    build_customer_candidate_records,
    build_internal_handoff_records,
    load_export_inputs,
    run_downstream_handoff_exporter_v1,
    validate_export,
)

EXPECTED_PROMOTABLE = [
    "trg_92469ca7eaab2c31",
    "trg_6720c2edd5e947d6",
    "trg_0f66487177c685e6",
    "trg_93d2e2b38d1de7b2",
    "trg_50fedb4c249d8fe4",
    "trg_ca18dce11deff8cf",
]
EXPECTED_NON_PROMOTABLE = [
    "trg_33975e86ffde5524",
    "trg_7426d9722059de76",
    "trg_12690fb418279750",
    "trg_adb3dfa03b0c7cf3",
]


def test_exporter_writes_all_formats_and_expected_counts() -> None:
    output = Path("output/test_downstream_handoff_exporter_counts")
    try:
        result = run_reduced_export(output, keep_output=True)
        internal = result["internal_handoff"]
        customer = result["customer_candidate_handoff"]
        by_id = {record["target_id"]: record for record in internal}

        assert len(internal) == 10
        assert len(customer) == 10
        assert [record["target_id"] for record in internal if record["promotable"]] == (
            EXPECTED_PROMOTABLE
        )
        assert [record["target_id"] for record in internal if not record["promotable"]] == (
            EXPECTED_NON_PROMOTABLE
        )
        assert result["execution_summary"]["promotion_counts"] == {
            "not_promotable": 4,
            "ready_for_candidate_handoff": 5,
            "ready_with_caveat": 1,
        }
        assert by_id["trg_92469ca7eaab2c31"]["final_value"] == "43830"
        assert by_id["trg_ca18dce11deff8cf"]["promotion_status"] == "ready_with_caveat"
        assert by_id["trg_ca18dce11deff8cf"]["reason_code"] == "accepted_with_caveat"
        assert by_id["trg_33975e86ffde5524"]["reason_code"] == (
            "dictionary_value_not_supported"
        )
        assert by_id["trg_adb3dfa03b0c7cf3"]["reason_code"] == "evidence_containment_failed"

        for filename in [
            "internal_handoff.json",
            "internal_handoff.jsonl",
            "internal_handoff.csv",
            "customer_candidate_handoff.json",
            "customer_candidate_handoff.jsonl",
            "customer_candidate_handoff.csv",
            "export_validation.json",
            "export_validation.md",
            "promotion_summary.json",
            "non_promotion_summary.json",
            "mapping_gaps.json",
            "export_provenance.json",
            "execution_summary.json",
        ]:
            assert (output / filename).exists()
        assert len(read_jsonl(output / "internal_handoff.jsonl")) == len(
            json.loads((output / "internal_handoff.json").read_text(encoding="utf-8"))
        )
        with (output / "customer_candidate_handoff.csv").open(encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == 10
    finally:
        if output.exists():
            shutil.rmtree(output)


def test_customer_candidate_rules_internal_only_policy() -> None:
    result = run_reduced_export(
        Path("output/test_downstream_handoff_exporter_internal_only"),
        customer_caveat_policy="internal_only",
    )
    customer_by_id = {
        record["target_id"]: record for record in result["customer_candidate_handoff"]
    }

    accepted = customer_by_id["trg_6720c2edd5e947d6"]
    assert accepted["display_value"] == "Y"
    assert accepted["boolean_rendering_status"] == "mapping_required"
    assert accepted["transformation_status"] == "mapping_required"

    caveated = customer_by_id["trg_ca18dce11deff8cf"]
    assert caveated["status"] == "accepted_with_caveat"
    assert caveated["caveat"] is None
    assert caveated["date_rendering_status"] == "mapping_required"

    for target_id in EXPECTED_NON_PROMOTABLE:
        assert customer_by_id[target_id]["display_value"] is None
        assert customer_by_id[target_id]["transformation_status"] == "not_promotable"
    assert all(record["asset_record_key"] is None for record in customer_by_id.values())
    assert all(record["customer_field_label"] is None for record in customer_by_id.values())
    assert all(
        record["source_reference"]["source_url"] is None for record in customer_by_id.values()
    )


def test_customer_candidate_include_caveat_policy() -> None:
    result = run_reduced_export(
        Path("output/test_downstream_handoff_exporter_include"),
        customer_caveat_policy="include",
    )
    customer_by_id = {
        record["target_id"]: record for record in result["customer_candidate_handoff"]
    }

    assert customer_by_id["trg_ca18dce11deff8cf"]["caveat"]
    assert customer_by_id["trg_93d2e2b38d1de7b2"]["caveat"] is None


def test_validation_rejects_unknown_reason_missing_provenance_and_duplicates() -> None:
    inputs = load_default_inputs()
    internal = build_internal_handoff_records(inputs)
    customer = build_customer_candidate_records(internal, customer_caveat_policy="internal_only")

    internal[0]["reason_code"] = "unknown_code"
    validation = validate_export(
        internal_records=internal,
        customer_records=customer,
        inputs=inputs,
        customer_caveat_policy="internal_only",
    )
    assert validation["status"] == "failed"
    assert any("unknown_reason_code" in error for error in validation["errors"])

    internal = build_internal_handoff_records(inputs)
    customer = build_customer_candidate_records(internal, customer_caveat_policy="internal_only")
    internal[0]["source_id"] = None
    validation = validate_export(
        internal_records=internal,
        customer_records=customer,
        inputs=inputs,
        customer_caveat_policy="internal_only",
    )
    assert validation["status"] == "failed"
    assert any(
        "missing_required_value" in error and "source_id" in error
        for error in validation["errors"]
    )

    internal = build_internal_handoff_records(inputs)
    customer = build_customer_candidate_records(internal, customer_caveat_policy="internal_only")
    internal.append(dict(internal[-1]))
    customer.append(dict(customer[-1]))
    validation = validate_export(
        internal_records=internal,
        customer_records=customer,
        inputs=inputs,
        customer_caveat_policy="internal_only",
    )
    assert validation["status"] == "failed"
    assert any("Duplicate target IDs" in error for error in validation["errors"])


def test_exporter_records_no_pipeline_invocations() -> None:
    result = run_reduced_export(Path("output/test_downstream_handoff_exporter_invocations"))
    summary = result["execution_summary"]
    provenance = result["export_provenance"]["pipeline_repeated_actions"]

    assert summary["model_calls"] == 0
    assert summary["extraction_invocations"] == 0
    assert summary["retrieval_invocations"] == 0
    assert summary["parser_invocations"] == 0
    assert summary["ocr_invocations"] == 0
    assert summary["vlm_invocations"] == 0
    assert summary["cache_expansion_invocations"] == 0
    assert summary["evidence_remapping_invocations"] == 0
    assert summary["target_reselection_invocations"] == 0
    assert not any(provenance.values())


def test_deterministic_ordering_and_trace_ids() -> None:
    first = run_reduced_export(Path("output/test_downstream_handoff_exporter_first"))
    second = run_reduced_export(Path("output/test_downstream_handoff_exporter_second"))

    assert first["internal_handoff"] == second["internal_handoff"]
    assert first["customer_candidate_handoff"] == second["customer_candidate_handoff"]
    assert [record["trace_id"] for record in first["internal_handoff"]] == [
        record["target_id"] for record in first["internal_handoff"]
    ]


def run_reduced_export(
    output_dir: Path,
    *,
    customer_caveat_policy: str = "internal_only",
    keep_output: bool = False,
) -> dict:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    try:
        return run_downstream_handoff_exporter_v1(
            output_dir=output_dir,
            customer_caveat_policy=cast(CustomerCaveatPolicy, customer_caveat_policy),
        )
    finally:
        if output_dir.exists() and not keep_output:
            shutil.rmtree(output_dir)


def load_default_inputs() -> dict:
    return load_export_inputs(
        adjudication_dir=DEFAULT_REDUCED_ADJUDICATION_OUTPUT_DIR,
        reduced_batch_dir=DEFAULT_REDUCED_BATCH_OUTPUT_DIR,
        review_dir=DEFAULT_HANDOFF_REVIEW_DIR,
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
