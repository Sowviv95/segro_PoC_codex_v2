from __future__ import annotations

import json
import shutil
from pathlib import Path

from segro_evidence_extraction.batch_v2_evidence_readiness import (
    DEFAULT_READINESS_AUDIT_OUTPUT_DIR,
)
from segro_evidence_extraction.batch_v2_false_positive_audit import (
    audit_false_positive,
    build_false_positive_audit,
    run_batch_v2_false_positive_audit,
)


def test_wrong_event_date_is_rejected() -> None:
    audited = audit_false_positive(
        row(
            "construction_date",
            value_shape="date",
            datatype="date",
            excerpt="Construction Methodology Ref: SWMP02, Issue 2 dated 15/03/19.",
        )
    )

    assert audited["false_positive_classification"] == "wrong_event_or_attribute"


def test_correctly_labelled_event_date_is_confirmed() -> None:
    audited = audit_false_positive(
        row(
            "fire_alarm_control_infra_installation_date",
            value_shape="date",
            datatype="date",
            excerpt="Fire alarm control infrastructure installed 13/03/2020.",
        )
    )

    assert audited["false_positive_classification"] == "confirmed_execution_ready"


def test_identifier_requires_label_and_rejects_drawing_number() -> None:
    labelled = audit_false_positive(
        row(
            "pump_model_number",
            value_shape="identifier_or_reference",
            excerpt="Pump model number ABC-123 is recorded in the asset schedule.",
        )
    )
    drawing = audit_false_positive(
        row(
            "pump_model_number",
            value_shape="identifier_or_reference",
            excerpt="Pump shown on Drawing No. 30803-PL-128.",
        )
    )

    assert labelled["false_positive_classification"] == "confirmed_execution_ready"
    assert drawing["false_positive_classification"] == "wrong_event_or_attribute"


def test_count_and_measurement_require_locality() -> None:
    count = audit_false_positive(
        row(
            "parking_space_count",
            value_shape="integer_count",
            datatype="integer",
            excerpt="Parking space count 42 No. spaces.",
        )
    )
    measurement = audit_false_positive(
        row(
            "office_area_value",
            value_shape="decimal_measurement",
            datatype="decimal",
            excerpt="Office area value 1200 m2.",
        )
    )
    unrelated = audit_false_positive(
        row(
            "parking_space_count",
            value_shape="integer_count",
            datatype="integer",
            excerpt="Parking areas are shown on drawing 1234.",
        )
    )

    assert count["false_positive_classification"] == "confirmed_execution_ready"
    assert measurement["false_positive_classification"] == "confirmed_execution_ready"
    assert unrelated["false_positive_classification"] == "component_only"


def test_component_only_and_generic_description_are_rejected() -> None:
    component = audit_false_positive(
        row(
            "loading_door_overhead_component_description",
            value_shape="descriptive_text",
            excerpt="There are loading doors and dock levellers to the warehouse.",
        )
    )
    generic = audit_false_positive(
        row(
            "gas_safety_component_description",
            value_shape="descriptive_text",
            excerpt="The Health and Safety File is provided for construction work.",
        )
    )

    assert component["false_positive_classification"] == "component_only"
    assert generic["false_positive_classification"] == "wrong_event_or_attribute"


def test_dictionary_ambiguity_and_deterministic_reconciliation() -> None:
    rows = [
        row("led_fluorescent_other_count", value_shape="categorical", datatype="integer"),
        row("pump_model_number", value_shape="identifier_or_reference", excerpt="Pump model ABC-1"),
    ]

    first = build_false_positive_audit(rows, target_count=75)
    second = build_false_positive_audit(rows, target_count=75)

    assert first["false_positive_metrics"] == second["false_positive_metrics"]
    assert first["false_positive_metrics"]["audited_execution_ready_count"] == 2
    assert first["false_positive_metrics"]["final_readiness_status"] == "blocked"
    assert first["confirmed_batch_v2_run_readiness"]["extraction_approved"] is False
    assert first["dictionary_ambiguous_targets"][0]["field_name"] == "led_fluorescent_other_count"


def test_real_false_positive_audit_runs_and_writes_outputs() -> None:
    if not DEFAULT_READINESS_AUDIT_OUTPUT_DIR.exists():
        return
    output_root = Path("output/test_batch_v2_false_positive_audit")
    if output_root.exists():
        shutil.rmtree(output_root)
    try:
        result = run_batch_v2_false_positive_audit(output_dir=output_root)

        assert result["false_positive_metrics"]["audited_execution_ready_count"] == 48
        assert (output_root / "target_false_positive_audit.json").exists()
        assert json.loads((output_root / "false_positive_metrics.json").read_text())[
            "audited_execution_ready_count"
        ] == 48
    finally:
        if output_root.exists():
            shutil.rmtree(output_root)


def row(
    field_name: str,
    *,
    value_shape: str,
    datatype: str = "string",
    excerpt: str = "Pump model number ABC-123.",
) -> dict[str, object]:
    return {
        "target_id": f"trg_{field_name}",
        "corrected_selection_rank": 1,
        "field_name": field_name,
        "definition": f"{field_name} target definition",
        "datatype": datatype,
        "unit": None,
        "value_shape": value_shape,
        "audit_source_file": "Building Manual - Part 1 General.pdf",
        "best_cached_page": 1,
        "best_cached_excerpt": excerpt,
        "audit_classification": "execution_ready",
        "likely_hierarchy_section": "Page 1",
    }
