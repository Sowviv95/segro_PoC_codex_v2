from __future__ import annotations

import json
import shutil
from pathlib import Path

from segro_evidence_extraction.batch_selection_strategy import (
    DEFAULT_ADJUDICATION_DIR,
    DEFAULT_BATCH_V1_DIR,
    DEFAULT_DICTIONARY_JSONL,
    DEFAULT_HIERARCHY_PATH,
    DEFAULT_SOURCE_MANIFEST,
    READY_CLASSES,
    build_batch_v2_selection,
    load_selection_inputs,
    run_batch_v2_selection_pack,
    score_target,
)
from segro_evidence_extraction.models.target import TargetSpecification


def test_prior_batch_visual_missing_and_ambiguous_candidates_are_not_selected() -> None:
    targets = [
        target("prior_target", row=1),
        target("visual_target", row=2, likely=["drawing"]),
        target("mri_reference", row=3),
        target("unclear_description", row=4, definition="Unclear target - x"),
        target("loading_door_count", row=5, datatype="integer"),
    ]
    inputs = synthetic_inputs(targets, prior_ids={"prior_target"})
    result = build_batch_v2_selection(inputs, target_count=1)
    by_id = {item["target_id"]: item for item in result["candidate_scores"]}

    assert by_id["prior_target"]["readiness_class"] == "exclude_prior_batch"
    assert by_id["visual_target"]["readiness_class"] == "defer_visual"
    assert by_id["mri_reference"]["readiness_class"] == "defer_missing_source"
    assert by_id["unclear_description"]["readiness_class"] == "defer_dictionary_clarification"
    assert result["selected_targets"][0]["target_id"] == "loading_door_count"


def test_scoring_rewards_counts_dates_identifiers_and_penalizes_bad_families() -> None:
    count_target = target("loading_door_count", row=1, datatype="integer")
    broad_target = target("general_description", row=2, definition="General component description")
    date_target = target("fire_certificate_issue_date", row=3, datatype="date")
    unit_bad = target("wall_description", row=4, unit="m")
    metadata_conflict = target(
        "pump_manufacturer",
        row=5,
        datatype="integer",
        definition="No. Pumps - Manufacturer - Number of pumps",
        field_label="Manufacturer",
    )
    model_number = target(
        "pump_model_number",
        row=6,
        datatype="string",
        definition="Pump model number or reference",
        field_label="Model Number / Reference",
    )
    outcomes = {"Component|integer_count|loading_door": {"component_present_attribute_absent": 2}}
    inputs = synthetic_inputs([count_target, broad_target, date_target, unit_bad])

    count_score = score_target(
        target=count_target,
        inputs=inputs,
        family_outcomes={},
    )
    broad_score = score_target(target=broad_target, inputs=inputs, family_outcomes={})
    date_score = score_target(target=date_target, inputs=inputs, family_outcomes={})
    unit_score = score_target(target=unit_bad, inputs=inputs, family_outcomes={})
    conflict_score = score_target(target=metadata_conflict, inputs=inputs, family_outcomes={})
    model_number_score = score_target(target=model_number, inputs=inputs, family_outcomes={})
    penalized = score_target(
        target=count_target,
        inputs=inputs,
        family_outcomes={
            "Component|integer_count|loading_door": counter_dict(
                outcomes["Component|integer_count|loading_door"]
            )
        },
    )

    assert count_score["total_score"] > broad_score["total_score"]
    assert date_score["readiness_class"] == "ready_identifier_or_date_extraction"
    assert unit_score["readiness_class"] == "defer_dictionary_clarification"
    assert conflict_score["readiness_class"] == "defer_dictionary_clarification"
    assert model_number_score["value_shape"] == "identifier_or_reference"
    assert model_number_score["readiness_class"] == "ready_identifier_or_date_extraction"
    assert penalized["readiness_class"] == "defer_low_evidence_readiness"


def test_exactly_75_selected_when_ready_candidates_exist_and_selection_is_deterministic() -> None:
    domains = ["Component", "Property", "Size", "Energy"]
    targets = [
        target(
            f"loading_door_{index}_count",
            row=index,
            datatype="integer",
            domain=domains[index % len(domains)],
        )
        for index in range(1, 91)
    ]
    inputs = synthetic_inputs(targets)

    first = build_batch_v2_selection(inputs, target_count=75)
    second = build_batch_v2_selection(inputs, target_count=75)

    assert len(first["selected_targets"]) == 75
    assert len({item["target_id"] for item in first["selected_targets"]}) == 75
    assert first["selected_targets"] == second["selected_targets"]
    assert first["selection_trace"][0]["total_score"] == first["selected_targets"][0]["total_score"]
    assert not first["selection_metrics"]["diversity_control_results"]["domain_cap_exception"]


def test_blocked_readiness_when_fewer_than_75_ready_candidates_exist() -> None:
    inputs = synthetic_inputs(
        [target(f"door_{index}_count", row=index, datatype="integer") for index in range(1, 10)]
    )
    result = build_batch_v2_selection(inputs, target_count=75)

    assert result["batch_v2_run_readiness"]["overall_status"] == "blocked"
    assert result["batch_v2_run_readiness"]["additional_ready_targets_needed"] == 66


def test_real_batch_v2_selection_pack_reconciles_and_is_stable() -> None:
    if not all(
        path.exists()
        for path in [
            DEFAULT_DICTIONARY_JSONL,
            DEFAULT_BATCH_V1_DIR,
            DEFAULT_ADJUDICATION_DIR,
            DEFAULT_SOURCE_MANIFEST,
            DEFAULT_HIERARCHY_PATH,
        ]
    ):
        return
    tmp_root = Path("output/test_batch_v2_selection_pack")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    first_dir = tmp_root / "first"
    second_dir = tmp_root / "second"
    try:
        first = run_batch_v2_selection_pack(output_dir=first_dir)
        second = run_batch_v2_selection_pack(output_dir=second_dir)

        assert len(first["selected_targets"]) == 75
        assert first["selected_targets"] == second["selected_targets"]
        assert first["selection_metrics"]["total_dictionary_targets"] == 1266
        assert first["selection_metrics"]["prior_batch_v1_targets_excluded"] == 75
        assert all(item["readiness_class"] in READY_CLASSES for item in first["selected_targets"])
        assert len(first["batch_v2_source_plan"]) == 75
        assert (first_dir / "selected_targets.json").exists()
        assert json.loads((first_dir / "selection_metrics.json").read_text())[
            "selected_count"
        ] == 75
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


def test_load_real_adjudication_inputs() -> None:
    if not DEFAULT_DICTIONARY_JSONL.exists():
        return
    inputs = load_selection_inputs(
        dictionary_jsonl=DEFAULT_DICTIONARY_JSONL,
        batch_v1_dir=DEFAULT_BATCH_V1_DIR,
        adjudication_dir=DEFAULT_ADJUDICATION_DIR,
        source_manifest=DEFAULT_SOURCE_MANIFEST,
        hierarchy_path=DEFAULT_HIERARCHY_PATH,
    )

    assert len(inputs["dictionary_targets"]) == 1266
    assert len(inputs["batch_v1_ids"]) == 75


def synthetic_inputs(
    targets: list[TargetSpecification],
    *,
    prior_ids: set[str] | None = None,
) -> dict[str, object]:
    return {
        "dictionary_targets": targets,
        "batch_v1_ids": prior_ids or set(),
        "manual_review_ids": set(),
        "final_dispositions": [],
        "final_metrics": {},
        "unsupported_targets": [],
        "ambiguous_targets": [],
        "rejected_prior_extractions": [],
        "evidence_audit": [],
        "disposition_trace": [],
        "lessons_for_next_batch": "",
        "source_by_id": {},
        "hierarchy_index": {},
        "dictionary_jsonl": "synthetic",
        "batch_v1_dir": "synthetic",
        "adjudication_dir": "synthetic",
        "source_manifest": "synthetic",
        "hierarchy_path": "synthetic",
    }


def target(
    field_name: str,
    *,
    row: int,
    datatype: str = "string",
    definition: str | None = None,
    unit: str | None = None,
    domain: str = "Component",
    likely: list[str] | None = None,
    field_label: str | None = None,
) -> TargetSpecification:
    return TargetSpecification.model_validate(
        {
            "target_row_id": field_name,
            "requirement_id": f"DR{row}",
            "sub_domain": domain,
            "requirement_text": definition or f"{field_name} explicit count or value",
            "expected_field": field_name,
            "expected_data_type": datatype,
            "unit": unit,
            "cardinality": "single",
            "accepted_values": [],
            "likely_evidence_types": likely or ["text"],
            "metadata": {"domain": domain, "field_label": field_label or field_name},
            "source_dictionary_provenance": {
                "dictionary_id": "synthetic",
                "dictionary_path": "synthetic.xlsx",
                "row_number": row,
            },
        }
    )


def counter_dict(values: dict[str, int]):
    from collections import Counter

    return Counter(values)
