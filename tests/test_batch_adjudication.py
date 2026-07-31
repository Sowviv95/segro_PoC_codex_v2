from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from segro_evidence_extraction.batch_adjudication import (
    DEFAULT_BATCH_V1_INPUT_DIR,
    DEFAULT_DIAGNOSTIC_INPUT_DIR,
    DEFAULT_ESCALATION_INPUT_DIR,
    DEFAULT_ESCALATION_TEXT_INPUT_DIR,
    DEFAULT_REEXTRACT_V1_INPUT_DIR,
    DEFAULT_REEXTRACT_V2_INPUT_DIR,
    LoadedArtifacts,
    adjudicate_loaded_artifacts,
    component_specific_description,
    materialize_accepted_values,
    run_batch_v1_final_adjudication,
)

REAL_OUTPUT_DIRS = [
    DEFAULT_BATCH_V1_INPUT_DIR,
    DEFAULT_DIAGNOSTIC_INPUT_DIR,
    DEFAULT_REEXTRACT_V1_INPUT_DIR,
    DEFAULT_REEXTRACT_V2_INPUT_DIR,
    DEFAULT_ESCALATION_INPUT_DIR,
    DEFAULT_ESCALATION_TEXT_INPUT_DIR,
]


def test_generic_integer_count_materialization_is_not_dock_specific() -> None:
    loading = materialize_accepted_values(
        target=target_stub("loading_door_count", "integer"),
        intent={"value_shape_family": "integer_count", "primary_component": "loading door"},
        extraction={"evidence_value": "12 No. loading doors", "display_value": "12"},
        normalized={"normalized_value": 12},
        evidence_excerpt="12 No. loading doors",
    )
    assert loading == ("12 No. loading doors", 12, 12, "12")

    parking = materialize_accepted_values(
        target=target_stub("parking_space_count", "integer"),
        intent={"value_shape_family": "integer_count", "primary_component": "parking space"},
        extraction={"evidence_value": "3 parking spaces", "display_value": "3"},
        normalized={"normalized_value": 3},
        evidence_excerpt="3 parking spaces",
    )
    assert parking == ("3 parking spaces", 3, 3, "3")

    words = materialize_accepted_values(
        target=target_stub("rooflight_count", "integer"),
        intent={"value_shape_family": "integer_count", "primary_component": "rooflight"},
        extraction={"evidence_value": "Seven rooflights", "display_value": "Seven rooflights"},
        normalized={"normalized_value": None},
        evidence_excerpt="Seven rooflights",
    )
    assert words == ("Seven rooflights", None, "Seven rooflights", "Seven rooflights")


def test_component_description_phrase_selection_is_generic_and_safe() -> None:
    roof_phrase = component_specific_description(
        intent={"requested_attribute": "description", "primary_component": "roof"},
        evidence_value="The building has a reinforced concrete frame",
        evidence_excerpt=(
            "The building has a reinforced concrete frame, with a standing-seam "
            "aluminium roof and polycarbonate rooflights."
        ),
    )
    assert roof_phrase == "Standing-seam aluminium roof and polycarbonate rooflights"

    absent_roof = component_specific_description(
        intent={"requested_attribute": "description", "primary_component": "roof"},
        evidence_value="The warehouse has a steel frame",
        evidence_excerpt=(
            "The warehouse has a steel frame with two-storey offices and loading docks."
        ),
    )
    assert absent_roof is None

    wall_phrase = component_specific_description(
        intent={"requested_attribute": "description", "primary_component": "wall"},
        evidence_value="The building has a concrete frame",
        evidence_excerpt=(
            "The building has a concrete frame and vertically laid insulated composite "
            "wall panels."
        ),
    )
    assert wall_phrase == "Vertically laid insulated composite wall panels"

    separated_roof = component_specific_description(
        intent={"requested_attribute": "description", "primary_component": "roof"},
        evidence_value="The building is steel framed",
        evidence_excerpt="The building is steel framed. Roof maintenance is required annually.",
    )
    assert separated_roof is None


def test_precedence_and_validation_with_synthetic_artifacts() -> None:
    tmp_path = workspace_tmp("precedence")
    ids = [
        "accepted",
        "normalized",
        "caveat",
        "component_only",
        "structured_abstain",
        "manual",
        "ambiguous",
        "conflict",
        "strict_floor",
        "strict_dock",
    ]
    dirs = write_synthetic_artifacts(tmp_path, ids)

    result = run_batch_v1_final_adjudication(
        batch_v1_dir=dirs["batch"],
        diagnostic_dir=dirs["diagnostic"],
        reextract_v1_dir=dirs["reextract_v1"],
        reextract_v2_dir=dirs["reextract_v2"],
        escalation_dir=dirs["escalation"],
        escalation_text_dir=dirs["text"],
        output_dir=tmp_path / "out",
        expected_target_count=len(ids),
    )

    by_id = {record["target_id"]: record for record in result.final_target_dispositions}
    assert by_id["accepted"]["final_disposition"] == "accepted"
    assert by_id["normalized"]["final_disposition"] == "accepted_after_normalization"
    assert by_id["caveat"]["final_disposition"] == "accepted_with_dictionary_caveat"
    assert by_id["component_only"]["final_disposition"] == "component_present_attribute_absent"
    assert by_id["structured_abstain"]["final_disposition"] == "insufficient_evidence"
    assert by_id["manual"]["final_disposition"] == "manual_visual_review"
    assert by_id["ambiguous"]["final_disposition"] == "dictionary_target_ambiguous"
    assert by_id["conflict"]["final_disposition"] == "conflicting_evidence"
    assert by_id["strict_floor"]["final_disposition"] == "insufficient_evidence"
    assert by_id["strict_dock"]["final_disposition"] == "insufficient_evidence"

    assert len(result.final_target_dispositions) == len(ids)
    assert len({record["target_id"] for record in result.final_target_dispositions}) == len(ids)
    assert len(result.rejected_prior_extractions) == 4
    assert {row["target_id"] for row in result.manual_review_queue} == {"manual"}
    assert result.final_metrics["total_targets"] == len(ids)
    assert sum(result.final_metrics["disposition_counts"].values()) == len(ids)
    assert result.final_metrics["total_accepted_count"] == 3
    assert result.final_metrics["rejected_prior_extraction_count"] == 4

    accepted_ids = {row["target_id"] for row in result.accepted_values}
    unsupported_ids = {row["target_id"] for row in result.unsupported_targets}
    assert not accepted_ids & unsupported_ids

    written_again = run_batch_v1_final_adjudication(
        batch_v1_dir=dirs["batch"],
        diagnostic_dir=dirs["diagnostic"],
        reextract_v1_dir=dirs["reextract_v1"],
        reextract_v2_dir=dirs["reextract_v2"],
        escalation_dir=dirs["escalation"],
        escalation_text_dir=dirs["text"],
        output_dir=tmp_path / "out",
        expected_target_count=len(ids),
    )
    assert written_again.final_target_dispositions == result.final_target_dispositions
    assert list((tmp_path / "out").glob("*.json"))
    assert (tmp_path / "out" / "accepted_values.csv").exists()
    assert (tmp_path / "out" / "final_adjudication_summary.md").exists()


def test_missing_and_duplicate_targets_fail_validation() -> None:
    tmp_path = workspace_tmp("coverage")
    dirs = write_synthetic_artifacts(tmp_path, ["one", "two"])
    artifacts = LoadedArtifacts.load(input_paths(dirs))
    with pytest.raises(ValueError, match="Expected 3 targets"):
        adjudicate_loaded_artifacts(artifacts, expected_target_count=3)

    selected_path = dirs["batch"] / "selected_targets.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected[1]["target"]["target_row_id"] = selected[0]["target"]["target_row_id"]
    selected_path.write_text(json.dumps(selected), encoding="utf-8")
    duplicate_artifacts = LoadedArtifacts.load(input_paths(dirs))
    with pytest.raises(ValueError, match="Duplicate selected target IDs"):
        adjudicate_loaded_artifacts(duplicate_artifacts, expected_target_count=2)


def test_accepted_records_require_canonical_evidence() -> None:
    tmp_path = workspace_tmp("accepted_evidence")
    dirs = write_synthetic_artifacts(tmp_path, ["accepted"])
    extractions = json.loads((dirs["batch"] / "extraction_results.json").read_text())
    extractions[0]["supporting_span_ids"] = []
    (dirs["batch"] / "extraction_results.json").write_text(
        json.dumps(extractions),
        encoding="utf-8",
    )
    artifacts = LoadedArtifacts.load(input_paths(dirs))

    with pytest.raises(ValueError, match="lacks canonical evidence span ID"):
        adjudicate_loaded_artifacts(artifacts, expected_target_count=1)


def test_real_artifact_known_outcomes_and_no_parser_or_llm_invocation() -> None:
    tmp_path = workspace_tmp("real")
    if not all(path.exists() for path in REAL_OUTPUT_DIRS):
        pytest.skip("Real Batch V1 artifacts are not present in this checkout.")

    result = run_batch_v1_final_adjudication(output_dir=tmp_path / "real")
    by_field = {record["field_name"]: record for record in result.final_target_dispositions}

    assert by_field["floor_construction_capacity"]["final_disposition"] == "insufficient_evidence"
    assert by_field["dock_count"]["final_disposition"] == "insufficient_evidence"
    dock_leveller = by_field["dock_leveller_count"]
    assert dock_leveller["final_evidence_value"] == "5No Dock Levellers"
    assert dock_leveller["normalized_value"] == 5
    assert dock_leveller["final_accepted_value"] == 5
    assert dock_leveller["display_value"] == "5"
    assert dock_leveller["final_disposition"] == "accepted"
    assert dock_leveller["source_document"] == "Building Manual - Part 1 General.pdf"
    assert dock_leveller["page"] == 8
    assert "5No Dock Levellers" in dock_leveller["evidence_excerpt"]

    roof = by_field["roof_construction_description"]
    assert roof["final_evidence_value"] == "The construction of a steel frame warehouse"
    assert roof["final_accepted_value"] == "Profiled metal clad roof with rooflights"
    assert roof["normalized_value"] == "Profiled metal clad roof with rooflights"
    assert roof["display_value"] == "Profiled metal clad roof with rooflights"
    assert roof["final_disposition"] == "accepted_with_dictionary_caveat"
    assert roof["dictionary_compatibility_status"] == "unit_not_applicable"
    assert "profiled metal clad elevations and roof, rooflights" in roof["evidence_excerpt"]
    assert "component-specific phrase" in roof["decision_rationale"]

    visual = {
        "green_wall_component_description",
        "pv_other_component_description",
        "pv_panel_component_description",
        "mansafe_roof_guardrails_component_description",
        "mansafe_roof_anchors_component_description",
    }
    assert {
        record["field_name"]
        for record in result.manual_review_queue
        if record["field_name"] in visual
    } == visual
    assert result.final_metrics["total_targets"] == 75
    assert sum(result.final_metrics["disposition_counts"].values()) == 75
    assert result.final_metrics["original_non_null_extractions"] == 24
    assert result.final_metrics["original_non_null_extractions_retained"] == len(
        result.accepted_values
    )

    with (tmp_path / "real" / "final_adjudication_review.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 75


def write_synthetic_artifacts(tmp_path: Path, ids: list[str]) -> dict[str, Path]:
    dirs = {
        "batch": tmp_path / "batch",
        "diagnostic": tmp_path / "diagnostic",
        "reextract_v1": tmp_path / "reextract_v1",
        "reextract_v2": tmp_path / "reextract_v2",
        "escalation": tmp_path / "escalation",
        "text": tmp_path / "text",
    }
    for directory in dirs.values():
        directory.mkdir()

    selected = [selected_target(target_id, index + 1) for index, target_id in enumerate(ids)]
    write_json(dirs["batch"] / "selected_targets.json", selected)
    write_json(
        dirs["batch"] / "extraction_results.json",
        [extraction(target_id) for target_id in ids],
    )
    write_json(
        dirs["batch"] / "validation_results.json",
        [validation(target_id) for target_id in ids],
    )
    write_json(
        dirs["batch"] / "evidence_validation.json",
        [evidence_validation(target_id) for target_id in ids],
    )
    write_json(
        dirs["batch"] / "shape_validation.json",
        [shape_validation(target_id) for target_id in ids],
    )
    write_json(
        dirs["batch"] / "schema_compatibility_results.json",
        [schema_result(target_id) for target_id in ids],
    )
    write_json(
        dirs["batch"] / "normalized_values.json",
        [normalized_value(target_id) for target_id in ids],
    )
    write_json(
        dirs["batch"] / "retrieval_results.json",
        [retrieval(target_id) for target_id in ids],
    )
    write_json(
        dirs["batch"] / "evidence_spans.json",
        [span(target_id) for target_id in ids],
    )

    write_json(
        dirs["diagnostic"] / "retrieval_eligibility.json",
        [eligibility(target_id) for target_id in ids],
    )
    write_json(dirs["diagnostic"] / "extracted_result_attribute_audit.json", audit_rows(ids))
    write_json(dirs["diagnostic"] / "target_failure_classification.json", [])
    write_json(dirs["diagnostic"] / "target_semantics.json", [])

    write_json(
        dirs["reextract_v1"] / "extraction_results.json",
        [reextract_v1("structured_abstain")],
    )
    write_json(dirs["reextract_v2"] / "extraction_results.json", [abstain("structured_abstain")])
    write_json(dirs["escalation"] / "escalation_decisions.json", [manual_decision("manual")])
    write_json(dirs["escalation"] / "visual_page_triage.json", [visual_triage("manual")])
    write_json(
        dirs["escalation"] / "reextract_readiness.json",
        [
            {
                "target_row_id": "manual",
                "readiness": "manual_review_only",
                "reason": "visual only",
            }
        ],
    )
    write_json(
        dirs["text"] / "extraction_results.json",
        [abstain("strict_floor"), abstain("strict_dock")],
    )
    write_json(
        dirs["text"] / "preflight_report.json",
        {"target_ids": ["strict_floor", "strict_dock"]},
    )
    write_json(dirs["text"] / "approved_evidence.json", [])
    return dirs


def input_paths(dirs: dict[str, Path]):
    from segro_evidence_extraction.batch_adjudication import AdjudicationInputs

    return AdjudicationInputs(
        batch_v1_dir=dirs["batch"],
        diagnostic_dir=dirs["diagnostic"],
        reextract_v1_dir=dirs["reextract_v1"],
        reextract_v2_dir=dirs["reextract_v2"],
        escalation_dir=dirs["escalation"],
        escalation_text_dir=dirs["text"],
    )


def selected_target(target_id: str, row: int) -> dict[str, object]:
    return {
        "target": {
            "target_row_id": target_id,
            "requirement_id": f"DR{row}",
            "sub_domain": "Component",
            "requirement_text": f"{target_id} definition",
            "expected_field": target_id,
            "expected_data_type": "string",
            "unit": None,
            "cardinality": "single",
            "accepted_values": [],
            "likely_evidence_types": ["text"],
            "metadata": {"domain": "Component", "field_label": target_id},
            "source_dictionary_provenance": {
                "dictionary_id": "test",
                "dictionary_path": "dict.xlsx",
                "row_number": row,
            },
        },
        "value_shape": {
            "target_row_id": target_id,
            "value_shape_family": "short_text",
            "reason": "test",
        },
        "selection_reason": "test",
        "expected_evidence_source_category": "bounded manuals",
        "expected_support_status": "uncertain",
    }


def target_stub(field_name: str, datatype: str):
    from segro_evidence_extraction.models.target import TargetSpecification

    return TargetSpecification.model_validate(
        {
            "target_row_id": f"trg_{field_name}",
            "requirement_id": "DRX",
            "sub_domain": "Component",
            "requirement_text": f"{field_name} definition",
            "expected_field": field_name,
            "expected_data_type": datatype,
            "cardinality": "single",
            "accepted_values": [],
            "likely_evidence_types": ["text"],
            "metadata": {"domain": "Component"},
        }
    )


def extraction(target_id: str) -> dict[str, object]:
    extracted_ids = {
        "accepted",
        "normalized",
        "caveat",
        "component_only",
        "manual",
        "ambiguous",
        "conflict",
    }
    status = "extracted" if target_id in extracted_ids else "insufficient_evidence"
    value = f"value-{target_id}" if status == "extracted" else None
    return {
        "target_row_id": target_id,
        "requirement_id": "DR1",
        "raw_model_value": value,
        "extracted_value": value,
        "evidence_value": value,
        "normalized_value": value,
        "display_value": value,
        "proposed_value_shape": "short_text",
        "value_bearing_quote": value,
        "status": status,
        "confidence": 0.9 if value else 0.0,
        "supporting_span_ids": [f"span-{target_id}"] if value else [],
        "supporting_evidence_excerpt": f"excerpt {target_id}" if value else None,
        "source_id": "src",
        "source_file": "source.pdf" if value else None,
        "page_number": 1 if value else None,
        "page_range": None,
        "hierarchy_node_id": f"node-{target_id}" if value else None,
        "unit": None,
        "ambiguity_or_caveat": None,
        "reasoning_summary": None,
        "model_provider": "test",
        "model_name": "none",
        "model_usage": {"input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0},
    }


def validation(target_id: str) -> dict[str, object]:
    statuses = {
        "normalized": "valid_after_normalization",
        "caveat": "valid_with_dictionary_caveat",
    }
    return {
        "target_row_id": target_id,
        "status": statuses.get(
            target_id,
            "valid" if target_id == "accepted" else "insufficient_evidence",
        ),
        "evidence_valid": True,
        "value_format_valid": True,
        "issues": [],
    }


def evidence_validation(target_id: str) -> dict[str, object]:
    return {
        "target_row_id": target_id,
        "status": "valid",
        "canonical_evidence_available": target_id in {"accepted", "normalized", "caveat"},
        "evidence_value_present": target_id in {"accepted", "normalized", "caveat"},
        "source_page_node_valid": True,
        "selected_span_ids": [f"span-{target_id}"],
        "issues": [],
    }


def shape_validation(target_id: str) -> dict[str, object]:
    return {
        "target_row_id": target_id,
        "status": "valid",
        "value_shape_family": "short_text",
        "issues": [],
    }


def schema_result(target_id: str) -> dict[str, object]:
    compatibility = "compatible_after_normalization" if target_id == "normalized" else "compatible"
    if target_id == "caveat":
        compatibility = "suspected_dictionary_metadata_mismatch"
    return {
        "target_row_id": target_id,
        "compatibility": compatibility,
        "dictionary_datatype": "string",
        "dictionary_unit": None,
        "observed_unit": None,
        "observed_value": f"value-{target_id}",
        "evidence_validity": "valid",
        "value_format_validity": "valid",
        "overall_review_status": validation(target_id)["status"],
        "issues": [],
    }


def normalized_value(target_id: str) -> dict[str, object]:
    return {
        "target_row_id": target_id,
        "value_shape_family": "short_text",
        "raw_model_value": f"value-{target_id}",
        "evidence_value": f"value-{target_id}",
        "normalized_value": f"value-{target_id}",
        "display_value": f"value-{target_id}",
        "unit": None,
        "normalization_status": "normalized",
        "issues": [],
    }


def retrieval(target_id: str) -> dict[str, object]:
    return {
        "target_row_id": target_id,
        "query": target_id,
        "retrieval_status": "evidence_found",
        "results": [],
        "retrieval_time_ms": 0.0,
        "top_score": 0.0,
    }


def span(target_id: str) -> dict[str, object]:
    return {
        "span_id": f"span-{target_id}",
        "source_id": "src",
        "source_file": "source.pdf",
        "page_number": 1,
        "hierarchy_node_id": f"node-{target_id}",
        "retrieval_rank": 1,
        "score": 1.0,
        "start_char": 0,
        "end_char": 20,
        "text": f"excerpt {target_id}",
    }


def eligibility(target_id: str) -> dict[str, object]:
    status = {
        "component_only": "component_present_attribute_absent",
        "ambiguous": "dictionary_target_ambiguous",
        "conflict": "conflicting_evidence",
    }.get(target_id, "eligible_for_extraction")
    support = {
        "component_only": "component_only",
        "conflict": "conflicting",
    }.get(target_id, "supports_requested_attribute")
    return {
        "target_row_id": target_id,
        "status": status,
        "support_classification": support,
        "source_id": "src",
        "page_number": 1,
        "node_id": f"node-{target_id}",
        "reason": f"reason {target_id}",
    }


def audit_rows(ids: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    classifications = {
        "accepted": "correctly_supported",
        "normalized": "correctly_supported",
        "caveat": "supported_with_caveat",
        "component_only": "component_only_overgeneralization",
        "manual": "correctly_supported",
        "ambiguous": "component_only_overgeneralization",
        "conflict": "wrong_attribute",
    }
    for target_id in ids:
        if target_id in classifications:
            rows.append(
                {
                    "target_row_id": target_id,
                    "field_name": target_id,
                    "audit_classification": classifications[target_id],
                    "support_classification": "supports_requested_attribute"
                    if classifications[target_id]
                    in {"correctly_supported", "supported_with_caveat"}
                    else "component_only",
                    "explanation": f"audit {target_id}",
                }
            )
    return rows


def reextract_v1(target_id: str) -> dict[str, object]:
    row = abstain(target_id)
    row["status"] = "invalid_format"
    return row


def abstain(target_id: str) -> dict[str, object]:
    row = extraction(target_id)
    row.update(
        {
            "status": "insufficient_evidence",
            "raw_model_value": None,
            "extracted_value": None,
            "evidence_value": None,
            "normalized_value": None,
            "display_value": None,
            "supporting_span_ids": [],
            "supporting_evidence_excerpt": None,
            "ambiguity_or_caveat": "strict abstention",
        }
    )
    return row


def manual_decision(target_id: str) -> dict[str, object]:
    return {
        "target_row_id": target_id,
        "field_name": target_id,
        "final_status": "manual_visual_review",
        "manual_review_required": True,
        "ocr_required": False,
        "vlm_required": False,
        "retry_extraction": False,
        "candidate_page_or_range": "src:1",
        "evidence_type": "visual",
        "recommended_next_action": "Perform manual visual review.",
        "notes": "Sparse text.",
        "confidence": 0.5,
    }


def visual_triage(target_id: str) -> dict[str, object]:
    return {
        "target_row_id": target_id,
        "source_page": "src:1",
        "page_type": "weak_or_sparse_text",
        "visual_need": "visual_review_helpful",
        "text_character_count": 10,
        "explanation": "Parsed text is sparse.",
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def workspace_tmp(name: str) -> Path:
    root = Path(".pytest_local_batch_adjudication").resolve()
    path = root / name
    if path.exists():
        resolved = path.resolve()
        if root not in resolved.parents and resolved != root:
            msg = f"Refusing to remove test directory outside {root}: {resolved}"
            raise ValueError(msg)
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path
