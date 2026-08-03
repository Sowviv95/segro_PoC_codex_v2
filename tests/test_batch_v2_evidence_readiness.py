from __future__ import annotations

import json
import shutil
from pathlib import Path

from segro_evidence_extraction.batch_v2_evidence_readiness import (
    DEFAULT_ADJUDICATION_DIR,
    DEFAULT_BATCH_V2_SELECTION_OUTPUT_DIR,
    DEFAULT_PAGE_CACHE_ROOT,
    ReadinessScanCache,
    audit_target,
    build_readiness_audit,
    load_readiness_inputs,
    prepare_page_excerpts,
    run_batch_v2_readiness_audit,
)

PART3 = "Building Manual - Part 3 Building Services.pdf"


def test_component_only_model_number_evidence_is_not_ready() -> None:
    inputs = synthetic_inputs(
        selected=[row("pump_model_number", value_shape="identifier_or_reference")],
        pages={
            "Building Manual - Part 1 General.pdf": [
                page("Pumps and valves are installed in the plant room.")
            ]
        },
    )

    audited = audit_target(inputs["selected_targets"][0], inputs, audit_stage="test")

    assert audited["audit_classification"] == "component_only_risk"


def test_labelled_model_evidence_is_execution_ready() -> None:
    inputs = synthetic_inputs(
        selected=[row("pump_model_number", value_shape="identifier_or_reference")],
        pages={
            "Building Manual - Part 1 General.pdf": [
                page("Pump schedule Model Number: ABC-123 Pump reference P-01.")
            ]
        },
    )

    audited = audit_target(inputs["selected_targets"][0], inputs, audit_stage="test")

    assert audited["audit_classification"] == "execution_ready"
    assert audited["local_association_observed"]


def test_contractor_name_is_not_treated_as_manufacturer() -> None:
    inputs = synthetic_inputs(
        selected=[row("pump_manufacturer", value_shape="short_text")],
        pages={
            "Building Manual - Part 1 General.pdf": [
                page("Pump installation contractor Acme Services attended site.")
            ]
        },
    )

    audited = audit_target(inputs["selected_targets"][0], inputs, audit_stage="test")

    assert audited["audit_classification"] == "component_only_risk"
    assert not audited["requested_attribute_observed"]


def test_event_date_requires_local_certificate_label() -> None:
    inputs = synthetic_inputs(
        selected=[row("fire_certificate_issue_date", value_shape="date", datatype="date")],
        pages={
            "Building Manual - Part 3 Building Services.pdf": [
                page("Fire certificate issue date 13/03/2020.", source=PART3)
            ]
        },
    )
    inputs["selected_targets"][0]["likely_source_file"] = PART3

    audited = audit_target(inputs["selected_targets"][0], inputs, audit_stage="test")

    assert audited["audit_classification"] == "execution_ready"


def test_measurement_requires_value_and_compatible_unit() -> None:
    inputs = synthetic_inputs(
        selected=[row("office_area_value", value_shape="decimal_measurement", datatype="decimal")],
        pages={
            "Building Manual - Part 1 General.pdf": [
                page("Office area value 1250 m2 recorded in accommodation schedule.")
            ]
        },
    )

    audited = audit_target(inputs["selected_targets"][0], inputs, audit_stage="test")

    assert audited["audit_classification"] == "execution_ready"


def test_count_label_does_not_match_inside_unrelated_word() -> None:
    inputs = synthetic_inputs(
        selected=[row("sprinkler_head_count", value_shape="integer_count", datatype="integer")],
        pages={
            "Building Manual - Part 1 General.pdf": [
                page("Water pressure is 10m head; developer should take account of this.")
            ]
        },
    )

    audited = audit_target(inputs["selected_targets"][0], inputs, audit_stage="test")

    assert audited["audit_classification"] == "component_only_risk"


def test_part1_source_correction_uses_stronger_cached_source() -> None:
    inputs = synthetic_inputs(
        selected=[row("fire_certificate_reference", value_shape="identifier_or_reference")],
        pages={
            "Building Manual - Part 1 General.pdf": [page("General fire safety information.")],
            "Building Manual - Part 3 Building Services.pdf": [
                page(
                    "Fire certificate reference FS-2020-77.",
                    source=PART3,
                )
            ],
        },
    )

    audited = audit_target(inputs["selected_targets"][0], inputs, audit_stage="test")

    assert audited["audit_classification"] == "wrong_source_mapping"
    assert audited["better_existing_source_file"] == PART3


def test_visual_uncached_and_dictionary_clarification_classifications() -> None:
    visual = row("roof_plant_area_value", value_shape="decimal_measurement")
    visual["readiness_class"] = "defer_visual"
    ambiguous = row("pump_manufacturer", value_shape="short_text", unit="m")
    uncached = row("dock_door_count", value_shape="integer_count")
    inputs = synthetic_inputs(selected=[visual, ambiguous, uncached], pages={})

    results = {
        item["field_name"]: audit_target(item, inputs, audit_stage="test")
        for item in inputs["selected_targets"]
    }

    assert results["roof_plant_area_value"]["audit_classification"] == "defer_visual"
    assert results["pump_manufacturer"]["audit_classification"] == "dictionary_clarification"
    assert (
        results["dock_door_count"]["audit_classification"]
        == "requires_additional_cached_pages"
    )


def test_replacements_blocked_readiness_and_deterministic_outputs() -> None:
    selected = [row("pump_model_number", value_shape="identifier_or_reference", rank=1)]
    candidates = selected + [
        row("door_count", value_shape="integer_count", rank=2, selected=False),
        row("bay_count", value_shape="integer_count", rank=3, selected=False),
    ]
    inputs = synthetic_inputs(
        selected=selected,
        candidates=candidates,
        pages={
            "Building Manual - Part 1 General.pdf": [
                page("Pump is present."),
                page("Door count 4 No doors."),
                page("Bay count 3 No bays."),
            ]
        },
    )

    first = build_readiness_audit(inputs, target_count=2)
    second = build_readiness_audit(inputs, target_count=2)

    assert first["readiness_metrics"] == second["readiness_metrics"]
    assert first["readiness_metrics"]["replacements_accepted"] == 2
    assert first["corrected_batch_v2_run_readiness"]["overall_status"] == "ready_with_caveats"

    blocked = build_readiness_audit(inputs, target_count=3)
    assert blocked["corrected_batch_v2_run_readiness"]["overall_status"] == "blocked"
    assert blocked["corrected_batch_v2_run_readiness"]["additional_ready_targets_needed"] == 1


def test_prepared_excerpts_normalize_text_once() -> None:
    excerpts = prepare_page_excerpts(
        page("Pump schedule\nModel Number: ABC-123", page_number=7)
    )

    assert len(excerpts) == 2
    assert excerpts[0].lower == excerpts[0].excerpt.lower()
    assert "pump" in excerpts[0].words
    assert excerpts[0].page_number == 7
    assert excerpts[0].source_file == "Building Manual - Part 1 General.pdf"


def test_readiness_scan_cache_reuses_and_clones_best_signal() -> None:
    pages = {
        "Building Manual - Part 1 General.pdf": [
            page("Pump schedule Model Number: ABC-123 Pump reference P-01.")
        ]
    }
    cache = ReadinessScanCache(pages)
    query = {
        "component_terms": ["pump"],
        "attribute_terms": ["model", "model number", "reference"],
        "value_patterns": [r"\b[A-Z]{1,6}[-/]?[A-Z0-9]{2,}(?:[-/][A-Z0-9]{2,})*\b"],
        "field_terms": ["pump_model_number"],
    }

    first = cache.best_signal(
        "Building Manual - Part 1 General.pdf", query, "identifier_or_reference"
    )
    first["component_hits"].append("mutated")
    second = cache.best_signal(
        "Building Manual - Part 1 General.pdf", query, "identifier_or_reference"
    )

    assert second["score"] == 12
    assert second["component_hits"] == ["pump"]
    assert second["page_number"] == 1
    prepared = cache.excerpts_by_source["Building Manual - Part 1 General.pdf"][0]
    assert query["value_patterns"][0] in prepared.value_pattern_matches


def test_real_readiness_audit_loads_and_reconciles() -> None:
    if not all(
        path.exists()
        for path in [
            DEFAULT_BATCH_V2_SELECTION_OUTPUT_DIR,
            DEFAULT_PAGE_CACHE_ROOT,
            DEFAULT_ADJUDICATION_DIR,
        ]
    ):
        return
    output_root = Path("output/test_batch_v2_readiness_audit")
    if output_root.exists():
        shutil.rmtree(output_root)
    try:
        inputs = load_readiness_inputs(
            selection_dir=DEFAULT_BATCH_V2_SELECTION_OUTPUT_DIR,
            page_cache_root=DEFAULT_PAGE_CACHE_ROOT,
            adjudication_dir=DEFAULT_ADJUDICATION_DIR,
        )
        assert len(inputs["selected_targets"]) == 75
        assert inputs["cached_pages"]

        result = run_batch_v2_readiness_audit(output_dir=output_root / "real")
        assert len(result["target_readiness_audit"]) == 75
        assert (output_root / "real" / "target_readiness_audit.json").exists()
        assert json.loads((output_root / "real" / "readiness_metrics.json").read_text())[
            "original_selected_count"
        ] == 75
    finally:
        if output_root.exists():
            shutil.rmtree(output_root)


def synthetic_inputs(
    *,
    selected: list[dict[str, object]],
    pages: dict[str, list[dict[str, object]]],
    candidates: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "selection_dir": "synthetic",
        "page_cache_root": "synthetic",
        "adjudication_dir": "synthetic",
        "selected_targets": selected,
        "candidate_scores": candidates or selected,
        "selection_metrics": {},
        "source_plan": [],
        "prior_lessons": "",
        "cached_pages": pages,
    }


def row(
    field_name: str,
    *,
    value_shape: str,
    datatype: str = "string",
    unit: str | None = None,
    rank: int = 1,
    selected: bool = True,
) -> dict[str, object]:
    return {
        "selection_rank": rank if selected else None,
        "target_id": f"trg_{field_name}",
        "dictionary_row": rank,
        "domain": "Component",
        "sub_domain": "Component",
        "field_name": field_name,
        "definition": f"{field_name} target definition",
        "datatype": datatype,
        "unit": unit,
        "value_shape": value_shape,
        "likely_source_file": "Building Manual - Part 1 General.pdf",
        "readiness_class": "ready_identifier_or_date_extraction"
        if value_shape in {"identifier_or_reference", "date"}
        else "ready_text_extraction",
        "evidence_source_readiness": 20,
        "attribute_locality": 15,
        "dictionary_clarity": 15,
        "value_shape_reliability": 14,
        "batch_v1_family_evidence": 8,
        "boundedness": 10,
        "total_score": 82 - rank,
        "expected_retrieval_query": field_name,
        "expected_extraction_route": "text evidence bundle then bounded extraction",
    }


def page(
    text: str,
    *,
    source: str = "Building Manual - Part 1 General.pdf",
    page_number: int = 1,
) -> dict[str, object]:
    return {
        "source_file": source,
        "page_number": page_number,
        "extracted_text": text,
        "cache_path": "synthetic",
    }
