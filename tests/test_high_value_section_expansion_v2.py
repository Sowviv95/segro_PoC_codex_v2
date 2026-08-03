from __future__ import annotations

import shutil

from segro_evidence_extraction.high_value_section_expansion_v2 import (
    DEFAULT_HIGH_VALUE_EXPANSION_OUTPUT_DIR,
    DEFAULT_PAGE_CACHE_ROOT,
    DEFAULT_SOURCE_MANIFEST,
    adaptive_section_window,
    adaptive_window_size,
    approve_v2_parse_plan,
    audit_priority_classifications,
    build_v2_section_candidates,
    classify_page_text_v3,
    correct_priority_classifications,
    evidence_family_coverage,
    rescore_section_candidates,
    run_high_value_section_expansion_v2,
    source_caps,
    split_v2_batches,
)
from segro_evidence_extraction.models.source import SourceRegistryEntry
from segro_evidence_extraction.source_coverage_pageindex import validate_parse_batches


def test_separator_cover_overclassification_correction_for_sparse_drawing() -> None:
    corrected = classify_page_text_v3(
        "",
        source_filename="Building Manual - Part 2 Building Fabric.pdf",
        page_number=96,
        prior_type="separator_or_cover",
    )

    assert corrected["primary_page_type"] == "drawing_visual_required"
    assert corrected["recommended_route"] == "visual"
    assert "not cover" in corrected["classification_reason"]


def test_certificate_title_and_element_sheet_classification() -> None:
    certificate = classify_page_text_v3(
        "FIRE ALARM COMMISSIONING TEST CERTIFICATE Certificate No Date Result",
        source_filename="Building Manual - Part 6 Appendices.pdf",
        prior_type="unknown",
    )
    element = classify_page_text_v3(
        "ELEMENT: 2.3.1 ROOF COVERINGS manufacturer product material finish",
        source_filename="Building Manual - Part 2 Building Fabric.pdf",
        prior_type="separator_or_cover",
    )

    assert certificate["primary_page_type"] == "certificate"
    assert "commissioning_results" in certificate["likely_target_families"]
    assert element["primary_page_type"] == "narrative"
    assert "materials_finishes" in element["likely_target_families"]


def test_audit_and_corrections_record_generic_reasons() -> None:
    rows = [
        {
            "source_id": "src_a",
            "source_filename": "Building Manual - Part 2 Building Fabric.pdf",
            "page_number": 96,
            "primary_page_type": "separator_or_cover",
            "text_character_count": 0,
        }
    ]
    audit = audit_priority_classifications(rows)
    corrections = correct_priority_classifications(
        prior_classifications=rows,
        cached_pages={"src_a": [{"page_number": 96, "extracted_text": ""}]},
        sources=[source("src_a", "Building Manual - Part 2 Building Fabric.pdf")],
    )

    assert audit["suspected_separator_cover_overclassification"] == 1
    assert corrections[0]["classification_changed"] is True
    assert corrections[0]["rule_signals"]["source_part"] == "part 2"


def test_adaptive_section_window_sizing_and_cache_reuse() -> None:
    size = adaptive_window_size(
        route="table",
        families={"manufacturer_model", "equipment_specification"},
        section_page_count=20,
    )
    pages = adaptive_section_window(
        {
            "source_id": "src_a",
            "pdf_page_start": 10,
            "pdf_page_end": 20,
            "section_title": "Equipment Schedule",
        },
        families={"manufacturer_model", "equipment_specification"},
        total_pages=40,
        cached_pages={10, 11, 12, 13, 14, 15, 16, 17},
    )

    assert size == 8
    assert pages == [18, 19, 20]


def test_evidence_family_gap_scoring_and_metrics() -> None:
    coverage = evidence_family_coverage(
        [
            {
                "evidence_bearing": True,
                "source_filename": "Part 3.pdf",
                "likely_target_families": ["manufacturer_model"],
            }
        ]
    )
    rescored = rescore_section_candidates(
        resolved_sections=[
            resolved_section("3.4", "FIRE ALARM COMMISSIONING AND TESTING", "src_a")
        ],
        family_gap=coverage,
        page_counts={"src_a": 100},
        cached_pages={"src_a": [{"page_number": page} for page in range(1, 30)]},
    )

    assert "commissioning_results" in coverage["weak_or_absent_families"]
    assert rescored[0]["v2_evidence_value_score"] > rescored[0]["evidence_value_score"]
    assert "gap_commissioning_results+12" in rescored[0]["v2_scoring_reasons"]


def test_balanced_page_allocation_caps_duplicates_and_batches() -> None:
    candidates = build_v2_section_candidates(
        rescored_sections=[
            {
                **resolved_section("2.3", "ROOF COVERINGS", "src_p2"),
                "source_filename": "Building Manual - Part 2 Building Fabric.pdf",
                "v2_evidence_value_score": 50,
                "expected_evidence_families": ["materials_finishes"],
                "adaptive_window_pages": list(range(96, 104)),
                "adaptive_window_page_count": 8,
                "adaptive_window_rationale": "element sheet",
                "recommended_route": "text",
            },
            {
                **resolved_section("3.2", "MECHANICAL EQUIPMENT SCHEDULE", "src_p3"),
                "source_filename": "Building Manual - Part 3 Building Services.pdf",
                "v2_evidence_value_score": 50,
                "expected_evidence_families": ["manufacturer_model"],
                "adaptive_window_pages": list(range(56, 66)),
                "adaptive_window_page_count": 10,
                "adaptive_window_rationale": "schedule",
                "recommended_route": "table",
            },
            {
                **resolved_section("D.1", "COMMISSIONING CERTIFICATES", "src_p6"),
                "source_filename": "Building Manual - Part 6 Appendices.pdf",
                "v2_evidence_value_score": 50,
                "expected_evidence_families": ["commissioning_results"],
                "adaptive_window_pages": list(range(36, 42)),
                "adaptive_window_page_count": 6,
                "adaptive_window_rationale": "certificate",
                "recommended_route": "certificate",
            },
        ],
        page_map=page_map(),
        page_counts={"src_p2": 200, "src_p3": 200, "src_p6": 200},
    )
    approved = approve_v2_parse_plan(
        candidates,
        max_new_pages=24,
        per_source_caps={"part 2": 8, "part 3": 8, "part 6": 8},
    )
    batches = split_v2_batches(approved)

    assert {row["source_id"] for row in approved} == {"src_p2", "src_p3", "src_p6"}
    assert sum(row["page_count"] for row in approved) <= 24
    assert all(batch["page_count"] <= 10 for batch in batches)
    validate_parse_batches(batches, {"src_p2": [{"page_number": 1}]})
    requested = [
        (batch["source_id"], page)
        for batch in batches
        for page in range(batch["page_start"], batch["page_end"] + 1)
    ]
    assert len(requested) == len(set(requested))


def test_source_caps_scale_total_cap() -> None:
    caps = source_caps(80)

    assert caps["part 2"] < 55
    assert caps["part 6"] < 45


def test_bounded_real_manifest_integration_dry_run_no_models() -> None:
    if not DEFAULT_SOURCE_MANIFEST.exists() or not DEFAULT_PAGE_CACHE_ROOT.exists():
        return
    output_root = DEFAULT_HIGH_VALUE_EXPANSION_OUTPUT_DIR.with_name(
        "test_high_value_section_expansion_v2"
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    try:
        result = run_high_value_section_expansion_v2(
            output_dir=output_root,
            max_new_pages=18,
            dry_run=True,
        )

        assert result["parse_execution_report"]["parser_batches_run"] == 0
        assert result["coverage_metrics"]["newly_parsed_pages"] == 0
        assert result["approved_section_parse_plan"]
        assert (output_root / "classification_corrections.json").exists()
    finally:
        if output_root.exists():
            shutil.rmtree(output_root)


def resolved_section(label: str, title: str, source_id: str) -> dict[str, object]:
    return {
        "entry_id": label,
        "source_id": source_id,
        "source_filename": "Building Manual - Part 3 Building Services.pdf",
        "section_label": label,
        "section_title": title,
        "normalized_section_title": title.lower(),
        "parent_section": "",
        "pdf_page_start": 30,
        "pdf_page_end": 40,
        "page_count": 11,
        "uncached_pages_in_range": list(range(30, 41)),
        "range_already_cached": False,
        "resolved": True,
    }


def page_map() -> list[dict[str, object]]:
    rows = []
    source_names = {
        "src_p2": "Building Manual - Part 2 Building Fabric.pdf",
        "src_p3": "Building Manual - Part 3 Building Services.pdf",
        "src_p6": "Building Manual - Part 6 Appendices.pdf",
    }
    for source_id, filename in source_names.items():
        for page_number in range(1, 210):
            rows.append(
                {
                    "source_id": source_id,
                    "source_filename": filename,
                    "page_number": page_number,
                    "cache_status": "cached" if page_number <= 10 else "uncached",
                }
            )
    return rows


def source(source_id: str, filename: str) -> SourceRegistryEntry:
    return SourceRegistryEntry.model_validate(
        {
            "source_id": source_id,
            "original_path": filename,
            "logical_path": filename,
            "file_hash": f"hash_{source_id}",
            "size_bytes": 100,
            "file_type": "pdf",
        }
    )
