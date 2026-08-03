from __future__ import annotations

import shutil
from pathlib import Path

from segro_evidence_extraction.evidence_gap_refinement_v1 import (
    DEFAULT_EVIDENCE_GAP_OUTPUT_DIR,
    approve_gap_parse_plan,
    build_evidence_gap_baseline,
    build_gap_section_candidates,
    frontier_gap_templates,
    load_source_frontier_templates,
    reconcile_cache_provenance,
    run_evidence_gap_refinement_v1,
    split_gap_batches,
    validate_family_text,
)
from segro_evidence_extraction.source_coverage_pageindex import (
    DEFAULT_PAGE_CACHE_ROOT,
    DEFAULT_SOURCE_MANIFEST,
    validate_parse_batches,
)


def test_cache_provenance_separates_sprint_and_cumulative_counts() -> None:
    provenance = reconcile_cache_provenance(
        DEFAULT_EVIDENCE_GAP_OUTPUT_DIR.with_name("missing_v2"),
        {"src": [{"page_number": 1}, {"page_number": 2}]},
    )

    assert provenance["current_cache_before_this_sprint"] == 2
    assert "not counted as this sprint parsing" in provenance["provenance_note"]


def test_family_validation_accepts_complete_commissioning_result() -> None:
    result = validate_family_text(
        "commissioning_results",
        "Fire alarm system commissioning test certificate ref FA-12 dated 12/03/2020 "
        "measured value satisfactory pass completed",
    )

    assert result["valid"] is True
    assert result["signals"]["commission_event"] is True
    assert result["signals"]["result"] is True


def test_family_validation_accepts_statutory_compliance() -> None:
    result = validate_family_text(
        "statutory_compliance",
        "Building control approved inspector certificate no BC-1 dated 13/03/2020 "
        "certified completion compliant",
    )

    assert result["valid"] is True
    assert result["signals"]["authority"] is True


def test_family_validation_accepts_installation_details() -> None:
    result = validate_family_text(
        "installation_details",
        "Loading door system installed at dock location type sectional mounted fixing "
        "configuration with steel frame",
    )

    assert result["valid"] is True
    assert result["signals"]["installation_detail"] is True


def test_rejects_heading_only_and_maintenance_only_evidence() -> None:
    heading_only = validate_family_text("commissioning_results", "Commissioning Certificates")
    maintenance = validate_family_text(
        "installation_details",
        "Maintenance cleaning inspection COSHH safety data sheet for equipment",
    )

    assert heading_only["valid"] is False
    assert maintenance["valid"] is False
    assert maintenance["reason"] == "maintenance-only evidence rejected"


def test_no_unnecessary_parsing_when_families_supported() -> None:
    baseline = {
        "families_requiring_parsing": [],
        "family_validation": {
            "commissioning_results": {"status": "valid"},
            "statutory_compliance": {"status": "valid"},
            "installation_details": {"status": "valid"},
        },
    }
    candidates = build_gap_section_candidates(
        v2_output_dir=DEFAULT_EVIDENCE_GAP_OUTPUT_DIR.with_name("missing_v2"),
        baseline=baseline,
        cached_pages={},
        page_counts={},
    )

    assert candidates == []


def test_frontier_gap_templates_are_source_config_driven(tmp_path: Path) -> None:
    config = tmp_path / "source_config.json"
    config.write_text(
        """
        {
          "schema_version": "segro_evidence_gap_source_config_v1",
          "config_version": "1.0.0",
          "unit_id": "portable_unit",
          "source_frontier_templates": [
            {
              "source_id": "portable_source_a",
              "source_filename": "Standalone Certificates.pdf",
              "section": "completion certificates",
              "trigger_families": ["statutory_compliance"],
              "evidence_families": ["statutory_compliance"],
              "recommended_route": "certificate",
              "priority_score": 70,
              "page_window_size": 3
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    rows = frontier_gap_templates(
        {"statutory_compliance"},
        {"portable_source_a": 4},
        {"portable_source_a": 20},
        load_source_frontier_templates(config),
    )

    assert rows == [
        {
            "source_id": "portable_source_a",
            "source_filename": "Standalone Certificates.pdf",
            "section": "completion certificates",
            "section_label": "",
            "page_start": 5,
            "page_end": 7,
            "page_count": 3,
            "approved_pages": [5, 6, 7],
            "evidence_families": ["statutory_compliance"],
            "expected_evidence": (
                "approval/certificate authority, date/reference and certified status"
            ),
            "recommended_route": "certificate",
            "priority_score": 70,
            "current_cache_insufficiency": (
                "family validation has fewer than two complete evidence sections"
            ),
            "selection_basis": "frontier_gap_template",
        }
    ]


def test_gap_planning_cap_duplicates_cached_pages_and_batches() -> None:
    candidates = [
        {
            "candidate_id": "gap_1",
            "source_id": "src",
            "source_filename": "Building Manual - Part 6 Appendices.pdf",
            "section": "Fire certificates",
            "section_label": "D",
            "approved_pages": list(range(10, 30)),
            "evidence_families": ["commissioning_results"],
            "expected_evidence": "commissioning body",
            "recommended_route": "certificate",
            "priority_score": 90,
            "current_cache_insufficiency": "gap",
            "selection_basis": "test",
        }
    ]
    approved = approve_gap_parse_plan(candidates, max_new_pages=12)
    batches = split_gap_batches(approved)

    assert sum(row["page_count"] for row in approved) <= 12
    assert all(batch["page_count"] <= 10 for batch in batches)
    validate_parse_batches(batches, {"src": [{"page_number": 1}]})
    requested = [
        (batch["source_id"], page)
        for batch in batches
        for page in range(batch["page_start"], batch["page_end"] + 1)
    ]
    assert len(requested) == len(set(requested))


def test_baseline_reports_supported_families() -> None:
    classifications = [
        {
            "source_id": "src",
            "source_filename": "Part 6.pdf",
            "page_number": 1,
            "primary_page_type": "certificate",
            "recommended_route": "certificate",
            "secondary_tags": [],
        },
        {
            "source_id": "src",
            "source_filename": "Part 6.pdf",
            "page_number": 2,
            "primary_page_type": "certificate",
            "recommended_route": "certificate",
            "secondary_tags": [],
        },
        {
            "source_id": "src",
            "source_filename": "Part 6.pdf",
            "page_number": 3,
            "primary_page_type": "certificate",
            "recommended_route": "certificate",
            "secondary_tags": [],
        },
    ]
    cached = {
        "src": [
            {
                "page_number": 1,
                "extracted_text": "Building control approved inspector certificate ref 1 "
                "12/03/2020 certified completion",
            },
        {
            "page_number": 2,
            "extracted_text": "Electrical system commissioning test ref 2 12/03/2020 "
            "measured result pass completed",
        },
        {
            "page_number": 3,
            "extracted_text": "Approved inspector certificate ref 3 dated 13/03/2020 "
            "certified compliant completion",
        },
        ]
    }
    baseline = build_evidence_gap_baseline(classifications, cached)

    assert "statutory_compliance" not in baseline["families_requiring_parsing"]
    assert baseline["family_validation"]["commissioning_results"]["valid_section_count"] == 1


def test_bounded_real_manifest_integration_dry_run_no_parser_workers() -> None:
    if not DEFAULT_SOURCE_MANIFEST.exists() or not DEFAULT_PAGE_CACHE_ROOT.exists():
        return
    output_root = DEFAULT_EVIDENCE_GAP_OUTPUT_DIR.with_name("test_evidence_gap_refinement")
    if output_root.exists():
        shutil.rmtree(output_root)
    try:
        result = run_evidence_gap_refinement_v1(
            output_dir=output_root,
            max_new_pages=6,
            dry_run=True,
        )

        assert result["parse_execution_report"]["parser_batches_run"] == 0
        assert result["coverage_metrics"]["newly_parsed_pages"] == 0
        assert (output_root / "evidence_family_validation.json").exists()
    finally:
        if output_root.exists():
            shutil.rmtree(output_root)
