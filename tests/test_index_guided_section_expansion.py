from __future__ import annotations

import json
import shutil
from pathlib import Path

from segro_evidence_extraction.index_guided_section_expansion import (
    DEFAULT_INDEX_EXPANSION_OUTPUT_DIR,
    DEFAULT_PAGE_CACHE_ROOT,
    DEFAULT_SOURCE_MANIFEST,
    approve_section_parse_plan,
    audit_v1_artifacts,
    build_baseline_reconciliation,
    build_uncached_section_candidates,
    classify_page_text_v2,
    extract_contents_index_entries,
    normalize_heading,
    parse_contents_entries,
    resolve_printed_page_to_pdf_page,
    resolve_section_boundaries,
    route_for_section,
    run_index_guided_section_expansion_v1,
    score_sections,
    split_approved_section_batches,
    split_section_label_title,
    validate_parse_batches,
)
from segro_evidence_extraction.models.source import SourceRegistryEntry


def test_v1_artifact_reconciliation_and_baseline_handling() -> None:
    output_root = Path("output/test_index_guided_artifact_audit")
    if output_root.exists():
        shutil.rmtree(output_root)
    v1 = output_root / "v1"
    v1.mkdir(parents=True)
    try:
        write_json(v1 / "page_classifications.json", [page("src_a", 1, "narrative")])
        write_json(
            v1 / "section_inventory.json",
            [{"source_id": "src_a", "page_start": 1, "page_end": 1}],
        )
        for name in [
            "table_route_pages.json",
            "certificate_route_pages.json",
            "visual_route_pages.json",
            "low_value_pages.json",
        ]:
            write_json(v1 / name, [])
        write_json(v1 / "coverage_metrics.json", {"page_type_distribution": {"narrative": 1}})
        write_json(v1 / "parse_execution_report.json", {"parse_failure_count": 0})

        audit = audit_v1_artifacts(
            v1_output_dir=v1,
            sources=[source("src_a", "Part 2.pdf")],
            page_counts={"src_a": 2},
            cached_pages={"src_a": [{"page_number": 1}]},
        )
        baseline = build_baseline_reconciliation(
            audit,
            [{"source_id": "src_a", "cached_page_count": 1}],
        )

        assert audit["status"] == "pass"
        assert baseline["historical_baseline_before_source_coverage_v1"]["status"] == (
            "historical baseline unavailable"
        )
        assert baseline["current_cache_before_index_expansion_v1"]["cached_pages"] == 1
    finally:
        if output_root.exists():
            shutil.rmtree(output_root)


def test_unknown_page_reclassification_reasons_are_generic() -> None:
    classified = classify_page_text_v2(
        "Equipment Schedule Item Description Manufacturer Model Ref 1 2 3",
        source_filename="Building Manual - Part 3 Building Services.pdf",
    )

    assert classified["primary_page_type"] == "schedule"
    assert classified["recommended_route"] == "table"
    assert "schedule" in classified["classification_reason"]
    assert classified["rule_signals"]["table_label_terms"] >= 2


def test_numbered_and_appendix_heading_detection() -> None:
    assert normalize_heading("2.3.1 ROOF COVERINGS") == "2 3 1 roof coverings"
    assert split_section_label_title("2.3.1 ROOF COVERINGS") == ("2.3.1", "ROOF COVERINGS")
    assert split_section_label_title("D COMMISSIONING / TEST CERTIFICATES") == (
        "D",
        "COMMISSIONING / TEST CERTIFICATES",
    )


def test_contents_entry_extraction_from_numbered_and_lettered_entries() -> None:
    entries = parse_contents_entries(
        "2.3\nROOF\n2.3.1\nROOF COVERINGS\nD COMMISSIONING / TEST CERTIFICATES 120",
        source_id="src_a",
        source_filename="Part 2.pdf",
        index_page=4,
    )

    labels = {entry["section_label"] for entry in entries}
    assert {"2.3", "2.3.1", "D"} <= labels
    assert any(entry["printed_page"] == 120 for entry in entries)


def test_printed_page_to_pdf_page_offset_resolution() -> None:
    assert resolve_printed_page_to_pdf_page({"printed_page": 12}, {7, 8, 9}) == 7
    assert resolve_printed_page_to_pdf_page({"printed_page": 50}, {1, 2}) == 50


def test_section_start_end_resolution_and_unresolved_handling() -> None:
    entries = [
        {
            "entry_id": "a",
            "source_id": "src_a",
            "section_label": "2.3",
            "section_title": "ROOF",
            "normalized_section_title": "roof",
            "parent_section": "2",
            "candidate_pdf_page": None,
            "printed_page": None,
        },
        {
            "entry_id": "b",
            "source_id": "src_a",
            "section_label": "9.9",
            "section_title": "MISSING",
            "normalized_section_title": "missing",
            "parent_section": "9",
            "candidate_pdf_page": None,
            "printed_page": None,
        },
    ]
    resolved = resolve_section_boundaries(
        index_entries=entries,
        hierarchy_nodes=[
            {"source_id": "src_a", "title": "2.3 ROOF", "page_start": 20, "page_end": 25}
        ],
        page_counts={"src_a": 100},
        cached_pages={"src_a": [{"page_number": 20}]},
        sources=[source("src_a", "Building Manual - Part 2 Building Fabric.pdf")],
    )
    by_label = {row["section_label"]: row for row in resolved}

    assert by_label["2.3"]["resolved"] is True
    assert by_label["2.3"]["pdf_page_start"] == 20
    assert by_label["2.3"]["pdf_page_end"] == 25
    assert by_label["9.9"]["resolved"] is False


def test_evidence_value_scoring_boosts_and_penalties() -> None:
    scored = score_sections(
        [
            resolved_section("2.6.2", "LOADING DOORS"),
            resolved_section("A", "GENERIC MAINTENANCE COSHH SAFETY DATA"),
            resolved_section("D", "COMMISSIONING TEST CERTIFICATES"),
        ]
    )
    by_title = {row["section_title"]: row for row in scored}

    assert by_title["COMMISSIONING TEST CERTIFICATES"]["evidence_value_score"] > 20
    assert by_title["LOADING DOORS"]["evidence_value_score"] > 10
    assert by_title["GENERIC MAINTENANCE COSHH SAFETY DATA"]["evidence_value_score"] < 10
    assert route_for_section("COMMISSIONING TEST CERTIFICATES") == "certificate"


def test_planning_caps_total_cap_overlap_cache_reuse_and_batches() -> None:
    candidates = build_uncached_section_candidates(
        scored_sections=[
            {**resolved_section("2.6.2", "LOADING DOORS"), "evidence_value_score": 50},
            {**resolved_section("2.6.4", "DOCK LEVELLERS"), "evidence_value_score": 45},
        ],
        page_map=[
            {
                "source_id": "src_a",
                "source_filename": "Building Manual - Part 2 Building Fabric.pdf",
                "page_number": page_number,
                "cache_status": "cached" if page_number == 10 else "uncached",
            }
            for page_number in range(10, 31)
        ],
        page_counts={"src_a": 100},
    )
    approved = approve_section_parse_plan(
        candidates,
        max_new_pages=8,
        per_source_caps={"part 2": 6},
    )
    batches = split_approved_section_batches(approved)

    assert sum(item["page_count"] for item in approved) <= 6
    assert all(batch["page_count"] <= 10 for batch in batches)
    validate_parse_batches(batches, {"src_a": [{"page_number": 10}]})
    assert {page for item in approved for page in item["approved_pages"]}.isdisjoint({10})


def test_no_duplicate_or_cached_parse_validation() -> None:
    duplicate_batches = [
        {"source_id": "src_a", "page_start": 1, "page_end": 2, "page_count": 2},
        {"source_id": "src_a", "page_start": 2, "page_end": 3, "page_count": 2},
    ]
    try:
        validate_parse_batches(duplicate_batches, {})
    except ValueError as exc:
        assert "Duplicate parse request" in str(exc)
    else:
        raise AssertionError("duplicate page parse request should fail")

    try:
        validate_parse_batches(
            [{"source_id": "src_a", "page_start": 1, "page_end": 1, "page_count": 1}],
            {"src_a": [{"page_number": 1}]},
        )
    except ValueError as exc:
        assert "Already cached page" in str(exc)
    else:
        raise AssertionError("cached page parse request should fail")


def test_extract_contents_entries_from_hierarchy_nodes() -> None:
    entries = extract_contents_index_entries(
        classifications=[],
        cached_pages={},
        hierarchy_nodes=[
            {"source_id": "src_a", "title": "ELEMENT:2.6.4 DOCK LEVELLERS", "page_start": 80}
        ],
    )

    assert entries[0]["section_label"] == "2.6.4"
    assert entries[0]["entry_type"] == "element_or_component_section"


def test_bounded_real_manifest_integration_dry_run() -> None:
    if not DEFAULT_SOURCE_MANIFEST.exists() or not DEFAULT_PAGE_CACHE_ROOT.exists():
        return
    output_root = DEFAULT_INDEX_EXPANSION_OUTPUT_DIR.with_name("test_index_guided_expansion")
    if output_root.exists():
        shutil.rmtree(output_root)
    try:
        result = run_index_guided_section_expansion_v1(
            output_dir=output_root,
            dry_run=True,
            max_new_pages=12,
        )

        assert result["coverage_metrics"]["newly_parsed_pages"] == 0
        assert result["parse_execution_report"]["parser_batches_run"] == 0
        assert result["coverage_metrics"]["detected_index_entries"] > 0
        assert (output_root / "v1_artifact_audit.json").exists()
        assert (output_root / "approved_section_parse_plan.json").exists()
    finally:
        if output_root.exists():
            shutil.rmtree(output_root)


def page(source_id: str, page_number: int, page_type: str) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_filename": "Part 2.pdf",
        "page_number": page_number,
        "primary_page_type": page_type,
    }


def resolved_section(label: str, title: str) -> dict[str, object]:
    return {
        "entry_id": label,
        "source_id": "src_a",
        "source_filename": "Building Manual - Part 2 Building Fabric.pdf",
        "section_label": label,
        "section_title": title,
        "normalized_section_title": normalize_heading(title),
        "parent_section": "",
        "pdf_page_start": 10,
        "pdf_page_end": 20,
        "page_count": 11,
        "uncached_pages_in_range": list(range(11, 21)),
        "range_already_cached": False,
        "resolved": True,
    }


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


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
