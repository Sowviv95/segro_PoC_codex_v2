from __future__ import annotations

import shutil
from pathlib import Path

from segro_evidence_extraction.models.source import SourceRegistryEntry
from segro_evidence_extraction.source_coverage_pageindex import (
    DEFAULT_PAGE_CACHE_ROOT,
    DEFAULT_SOURCE_MANIFEST,
    approve_parse_plan,
    build_evidence_section_map,
    build_page_coverage_map,
    build_source_family_support_map,
    build_source_inventory,
    build_uncached_page_candidates,
    cache_coverage_snapshot,
    classify_cached_pages,
    classify_page_text,
    detect_sections,
    run_source_coverage_pageindex_v1,
    split_approved_parse_batches,
    validate_parse_batches,
)


def test_source_page_count_inventory_and_cache_reconciliation() -> None:
    sources = [source("src_a", "Building Manual - Part 3 Building Services.pdf")]
    cached = {
        "src_a": [
            {"page_number": 1, "extracted_text": "Contents 1 2 3", "text_character_count": 14}
        ]
    }

    inventory = build_source_inventory(
        sources=sources,
        page_counts={"src_a": 12},
        cached_pages=cached,
        hierarchy_nodes=[],
        parser_fingerprint={"parser_name": "pymupdf"},
    )

    assert inventory[0]["total_page_count"] == 12
    assert inventory[0]["cached_page_count"] == 1
    assert inventory[0]["uncached_page_count"] == 11
    assert inventory[0]["cache_percentage"] == 8.33


def test_missing_page_detection_and_cache_snapshot_metrics() -> None:
    sources = [source("src_a", "Building Manual - Part 6 Appendices.pdf")]
    page_map = build_page_coverage_map(
        sources,
        {"src_a": 5},
        {"src_a": [{"page_number": 1}, {"page_number": 3}]},
    )

    assert [row["cache_status"] for row in page_map] == [
        "cached",
        "uncached",
        "cached",
        "uncached",
        "uncached",
    ]
    snapshot = cache_coverage_snapshot(
        [
            {
                "source_id": "src_a",
                "source_filename": "a.pdf",
                "total_page_count": 5,
                "cached_page_count": 2,
                "uncached_page_count": 3,
                "cache_percentage": 40.0,
            }
        ]
    )
    assert snapshot["uncached_pages"] == 3


def test_deterministic_page_classification_rules() -> None:
    examples = {
        "index_or_contents": "Contents 1.0 General 2.0 Roof 3.0 Doors ........ 12",
        "certificate": "Building Control Final Certificate Date 13/03/2020",
        "statutory_planning_decision": (
            "Planning granted by the Local Planning Authority. "
            "The equipment shall be installed prior to occupation as hereby approved."
        ),
        "blank_unusable": "",
        "schedule": "Equipment Schedule Item Description Manufacturer Model Reference",
        "structured_table": "Item   Qty   Ref 1 2 3 4 5 6 7 8",
        "drawing_text_extractable": (
            "Drawing No A100 Title Ground floor plan Scale: 1:100 Revision P01"
        ),
        "drawing_visual_required": "Floor plan see drawing symbol legend layout dependency",
        "product_datasheet": "Technical data sheet manufacturer model specification",
        "maintenance_guidance": "Maintenance recommendations inspect monthly clean annually",
        "low_value_repetitive": "copyright notice Â© Â© Â© repeated publisher footer",
        "separator_or_cover": "Building Manual Part 2 Building Fabric",
    }

    for expected, text in examples.items():
        assert classify_page_text(text)["primary_page_type"] == expected

    index = classify_page_text("Contents 1.0 General 2.0 Roof 3.0 Doors ........ 12")
    assert index["recommended_route"] == "deprioritized"
    assert index["evidence_bearing"] is False
    assert index["evidence_role"] == "navigation only"

    blank = classify_page_text("")
    assert blank["recommended_route"] == "deprioritized"
    assert blank["evidence_bearing"] is False
    assert blank["evidence_role"] == "unusable"


def test_part4_element_safety_literature_and_table_classification_rules() -> None:
    element = classify_page_text(
        "ELEMENT:4.1.7 EXTERNAL WORKS BARRIER "
        "1 NATURE OF INSTALLATION Barrier. "
        "3 PRODUCT DESCRIPTION 1100mm high Armco barrier with handrail, galvanised finish. "
        "7 AS BUILT DRAWINGS Refer to Part 6 Appendix F for Architect's drawings.",
        source_filename="Building Manual - Part 4 External Works.pdf",
    )
    assert element["primary_page_type"] == "project_element_sheet"
    assert element["recommended_route"] == "text"
    assert element["evidence_bearing"] is True
    assert element["evidence_role"] == "direct evidence"

    safety = classify_page_text(
        "SAFETY DATA SHEET according to Regulation (EC) No. 1907/2006 "
        "Sikafloor CureHard-24 Revision Date 05.01.2016 SECTION 2 Hazards identification",
        source_filename="Building Manual - Part 4 External Works.pdf",
    )
    assert safety["primary_page_type"] == "safety_data"
    assert safety["recommended_route"] == "deprioritized"
    assert safety["evidence_bearing"] is False
    assert safety["evidence_role"] == "generic reference literature"

    literature = classify_page_text(
        "Declaration of Performance Unique identification code of product type. "
        "Manufacturer: Hanlon Concrete Products Ltd. Aggregates for Concrete.",
        source_filename="Building Manual - Part 4 External Works.pdf",
    )
    assert literature["primary_page_type"] == "manufacturer_literature"
    assert literature["recommended_route"] == "deprioritized"
    assert literature["evidence_role"] == "generic reference literature"

    table = classify_page_text(
        "Sieve Size (mm) Percent passing Specification Complies Control limits "
        "8 100 100 6.3 100 95 4 93 87 2 80 1 68 Gradation Analysis Test Report",
        source_filename="Building Manual - Part 4 External Works.pdf",
    )
    assert table["primary_page_type"] == "structured_table"
    assert table["recommended_route"] == "table"


def test_part5_health_safety_structural_and_reference_classification_rules() -> None:
    hazard = classify_page_text(
        "5.1 REMAINING IDENTIFIED HAZARDS REMAINING IDENTIFIED HAZARD "
        "PROPOSED CONTROL MEASURE SECTION Roof access by cat ladder danger of falls",
        source_filename="Building Manual - Part 5 The Health & Safety File.pdf",
    )
    assert hazard["primary_page_type"] == "residual_hazard_schedule"
    assert hazard["recommended_route"] == "table"
    assert hazard["evidence_role"] == "operational or safety guidance"

    reference = classify_page_text(
        "5.4 - FIRE SAFETY STRATEGY Refer to Michael Sparks Associates drawings overleaf.",
        source_filename="Building Manual - Part 5 The Health & Safety File.pdf",
    )
    assert reference["primary_page_type"] == "reference_only"
    assert reference["evidence_bearing"] is False
    assert reference["evidence_role"] == "cross-reference only"

    contacts = classify_page_text(
        "5.3 - EMERGENCY CONTACTS Gas Supplier - National Grid "
        "Electricity Supplier - UK Power Networks Emergency Number 0800 3163 105",
        source_filename="Building Manual - Part 5 The Health & Safety File.pdf",
    )
    assert contacts["primary_page_type"] == "emergency_contacts"
    assert contacts["evidence_role"] == "direct evidence"

    asbestos = classify_page_text(
        "ASBESTOS STATEMENT The designers / principal contractor have confirmed "
        "that no asbestos containing products were specified / used within the contract.",
        source_filename="Building Manual - Part 5 The Health & Safety File.pdf",
    )
    assert asbestos["primary_page_type"] == "hazardous_material_statement"

    loading = classify_page_text(
        "OFFICE SLAB 200 THICK R.C. SUSPENDED SLAB designed for imposed load of 7.5KN/m2. "
        "WAREHOUSE GROUND SLAB SPECIFICATION Live warehouse 50kN/m2.",
        source_filename="Building Manual - Part 5 The Health & Safety File.pdf",
    )
    assert loading["primary_page_type"] == "loading_schedule"
    assert loading["recommended_route"] == "table"

    fire = classify_page_text(
        "Fire strategy drawing FD60s FD30s BS 5839 Drawing No MSA-0700 Scale 1:100",
        source_filename="Building Manual - Part 5 The Health & Safety File.pdf",
    )
    assert fire["primary_page_type"] == "fire_strategy_drawing"
    assert fire["recommended_route"] == "drawing_text"


def test_part6_appendix_certificate_schedule_and_template_classification_rules() -> None:
    appendix_index = classify_page_text(
        "PART 6 - INDEX - APPENDICES A POST CONSTRUCTION ENVIRONMENTAL AUDIT "
        "D COMMISSIONING / TEST CERTIFICATES E WORK PERMITS",
        source_filename="Building Manual - Part 6 Appendices.pdf",
    )
    certificate_index = classify_page_text(
        "D - COMMISSIONING / TEST CERTIFICATES Certificates from the following companies "
        "are included within this section. Building Control Certificate Photovoltaic "
        "Commissioning Certificates",
        source_filename="Building Manual - Part 6 Appendices.pdf",
    )
    pv_record = classify_page_text(
        "PV COMMISSIONING FORM - INVERTER 01 Project Segro Park, Enfield - Unit 1 "
        "Array Module 275Wp Quantity 21 Inverter Solis 50k",
        source_filename="Building Manual - Part 6 Appendices.pdf",
    )
    bms_schedule = classify_page_text(
        "Segro Park - Enfield Unit 1 OUTSTATION: 11 Trend IQ4E/64/BAC/230 "
        "Points Schedule Input Device Point Pre Comm Software System Graphic",
        source_filename="Building Manual - Part 6 Appendices.pdf",
    )
    permit = classify_page_text(
        "ROOF WORK PERMIT VALID FOR DAY OF ISSUE ONLY Nature of work Date "
        "ESTIMATED TIME PERIOD Start Finish",
        source_filename="Building Manual - Part 6 Appendices.pdf",
    )

    assert appendix_index["primary_page_type"] == "appendix_index"
    assert appendix_index["evidence_bearing"] is False
    assert certificate_index["primary_page_type"] == "certificate_index"
    assert certificate_index["evidence_bearing"] is False
    assert pv_record["primary_page_type"] == "pv_commissioning_record"
    assert pv_record["recommended_route"] == "table"
    assert bms_schedule["primary_page_type"] == "bms_points_schedule"
    assert bms_schedule["recommended_route"] == "table"
    assert permit["primary_page_type"] == "work_permit_template"
    assert permit["evidence_role"] == "template only"


def test_part3_service_element_schedule_and_literature_classification_rules() -> None:
    element = classify_page_text(
        "ELEMENT:3.1 BUILDING SERVICES MECHANICAL & PUBLIC HEALTH "
        "1 NATURE OF INSTALLATION incoming gas and water services. "
        "3 PRODUCT DESCRIPTION As specialist O&M Manual.",
        source_filename="Building Manual - Part 3 Building Services.pdf",
    )
    test_sheet = classify_page_text(
        "Contract Title: Segro Park Unit 1 System Title: AHU Supply Fan "
        "Fan Manufacturer Fan Type Fan Total Pressure Design Volume "
        "Measured Volume Commissioning Engineer Date",
        source_filename="Building Manual - Part 3 Building Services.pdf",
    )
    literature = classify_page_text(
        "Installation manual for crystalline solar photovoltaic modules "
        "Manufacturer model safety precautions warranty conditions.",
        source_filename="Building Manual - Part 3 Building Services.pdf",
    )

    assert element["primary_page_type"] == "project_element_sheet"
    assert element["evidence_role"] == "direct evidence"
    assert test_sheet["primary_page_type"] == "mechanical_test_sheet"
    assert test_sheet["recommended_route"] == "table"
    assert literature["primary_page_type"] == "manufacturer_literature"
    assert literature["evidence_role"] == "generic reference literature"


def test_part2_index_continuation_and_element_sheet_classification_rules() -> None:
    index_continuation = classify_page_text(
        "BUILDING MANUAL 2.8 JOINERY 2.8.1 2ND FIX CARPENTRY "
        "2.8.2 INTERNAL DOORS INC IRONMONGERY 2.9 INTERNAL FINISHES "
        "2.9.1 WALL FINISHES 2.9.2 FLOOR FINISHES",
        source_filename="Building Manual - Part 2 Building Fabric.pdf",
    )
    assert index_continuation["primary_page_type"] == "index_or_contents"
    assert index_continuation["recommended_route"] == "deprioritized"
    assert index_continuation["evidence_bearing"] is False

    fire_rated_element = classify_page_text(
        "ELEMENT:2.8.2 JOINERY INTERNAL DOORS INC IRONMONGERY "
        "1 NATURE OF INSTALLATION Internal doors including ironmongery. "
        "3 PRODUCT DESCRIPTION FD30 & FD60 solid core American white oak, "
        "with Polyrey Laminate. Ironmongery is Eatilo range satin stainless steel. "
        "7 AS BUILT DRAWINGS Refer to Part 6 Appendix F for Architect's drawings.",
        source_filename="Building Manual - Part 2 Building Fabric.pdf",
    )
    assert fire_rated_element["primary_page_type"] == "project_element_sheet"
    assert fire_rated_element["recommended_route"] == "text"
    assert fire_rated_element["evidence_role"] == "direct evidence"


def test_index_element_certificate_table_schedule_and_section_detection() -> None:
    cached = {
        "src_a": [
            {"page_number": 1, "extracted_text": "Contents 1.0 General 2.0 Doors 3.0 Roof"},
            {"page_number": 2, "extracted_text": "Element Sheet Roof manufacturer product finish"},
            {"page_number": 3, "extracted_text": "Certificate of test date result pass"},
            {
                "page_number": 4,
                "extracted_text": "Equipment Schedule Item Description Manufacturer Model",
            },
        ]
    }
    sources = [source("src_a", "Building Manual - Part 2 Building Fabric.pdf")]
    pages = classify_cached_pages(cached, sources)
    sections = detect_sections(pages, sources, [])

    assert pages[0]["primary_page_type"] == "index_or_contents"
    assert "materials_finishes" in pages[1]["likely_target_families"]
    assert pages[2]["primary_page_type"] == "certificate"
    assert pages[3]["primary_page_type"] == "schedule"
    assert len(sections) >= 3


def test_bounded_parse_planning_overlap_merging_and_no_duplicate_cached_pages() -> None:
    inventory = [
        {
            "source_id": "src_a",
            "source_filename": "Building Manual - Part 6 Appendices.pdf",
            "total_page_count": 15,
            "cached_page_count": 2,
            "uncached_page_count": 13,
            "cache_percentage": 13.33,
        }
    ]
    page_map = [
        {
            "source_id": "src_a",
            "source_filename": "Building Manual - Part 6 Appendices.pdf",
            "page_number": page,
            "cache_status": "cached" if page in {1, 2} else "uncached",
        }
        for page in range(1, 16)
    ]

    candidates = build_uncached_page_candidates(
        inventory=inventory,
        page_map=page_map,
        sections=[],
    )
    approved = approve_parse_plan(candidates, max_new_pages=20)
    batches = split_approved_parse_batches(approved)

    assert [(batch["page_start"], batch["page_end"]) for batch in batches] == [(3, 10)]
    assert all(batch["page_count"] <= 10 for batch in batches)
    validate_parse_batches(
        batches,
        {"src_a": [{"page_number": 1}, {"page_number": 2}]},
    )


def test_validate_parse_batches_rejects_duplicates_and_cached_pages() -> None:
    duplicate = [
        {"source_id": "src_a", "page_start": 1, "page_end": 2, "page_count": 2},
        {"source_id": "src_a", "page_start": 2, "page_end": 3, "page_count": 2},
    ]

    try:
        validate_parse_batches(duplicate, {})
    except ValueError as exc:
        assert "Duplicate parse request" in str(exc)
    else:
        raise AssertionError("expected duplicate parse request to fail")

    try:
        validate_parse_batches(
            [{"source_id": "src_a", "page_start": 1, "page_end": 1, "page_count": 1}],
            {"src_a": [{"page_number": 1}]},
        )
    except ValueError as exc:
        assert "Already cached page" in str(exc)
    else:
        raise AssertionError("expected cached page parse request to fail")


def test_hierarchy_enrichment_and_source_family_support_mapping() -> None:
    sections = [
        {
            "source_id": "src_a",
            "source_filename": "a.pdf",
            "title": "Certificate",
            "page_start": 1,
            "page_end": 1,
            "evidence_bearing": True,
            "likely_target_families": ["dates_certificates"],
            "likely_dictionary_domains": ["appendices_certificates"],
            "recommended_route": "text",
        }
    ]

    evidence = build_evidence_section_map(sections)
    support = build_source_family_support_map(evidence)

    assert evidence[0]["supported_target_family_hints"] == ["dates_certificates"]
    assert support[0]["likely_dictionary_domains"] == ["appendices_certificates"]


def test_real_source_manifest_loading_dry_run_without_model_calls() -> None:
    if not DEFAULT_SOURCE_MANIFEST.exists() or not DEFAULT_PAGE_CACHE_ROOT.exists():
        return
    output_root = Path("output/test_source_coverage_pageindex")
    if output_root.exists():
        shutil.rmtree(output_root)
    try:
        result = run_source_coverage_pageindex_v1(
            output_dir=output_root,
            dry_run=True,
            max_new_pages=5,
        )
        metrics = result["coverage_metrics"]

        assert metrics["total_registered_sources"] >= 1
        assert metrics["total_pdf_sources"] >= 1
        assert metrics["new_pages_parsed"] == 0
        assert result["parse_execution_report"]["parser_batches_run"] == 0
        assert (output_root / "source_coverage_inventory.json").exists()
        assert (output_root / "next_target_selection_readiness.md").exists()
    finally:
        if output_root.exists():
            shutil.rmtree(output_root)


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
