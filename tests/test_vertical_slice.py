from __future__ import annotations

from pathlib import Path

import pytest

from segro_evidence_extraction.models.common import ExpectedDataType, ProvenanceRef
from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.parsing.models import (
    OcrRouting,
    ParsedPage,
    ParsingStatus,
    TextQualityMetrics,
)
from segro_evidence_extraction.parsing.service import load_source_registry
from segro_evidence_extraction.vertical_slice import (
    AbstainingExtractionClient,
    EvidenceBundleRecord,
    EvidenceFirstVerticalSliceService,
    EvidenceValidationResult,
    ExtractionResult,
    RetrievalResult,
    RetrievalScoreBreakdown,
    RetrievedEvidence,
    SelectedTarget,
    ShapeValidationResult,
    ValidationResult,
    assess_schema_compatibility,
    assess_schema_compatibility_v3,
    build_evidence_bundle,
    build_lightweight_hierarchy,
    build_overall_validation_results,
    calibrate_extraction_value,
    derive_evidence_value_and_normalization,
    infer_value_shape_assignment,
    materialize_selected_spans,
    parse_extraction_response,
    retrieve_evidence,
    select_source_ranges,
    select_vertical_slice_targets,
    split_range,
    validate_evidence_layer,
    validate_extractions,
    validate_shape_layer,
)


def test_hierarchy_node_ids_and_relationships_are_deterministic() -> None:
    pages = {"src-test": [_page(1, "1.1 - THE WORKS\nSteel frame warehouse.")]}
    registry = {"src-test": _source("src-test", "Building Manual - Part 1 General.pdf")}

    first = build_lightweight_hierarchy(pages, registry)
    second = build_lightweight_hierarchy(pages, registry)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    ids = {node.node_id for node in first.nodes}
    assert all(node.parent_node_id is None or node.parent_node_id in ids for node in first.nodes)
    document = next(node for node in first.nodes if node.node_type == "document")
    assert document.child_node_ids


def test_headings_and_certificate_titles_create_nodes() -> None:
    pages = {
        "src-test": [
            _page(
                1,
                "Certificate of Practical Completion\n"
                "Date of practical completion: 23rd April 2020",
            )
        ]
    }
    registry = {"src-test": _source("src-test", "Building Manual - Part 1 General.pdf")}

    hierarchy = build_lightweight_hierarchy(pages, registry)

    assert any("Certificate of Practical Completion" in node.title for node in hierarchy.nodes)


def test_retrieval_ranks_matching_section_above_unrelated_and_uses_title_match() -> None:
    target = _target("dock_leveller_count", ExpectedDataType.INTEGER, "Dock Levellers - Count")
    pages = {
        "src-test": [
            _page(1, "1.1.1 - DESCRIPTION\nUnrelated office maintenance text."),
            _page(2, "Dock Levellers\nThe warehouse has 5No Dock Levellers."),
        ]
    }
    registry = {"src-test": _source("src-test", "Building Manual - Part 1 General.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)

    retrieval = retrieve_evidence(target, hierarchy, pages, registry, max_evidence_chars=1200)

    assert retrieval.results[0].page_start == 2
    assert retrieval.results[0].score_components.title_match > 0


def test_component_attribute_proximity_beats_generic_numeric_floor_match() -> None:
    target = _target("office_area_value", ExpectedDataType.DECIMAL, "Office Area", "m2")
    pages = {
        "src-test": [
            _page(1, "Concrete floor slab loading is 50kN/m2 with bay joints."),
            _page(2, "Office area\nThe office floor area is 125 m2 GEA."),
        ]
    }
    registry = {"src-test": _source("src-test", "Building Manual - Part 1 General.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)

    retrieval = retrieve_evidence(target, hierarchy, pages, registry, max_evidence_chars=1200)

    assert retrieval.retrieval_status == "evidence_found"
    assert retrieval.results[0].page_start == 2
    assert retrieval.results[0].score_components.component_attribute_proximity > 0


def test_weak_unrelated_office_area_evidence_is_not_forced() -> None:
    target = _target("office_area_value", ExpectedDataType.DECIMAL, "Office Area", "m2")
    pages = {"src-test": [_page(1, "Concrete floor slab loading is 50kN/m2.")]}
    registry = {"src-test": _source("src-test", "Building Manual - Part 4 External Works.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)

    retrieval = retrieve_evidence(target, hierarchy, pages, registry, max_evidence_chars=1200)

    assert retrieval.retrieval_status in {"weak_evidence", "no_relevant_evidence"}
    assert not retrieval.results or retrieval.results[0].page_start != 1


def test_dock_leveller_and_fire_alarm_specific_evidence_rank_above_generic_text() -> None:
    dock = _target("dock_leveller_count", ExpectedDataType.INTEGER, "Dock Levellers - Count")
    fire = _target(
        "fire_alarm_system_description",
        ExpectedDataType.STRING,
        "Fire alarm system category",
    )
    pages = {
        "src-test": [
            _page(1, "PV maintenance access and roof warranties."),
            _page(2, "5No Dock Levellers Warehouse loading bay."),
            _page(3, "Generic fire safety responsibilities only."),
            _page(4, "Fire Alarm System\nCategory L2 fire detection and alarm system."),
        ]
    }
    registry = {"src-test": _source("src-test", "manual.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)

    dock_retrieval = retrieve_evidence(dock, hierarchy, pages, registry, max_evidence_chars=1200)
    fire_retrieval = retrieve_evidence(fire, hierarchy, pages, registry, max_evidence_chars=1200)

    assert dock_retrieval.results[0].page_start == 2
    assert fire_retrieval.results[0].page_start == 4


def test_roofing_system_phrase_beats_generic_roof_description() -> None:
    target = _target(
        "roof_construction_description",
        ExpectedDataType.STRING,
        "Roof construction description",
    )
    pages = {
        "src-test": [
            _page(
                1,
                "Description of the Building and Facilities\n"
                "The construction of a steel frame warehouse with profiled metal clad "
                "elevations and roof, rooflights.",
            ),
            _page(
                2,
                "Roof Plan\nBuilt-up curved standing roofing system. "
                "Class B non-fragile rooflights.",
            ),
        ]
    }
    registry = {"src-test": _source("src-test", "Building Manual - Part 1 General.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)

    retrieval = retrieve_evidence(target, hierarchy, pages, registry, max_evidence_chars=1200)

    assert retrieval.results[0].page_start == 2


def test_certificate_value_page_beats_overleaf_cross_reference() -> None:
    target = _target(
        "practical_completion_date",
        ExpectedDataType.DATE,
        "Practical completion date",
    )
    pages = {
        "src-test": [
            _page(
                1,
                "1.2.5 - PRACTICAL COMPLETION CERTIFICATE "
                "Refer to Certificate of Practical Completion dated 23rd April 2020 overleaf.",
            ),
            _page(
                2,
                "Certificate of Practical Completion\n"
                "Issue date: 23rd April 2020\n"
                "Date of practical completion: 5:00pm on 23rd April 2020",
            ),
        ]
    }
    registry = {"src-test": _source("src-test", "Building Manual - Part 1 General.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)

    retrieval = retrieve_evidence(target, hierarchy, pages, registry, max_evidence_chars=1200)

    assert retrieval.results[0].page_start == 2


def test_building_control_certificate_reference_is_retrievable() -> None:
    target = _target(
        "building_control_certificate_reference_date",
        ExpectedDataType.STRING,
        "Building Control final certificate reference and date",
    )
    pages = {
        "src-test": [
            _page(1, "Building Control Final Certificate Date: 13/03/2020 Assent Ref: B152318.")
        ]
    }
    registry = {"src-test": _source("src-test", "Building Manual - Part 1 General.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)

    retrieval = retrieve_evidence(target, hierarchy, pages, registry, max_evidence_chars=1200)

    assert retrieval.retrieval_status == "evidence_found"
    assert retrieval.results[0].page_start == 1


def test_planning_requirement_does_not_satisfy_installed_model_target() -> None:
    target = _target(
        "ev_charger_installed_model_name",
        ExpectedDataType.STRING,
        "EV charger installed model",
    )
    pages = {
        "src-test": [
            _page(
                1,
                "Planning granted by the Local Planning Authority. Electric vehicle "
                "charging points shall be installed prior to occupation in accordance "
                "with approved details.",
            )
        ]
    }
    registry = {"src-test": _source("src-test", "Building Manual - Part 1 General.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)

    retrieval = retrieve_evidence(target, hierarchy, pages, registry, max_evidence_chars=1200)

    assert retrieval.retrieval_status in {"weak_evidence", "no_relevant_evidence"}
    assert retrieval.results == []


def test_certificate_date_does_not_satisfy_generic_equipment_installation_date() -> None:
    target = _target(
        "equipment_installation_date",
        ExpectedDataType.DATE,
        "Equipment installation date",
    )
    pages = {
        "src-test": [
            _page(
                1,
                "Assent Building Control Ltd Final Certificate Date: 13/03/2020 "
                "Assent Ref: B152318.",
            )
        ]
    }
    registry = {"src-test": _source("src-test", "Building Manual - Part 1 General.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)

    retrieval = retrieve_evidence(target, hierarchy, pages, registry, max_evidence_chars=1200)

    assert retrieval.retrieval_status in {"weak_evidence", "no_relevant_evidence"}
    assert retrieval.results == []


def test_retrieval_deduplicates_page_section_and_text_block_overlap() -> None:
    target = _target("dock_leveller_count", ExpectedDataType.INTEGER, "Dock Levellers - Count")
    pages = {"src-test": [_page(1, "Dock Levellers\nThe warehouse has 5No Dock Levellers.")]}
    registry = {"src-test": _source("src-test", "Building Manual - Part 1 General.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)

    retrieval = retrieve_evidence(target, hierarchy, pages, registry, max_evidence_chars=1200)

    assert retrieval.retrieval_status == "evidence_found"
    assert len(retrieval.results) == 1


def test_retrieval_bundle_is_bounded_and_preserves_provenance() -> None:
    target = _target("roof_construction_description", ExpectedDataType.STRING, "Roof")
    evidence = RetrievedEvidence(
        target_row_id=target.target_row_id,
        rank=1,
        node_id="node-1",
        source_id="src-test",
        source_file="manual.pdf",
        page_start=9,
        page_end=9,
        score=10,
        score_components=RetrievalScoreBreakdown(title_match=3),
        matched_terms=["roof"],
        hierarchy_path=["manual", "roof"],
        excerpt="roof " * 500,
    )
    retrieval = RetrievalResult(
        target_row_id=target.target_row_id,
        query="roof",
        results=[evidence],
        retrieval_time_ms=1,
        top_score=10,
    )

    bundle = build_evidence_bundle(target, retrieval, max_evidence_chars=300)

    assert bundle.character_count <= 300
    assert bundle.evidence_items[0].source_id == "src-test"
    assert bundle.evidence_items[0].node_id == "node-1"


def test_evidence_span_ids_offsets_and_materialization_are_canonical() -> None:
    target = _target("dock_leveller_count", ExpectedDataType.INTEGER, "Dock Levellers - Count")
    pages = {"src-test": [_page(2, "Intro\nThe warehouse has 5No Dock Levellers.\nEnd")]}
    registry = {"src-test": _source("src-test", "manual.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)
    retrieval = retrieve_evidence(target, hierarchy, pages, registry, max_evidence_chars=1200)
    bundle = build_evidence_bundle(target, retrieval, 1200, pages)
    span = bundle.evidence_spans[0]
    second = build_evidence_bundle(target, retrieval, 1200, pages).evidence_spans[0]

    assert span.model_dump(mode="json") == second.model_dump(mode="json")
    assert pages["src-test"][0].text[span.start_char : span.end_char] == span.text

    parsed = parse_extraction_response(
        target=target,
        bundle=bundle,
        provider="mock",
        model_name="mock-model",
        response_text=(
            '{"status":"extracted","extracted_value":"5","normalized_value":5,'
            f'"confidence":0.8,"selected_supporting_span_ids":["{span.span_id}"],'
            '"supporting_evidence_excerpt":"paraphrased by model"}'
        ),
    )
    materialized = materialize_selected_spans(parsed, bundle)

    assert materialized.supporting_evidence_excerpt == span.text
    assert materialized.source_id == span.source_id
    assert materialized.page_number == span.page_number


def test_structured_extraction_response_is_parsed_safely() -> None:
    target = _target("dock_leveller_count", ExpectedDataType.INTEGER, "Dock Levellers - Count")
    bundle = _bundle(target, "The warehouse has 5No Dock Levellers.")

    result = parse_extraction_response(
        target=target,
        bundle=bundle,
        provider="mock",
        model_name="mock-model",
        response_text=(
            '{"status":"extracted","extracted_value":"5","normalized_value":5,'
            '"confidence":0.8,"supporting_evidence_excerpt":"5No Dock Levellers",'
            '"source_id":"src-test","page_number":2,"hierarchy_node_id":"node-1"}'
        ),
    )

    assert result.status == "extracted"
    assert result.page_number == 2
    assert result.hierarchy_node_id == "node-1"


def test_unknown_span_id_fails_validation() -> None:
    target = _target("dock_leveller_count", ExpectedDataType.INTEGER, "Dock Levellers - Count")
    selected = _selected(target)
    pages = {"src-test": [_page(2, "The warehouse has 5No Dock Levellers.")]}
    registry = {"src-test": _source("src-test", "manual.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)
    node_id = next(node.node_id for node in hierarchy.nodes if node.node_type == "text_block")
    extraction = ExtractionResult(
        target_row_id=target.target_row_id,
        requirement_id=target.requirement_id,
        status="extracted",
        confidence=0.7,
        extracted_value="5",
        source_id="src-test",
        page_number=2,
        hierarchy_node_id=node_id,
        supporting_span_ids=["span_missing"],
        supporting_evidence_excerpt="The warehouse has 5No Dock Levellers.",
        model_provider="mock",
        model_name="mock",
    )
    bundle = EvidenceBundleRecord(
        target_row_id=target.target_row_id,
        evidence_items=[],
        evidence_spans=[],
        combined_text="",
        character_count=0,
        token_estimate=1,
    )

    result = validate_extractions([extraction], [selected], pages, hierarchy, [bundle])[0]

    assert result.status == "invalid"
    assert "selected span id is not in supplied bundle" in " ".join(result.issues)


def test_malformed_model_output_becomes_invalid_format() -> None:
    target = _target("dock_leveller_count", ExpectedDataType.INTEGER, "Dock Levellers - Count")
    bundle = _bundle(target, "The warehouse has 5No Dock Levellers.")

    result = parse_extraction_response(
        target=target,
        bundle=bundle,
        provider="mock",
        model_name="mock-model",
        response_text="not json",
    )

    assert result.status == "invalid_format"


def test_unsupported_evidence_abstains_with_mock_client() -> None:
    target = _target("office_area_value", ExpectedDataType.DECIMAL, "Office Area")
    result = AbstainingExtractionClient().extract(
        target=target,
        bundle=EvidenceBundleRecord(
            target_row_id=target.target_row_id,
            evidence_items=[],
            combined_text="",
            character_count=0,
            token_estimate=1,
        ),
        max_output_tokens=100,
    )

    assert result.status == "insufficient_evidence"


def test_validation_rejects_missing_or_wrong_evidence_excerpt() -> None:
    target = _target("dock_leveller_count", ExpectedDataType.INTEGER, "Dock Levellers - Count")
    selected = _selected(target)
    pages = {"src-test": [_page(2, "The warehouse has 5No Dock Levellers.")]}
    registry = {"src-test": _source("src-test", "manual.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)
    node_id = next(node.node_id for node in hierarchy.nodes if node.node_type == "text_block")
    no_evidence = ExtractionResult(
        target_row_id=target.target_row_id,
        requirement_id=target.requirement_id,
        status="extracted",
        confidence=0.7,
        extracted_value="5",
        model_provider="mock",
        model_name="mock",
    )
    wrong_excerpt = no_evidence.model_copy(
        update={
            "source_id": "src-test",
            "page_number": 2,
            "hierarchy_node_id": node_id,
            "supporting_evidence_excerpt": "not on the page",
        }
    )

    results = validate_extractions([no_evidence, wrong_excerpt], [selected], pages, hierarchy)

    assert results[0].status == "invalid"
    assert "supporting evidence excerpt is not present" in " ".join(results[1].issues)


def test_validation_checks_dates_numerics_and_units_without_changing_value() -> None:
    date_target = _target("construction_date", ExpectedDataType.DATE, "Construction Date")
    numeric_target = _target("pv_panel_capacity_kwp", ExpectedDataType.DECIMAL, "PV Capacity", "kW")
    pages = {"src-test": [_page(1, "Issue date: 2020-04-23"), _page(2, "PV capacity: not known")]}
    registry = {"src-test": _source("src-test", "manual.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)
    node_id = next(node.node_id for node in hierarchy.nodes if node.node_type == "text_block")
    valid_date = ExtractionResult(
        target_row_id=date_target.target_row_id,
        requirement_id=date_target.requirement_id,
        status="extracted",
        confidence=0.8,
        extracted_value="2020-04-23",
        source_id="src-test",
        page_number=1,
        hierarchy_node_id=node_id,
        supporting_evidence_excerpt="Issue date: 2020-04-23",
        model_provider="mock",
        model_name="mock",
    )
    bad_number = ExtractionResult(
        target_row_id=numeric_target.target_row_id,
        requirement_id=numeric_target.requirement_id,
        status="extracted",
        confidence=0.8,
        extracted_value="not known",
        unit="MW",
        source_id="src-test",
        page_number=2,
        hierarchy_node_id=node_id,
        supporting_evidence_excerpt="PV capacity: not known",
        model_provider="mock",
        model_name="mock",
    )

    results = validate_extractions(
        [valid_date, bad_number],
        [_selected(date_target), _selected(numeric_target)],
        pages,
        hierarchy,
    )

    assert results[0].status == "valid"
    assert results[1].status == "invalid"
    assert bad_number.extracted_value == "not known"
    assert "decimal value is not parseable" in results[1].issues
    assert "unit does not match target unit" in results[1].issues


def test_schema_compatibility_reports_dictionary_caveat_separately() -> None:
    target = _target(
        "frame_construction_description",
        ExpectedDataType.DECIMAL,
        "Frame construction description",
        "m",
    )
    extraction = ExtractionResult(
        target_row_id=target.target_row_id,
        requirement_id=target.requirement_id,
        status="extracted",
        confidence=0.8,
        extracted_value="steel portal frame",
        unit="m",
        source_id="src-test",
        page_number=1,
        hierarchy_node_id="node-1",
        supporting_span_ids=["span-1"],
        supporting_evidence_excerpt="steel portal frame",
        model_provider="mock",
        model_name="mock",
    )
    validation = ValidationResult(
        target_row_id=target.target_row_id,
        status="invalid",
        evidence_valid=True,
        value_format_valid=False,
        issues=["decimal value is not parseable"],
    )

    result = assess_schema_compatibility([extraction], [_selected(target)], [validation])[0]

    assert target.expected_data_type == ExpectedDataType.DECIMAL
    assert target.unit == "m"
    assert result.compatibility == "narrative_value_against_structured_constraint"
    assert result.overall_review_status == "valid_with_dictionary_caveat"


def test_v3_count_and_date_normalization_from_evidence() -> None:
    count_target = _target("dock_leveller_count", ExpectedDataType.INTEGER, "Dock count")
    count_assignment = infer_value_shape_assignment(count_target)
    count = derive_evidence_value_and_normalization(
        target=count_target,
        assignment=count_assignment,
        raw_model_value="There are five dock levellers.",
        value_bearing_quote="5No Dock Levellers",
        span_text="Grid\n5No Dock Levellers\nWarehouse",
    )
    assert count.evidence_value == "5No Dock Levellers"
    assert count.normalized_value == 5

    date_target = _target("construction_date", ExpectedDataType.DATE, "Construction date")
    date = derive_evidence_value_and_normalization(
        target=date_target,
        assignment=infer_value_shape_assignment(date_target),
        raw_model_value="23rd April 2020",
        value_bearing_quote=None,
        span_text="Certificate dated 23rd April 2020",
    )
    assert date.normalized_value == "2020-04-23"
    iso = derive_evidence_value_and_normalization(
        target=date_target,
        assignment=infer_value_shape_assignment(date_target),
        raw_model_value="2020-04-23",
        value_bearing_quote=None,
        span_text="Date: 2020-04-23",
    )
    assert iso.normalized_value == "2020-04-23"


def test_v3_use_classes_normalize_to_list_without_losing_evidence() -> None:
    target = _target("approved_use_classes", ExpectedDataType.ENUM, "Approved use classes")
    assignment = infer_value_shape_assignment(target)

    result = derive_evidence_value_and_normalization(
        target=target,
        assignment=assignment,
        raw_model_value="B1c, B2 and B8",
        value_bearing_quote=None,
        span_text="3 industrial units for B1c, B2 or B8 uses",
    )

    assert result.normalized_value == ["B1c", "B2", "B8"]
    assert result.evidence_value == "B1c, B2 or B8"


def test_v3_descriptive_value_does_not_require_full_verbatim_model_value() -> None:
    target = _target(
        "fire_alarm_system_description",
        ExpectedDataType.STRING,
        "Fire alarm system description",
        "m",
    )
    assignment = infer_value_shape_assignment(target)
    pages = {"src-test": [_page(10, "Fire Alarm & Associated Systems\nMust be tested annually.")]}
    registry = {"src-test": _source("src-test", "manual.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)
    node_id = next(node.node_id for node in hierarchy.nodes if node.node_type == "text_block")
    span = EvidenceBundleRecord(
        target_row_id=target.target_row_id,
        evidence_items=[],
        evidence_spans=[
            _span(
                "span-1",
                "src-test",
                10,
                node_id,
                "Fire Alarm & Associated Systems\nMust be tested annually.",
            )
        ],
        combined_text="",
        character_count=0,
        token_estimate=1,
    )
    extraction = ExtractionResult(
        target_row_id=target.target_row_id,
        requirement_id=target.requirement_id,
        status="extracted",
        confidence=0.8,
        raw_model_value="The fire alarm system must be tested annually.",
        extracted_value="The fire alarm system must be tested annually.",
        value_bearing_quote="Fire Alarm & Associated Systems",
        supporting_span_ids=["span-1"],
        supporting_evidence_excerpt="Fire Alarm & Associated Systems\nMust be tested annually.",
        source_id="src-test",
        source_file="manual.pdf",
        page_number=10,
        hierarchy_node_id=node_id,
        model_provider="mock",
        model_name="mock",
    )
    calibrated = calibrate_extraction_value(
        extraction=extraction,
        target=target,
        assignment=assignment,
        bundle=span,
    )
    evidence = validate_evidence_layer([calibrated], [_selected(target)], pages, hierarchy, [span])
    shape = validate_shape_layer([calibrated], [assignment])

    assert evidence[0].status == "valid"
    assert shape[0].status == "valid"


def test_v3_evidence_value_must_occur_in_selected_span() -> None:
    target = _target("dock_leveller_count", ExpectedDataType.INTEGER, "Dock count")
    pages = {"src-test": [_page(2, "5No Dock Levellers")]}
    registry = {"src-test": _source("src-test", "manual.pdf")}
    hierarchy = build_lightweight_hierarchy(pages, registry)
    node_id = next(node.node_id for node in hierarchy.nodes if node.node_type == "text_block")
    bundle = EvidenceBundleRecord(
        target_row_id=target.target_row_id,
        evidence_items=[],
        evidence_spans=[_span("span-1", "src-test", 2, node_id, "5No Dock Levellers")],
        combined_text="",
        character_count=0,
        token_estimate=1,
    )
    extraction = ExtractionResult(
        target_row_id=target.target_row_id,
        requirement_id=target.requirement_id,
        status="extracted",
        confidence=0.8,
        raw_model_value="5",
        evidence_value="6No Dock Levellers",
        normalized_value=5,
        supporting_span_ids=["span-1"],
        source_id="src-test",
        page_number=2,
        hierarchy_node_id=node_id,
        model_provider="mock",
        model_name="mock",
    )

    result = validate_evidence_layer([extraction], [_selected(target)], pages, hierarchy, [bundle])

    assert result[0].status == "invalid"


def test_v3_ambiguous_numbers_require_review_and_units_are_retained() -> None:
    count_target = _target("dock_leveller_count", ExpectedDataType.INTEGER, "Dock count")
    ambiguous = derive_evidence_value_and_normalization(
        target=count_target,
        assignment=infer_value_shape_assignment(count_target),
        raw_model_value="dock levellers",
        value_bearing_quote=None,
        span_text="2 dock levellers and 5 dock levellers are referenced",
    )
    assert ambiguous.normalization_status == "ambiguous"

    measurement_target = _target("office_area_value", ExpectedDataType.DECIMAL, "Office area", "m2")
    measured = derive_evidence_value_and_normalization(
        target=measurement_target,
        assignment=infer_value_shape_assignment(measurement_target),
        raw_model_value="125 m2",
        value_bearing_quote=None,
        span_text="Office floor area 125 m2",
    )
    assert measured.normalized_value == 125
    assert measured.unit == "m2"


def test_v3_metadata_mismatch_becomes_dictionary_caveat_not_invalid() -> None:
    target = _target(
        "frame_construction_description",
        ExpectedDataType.DECIMAL,
        "Frame construction description",
        "m",
    )
    assignment = infer_value_shape_assignment(target)
    extraction = ExtractionResult(
        target_row_id=target.target_row_id,
        requirement_id=target.requirement_id,
        status="extracted",
        confidence=0.8,
        raw_model_value="steel frame warehouse",
        evidence_value="steel frame warehouse",
        normalized_value="steel frame warehouse",
        display_value="steel frame warehouse",
        unit="m",
        model_provider="mock",
        model_name="mock",
    )
    evidence = EvidenceValidationResult(
        target_row_id=target.target_row_id,
        status="valid",
        canonical_evidence_available=True,
        evidence_value_present=True,
        source_page_node_valid=True,
    )
    shape = ShapeValidationResult(
        target_row_id=target.target_row_id,
        value_shape_family=assignment.value_shape_family,
        status="valid",
    )
    schema = assess_schema_compatibility_v3(
        [extraction], [_selected(target)], [assignment], [evidence], [shape]
    )
    overall = build_overall_validation_results(
        [extraction], [_selected(target)], [evidence], [shape], schema
    )

    assert target.expected_data_type == ExpectedDataType.DECIMAL
    assert target.unit == "m"
    assert schema[0].compatibility == "suspected_dictionary_metadata_mismatch"
    assert overall[0].status == "valid_with_dictionary_caveat"


def test_no_whole_manual_or_worker_when_cached_service_is_fully_warm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class FakeCachedService:
        def __init__(self, **kwargs: object) -> None:
            _ = kwargs

        def parse(self, request: object) -> object:
            page_start = request.page_start  # type: ignore[attr-defined]
            page_end = request.page_end  # type: ignore[attr-defined]
            source_id = request.source.source_id  # type: ignore[attr-defined]
            calls.append((page_start, page_end))
            assert page_end - page_start + 1 <= 10
            parsed_pages = [
                _page(page, f"cached page {page}", source_id=source_id)
                for page in range(page_start, page_end + 1)
            ]

            class Result:
                cache_hits = len(parsed_pages)
                cache_misses = 0
                worker_invocation_count = 0
                pages = parsed_pages

            return Result()

    monkeypatch.setattr(
        "segro_evidence_extraction.vertical_slice.CachedBatchParsingService",
        FakeCachedService,
    )
    output_dir = Path("output/test_vertical_slice/no_worker_when_warm")
    service = EvidenceFirstVerticalSliceService(
        source_manifest=Path("output/sprint3_source_ingestion/source_pack_manifest.json"),
        dictionary_path=Path("data/input/data_dictionary/SEGRO_Extraction_Template.xlsx"),
        output_dir=output_dir,
        cache_root=output_dir / "cache",
        extraction_client=AbstainingExtractionClient(),
    )

    result = service.run()

    assert calls
    assert all(end - start + 1 <= 10 for start, end in calls)
    assert result.telemetry.parser_worker_invocations == 0


def test_frozen_targets_and_source_ranges_are_unchanged() -> None:
    output_dir = Path("output/test_vertical_slice/frozen_preview")
    targets = select_vertical_slice_targets(
        Path("data/input/data_dictionary/SEGRO_Extraction_Template.xlsx"),
        output_dir,
    )
    registry = {
        source.source_id: source
        for source in load_source_registry(
            Path("output/sprint3_source_ingestion/source_pack_manifest.json")
        )
    }
    ranges = select_source_ranges(registry)

    assert [item.target.expected_field for item in targets] == [
        "frame_primary_construction_type",
        "frame_construction_description",
        "roof_construction_description",
        "external_cladding_description",
        "wall_construction_description",
        "dock_leveller_count",
        "pv_panel_component_description",
        "floor_construction_description",
        "fire_alarm_system_description",
        "approved_use_classes",
        "landlord_planning_consent_obligations",
        "pv_certification_component_description",
        "construction_date",
        "office_area_value",
    ]
    assert [(item.logical_path, item.page_start, item.page_end) for item in ranges] == [
        ("Building Manual - Part 1 General.pdf", 4, 11),
        ("Building Manual - Part 1 General.pdf", 14, 22),
        ("Building Manual - Part 1 General.pdf", 41, 46),
        ("Building Manual - Part 4 External Works.pdf", 4, 14),
        ("Building Manual - Part 5 The Health & Safety File.pdf", 1, 10),
        ("Building Manual - Part 6 Appendices.pdf", 4, 10),
    ]


def test_split_range_never_exceeds_ten_pages() -> None:
    assert split_range(1, 25) == [(1, 10), (11, 20), (21, 25)]


def _page(page_number: int, text: str, source_id: str = "src-test") -> ParsedPage:
    return ParsedPage(
        page_id=f"{source_id}:p{page_number}",
        source_id=source_id,
        content_hash="hash",
        page_number=page_number,
        physical_page_index=page_number - 1,
        parser_name="fake",
        parser_version="fake-1",
        text=text,
        character_count=len(text),
        word_count=len(text.split()),
        quality=TextQualityMetrics(
            character_count=len(text),
            word_count=len(text.split()),
            line_count=max(1, text.count("\n") + 1),
            average_line_length=10,
            alphabetic_ratio=0.5,
            numeric_ratio=0.1,
            repeated_character_ratio=0,
            text_density=1,
            scan_likelihood=0,
        ),
        scan_likelihood=0,
        ocr_routing=OcrRouting.NOT_REQUIRED,
        ocr_rationale="fixture",
        status=ParsingStatus.PARSED,
        parsing_duration_ms=1,
        provenance=[ProvenanceRef(source_id=source_id, page_or_sheet=f"page-{page_number}")],
    )


def _source(source_id: str, logical_path: str) -> SourceRegistryEntry:
    return SourceRegistryEntry(
        source_id=source_id,
        original_path=logical_path,
        logical_path=logical_path,
        file_type=FileType.PDF,
        extension=".pdf",
        size_bytes=100,
        file_hash="hash",
    )


def _target(
    expected_field: str,
    datatype: ExpectedDataType,
    requirement_text: str,
    unit: str | None = None,
) -> TargetSpecification:
    return TargetSpecification(
        target_row_id=f"target-{expected_field}",
        requirement_id="REQ-1",
        sub_domain="Technical Specification",
        requirement_text=requirement_text,
        expected_field=expected_field,
        expected_data_type=datatype,
        unit=unit,
    )


def _selected(target: TargetSpecification) -> SelectedTarget:
    return SelectedTarget(target=target, selection_reason="test")


def _bundle(target: TargetSpecification, excerpt: str) -> EvidenceBundleRecord:
    return EvidenceBundleRecord(
        target_row_id=target.target_row_id,
        evidence_items=[
            RetrievedEvidence(
                target_row_id=target.target_row_id,
                rank=1,
                node_id="node-1",
                source_id="src-test",
                source_file="manual.pdf",
                page_start=2,
                page_end=2,
                score=10,
                score_components=RetrievalScoreBreakdown(token_overlap=5),
                matched_terms=["dock"],
                hierarchy_path=["manual", "Dock Levellers"],
                excerpt=excerpt,
            )
        ],
        combined_text=excerpt,
        character_count=len(excerpt),
        token_estimate=1,
    )


def _span(
    span_id: str,
    source_id: str,
    page_number: int,
    node_id: str,
    text: str,
):
    from segro_evidence_extraction.vertical_slice import EvidenceSpan

    return EvidenceSpan(
        span_id=span_id,
        source_id=source_id,
        source_file="manual.pdf",
        page_number=page_number,
        hierarchy_node_id=node_id,
        text=text,
        start_char=0,
        end_char=len(text),
        retrieval_rank=1,
        score=10,
    )
