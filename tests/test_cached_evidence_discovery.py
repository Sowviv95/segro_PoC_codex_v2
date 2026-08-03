from __future__ import annotations

import json
from pathlib import Path

from segro_evidence_extraction.cached_evidence_discovery import (
    DiscoveryRequest,
    load_discovery_request,
    reconcile_discovery_with_manifest,
    run_cached_evidence_discovery,
)
from segro_evidence_extraction.cluster_manifest import load_cluster_manifest
from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry


def test_discovery_surfaces_schedule_above_lower_signal_pages(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    write_page(cache_root, "src_a", 1, "Unit 1 GA drawing with dimensions 1000 2000")
    write_page(
        cache_root,
        "src_b",
        7,
        "Schedule of Accommodation (GEA) Unit 1 5,193m² 55,890ft² Total A 6,002m²",
    )
    request = discovery_request(cache_root)

    result = run_cached_evidence_discovery(
        request,
        sources=[source("src_a", "Drawings.pdf"), source("src_b", "Unexpected Schedule.pdf")],
    )

    candidates = result["ranked_candidate_pages"]
    assert candidates[0]["source_id"] == "src_b"
    assert candidates[0]["page_number"] == 7
    assert candidates[0]["readiness_decision"] is None
    assert candidates[0]["candidate_score"] > candidates[1]["candidate_score"]


def test_text_empty_pages_receive_quality_warning(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    write_page(cache_root, "src_empty", 4, "")
    request = discovery_request(cache_root, minimum_score=0)

    result = run_cached_evidence_discovery(
        request,
        sources=[source("src_empty", "Blank Forms.pdf")],
    )

    candidate = result["ranked_candidate_pages"][0]
    assert candidate["text_empty"] is True
    assert candidate["page_quality"] == "text_empty"
    assert "text_empty_page_quality_warning" in candidate["exclusion_reasons"]


def test_manifest_reconciliation_identifies_omitted_higher_signal_page(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    write_page(cache_root, "src_a", 1, "Unit 1 GA drawing with dimensions 1000 2000")
    write_page(cache_root, "src_b", 7, "Schedule of Accommodation (GEA) Unit 1 5,193m²")
    request = discovery_request(cache_root)
    result = run_cached_evidence_discovery(
        request,
        sources=[source("src_a", "Drawings.pdf"), source("src_b", "Unexpected Schedule.pdf")],
    )
    manifest_path = write_manifest(tmp_path)
    manifest = load_cluster_manifest(manifest_path)

    reconciliation = reconcile_discovery_with_manifest(
        discovery_result=result,
        manifest=manifest,
    )

    omitted = reconciliation["omitted_higher_signal_pages"]
    assert any(row["source_id"] == "src_b" and row["page_number"] == 7 for row in omitted)
    assert reconciliation["manifest_was_altered"] is False


def test_property_area_regression_surfaces_part5_page_21() -> None:
    request = DiscoveryRequest(
        schema_version="segro_cached_evidence_family_discovery_request_v1",
        contract_version="1.0.0",
        discovery_run_name="property_area_regression",
        source_inventory_path="output/sprint3_source_ingestion/source_pack_manifest.json",
        cache_root="output/enfield_unit1_evidence_first_vertical_slice_v1/page_cache",
        output_path="output/test_cached_discovery",
        descriptors=[
            {
                "descriptor_id": "property_area_schedule",
                "evidence_family": "property_area_schedule",
                "target_ids": ["property_target"],
                "field_names": ["area_value"],
                "requested_attribute_phrases": ["area", "unit"],
                "expected_value_shapes": ["decimal_measurement"],
                "unit_patterns": [r"m²", r"ft²", r"\bHA\b", r"Acres"],
                "positive_labels": ["Schedule of Accommodation", "GEA", "Unit 1", "Total A"],
                "negative_labels": ["planning condition"],
                "indicators": ["schedule"],
                "max_candidates": 5,
                "minimum_score": 1,
            }
        ],
    )

    result = run_cached_evidence_discovery(
        request,
        sources=[
            source("src_7216bce3ba88ba2d", "Building Manual - Part 1 General.pdf"),
            source("src_1fd0bf352104ae02", "Building Manual - Part 2 Building Fabric.pdf"),
            source("src_289b4022953f308c", "Building Manual - Part 5 The Health & Safety File.pdf"),
        ],
    )

    top = result["ranked_candidate_pages"][0]
    assert top["source_id"] == "src_289b4022953f308c"
    assert top["page_number"] == 21
    assert top["readiness_decision"] is None


def test_synthetic_second_unit_unexpected_schedule_document(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    write_page(cache_root, "alpha_main", 2, "General arrangement narrative only")
    write_page(
        cache_root,
        "alpha_unexpected",
        12,
        "Area Register Schedule Bay Alpha 123.4 sqm loading apron 55 sqm",
    )
    request = DiscoveryRequest(
        schema_version="segro_cached_evidence_family_discovery_request_v1",
        contract_version="1.0.0",
        discovery_run_name="alpha_discovery",
        source_inventory_path="alpha_inventory.jsonl",
        cache_root=str(cache_root),
        output_path="output/alpha",
        descriptors=[
            {
                "descriptor_id": "alpha_area",
                "evidence_family": "area_register",
                "target_ids": ["alpha_target_001"],
                "field_names": ["alpha_area_measure"],
                "requested_attribute_phrases": ["area", "loading apron"],
                "expected_value_shapes": ["decimal_measurement"],
                "unit_patterns": [r"sqm"],
                "positive_labels": ["Area Register", "Schedule"],
                "negative_labels": [],
                "indicators": ["schedule"],
                "max_candidates": 3,
                "minimum_score": 1,
            }
        ],
    )

    result = run_cached_evidence_discovery(
        request,
        sources=[
            source("alpha_main", "Main Narrative.pdf"),
            source("alpha_unexpected", "Standalone Measurements.pdf"),
        ],
    )

    assert result["ranked_candidate_pages"][0]["source_id"] == "alpha_unexpected"
    assert result["readiness_decisions_made"] is False
    assert result["extraction_decisions_made"] is False


def test_request_loader_validates_schema(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "segro_cached_evidence_family_discovery_request_v1",
                "contract_version": "1.0.0",
                "discovery_run_name": "fixture",
                "source_inventory_path": "sources.jsonl",
                "cache_root": "cache",
                "output_path": "output",
                "descriptors": [
                    {
                        "descriptor_id": "descriptor",
                        "evidence_family": "family",
                        "requested_attribute_phrases": [],
                        "expected_value_shapes": [],
                        "unit_patterns": [],
                        "positive_labels": [],
                        "negative_labels": [],
                        "indicators": [],
                        "max_candidates": 1,
                        "minimum_score": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert load_discovery_request(path).discovery_run_name == "fixture"


def discovery_request(cache_root: Path, minimum_score: int = 1) -> DiscoveryRequest:
    return DiscoveryRequest(
        schema_version="segro_cached_evidence_family_discovery_request_v1",
        contract_version="1.0.0",
        discovery_run_name="fixture",
        source_inventory_path="sources.jsonl",
        cache_root=str(cache_root),
        output_path="output",
        descriptors=[
            {
                "descriptor_id": "property_area",
                "evidence_family": "property_area_schedule",
                "target_ids": ["target_a"],
                "field_names": ["area_value"],
                "requested_attribute_phrases": ["area", "unit"],
                "expected_value_shapes": ["decimal_measurement"],
                "unit_patterns": [r"m²", r"ft²", r"sqm"],
                "positive_labels": ["Schedule of Accommodation", "GEA", "Unit 1"],
                "negative_labels": ["planning condition"],
                "indicators": ["schedule"],
                "max_candidates": 10,
                "minimum_score": minimum_score,
            }
        ],
    )


def write_page(cache_root: Path, source_id: str, page_number: int, text: str) -> None:
    page_dir = cache_root / source_id / "cache"
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / f"page_{page_number:06d}.json").write_text(
        json.dumps(
            {
                "source_id": source_id,
                "page_number": page_number,
                "extracted_text": text,
            }
        ),
        encoding="utf-8",
    )


def write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "segro_evidence_cluster_manifest_v1",
                "contract_version": "1.0.0",
                "run_name": "fixture_manifest",
                "unit_id": "fixture_unit",
                "output_dir": "output/fixture",
                "total_page_cap": 1,
                "allowed_readiness_dispositions": ["evidence_ready", "evidence_not_found"],
                "targets": [
                    {
                        "target_id": "target_a",
                        "field_name": "area_value",
                        "domain": "Property",
                        "value_shape": "decimal_measurement",
                        "requested_attribute": "area value",
                        "expected_evidence_families": ["property_area_schedule"],
                        "max_pages_per_target": 1,
                        "source_assignments": [
                            {
                                "source_id": "src_a",
                                "source_document_role": "drawing",
                                "section_label": "drawing page",
                                "candidate_page_ranges": [
                                    {
                                        "page_start": 1,
                                        "page_end": 1,
                                        "rationale": "fixture",
                                        "expected_cache_state": "cached",
                                    }
                                ],
                                "allowed_methods": ["cached_text"],
                                "max_pages_for_assignment": 1,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def source(source_id: str, logical_path: str) -> SourceRegistryEntry:
    return SourceRegistryEntry(
        source_id=source_id,
        original_path=f"data/input/{logical_path}",
        logical_path=logical_path,
        file_type=FileType.PDF,
        file_hash="0" * 64,
        size_bytes=10,
    )
