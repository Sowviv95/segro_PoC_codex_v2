from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from segro_evidence_extraction.cluster_manifest import (
    ClusterManifestError,
    build_cluster_preparation_plan,
    load_cluster_manifest,
    manifest_identity,
)
from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry


def test_manifest_schema_validation_rejects_unknown_field(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, {"unexpected": True})

    with pytest.raises(ClusterManifestError, match="JSON Schema validation failed"):
        load_cluster_manifest(path)


def test_semantic_validation_rejects_duplicate_target_ids(tmp_path: Path) -> None:
    manifest = base_manifest()
    manifest["targets"].append(dict(manifest["targets"][0]))
    path = write_manifest(tmp_path, manifest)

    with pytest.raises(ClusterManifestError, match="duplicate target IDs"):
        load_cluster_manifest(path)


def test_semantic_validation_rejects_unknown_registered_source(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, base_manifest())

    with pytest.raises(ClusterManifestError, match="unknown source_id"):
        load_cluster_manifest(path, registered_sources=[source("other_source", "Other.pdf")])


def test_semantic_validation_rejects_reversed_page_range(tmp_path: Path) -> None:
    manifest = base_manifest()
    range_row = manifest["targets"][0]["source_assignments"][0]["candidate_page_ranges"][0]
    range_row["page_start"] = 9
    range_row["page_end"] = 3
    path = write_manifest(tmp_path, manifest)

    with pytest.raises(ClusterManifestError, match="page_end precedes page_start"):
        load_cluster_manifest(path)


def test_semantic_validation_rejects_page_cap_excess(tmp_path: Path) -> None:
    manifest = base_manifest()
    manifest["targets"][0]["max_pages_per_target"] = 1
    path = write_manifest(tmp_path, manifest)

    with pytest.raises(ClusterManifestError, match="exceeds target cap"):
        load_cluster_manifest(path)


def test_semantic_validation_rejects_absolute_output_dir(tmp_path: Path) -> None:
    manifest = base_manifest()
    manifest["output_dir"] = str(tmp_path.resolve())
    path = write_manifest(tmp_path, manifest)

    with pytest.raises(ClusterManifestError, match="output_dir"):
        load_cluster_manifest(path)


def test_deterministic_manifest_identity(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, base_manifest())
    manifest = load_cluster_manifest(path)

    assert manifest_identity(manifest) == manifest_identity(load_cluster_manifest(path))


def test_preparation_plan_deduplicates_and_reports_cache_state(tmp_path: Path) -> None:
    manifest = base_manifest()
    manifest["targets"].append(
        {
            **manifest["targets"][0],
            "target_id": "portable_target_b",
            "field_name": "different_attribute_name",
            "requested_attribute": "different attribute",
        }
    )
    path = write_manifest(tmp_path, manifest)
    loaded = load_cluster_manifest(path, registered_sources=[source("alpha_source", "Alpha.pdf")])

    plan = build_cluster_preparation_plan(
        loaded,
        cached_pages={
            "alpha_source": [
                {"page_number": 3, "extracted_text": "Readable certificate reference A-1"},
                {"page_number": 4, "extracted_text": ""},
            ]
        },
    )

    summary = plan["page_deduplication_summary"]
    assert summary["candidate_page_references"] == 6
    assert summary["deduplicated_page_count"] == 3
    by_page = {row["page_number"]: row for row in summary["deduplicated_pages"]}
    assert by_page[3]["acquisition_disposition"] == "cached_text_available"
    assert by_page[4]["acquisition_disposition"] == "already_cached_but_text_empty"
    assert by_page[5]["acquisition_disposition"] == "uncached_parsing_candidate"


def test_no_target_or_field_name_specific_decisions(tmp_path: Path) -> None:
    first = base_manifest()
    second = base_manifest()
    second["targets"][0]["target_id"] = "unrelated_target_999"
    second["targets"][0]["field_name"] = "unrelated_field"
    second["targets"][0]["requested_attribute"] = "unrelated requested attribute"
    first_path = write_manifest(tmp_path, first, "first.json")
    second_path = write_manifest(tmp_path, second, "second.json")

    first_plan = build_cluster_preparation_plan(load_cluster_manifest(first_path))
    second_plan = build_cluster_preparation_plan(load_cluster_manifest(second_path))

    first_dispositions = [
        row["acquisition_disposition"] for row in first_plan["page_acquisition_manifest"]
    ]
    second_dispositions = [
        row["acquisition_disposition"] for row in second_plan["page_acquisition_manifest"]
    ]
    assert first_dispositions == second_dispositions


def test_synthetic_second_unit_portability_manifest_has_no_enfield_strings() -> None:
    path = Path("config/synthetic_second_unit_evidence_cluster_manifest_v1.json")
    text = path.read_text(encoding="utf-8")

    assert "Enfield" not in text
    assert "Building Manual - Part" not in text
    manifest = load_cluster_manifest(
        path,
        registered_sources=[
            source("alpha_src_cert_pack", "Certificates_Final_Alpha.pdf"),
            source("alpha_src_schedule", "Plant_Schedules_Alpha.pdf"),
        ],
    )
    plan = build_cluster_preparation_plan(manifest)

    assert plan["unit_id"] == "unit_alpha_warehouse_04"
    assert len(plan["frozen_target_manifest"]) == 2
    assert plan["page_deduplication_summary"]["deduplicated_page_count"] == 8


def test_same_generic_code_path_processes_enfield_and_second_unit() -> None:
    enfield = load_cluster_manifest(Path("config/enfield_unit1_evidence_cluster_manifest_v1.json"))
    synthetic = load_cluster_manifest(
        Path("config/synthetic_second_unit_evidence_cluster_manifest_v1.json")
    )

    enfield_plan = build_cluster_preparation_plan(enfield)
    synthetic_plan = build_cluster_preparation_plan(synthetic)

    assert enfield_plan["schema_version"] == synthetic_plan["schema_version"]
    assert enfield_plan["business_outcomes_generated"] is False
    assert synthetic_plan["business_outcomes_generated"] is False


def test_hard_coding_guard_for_new_reusable_modules() -> None:
    forbidden = [
        "trg_30093f2f106d4cb3",
        "trg_6790fedc7a5d7574",
        "trg_8b6f21644152f6e9",
        "trg_f6ce0dfb9db88fac",
        "trg_97fd495e53beab34",
        "trg_60a5bc3ae3ff9154",
        "trg_471356dd291ff34b",
        "src_48403a16aee1d4b3",
        "Building Manual - Part",
        "Appendix D",
        "enfield_unit1",
    ]
    text = Path("src/segro_evidence_extraction/cluster_manifest.py").read_text(encoding="utf-8")

    assert not [pattern for pattern in forbidden if pattern in text]


def base_manifest() -> dict[str, Any]:
    return {
        "schema_version": "segro_evidence_cluster_manifest_v1",
        "contract_version": "1.0.0",
        "run_name": "portable_cluster_fixture",
        "unit_id": "portable_unit",
        "project_number": "P-100",
        "output_dir": "output/portable_cluster_fixture",
        "total_page_cap": 3,
        "allowed_readiness_dispositions": [
            "evidence_ready",
            "evidence_not_found",
            "requires_targeted_parsing",
        ],
        "targets": [
            {
                "target_id": "portable_target_a",
                "field_name": "certificate_reference",
                "domain": "Compliance",
                "value_shape": "identifier_or_reference",
                "requested_attribute": "certificate reference",
                "expected_evidence_families": ["completion_certificate"],
                "max_pages_per_target": 3,
                "source_assignments": [
                    {
                        "source_id": "alpha_source",
                        "source_document_role": "standalone_certificate_pdf",
                        "logical_source_path": "Alpha Certificates.pdf",
                        "section_label": "handover certificates",
                        "max_pages_for_assignment": 3,
                        "allowed_methods": ["cached_text", "native_parser"],
                        "candidate_page_ranges": [
                            {
                                "page_start": 3,
                                "page_end": 5,
                                "rationale": "Bounded source-specific section.",
                                "expected_cache_state": "either",
                            }
                        ],
                    }
                ],
            }
        ],
    }


def write_manifest(
    tmp_path: Path,
    overrides: dict[str, Any],
    filename: str = "manifest.json",
) -> Path:
    data = base_manifest()
    data.update(overrides)
    path = tmp_path / filename
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
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
