import json
from pathlib import Path

from segro_evidence_extraction.extraction_batch_construction_v1 import (
    build_bounded_extraction_requests,
    build_canonical_evidence_payloads,
    build_evidence_to_target_mappings,
    build_observed_value_evidence,
    check_attribute_support,
    load_targets,
    run_extraction_batch_construction_v1,
    select_extraction_targets,
)


def _target(target_id: str, field: str, dtype: str = "string") -> dict[str, object]:
    return {
        "target_row_id": target_id,
        "requirement_id": f"req_{target_id}",
        "sub_domain": "Synthetic",
        "requirement_text": f"Capture {field}",
        "expected_field": field,
        "expected_data_type": dtype,
        "unit": None,
        "cardinality": "single",
        "component_type": None,
        "component_subtype": None,
        "source_guidance": None,
        "accepted_values": [],
        "likely_evidence_types": [],
        "metadata": {},
        "source_dictionary_provenance": None,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_page(cache_root: Path, source_id: str, page: int, text: str) -> None:
    _write_json(
        cache_root / source_id / f"page_{page:04d}.json",
        {
            "source_id": source_id,
            "source_path": f"{source_id}.pdf",
            "page_number": page,
            "extracted_text": text,
            "text_character_count": len(text),
            "parser_name": "synthetic",
            "parser_version": "1",
            "parser_config_fingerprint": "synthetic",
        },
    )


def _section(source_id: str, page: int, route: str = "certificate") -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_filename": f"{source_id}.pdf",
        "page_start": page,
        "page_end": page,
        "recommended_route": route,
        "primary_page_types": {"certificate": 1},
        "section": "Synthetic certificate",
    }


def _write_evidence_dir(evidence_dir: Path, section: dict[str, object]) -> None:
    _write_json(evidence_dir / "commissioning_evidence_sections.json", [section])
    _write_json(evidence_dir / "statutory_evidence_sections.json", [])
    _write_json(evidence_dir / "installation_evidence_sections.json", [])


def _write_targets(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_evidence_first_mapping_requires_observed_value_bearing_evidence() -> None:
    workspace = _test_workspace("evidence_first")
    evidence_dir = workspace / "evidence"
    cache_root = workspace / "cache"
    dictionary = workspace / "targets.jsonl"
    text = (
        "Fire alarm commissioning certificate for Enfield Unit 1. Certificate number "
        "43830 dated 13/03/2020. The fire detection and alarm system commissioning "
        "was completed and passed with satisfactory result. System type addressable."
    )
    _write_page(cache_root, "src_fire", 54, text)
    _write_evidence_dir(evidence_dir, _section("src_fire", 54))
    _write_targets(
        dictionary,
        [
            _target("trg_fire_cert_no", "fire_safety_certificate_name_or_number"),
            _target("trg_fire_date", "fire_safety_certificate_issue_date", "date"),
        ],
    )

    result = run_extraction_batch_construction_v1(
        evidence_dir=evidence_dir,
        dictionary_jsonl=dictionary,
        cache_root=cache_root,
        output_dir=workspace / "out",
        max_targets=30,
    )

    assert result["batch_metrics"]["selected_target_count"] == 2
    assert {row["target_id"] for row in result["selected_extraction_targets"]} == {
        "trg_fire_cert_no",
        "trg_fire_date",
    }


def test_component_only_and_heading_only_evidence_are_rejected() -> None:
    workspace = _test_workspace("component_only")
    evidence_dir = workspace / "evidence"
    cache_root = workspace / "cache"
    dictionary = workspace / "targets.jsonl"
    _write_page(cache_root, "src_fire", 1, "Fire alarm commissioning certificate")
    _write_evidence_dir(evidence_dir, _section("src_fire", 1))
    _write_targets(
        dictionary,
        [_target("trg_fire_cert_no", "fire_safety_certificate_name_or_number")],
    )

    observed = build_observed_value_evidence(evidence_dir, build_cache_dict(cache_root))
    mappings = build_evidence_to_target_mappings(observed, load_targets(dictionary))

    assert observed[0]["value_bearing"] is False
    assert mappings == []


def test_wrong_event_support_is_rejected() -> None:
    mapping = {
        "route": "certificate",
        "evidence_page_exists": True,
        "value_bearing_text": (
            "Electrical certificate for Enfield Unit 1. Certificate number E123 dated "
            "12/03/2020. The electrical installation was completed and approved. "
            "The certificate records the completed electrical installation test and "
            "approval result for the distribution system."
        ),
        "requested_attribute": "certificate_name_or_number",
        "component_system_identity": "fire_safety",
        "field": "fire_safety_certificate_name_or_number",
    }

    check = check_attribute_support([mapping])[0]

    assert check["attribute_support_status"] == "rejected_wrong_component"


def test_visual_only_candidates_are_deferred() -> None:
    row = supported_row("trg_fire", "obs_1")
    row["route"] = "visual"

    selected, rejected, visual = select_extraction_targets([row], max_targets=30)

    assert selected == []
    assert rejected == []
    assert len(visual) == 1


def test_contract_generation_uses_canonical_models() -> None:
    selected = [supported_row("trg_fire", "obs_1")]
    payloads = build_canonical_evidence_payloads(selected)
    requests = build_bounded_extraction_requests(selected, payloads)

    assert requests[0]["evidence_bundle"]["target_specification"]["target_row_id"] == "trg_fire"
    assert requests[0]["retrieved_evidence"]["score_components"]["final_score"] == 1.0


def test_selection_caps_duplicates_and_balances_sources_and_families() -> None:
    rows = []
    for index in range(40):
        row = supported_row(f"trg_{index}", f"obs_{index}")
        row["source_id"] = "src_a" if index < 20 else f"src_{index}"
        row["evidence_family"] = "commissioning_results" if index < 20 else "statutory_compliance"
        rows.append(row)
    rows.append(supported_row("trg_1", "obs_duplicate"))

    selected, rejected, _visual = select_extraction_targets(rows, max_targets=30)

    assert len(selected) <= 30
    assert len({row["target_id"] for row in selected}) == len(selected)
    assert sum(1 for row in selected if row["source_id"] == "src_a") <= 8
    assert sum(1 for row in selected if row["evidence_family"] == "commissioning_results") <= 8
    assert any(row["rejection_reason"] == "duplicate target" for row in rejected)


def test_deterministic_reruns() -> None:
    workspace = _test_workspace("deterministic")
    evidence_dir = workspace / "evidence"
    cache_root = workspace / "cache"
    dictionary = workspace / "targets.jsonl"
    text = (
        "Lighting commissioning certificate for Enfield Unit 1. Commissioned on "
        "28/02/2020 by CP Electronics. The lighting control system test completed "
        "with satisfactory results and system type networked controls."
    )
    _write_page(cache_root, "src_light", 59, text)
    _write_evidence_dir(evidence_dir, _section("src_light", 59))
    _write_targets(
        dictionary,
        [_target("trg_light_date", "internal_luminaire_installation_date", "date")],
    )

    first = run_extraction_batch_construction_v1(
        evidence_dir=evidence_dir,
        dictionary_jsonl=dictionary,
        cache_root=cache_root,
        output_dir=workspace / "out1",
        max_targets=30,
    )
    second = run_extraction_batch_construction_v1(
        evidence_dir=evidence_dir,
        dictionary_jsonl=dictionary,
        cache_root=cache_root,
        output_dir=workspace / "out2",
        max_targets=30,
    )

    assert first["selected_extraction_targets"] == second["selected_extraction_targets"]
    assert first["bounded_extraction_requests"] == second["bounded_extraction_requests"]


def test_no_extraction_or_external_calls_are_recorded() -> None:
    result = run_empty(_test_workspace("empty"))

    assert result["batch_metrics"]["extraction_calls_made"] == 0
    assert result["batch_metrics"]["llm_calls_made"] == 0
    assert result["batch_metrics"]["ocr_calls_made"] == 0
    assert result["batch_metrics"]["vlm_calls_made"] == 0
    assert result["batch_metrics"]["external_api_calls_made"] == 0


def test_bounded_real_artifact_integration() -> None:
    evidence_dir = Path("output/enfield_unit1_evidence_gap_refinement_v1")
    dictionary = Path("output/sprint2_dictionary_validation/normalized_targets.jsonl")
    cache_root = Path("output/enfield_unit1_evidence_first_vertical_slice_v1/page_cache")
    if not evidence_dir.exists() or not dictionary.exists() or not cache_root.exists():
        return

    result = run_extraction_batch_construction_v1(
        evidence_dir=evidence_dir,
        dictionary_jsonl=dictionary,
        cache_root=cache_root,
        output_dir=_test_workspace("real") / "real_out",
        max_targets=5,
    )

    assert result["batch_metrics"]["selected_target_count"] <= 5
    assert result["batch_metrics"]["extraction_calls_made"] == 0


def build_cache_dict(cache_root: Path) -> dict[str, list[dict[str, object]]]:
    pages = {}
    for path in cache_root.rglob("page_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["cache_path"] = str(path)
        pages.setdefault(data["source_id"], []).append(data)
    return pages


def supported_row(target_id: str, evidence_id: str) -> dict[str, object]:
    return {
        "target_id": target_id,
        "evidence_id": evidence_id,
        "route": "certificate",
        "evidence_page_exists": True,
        "attribute_support_status": "supported",
        "attribute_support_reason": "component and requested attribute co-occur",
        "executable_route": True,
        "dictionary_ambiguity": False,
        "source_id": "src_fire",
        "source_filename": "src_fire.pdf",
        "page_start": 54,
        "page_end": 54,
        "field": "fire_safety_certificate_name_or_number",
        "requested_attribute": "certificate_name_or_number",
        "component_system_identity": "fire_safety",
        "evidence_family": "commissioning_results",
        "page_type": "certificate",
        "section": "Fire certificate",
        "value_bearing_text": (
            "Fire alarm commissioning certificate for Enfield Unit 1. Certificate "
            "number 43830 dated 13/03/2020. The system commissioning completed and passed."
        ),
        "asset_applicability": "asset_applicable",
        "cache_path": "cache/page_0054.json",
        "requirement_id": f"req_{target_id}",
        "requirement": "Capture fire certificate number",
        "expected_data_type": "string",
        "unit": None,
        "dictionary_target": _target(target_id, "fire_safety_certificate_name_or_number"),
        "validation_expectations": [
            "evidence span must come from the specified source and page range"
        ],
    }


def run_empty(tmp_path: Path) -> dict[str, object]:
    evidence_dir = tmp_path / "evidence"
    dictionary = tmp_path / "targets.jsonl"
    _write_evidence_dir(evidence_dir, _section("src_empty", 1))
    _write_targets(dictionary, [_target("trg_fire", "fire_safety_certificate_name_or_number")])
    return run_extraction_batch_construction_v1(
        evidence_dir=evidence_dir,
        dictionary_jsonl=dictionary,
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "out",
        max_targets=30,
    )


def _test_workspace(name: str) -> Path:
    path = Path("output/test_extraction_batch_construction_v1") / name
    path.mkdir(parents=True, exist_ok=True)
    return path
