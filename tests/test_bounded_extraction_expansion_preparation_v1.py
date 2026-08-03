from __future__ import annotations

import shutil
from pathlib import Path

from segro_evidence_extraction.bounded_extraction_expansion_preparation_v1 import (
    classify_candidate,
    dry_run_validate_selected,
    run_bounded_extraction_expansion_preparation_v1,
)

COMPLETED_TARGETS = {
    "trg_92469ca7eaab2c31",
    "trg_33975e86ffde5524",
    "trg_6720c2edd5e947d6",
    "trg_0f66487177c685e6",
    "trg_7426d9722059de76",
    "trg_93d2e2b38d1de7b2",
    "trg_50fedb4c249d8fe4",
    "trg_ca18dce11deff8cf",
    "trg_12690fb418279750",
    "trg_adb3dfa03b0c7cf3",
}


def test_expansion_preparation_selects_only_evidence_ready_targets() -> None:
    output = Path("output/test_bounded_expansion_preparation")
    try:
        result = run_bounded_extraction_expansion_preparation_v1(output_dir=output)
        selected = result["selected_batch"]
        selected_ids = {record["target_id"] for record in selected}

        assert selected
        assert not selected_ids & COMPLETED_TARGETS
        assert all(record["classification"] == "execution_ready" for record in selected)
        assert all(record["evidence_bundle"] for record in selected)
        assert all(Path(record["cache_path"]).exists() for record in selected)
        assert result["dry_run_validation"]["overall_status"] == "passed"
        assert result["execution_manifest"]["do_not_execute_in_this_sprint"] is True
        assert (output / "next_execution_command.ps1").exists()
    finally:
        if output.exists():
            shutil.rmtree(output)


def test_expansion_preparation_is_deterministic() -> None:
    first = run_bounded_extraction_expansion_preparation_v1(
        output_dir=Path("output/test_bounded_expansion_first")
    )
    second = run_bounded_extraction_expansion_preparation_v1(
        output_dir=Path("output/test_bounded_expansion_second")
    )
    try:
        assert first["selected_batch"] == second["selected_batch"]
        assert first["selection_summary"]["selected_target_ids"] == [
            record["target_id"] for record in first["selected_batch"]
        ]
    finally:
        for output in [
            Path("output/test_bounded_expansion_first"),
            Path("output/test_bounded_expansion_second"),
        ]:
            if output.exists():
                shutil.rmtree(output)


def test_classification_excludes_component_wrong_event_wrong_system_and_dictionary_gaps() -> None:
    base = candidate(
        attribute_support_status="supported",
        asset_applicability="asset_applicable",
        value_bearing_text="value",
        cache_file_exists=True,
    )

    assert (
        classify_candidate(
            {**base, "attribute_support_status": "rejected_wrong_component"},
            "a",
            set(),
            set(),
            set(),
            {},
        )
        == "not_ready_component_only"
    )
    assert (
        classify_candidate(
            {**base, "attribute_support_status": "rejected_wrong_event"},
            "a",
            set(),
            set(),
            set(),
            {},
        )
        == "not_ready_wrong_event"
    )
    assert (
        classify_candidate(
            {**base, "attribute_support_reason": "wrong system"}, "a", set(), set(), set(), {}
        )
        == "not_ready_wrong_system"
    )
    assert (
        classify_candidate({**base, "dictionary_ambiguity": True}, "a", set(), set(), set(), {})
        == "not_ready_dictionary_clarification"
    )
    assert (
        classify_candidate({**base, "cache_file_exists": False}, "a", set(), set(), set(), {})
        == "not_ready_missing_cached_page"
    )


def test_dry_run_removes_invalid_missing_cache_candidate() -> None:
    selected = [
        {
            "target_id": "trg_missing_cache",
            "requirement_id": "req",
            "target": {"expected_field": "field"},
            "value_shape": "short_text",
            "datatype": "string",
            "evidence_bundle": {"bundle_id": "bundle"},
            "source_id": "src",
            "source_filename": "source.pdf",
            "page_number": 1,
            "cache_path": "output/does_not_exist/page.json",
            "canonical_evidence_payload": {"bounded_text": "value"},
        }
    ]

    validation = dry_run_validate_selected(selected)

    assert validation["overall_status"] == "failed"
    assert validation["failed_target_ids"] == ["trg_missing_cache"]
    assert validation["parser_invocations"] == 0
    assert validation["retrieval_invocations"] == 0
    assert validation["model_calls"] == 0
    assert validation["ocr_invocations"] == 0
    assert validation["vlm_invocations"] == 0
    assert validation["cache_expansion_invocations"] == 0


def test_expansion_outputs_expected_distribution_and_no_invocations() -> None:
    output = Path("output/test_bounded_expansion_distribution")
    try:
        result = run_bounded_extraction_expansion_preparation_v1(output_dir=output)
        selected = result["selected_batch"]

        assert result["value_shape_distribution"]["counts"]
        assert result["domain_distribution"]["counts"]
        assert result["evidence_source_distribution"]["counts"]
        assert result["execution_estimate"]["expected_model_calls"] == len(selected)
        assert result["dry_run_validation"]["model_calls"] == 0
        assert result["dry_run_validation"]["retrieval_invocations"] == 0
        assert result["dry_run_validation"]["parser_invocations"] == 0
        assert result["dry_run_validation"]["ocr_invocations"] == 0
        assert result["dry_run_validation"]["vlm_invocations"] == 0
    finally:
        if output.exists():
            shutil.rmtree(output)


def candidate(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "already_completed": False,
        "checkpoint_rejected": False,
        "cache_file_exists": True,
        "dictionary_ambiguity": False,
        "attribute_support_status": "supported",
        "attribute_support_reason": "supported",
        "asset_applicability": "asset_applicable",
        "value_bearing_text": "value",
    }
    record.update(overrides)
    return record
