from __future__ import annotations

import json
import shutil
from pathlib import Path

from segro_evidence_extraction.reduced_extraction_adjudication_v1 import (
    DEFAULT_REDUCED_BATCH_OUTPUT_DIR,
    evidence_containment_key,
    normalized_evidence_contains,
    repair_response_payload,
    run_reduced_extraction_adjudication_v1,
)


def test_normalized_evidence_contains_pdf_no_spacing_artifact() -> None:
    evidence = "8335 5No Dock Levellers Warehouse Undercroft"

    assert normalized_evidence_contains(evidence, "5 No Dock Levellers")
    assert normalized_evidence_contains(evidence, "5No Dock Levellers")
    assert not normalized_evidence_contains(evidence, "6 No Dock Levellers")
    assert not normalized_evidence_contains(evidence, "5 No Warehouse Dock Levellers")


def test_evidence_containment_key_preserves_word_order_without_fuzzy_matching() -> None:
    assert evidence_containment_key("A\u2022B 5 No C") == "a b 5no c"
    assert not normalized_evidence_contains("alpha beta gamma", "alpha gamma")


def test_repair_response_payload_only_repairs_known_shape_literal() -> None:
    repaired, reason = repair_response_payload(
        {"proposed_value_shape": "unordered_or_unordered_list", "status": "extracted"}
    )

    assert repaired["proposed_value_shape"] == "ordered_or_unordered_list"
    assert "unordered_or_unordered_list" in reason

    unchanged, unchanged_reason = repair_response_payload({"proposed_value_shape": "bad"})
    assert unchanged["proposed_value_shape"] == "bad"
    assert unchanged_reason == "No deterministic schema repair was available."


def test_real_adjudication_confirms_abstentions_repairs_containment_and_schema() -> None:
    if not (DEFAULT_REDUCED_BATCH_OUTPUT_DIR / "final_adjudication.json").exists():
        return
    output = Path("output/test_reduced_extraction_adjudication_v1")
    if output.exists():
        shutil.rmtree(output)
    try:
        result = run_reduced_extraction_adjudication_v1(output_dir=output)
        review = {item["target_id"]: item for item in result["target_review"]}
        final = {item["target_id"]: item for item in result["final_adjudication"]}

        assert result["adjudication_summary"]["new_model_calls"] == 0
        assert review["trg_33975e86ffde5524"]["substantive_outcome"] == (
            "confirmed_abstention"
        )
        assert review["trg_7426d9722059de76"]["substantive_outcome"] == (
            "confirmed_abstention"
        )
        assert review["trg_12690fb418279750"]["substantive_outcome"] == (
            "confirmed_abstention"
        )
        assert final["trg_93d2e2b38d1de7b2"]["final_decision"] == "accepted"
        assert final["trg_93d2e2b38d1de7b2"]["deterministic_repair_applied"] is True
        assert review["trg_adb3dfa03b0c7cf3"]["deterministic_repair_applied"] is True
        assert review["trg_adb3dfa03b0c7cf3"]["model_retry_occurred"] is False
        assert (output / "model_retry_requests.jsonl").read_text(encoding="utf-8") == ""
        assert (output / "model_retry_responses.jsonl").read_text(encoding="utf-8") == ""
    finally:
        if output.exists():
            shutil.rmtree(output)


def test_adjudication_outputs_are_deterministic() -> None:
    first_dir = Path("output/test_reduced_extraction_adjudication_first")
    second_dir = Path("output/test_reduced_extraction_adjudication_second")
    for path in [first_dir, second_dir]:
        if path.exists():
            shutil.rmtree(path)
    try:
        first = run_reduced_extraction_adjudication_v1(output_dir=first_dir)
        second = run_reduced_extraction_adjudication_v1(output_dir=second_dir)

        assert json.dumps(first["final_adjudication"], sort_keys=True) == json.dumps(
            second["final_adjudication"], sort_keys=True
        )
        assert first["repair_audit"] == second["repair_audit"]
    finally:
        for path in [first_dir, second_dir]:
            if path.exists():
                shutil.rmtree(path)
