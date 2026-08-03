from __future__ import annotations

import json
import shutil
from pathlib import Path

from segro_evidence_extraction.evidence_ready_batch_expansion_v2 import (
    EXECUTED_TARGET_IDS,
    run_evidence_ready_batch_expansion_v2,
)


def test_scale_trial_v2_selects_largest_defensible_empty_batch() -> None:
    output = Path("output/test_evidence_ready_batch_expansion_v2")
    config_path = Path("output/test_evidence_ready_batch_expansion_v2_config.json")
    try:
        result = run_evidence_ready_batch_expansion_v2(
            output_dir=output,
            config_path=config_path,
        )

        summary = result["selection_summary"]
        assert summary["candidate_inventory_count"] == 75
        assert summary["reviewed_target_count"] == 75
        assert summary["selected_count"] == 0
        assert summary["selected_target_ids"] == []
        assert summary["excluded_counts_by_reason"] == {
            "component_only": 2,
            "dictionary_clarification": 8,
            "insufficient_attribute_evidence": 63,
            "wrong_event": 2,
        }
        assert result["selected_batch"] == []
        assert result["dry_run_validation"]["overall_status"] == "passed"
        assert result["execution_estimate"]["expected_model_calls"] == 0
        assert config_path.exists()
    finally:
        cleanup(output, config_path)


def test_scale_trial_v2_retains_specific_false_positive_exclusions() -> None:
    output = Path("output/test_evidence_ready_batch_expansion_v2_exclusions")
    config_path = Path("output/test_evidence_ready_batch_expansion_v2_exclusions_config.json")
    try:
        result = run_evidence_ready_batch_expansion_v2(
            output_dir=output,
            config_path=config_path,
        )
        excluded = {row["target_id"]: row for row in result["excluded_candidates"]}

        assert excluded["trg_3d47ed5c7877ebca"]["classification"] == "wrong_event"
        assert excluded["trg_471356dd291ff34b"]["classification"] == "wrong_event"
        assert excluded["trg_75b5a1d84f166ff3"]["classification"] == "component_only"
        assert excluded["trg_c969962802095788"]["classification"] == "component_only"
        selected_ids = {record["target_id"] for record in result["selected_batch"]}
        assert not selected_ids & EXECUTED_TARGET_IDS
    finally:
        cleanup(output, config_path)


def test_scale_trial_v2_writes_deterministic_runner_config_and_artifacts() -> None:
    output = Path("output/test_evidence_ready_batch_expansion_v2_artifacts")
    config_path = Path("output/test_evidence_ready_batch_expansion_v2_artifacts_config.json")
    try:
        result = run_evidence_ready_batch_expansion_v2(
            output_dir=output,
            config_path=config_path,
        )
        selected_jsonl = output / "selected_batch.jsonl"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        assert json.loads((output / "selected_batch.json").read_text(encoding="utf-8")) == []
        assert selected_jsonl.read_text(encoding="utf-8") == ""
        assert config["unit_id"] == "enfield_unit1_scale_trial_v2"
        assert config["selected_batch_path"] == str(output / "selected_batch.json")
        assert result["input_provenance"]["retrieval_repeated"] is False
        assert result["input_provenance"]["parsing_repeated"] is False
        assert result["input_provenance"]["ocr_repeated"] is False
        assert result["input_provenance"]["vlm_repeated"] is False
        assert result["input_provenance"]["cache_expansion_repeated"] is False
    finally:
        cleanup(output, config_path)


def test_scale_trial_v2_is_deterministic() -> None:
    first_output = Path("output/test_evidence_ready_batch_expansion_v2_first")
    second_output = Path("output/test_evidence_ready_batch_expansion_v2_second")
    first_config = Path("output/test_evidence_ready_batch_expansion_v2_first_config.json")
    second_config = Path("output/test_evidence_ready_batch_expansion_v2_second_config.json")
    try:
        first = run_evidence_ready_batch_expansion_v2(
            output_dir=first_output,
            config_path=first_config,
        )
        second = run_evidence_ready_batch_expansion_v2(
            output_dir=second_output,
            config_path=second_config,
        )

        assert first["candidate_review"] == second["candidate_review"]
        assert first["selected_batch"] == second["selected_batch"]
        assert first["selection_summary"] == second["selection_summary"]
    finally:
        cleanup(first_output, first_config)
        cleanup(second_output, second_config)


def cleanup(output: Path, config_path: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    if config_path.exists():
        config_path.unlink()
