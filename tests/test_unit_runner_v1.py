from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from segro_evidence_extraction.expanded_bounded_extraction_batch_v1 import (
    EXPECTED_SELECTED_TARGET_IDS,
)
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.reduced_bounded_extraction_batch_v1 import (
    build_model_request_payload,
)
from segro_evidence_extraction.unit_runner_v1 import (
    EXIT_CONFIG_VALIDATION_FAILURE,
    EXIT_INPUT_VALIDATION_FAILURE,
    EXIT_PREFLIGHT_FAILURE,
    EXIT_RESUME_STATE_CONFLICT,
    EXIT_SUCCESS,
    deterministic_run_id,
    load_and_validate_config,
    print_console_summary,
    run_unit_runner_v1,
)
from segro_evidence_extraction.vertical_slice import (
    EvidenceBundleRecord,
    ExtractionResult,
    parse_extraction_response,
)


class MockRunnerClient:
    provider = "mock"
    model_name = "mock-unit-runner-model"

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def extract(
        self,
        *,
        target: TargetSpecification,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> tuple[ExtractionResult, dict[str, Any], dict[str, Any]]:
        if self.fail:
            raise RuntimeError("provider failure")
        self.calls.append(target.target_row_id)
        request = build_model_request_payload(target, bundle, self.model_name, max_output_tokens)
        payload = response_payload_for_target(target, bundle)
        response = {
            "provider": self.provider,
            "model": self.model_name,
            "choices": [{"message": {"content": json.dumps(payload, sort_keys=True)}}],
            "usage": {"prompt_tokens": 88, "completion_tokens": 12},
        }
        parsed = parse_extraction_response(
            target=target,
            bundle=bundle,
            provider=self.provider,
            model_name=self.model_name,
            response_text=json.dumps(payload, sort_keys=True),
        )
        parsed.model_usage.input_tokens = 88
        parsed.model_usage.output_tokens = 12
        return parsed, request, response


def test_valid_and_invalid_configuration(tmp_path: Path) -> None:
    valid = write_config(tmp_path)
    invalid = write_config(tmp_path, {"schema_version": "segro_unit_runner_v2"})

    assert load_and_validate_config(valid)["status"] == "passed"
    assert load_and_validate_config(invalid)["status"] == "failed"


def test_json_schema_validation_reports_field_paths(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        {
            "model_provider": "unsupported",
            "unexpected_field": True,
        },
    )

    validation = load_and_validate_config(config_path)

    assert validation["status"] == "failed"
    assert any("$.model_provider" in error for error in validation["errors"])
    assert any("$.unexpected_field" in error for error in validation["errors"])


def test_deterministic_run_identity(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    config = load_and_validate_config(config_path)["config"]

    assert deterministic_run_id(config, config_path) == deterministic_run_id(config, config_path)


def test_dry_run_makes_zero_model_calls_and_writes_stage_manifests(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    client = MockRunnerClient()

    result = run_unit_runner_v1(
        config_path=config_path,
        mode="dry_run",
        extraction_client=client,
    )

    summary = result["execution_summary"]
    run_dir = Path(summary["output_path"])
    assert result["exit_code"] == EXIT_SUCCESS
    assert summary["final_status"] == "execution_ready"
    assert summary["model_calls"] == 0
    assert client.calls == []
    assert (run_dir / "stage_03_dry_run_extraction_contract_validation.json").exists()
    assert not (run_dir / "expanded_bounded_extraction_batch_v1").exists()


def test_contract_only_makes_zero_model_calls_and_skips_live_stages(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    client = MockRunnerClient()

    result = run_unit_runner_v1(
        config_path=config_path,
        mode="contract_only",
        extraction_client=client,
    )

    run_dir = Path(result["execution_summary"]["output_path"])
    assert result["exit_code"] == EXIT_SUCCESS
    assert result["execution_summary"]["final_status"] == "contract_ready"
    assert result["execution_summary"]["model_calls"] == 0
    assert client.calls == []
    stage_4 = json.loads((run_dir / "stage_04_bounded_extraction.json").read_text())
    assert stage_4["stage_status"] == "skipped"
    assert stage_4["skip_reason"]
    assert "contract_only" in result["execution_summary"]["run_id"]


def test_execute_mode_calls_only_selected_three_targets_and_validates_contract(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path)
    client = MockRunnerClient()

    result = run_unit_runner_v1(
        config_path=config_path,
        mode="execute",
        extraction_client=client,
    )

    summary = result["execution_summary"]
    assert result["exit_code"] == EXIT_SUCCESS
    assert client.calls == EXPECTED_SELECTED_TARGET_IDS
    assert summary["model_calls"] == 3
    assert summary["accepted_count"] == 2
    assert summary["abstained_count"] == 1
    assert result["handoff"]["export_validation"]["contract_validation"]["status"] == "passed"
    assert all(summary[key] == 0 for key in prohibited_keys())


def test_metadata_and_asset_record_key_propagate(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, {"asset_record_key": "asset-123"})

    result = run_unit_runner_v1(
        config_path=config_path,
        mode="execute",
        extraction_client=MockRunnerClient(),
    )

    internal = result["handoff"]["internal_handoff"]
    customer = result["handoff"]["customer_candidate_handoff"]
    assert all(record["model"] == "mock-unit-runner-model" for record in internal)
    assert all(record["source_path"] for record in internal)
    assert all(
        record["input_provenance"]["runner_metadata"]["runner_run_id"]
        == result["execution_summary"]["run_id"]
        for record in internal
    )
    assert all(record["asset_record_key"] == "asset-123" for record in customer)


def test_resume_from_completed_preflight_reuses_stage(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)

    first = run_unit_runner_v1(config_path=config_path, mode="dry_run")
    second = run_unit_runner_v1(config_path=config_path, mode="dry_run", resume=True)

    assert first["execution_summary"]["run_id"] == second["execution_summary"]["run_id"]
    assert any(stage["executed_or_reused"] == "reused" for stage in second["stages"])


def test_resume_from_persisted_raw_responses_uses_zero_model_calls(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    first_client = MockRunnerClient()
    first = run_unit_runner_v1(
        config_path=config_path,
        mode="execute",
        extraction_client=first_client,
    )
    run_dir = Path(first["execution_summary"]["output_path"])
    shutil.rmtree(run_dir / "expanded_bounded_extraction_adjudication_v1")
    shutil.rmtree(run_dir / "expanded_downstream_handoff_exporter_v1")
    for stage_number in range(5, 9):
        for path in run_dir.glob(f"stage_{stage_number:02d}_*.json"):
            path.unlink()
    resume_client = MockRunnerClient(fail=True)

    resumed = run_unit_runner_v1(
        config_path=config_path,
        mode="execute",
        resume=True,
        extraction_client=resume_client,
    )

    assert resumed["exit_code"] == EXIT_SUCCESS
    assert resume_client.calls == []
    assert resumed["execution_summary"]["model_calls"] == 3


def test_resume_from_completed_adjudication_resumes_export(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    first = run_unit_runner_v1(
        config_path=config_path,
        mode="execute",
        extraction_client=MockRunnerClient(),
    )
    run_dir = Path(first["execution_summary"]["output_path"])
    shutil.rmtree(run_dir / "expanded_downstream_handoff_exporter_v1")
    for stage_number in range(6, 9):
        for path in run_dir.glob(f"stage_{stage_number:02d}_*.json"):
            path.unlink()
    resume_client = MockRunnerClient(fail=True)

    resumed = run_unit_runner_v1(
        config_path=config_path,
        mode="execute",
        resume=True,
        extraction_client=resume_client,
    )

    assert resumed["exit_code"] == EXIT_SUCCESS
    assert resume_client.calls == []
    assert (run_dir / "expanded_downstream_handoff_exporter_v1" / "internal_handoff.json").exists()


def test_resume_conflict_returns_exit_code(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    result = run_unit_runner_v1(config_path=config_path, mode="dry_run")
    run_dir = Path(result["execution_summary"]["output_path"])
    stage_path = run_dir / "stage_03_dry_run_extraction_contract_validation.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["stage_status"] = "running"
    stage_path.write_text(json.dumps(stage, sort_keys=True), encoding="utf-8")

    resumed = run_unit_runner_v1(config_path=config_path, mode="dry_run", resume=True)

    assert resumed["exit_code"] == EXIT_RESUME_STATE_CONFLICT
    assert (run_dir / "resume_conflict_report.json").exists()


def test_resume_partial_provider_state_fails_with_conflict(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    result = run_unit_runner_v1(
        config_path=config_path,
        mode="execute",
        extraction_client=MockRunnerClient(),
    )
    run_dir = Path(result["execution_summary"]["output_path"])
    raw_path = run_dir / "expanded_bounded_extraction_batch_v1" / "raw_model_responses.jsonl"
    rows = raw_path.read_text(encoding="utf-8").splitlines()
    raw_path.write_text("\n".join(rows[:2]) + "\n", encoding="utf-8")
    shutil.rmtree(run_dir / "expanded_bounded_extraction_adjudication_v1")

    resumed = run_unit_runner_v1(config_path=config_path, mode="execute", resume=True)

    assert resumed["exit_code"] == EXIT_RESUME_STATE_CONFLICT


def test_explicit_exit_codes_for_config_input_preflight_and_extraction_failures(
    tmp_path: Path,
) -> None:
    bad_config = write_config(tmp_path, {"config_version": "2.0.0"})
    assert run_unit_runner_v1(config_path=bad_config, mode="dry_run")["exit_code"] == (
        EXIT_CONFIG_VALIDATION_FAILURE
    )

    missing_input = write_config(tmp_path, {"selected_batch_path": "output/missing.json"})
    assert run_unit_runner_v1(config_path=missing_input, mode="dry_run")["exit_code"] == (
        EXIT_INPUT_VALIDATION_FAILURE
    )

    bad_batch_path = tmp_path / "bad_selected.json"
    bad_batch_path.write_text("[]", encoding="utf-8")
    bad_preflight = write_config(tmp_path, {"selected_batch_path": str(bad_batch_path)})
    assert run_unit_runner_v1(config_path=bad_preflight, mode="dry_run")["exit_code"] in {
        EXIT_INPUT_VALIDATION_FAILURE,
        EXIT_PREFLIGHT_FAILURE,
    }

    provider_fail = run_unit_runner_v1(
        config_path=write_config(tmp_path),
        mode="execute",
        extraction_client=MockRunnerClient(fail=True),
    )
    assert provider_fail["exit_code"] != EXIT_SUCCESS


def test_json_jsonl_csv_consistency_and_console_summary(tmp_path: Path) -> None:
    result = run_unit_runner_v1(
        config_path=write_config(tmp_path),
        mode="execute",
        extraction_client=MockRunnerClient(),
    )
    run_dir = Path(result["execution_summary"]["output_path"])
    handoff_dir = run_dir / "expanded_downstream_handoff_exporter_v1"
    internal_json = json.loads((handoff_dir / "internal_handoff.json").read_text())
    internal_jsonl = [
        json.loads(line)
        for line in (handoff_dir / "internal_handoff.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert len(internal_json) == len(internal_jsonl) == 3
    assert "exit_code: 0" in print_console_summary(result)


def test_powershell_wrapper_contains_no_secrets_and_preserves_exit_code() -> None:
    script = Path("scripts/run_enfield_unit1_v1.ps1").read_text(encoding="utf-8")

    assert "OPENAI_API_KEY" not in script
    assert "exit $ExitCode" in script
    assert "--dry-run" in script
    assert "--execute" in script
    assert "--contract-only" in script


def response_payload_for_target(
    target: TargetSpecification,
    bundle: EvidenceBundleRecord,
) -> dict[str, Any]:
    if target.expected_field.endswith("certificate_available"):
        value = "Y"
    elif target.expected_field.endswith("certificate_type"):
        value = "Certificate of Installation & Testing of Horizontal Lifeline System"
    else:
        value = "Horizontal Lifeline System"
    return {
        "raw_value": value,
        "extracted_value": value,
        "normalized_value": value,
        "status": "extracted",
        "confidence": 0.92,
        "selected_supporting_span_ids": [bundle.evidence_spans[0].span_id],
        "value_bearing_quote": value,
        "reasoning_summary": "Mocked unit-runner response.",
        "ambiguity_or_caveat": None,
    }


def write_config(tmp_path: Path, overrides: dict[str, Any] | None = None) -> Path:
    output_root = tmp_path / "unit_runs"
    if output_root.exists():
        shutil.rmtree(output_root)
    config = {
        "schema_version": "segro_unit_runner_v1",
        "config_version": "1.0.0",
        "unit_id": "enfield_unit1_test",
        "asset_record_key": "enfield-unit1-test",
        "repository_root": ".",
        "selected_batch_path": (
            "output/enfield_unit1_bounded_extraction_expansion_preparation_v1/"
            "selected_batch.json"
        ),
        "source_registry_path": "output/sprint3_source_ingestion/source_pack_manifest.json",
        "cache_root": "output/enfield_unit1_evidence_first_vertical_slice_v1/page_cache",
        "dictionary_artifact_path": (
            "output/enfield_unit1_extraction_batch_construction_v1/"
            "normalization_validation_plan.json"
        ),
        "output_root": str(output_root),
        "model_provider": "mock",
        "model_name": "mock-unit-runner-model",
        "execution_mode": "dry_run",
        "customer_caveat_policy": "internal_only",
        "max_model_retries": 1,
        "expected_branch": "feature/evidence-first-foundation",
        "frozen_internal_contract_version": "1.0.0",
        "frozen_customer_contract_version": "1.0.0",
    }
    config.update(overrides or {})
    path = tmp_path / f"runner_{abs(hash(json.dumps(config, sort_keys=True)))}.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return path


def prohibited_keys() -> list[str]:
    return [
        "retrieval_invocations",
        "parser_invocations",
        "ocr_invocations",
        "vlm_invocations",
        "cache_expansion_invocations",
        "evidence_remapping_invocations",
        "target_reselection_invocations",
    ]
