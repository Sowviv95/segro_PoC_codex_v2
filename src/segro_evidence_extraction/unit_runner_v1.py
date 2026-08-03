"""PowerShell-friendly runner for the bounded Enfield Unit 1 workflow."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator

from segro_evidence_extraction.downstream_handoff_contract_v1 import (
    CONTRACT_VERSION,
    load_contract_schemas,
)
from segro_evidence_extraction.expanded_bounded_extraction_batch_v1 import (
    CHECKPOINT_REJECTED_TARGET_IDS,
    COMPLETED_TARGET_IDS,
    build_adjudication_package,
    build_handoff_package,
    read_json_list,
    run_expanded_bounded_extraction_batch_v1,
    source_paths_by_id,
    validate_expanded_preflight,
    write_adjudication_outputs,
    write_handoff_outputs,
)
from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.reduced_bounded_extraction_batch_v1 import (
    ReducedExtractionClient,
)
from segro_evidence_extraction.source_coverage_pageindex import DEFAULT_SOURCE_MANIFEST
from segro_evidence_extraction.vertical_slice import _atomic_write_json

RUNNER_SCHEMA_VERSION = "segro_unit_runner_v1"
RUNNER_CONFIG_VERSION = "1.0.0"
DEFAULT_RUNNER_CONFIG_PATH = Path("config/enfield_unit1_runner_v1.json")

EXIT_SUCCESS = 0
EXIT_CONFIG_VALIDATION_FAILURE = 2
EXIT_INPUT_VALIDATION_FAILURE = 3
EXIT_PREFLIGHT_FAILURE = 4
EXIT_EXTRACTION_PROVIDER_FAILURE = 5
EXIT_ADJUDICATION_FAILURE = 6
EXIT_EXPORT_FAILURE = 7
EXIT_CONTRACT_VALIDATION_FAILURE = 8
EXIT_RESUME_STATE_CONFLICT = 9

StageStatus = Literal["pending", "running", "passed", "failed", "skipped"]
ExecutionMode = Literal["dry_run", "execute", "contract_only"]
CustomerCaveatPolicy = Literal["internal_only", "include"]

PROHIBITED_INVOCATION_KEYS = [
    "retrieval_invocations",
    "parser_invocations",
    "ocr_invocations",
    "vlm_invocations",
    "cache_expansion_invocations",
    "evidence_remapping_invocations",
    "target_reselection_invocations",
]


class UnitRunnerConfig(StrictBaseModel):
    schema_version: str = RUNNER_SCHEMA_VERSION
    config_version: str = RUNNER_CONFIG_VERSION
    unit_id: str
    asset_record_key: str | None = None
    repository_root: Path = Path(".")
    selected_batch_path: Path
    source_registry_path: Path = DEFAULT_SOURCE_MANIFEST
    cache_root: Path
    dictionary_artifact_path: Path
    output_root: Path = Path("output/unit_runs")
    model_provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    execution_mode: ExecutionMode = "dry_run"
    customer_caveat_policy: CustomerCaveatPolicy = "internal_only"
    max_model_retries: int = Field(default=1, ge=0, le=1)
    expected_branch: str = "feature/evidence-first-foundation"
    frozen_internal_contract_version: str = CONTRACT_VERSION
    frozen_customer_contract_version: str = CONTRACT_VERSION

    @field_validator("schema_version")
    @classmethod
    def schema_version_must_be_v1(cls, value: str) -> str:
        if value != RUNNER_SCHEMA_VERSION:
            raise ValueError(f"unsupported runner schema_version: {value}")
        return value

    @field_validator("config_version")
    @classmethod
    def config_version_must_be_compatible_v1(cls, value: str) -> str:
        if not value.startswith("1."):
            raise ValueError(f"unsupported runner config_version: {value}")
        return value


def run_unit_runner_v1(
    *,
    config_path: Path = DEFAULT_RUNNER_CONFIG_PATH,
    mode: ExecutionMode,
    resume: bool = False,
    extraction_client: ReducedExtractionClient | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    loaded = load_and_validate_config(config_path)
    if loaded["status"] != "passed":
        return failed_result(
            exit_code=EXIT_CONFIG_VALIDATION_FAILURE,
            config=None,
            run_id="unavailable",
            run_dir=Path("output/unit_runs/unavailable"),
            stage_results=[loaded],
            started=started,
        )
    config: UnitRunnerConfig = loaded["config"]
    config = config.model_copy(update={"execution_mode": mode})
    run_id = deterministic_run_id(config, config_path)
    run_dir = config.output_root / config.unit_id / run_id
    context = RunnerContext(
        config=config,
        config_path=config_path,
        run_id=run_id,
        run_dir=run_dir,
        resume=resume,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_log(context, "runner_started", {"mode": mode, "resume": resume})
    stage_results: list[dict[str, Any]] = []
    try:
        if resume:
            resume_status = validate_resume_state(context)
            if resume_status["status"] != "passed":
                stage_results.append(resume_status)
                return failed_result(
                    exit_code=EXIT_RESUME_STATE_CONFLICT,
                    config=config,
                    run_id=run_id,
                    run_dir=run_dir,
                    stage_results=stage_results,
                    started=started,
                )
        stage_results.append(write_stage(context, 1, "configuration_validation", loaded))
        input_validation = validate_inputs(context)
        stage_results.append(
            write_stage(context, 2, "repository_input_validation", input_validation)
        )
        if input_validation["status"] != "passed":
            return failed_result(
                exit_code=EXIT_INPUT_VALIDATION_FAILURE,
                config=config,
                run_id=run_id,
                run_dir=run_dir,
                stage_results=stage_results,
                started=started,
            )
        selected = read_json_list(config.selected_batch_path)
        source_paths = source_paths_by_id(config.source_registry_path)
        expected_target_ids = [str(item.get("target_id")) for item in selected]
        preflight = validate_expanded_preflight(
            selected,
            source_paths,
            expected_target_ids=expected_target_ids,
        )
        preflight_stage = {
            "status": "passed" if preflight["overall_status"] == "passed" else "failed",
            "counts": {
                "target_count": len(selected),
                "failed_target_count": preflight["failed_target_count"],
            },
            "warnings": [],
            "errors": preflight["global_errors"],
            "preflight": preflight,
            "input_fingerprint": input_fingerprint(context),
        }
        stage_results.append(
            write_stage(context, 3, "dry_run_extraction_contract_validation", preflight_stage)
        )
        if preflight_stage["status"] != "passed":
            return failed_result(
                exit_code=EXIT_PREFLIGHT_FAILURE,
                config=config,
                run_id=run_id,
                run_dir=run_dir,
                stage_results=stage_results,
                started=started,
            )
        if mode in {"dry_run", "contract_only"}:
            if mode == "contract_only":
                stage_results.extend(contract_only_stage_results(context))
            summary = build_execution_summary(
                config=config,
                run_id=run_id,
                run_dir=run_dir,
                mode=mode,
                final_status="contract_ready" if mode == "contract_only" else "execution_ready",
                exit_code=EXIT_SUCCESS,
                stage_results=stage_results,
                selected=selected,
                extraction_result=None,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
            write_run_outputs(context, stage_results, summary)
            return {
                "exit_code": EXIT_SUCCESS,
                "execution_summary": summary,
                "stages": stage_results,
            }
        if resume:
            resume_result = resume_existing_execution(context, selected, source_paths)
            if resume_result is not None:
                stage_results.extend(resume_result["stages"])
                extraction_result = resume_result.get("extraction_result")
                summary = build_execution_summary(
                    config=config,
                    run_id=run_id,
                    run_dir=run_dir,
                    mode=mode,
                    final_status="passed",
                    exit_code=EXIT_SUCCESS,
                    stage_results=stage_results,
                    selected=selected,
                    extraction_result=extraction_result,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                )
                write_run_outputs(context, stage_results, summary)
                return {
                    "exit_code": EXIT_SUCCESS,
                    "execution_summary": summary,
                    "stages": stage_results,
                    **(extraction_result or {}),
                }
        extraction_result = run_expanded_bounded_extraction_batch_v1(
            selected_batch_path=config.selected_batch_path,
            source_manifest=config.source_registry_path,
            batch_output_dir=batch_output_dir(context),
            adjudication_output_dir=adjudication_output_dir(context),
            handoff_output_dir=handoff_output_dir(context),
            extraction_client=extraction_client,
            runner_metadata=runner_metadata(context),
            expected_target_ids=expected_target_ids,
        )
        stage_results.extend(execution_stage_results(context, extraction_result))
        contract = extraction_result["handoff"]["export_validation"]["contract_validation"]
        if contract["status"] != "passed":
            return failed_result(
                exit_code=EXIT_CONTRACT_VALIDATION_FAILURE,
                config=config,
                run_id=run_id,
                run_dir=run_dir,
                stage_results=stage_results,
                started=started,
                extraction_result=extraction_result,
            )
        summary = build_execution_summary(
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            mode=mode,
            final_status="passed",
            exit_code=EXIT_SUCCESS,
            stage_results=stage_results,
            selected=selected,
            extraction_result=extraction_result,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        write_run_outputs(context, stage_results, summary)
        return {
            "exit_code": EXIT_SUCCESS,
            "execution_summary": summary,
            "stages": stage_results,
            **extraction_result,
        }
    except RuntimeError as exc:
        code = exit_code_for_runtime_error(str(exc))
        failure = {"status": "failed", "errors": [str(exc)], "warnings": [], "counts": {}}
        stage_results.append(
            write_stage(context, len(stage_results) + 1, "runtime_failure", failure)
        )
        return failed_result(
            exit_code=code,
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            stage_results=stage_results,
            started=started,
        )


class RunnerContext(StrictBaseModel):
    config: UnitRunnerConfig
    config_path: Path
    run_id: str
    run_dir: Path
    resume: bool = False


def load_and_validate_config(config_path: Path) -> dict[str, Any]:
    try:
        data = read_json_object(config_path)
        schema = read_json_object(Path("schemas/unit_runner/segro_unit_runner_v1.schema.json"))
        schema_errors = validate_json_schema_subset(data, schema)
        if schema_errors:
            return {
                "status": "failed",
                "errors": schema_errors,
                "warnings": [],
                "counts": {"schema_errors": len(schema_errors)},
                "config_path": str(config_path),
            }
        config = UnitRunnerConfig.model_validate(data)
    except (OSError, ValueError, ValidationError) as exc:
        return {
            "status": "failed",
            "errors": [str(exc)],
            "warnings": [],
            "counts": {},
            "config_path": str(config_path),
        }
    return {
        "status": "passed",
        "errors": [],
        "warnings": [],
        "counts": {"config_files": 1},
        "config": config,
        "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
    }


def validate_json_schema_subset(data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})
    for field in sorted(required):
        if field not in data:
            errors.append(f"$.{field}: missing required field")
    if schema.get("additionalProperties") is False:
        for field in sorted(set(data) - set(properties)):
            errors.append(f"$.{field}: unsupported extra field")
    for field, value in data.items():
        if field not in properties:
            continue
        validate_schema_property(f"$.{field}", value, properties[field], errors)
    return errors


def validate_schema_property(
    path: str,
    value: Any,
    property_schema: dict[str, Any],
    errors: list[str],
) -> None:
    if "const" in property_schema and value != property_schema["const"]:
        errors.append(f"{path}: expected {property_schema['const']!r}, got {value!r}")
    allowed = property_schema.get("enum")
    if allowed is not None and value not in allowed:
        errors.append(f"{path}: expected one of {allowed}, got {value!r}")
    expected_type = property_schema.get("type")
    if expected_type is not None and not json_type_matches(value, expected_type):
        errors.append(f"{path}: expected type {expected_type}, got {type(value).__name__}")
    pattern = property_schema.get("pattern")
    if pattern == "^1\\." and isinstance(value, str) and not value.startswith("1."):
        errors.append(f"{path}: unsupported V1-compatible version {value!r}")


def json_type_matches(value: Any, expected: str | list[str]) -> bool:
    expected_types = expected if isinstance(expected, list) else [expected]
    for expected_type in expected_types:
        if expected_type == "null" and value is None:
            return True
        if expected_type == "string" and isinstance(value, str):
            return True
        if expected_type == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if expected_type == "object" and isinstance(value, dict):
            return True
        if expected_type == "array" and isinstance(value, list):
            return True
        if expected_type == "boolean" and isinstance(value, bool):
            return True
    return False


def validate_inputs(context: RunnerContext) -> dict[str, Any]:
    config = context.config
    errors: list[str] = []
    warnings: list[str] = []
    observed_branch = current_branch(config.repository_root)
    if observed_branch != config.expected_branch:
        errors.append(
            f"expected branch {config.expected_branch}, observed {observed_branch}"
        )
    for label, path in {
        "selected_batch_path": config.selected_batch_path,
        "source_registry_path": config.source_registry_path,
        "cache_root": config.cache_root,
        "dictionary_artifact_path": config.dictionary_artifact_path,
    }.items():
        if not path.exists():
            errors.append(f"{label} does not exist: {path}")
    if config.model_provider != "openai":
        warnings.append(f"model_provider is configured as {config.model_provider}")
    selected_ids: list[str] = []
    if config.selected_batch_path.exists():
        selected = read_json_list(config.selected_batch_path)
        selected_ids = [str(item.get("target_id")) for item in selected]
        if len(selected_ids) != len(set(selected_ids)):
            errors.append(f"duplicate selected targets are not allowed: {selected_ids}")
        disallowed = sorted(
            set(selected_ids).intersection(COMPLETED_TARGET_IDS | CHECKPOINT_REJECTED_TARGET_IDS)
        )
        if disallowed:
            errors.append(f"disallowed targets present: {disallowed}")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "counts": {"selected_target_count": len(selected_ids)},
        "selected_target_ids": selected_ids,
        "input_fingerprint": input_fingerprint(context),
        **zero_prohibited_invocations(),
    }


def deterministic_run_id(config: UnitRunnerConfig, config_path: Path) -> str:
    payload = {
        "unit_id": config.unit_id,
        "schema_version": config.schema_version,
        "config_version": config.config_version,
        "selected_batch_hash": sha256_file(config.selected_batch_path)
        if config.selected_batch_path.exists()
        else None,
        "dictionary_hash": sha256_file(config.dictionary_artifact_path)
        if config.dictionary_artifact_path.exists()
        else None,
        "config_hash": sha256_file(config_path) if config_path.exists() else None,
        "internal_contract": config.frozen_internal_contract_version,
        "customer_contract": config.frozen_customer_contract_version,
        "execution_mode": config.execution_mode,
    }
    digest = sha256_json(payload)[:16]
    return f"{config.unit_id}_{config.execution_mode}_{digest}"


def input_fingerprint(context: RunnerContext) -> str:
    config = context.config
    return sha256_json(
        {
            "config": sha256_file(context.config_path) if context.config_path.exists() else None,
            "selected_batch": sha256_file(config.selected_batch_path)
            if config.selected_batch_path.exists()
            else None,
            "source_registry": sha256_file(config.source_registry_path)
            if config.source_registry_path.exists()
            else None,
            "dictionary": sha256_file(config.dictionary_artifact_path)
            if config.dictionary_artifact_path.exists()
            else None,
            "internal_contract": config.frozen_internal_contract_version,
            "customer_contract": config.frozen_customer_contract_version,
            "mode": config.execution_mode,
        }
    )


def write_stage(
    context: RunnerContext,
    number: int,
    name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    started = utc_now()
    status = payload.get("status", "passed")
    stage = {
        "schema_version": "segro_unit_runner_stage_manifest_v1",
        "run_id": context.run_id,
        "stage_number": number,
        "stage_name": name,
        "stage_status": status,
        "started_at": started,
        "completed_at": utc_now(),
        "input_artifact_references": input_artifacts(context),
        "output_artifact_references": output_artifacts_for_stage(context, number),
        "warnings": payload.get("warnings", []),
        "errors": payload.get("errors", []),
        "counts": payload.get("counts", {}),
        "invocation_metrics": prohibited_metrics_from_payload(payload),
        "deterministic_stage_fingerprint": sha256_json(
            {
                "stage": name,
                "input_fingerprint": payload.get("input_fingerprint")
                or input_fingerprint(context),
                "status": status,
                "counts": payload.get("counts", {}),
            }
        ),
        "executed_or_reused": "reused" if payload.get("reused") else "executed",
        "reused": bool(payload.get("reused")),
        "skip_reason": payload.get("skip_reason"),
        "error_code": payload.get("error_code"),
    }
    for key, value in payload.items():
        if key not in {"config", "status", "warnings", "errors", "counts"}:
            stage[key] = jsonable(value)
    existing_path = context.run_dir / f"stage_{number:02d}_{name}.json"
    if context.resume and existing_path.exists():
        existing = read_json_object(existing_path)
        if (
            existing.get("stage_status") == "passed"
            and existing.get("deterministic_stage_fingerprint")
            == stage["deterministic_stage_fingerprint"]
        ):
            existing["executed_or_reused"] = "reused"
            existing["reused"] = True
            _atomic_write_json(existing_path, existing)
            write_log(context, "stage_reused", {"stage": name, "status": "passed"})
            return existing
    _atomic_write_json(existing_path, stage)
    write_log(context, "stage_completed", {"stage": name, "status": status})
    return stage


def execution_stage_results(
    context: RunnerContext,
    extraction_result: dict[str, Any],
) -> list[dict[str, Any]]:
    batch_summary = extraction_result["batch"]["execution_summary"]
    adjudication_summary = extraction_result["adjudication"]["adjudication_summary"]
    handoff = extraction_result["handoff"]
    contract = handoff["export_validation"]["contract_validation"]
    return [
        write_stage(
            context,
            4,
            "bounded_extraction",
            {
                "status": "passed",
                "counts": {
                    "target_count": batch_summary["selected_count"],
                    "executed_count": batch_summary["executed_count"],
                    "model_calls": batch_summary["model_calls"],
                    "input_tokens": batch_summary["input_tokens"],
                    "output_tokens": batch_summary["output_tokens"],
                },
                "warnings": [],
                "errors": [],
                "input_fingerprint": input_fingerprint(context),
                **zero_prohibited_invocations(),
            },
        ),
        write_stage(
            context,
            5,
            "deterministic_adjudication",
            {
                "status": "passed",
                "counts": adjudication_summary,
                "warnings": [],
                "errors": [],
                "input_fingerprint": input_fingerprint(context),
                **zero_prohibited_invocations(),
            },
        ),
        write_stage(
            context,
            6,
            "internal_handoff_export",
            {
                "status": "passed",
                "counts": {"internal_records": len(handoff["internal_handoff"])},
                "warnings": [],
                "errors": [],
                "input_fingerprint": input_fingerprint(context),
                **zero_prohibited_invocations(),
            },
        ),
        write_stage(
            context,
            7,
            "customer_candidate_export",
            {
                "status": "passed",
                "counts": {"customer_records": len(handoff["customer_candidate_handoff"])},
                "warnings": [],
                "errors": [],
                "input_fingerprint": input_fingerprint(context),
                **zero_prohibited_invocations(),
            },
        ),
        write_stage(
            context,
            8,
            "frozen_contract_validation",
            {
                "status": contract["status"],
                "counts": {
                    "internal_record_count": contract["internal_record_count"],
                    "customer_record_count": contract["customer_record_count"],
                },
                "warnings": contract["warnings"],
                "errors": contract["errors"],
                "input_fingerprint": input_fingerprint(context),
                **zero_prohibited_invocations(),
            },
        ),
    ]


def contract_only_stage_results(context: RunnerContext) -> list[dict[str, Any]]:
    load_contract_schemas()
    skipped = [
        (4, "bounded_extraction", "contract-only mode does not make model calls"),
        (5, "deterministic_adjudication", "contract-only mode has no extraction results"),
        (6, "internal_handoff_export", "contract-only mode does not emit value exports"),
        (7, "customer_candidate_export", "contract-only mode does not emit value exports"),
    ]
    stages = [
        write_stage(
            context,
            number,
            name,
            {
                "status": "skipped",
                "skip_reason": reason,
                "counts": {},
                "warnings": [],
                "errors": [],
                "input_fingerprint": input_fingerprint(context),
                **zero_prohibited_invocations(),
            },
        )
        for number, name, reason in skipped
    ]
    stages.append(
        write_stage(
            context,
            8,
            "frozen_contract_validation",
            {
                "status": "passed",
                "counts": {
                    "internal_contract_version": CONTRACT_VERSION,
                    "customer_contract_version": CONTRACT_VERSION,
                },
                "warnings": [],
                "errors": [],
                "input_fingerprint": input_fingerprint(context),
                **zero_prohibited_invocations(),
            },
        )
    )
    return stages


def resume_existing_execution(
    context: RunnerContext,
    selected: list[dict[str, Any]],
    source_paths: dict[str, str],
) -> dict[str, Any] | None:
    state = classify_resume_artifacts(context)
    if state == "start_extraction":
        return None
    if state == "all_complete":
        extraction_result = load_existing_extraction_result(context)
        return {
            "stages": reused_execution_stage_results(context, extraction_result),
            "extraction_result": extraction_result,
        }
    if state == "resume_from_adjudication":
        batch_result = load_batch_result(context)
        provenance = batch_result["input_provenance"]
        adjudication = build_adjudication_package(batch_result, provenance)
        write_adjudication_outputs(adjudication, adjudication_output_dir(context))
        handoff = build_handoff_package(
            selected=selected,
            final_adjudication=adjudication["final_adjudication"],
            repair_audit=adjudication["repair_audit"],
            target_review=adjudication["target_review"],
            input_provenance=provenance,
            source_paths=source_paths,
            batch_output_dir=batch_output_dir(context),
            adjudication_output_dir=adjudication_output_dir(context),
            asset_record_key=context.config.asset_record_key,
        )
        write_handoff_outputs(handoff, handoff_output_dir(context))
        extraction_result = {
            "batch": batch_result,
            "adjudication": adjudication,
            "handoff": handoff,
        }
        return {
            "stages": reused_execution_stage_results(context, extraction_result),
            "extraction_result": extraction_result,
        }
    if state == "resume_from_export":
        batch_result = load_batch_result(context)
        adjudication = load_adjudication_result(context)
        handoff = build_handoff_package(
            selected=selected,
            final_adjudication=adjudication["final_adjudication"],
            repair_audit=adjudication["repair_audit"],
            target_review=adjudication["target_review"],
            input_provenance=batch_result["input_provenance"],
            source_paths=source_paths,
            batch_output_dir=batch_output_dir(context),
            adjudication_output_dir=adjudication_output_dir(context),
            asset_record_key=context.config.asset_record_key,
        )
        write_handoff_outputs(handoff, handoff_output_dir(context))
        extraction_result = {
            "batch": batch_result,
            "adjudication": adjudication,
            "handoff": handoff,
        }
        return {
            "stages": reused_execution_stage_results(context, extraction_result),
            "extraction_result": extraction_result,
        }
    if state == "resume_from_contract_validation":
        extraction_result = load_existing_extraction_result(context)
        return {
            "stages": reused_execution_stage_results(context, extraction_result),
            "extraction_result": extraction_result,
        }
    return None


def reused_execution_stage_results(
    context: RunnerContext,
    extraction_result: dict[str, Any],
) -> list[dict[str, Any]]:
    stages = execution_stage_results(context, extraction_result)
    for stage in stages:
        stage["executed_or_reused"] = "reused"
        stage["reused"] = True
        _atomic_write_json(context.run_dir / stage_path(stage), stage)
    return stages


def validate_resume_state(context: RunnerContext) -> dict[str, Any]:
    if not (context.run_dir / "run_manifest.json").exists():
        return {
            "status": "passed",
            "errors": [],
            "warnings": ["no existing run manifest; starting fresh"],
            "counts": {},
        }
    conflicts = []
    for path in sorted(context.run_dir.glob("stage_*.json")):
        try:
            stage = read_json_object(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            conflicts.append(f"corrupted manifest {path.name}: {exc}")
            continue
        if stage.get("run_id") not in {None, context.run_id}:
            conflicts.append(f"stage from another run ID: {path.name}")
        if stage.get("stage_status") == "running" or not stage.get("completed_at"):
            conflicts.append(f"partial stage is not resumable: {path.name}")
        if stage.get("stage_status") == "failed":
            conflicts.append(f"failed stage requires explicit rerun, not resume: {path.name}")
        if stage.get("deterministic_stage_fingerprint") is None:
            conflicts.append(f"missing fingerprint: {path.name}")
    conflicts.extend(resume_artifact_conflicts(context))
    if conflicts:
        write_resume_conflict_report(context, conflicts)
    return {
        "status": "passed" if not conflicts else "failed",
        "errors": conflicts,
        "warnings": [],
        "counts": {"existing_stage_manifests": len(list(context.run_dir.glob("stage_*.json")))},
    }


def resume_artifact_conflicts(context: RunnerContext) -> list[str]:
    conflicts: list[str] = []
    raw_path = batch_output_dir(context) / "raw_model_responses.jsonl"
    request_path = batch_output_dir(context) / "model_requests.jsonl"
    if raw_path.exists():
        raw_rows = safe_read_jsonl(raw_path, conflicts)
        request_rows = safe_read_jsonl(request_path, conflicts) if request_path.exists() else []
        raw_ids = [str(row.get("target_id")) for row in raw_rows]
        expected_ids = configured_target_ids(context)
        if len(raw_ids) != len(expected_ids):
            conflicts.append("response-count mismatch")
        if raw_ids != expected_ids:
            conflicts.append(f"missing, duplicate or reordered target responses: {raw_ids}")
        if len(raw_ids) != len(set(raw_ids)):
            conflicts.append("duplicate target responses")
        request_ids = [str(row.get("target_id")) for row in request_rows]
        if request_rows and request_ids != expected_ids:
            conflicts.append(f"model request set mismatch: {request_ids}")
    stage_4 = context.run_dir / "stage_04_bounded_extraction.json"
    if stage_4.exists() and not raw_path.exists():
        conflicts.append("extraction marked complete without raw provider evidence")
    if context.run_dir.exists() and input_fingerprint_changed(context):
        conflicts.append("changed selected-batch or model fingerprint")
    return conflicts


def classify_resume_artifacts(context: RunnerContext) -> str:
    raw_complete = complete_raw_responses_present(context)
    adjudication_complete = (adjudication_output_dir(context) / "final_adjudication.json").exists()
    export_complete = (handoff_output_dir(context) / "internal_handoff.json").exists() and (
        handoff_output_dir(context) / "customer_candidate_handoff.json"
    ).exists()
    contract_complete = (context.run_dir / "stage_08_frozen_contract_validation.json").exists()
    if raw_complete and adjudication_complete and export_complete and contract_complete:
        return "all_complete"
    if raw_complete and not adjudication_complete:
        return "resume_from_adjudication"
    if raw_complete and adjudication_complete and not export_complete:
        return "resume_from_export"
    if raw_complete and adjudication_complete and export_complete and not contract_complete:
        return "resume_from_contract_validation"
    return "start_extraction"


def complete_raw_responses_present(context: RunnerContext) -> bool:
    path = batch_output_dir(context) / "raw_model_responses.jsonl"
    if not path.exists():
        return False
    rows = read_jsonl(path)
    return [str(row.get("target_id")) for row in rows] == configured_target_ids(context)


def configured_target_ids(context: RunnerContext) -> list[str]:
    if not context.config.selected_batch_path.exists():
        return []
    return [
        str(item.get("target_id"))
        for item in read_json_list(context.config.selected_batch_path)
    ]


def input_fingerprint_changed(context: RunnerContext) -> bool:
    manifest_path = context.run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = read_json_object(manifest_path)
    return manifest.get("input_fingerprint") not in {None, input_fingerprint(context)}


def write_resume_conflict_report(context: RunnerContext, conflicts: list[str]) -> None:
    _atomic_write_json(
        context.run_dir / "resume_conflict_report.json",
        {
            "schema_version": "segro_unit_runner_resume_conflict_report_v1",
            "run_id": context.run_id,
            "status": "failed",
            "exit_code": EXIT_RESUME_STATE_CONFLICT,
            "conflicts": conflicts,
            "created_at": utc_now(),
        },
    )


def failed_result(
    *,
    exit_code: int,
    config: UnitRunnerConfig | None,
    run_id: str,
    run_dir: Path,
    stage_results: list[dict[str, Any]],
    started: float,
    extraction_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = build_execution_summary(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        mode=config.execution_mode if config else "dry_run",
        final_status="failed",
        exit_code=exit_code,
        stage_results=stage_results,
        selected=[],
        extraction_result=extraction_result,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
    if config:
        context = RunnerContext(
            config=config,
            config_path=DEFAULT_RUNNER_CONFIG_PATH,
            run_id=run_id,
            run_dir=run_dir,
        )
        write_run_outputs(context, stage_results, summary)
    return {"exit_code": exit_code, "execution_summary": summary, "stages": stage_results}


def build_execution_summary(
    *,
    config: UnitRunnerConfig | None,
    run_id: str,
    run_dir: Path,
    mode: str,
    final_status: str,
    exit_code: int,
    stage_results: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    extraction_result: dict[str, Any] | None,
    elapsed_ms: float,
) -> dict[str, Any]:
    final_rows = (
        extraction_result["adjudication"]["final_adjudication"]
        if extraction_result
        else []
    )
    counts = Counter(row.get("final_decision") for row in final_rows)
    batch_summary = extraction_result["batch"]["execution_summary"] if extraction_result else {}
    return {
        "schema_version": "segro_unit_runner_execution_summary_v1",
        "unit_id": config.unit_id if config else None,
        "asset_record_key": config.asset_record_key if config else None,
        "run_id": run_id,
        "mode": mode,
        "final_status": final_status,
        "output_path": str(run_dir),
        "selected_target_ids": [item.get("target_id") for item in selected]
        if selected
        else [],
        "accepted_count": counts.get("accepted", 0),
        "accepted_with_caveat_count": counts.get("accepted_with_caveat", 0),
        "rejected_count": counts.get("rejected", 0),
        "abstained_count": counts.get("abstained", 0),
        "model_calls": int(batch_summary.get("model_calls", 0)),
        "input_tokens": int(batch_summary.get("input_tokens", 0)),
        "output_tokens": int(batch_summary.get("output_tokens", 0)),
        "exit_code": exit_code,
        "elapsed_ms": round(elapsed_ms, 3),
        "stage_statuses": {
            stage.get("stage_name", f"stage_{index}"): stage.get(
                "stage_status", stage.get("status", "failed")
            )
            for index, stage in enumerate(stage_results, start=1)
        },
        **zero_prohibited_invocations(),
    }


def write_run_outputs(
    context: RunnerContext,
    stages: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    _atomic_write_json(
        context.run_dir / "config_snapshot.json",
        context.config.model_dump(mode="json"),
    )
    _atomic_write_json(context.run_dir / "execution_summary.json", summary)
    reconciliation = reconcile_stage_manifests(context, stages)
    manifest = {
        "schema_version": "segro_unit_runner_run_manifest_v1",
        "unit_id": context.config.unit_id,
        "run_id": context.run_id,
        "mode": context.config.execution_mode,
        "status": summary["final_status"],
        "created_at": utc_now(),
        "deterministic_run_identity": context.run_id,
        "stage_manifests": reconciliation["stage_manifests"],
        "stage_reconciliation": reconciliation,
        "stage_statuses": summary["stage_statuses"],
        "output_path": str(context.run_dir),
        "configuration_hash": sha256_file(context.config_path),
        "input_fingerprint": input_fingerprint(context),
    }
    _atomic_write_json(context.run_dir / "run_manifest.json", manifest)


def reconcile_stage_manifests(
    context: RunnerContext,
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    materialized = [stage for stage in stages if "stage_number" in stage]
    manifest_paths = [stage_path(stage) for stage in materialized]
    errors = []
    for stage, path in zip(materialized, manifest_paths, strict=True):
        disk_path = context.run_dir / path
        if not disk_path.exists():
            errors.append(f"missing materialized stage manifest: {path}")
            continue
        disk_stage = read_json_object(disk_path)
        if disk_stage.get("run_id") != context.run_id:
            errors.append(f"stage run_id mismatch: {path}")
        if disk_stage.get("deterministic_stage_fingerprint") != stage.get(
            "deterministic_stage_fingerprint"
        ):
            errors.append(f"stage fingerprint mismatch: {path}")
    return {
        "status": "passed" if not errors else "failed",
        "stage_manifests": manifest_paths,
        "errors": errors,
    }


def print_console_summary(result: dict[str, Any]) -> str:
    summary = result["execution_summary"]
    lines = [
        f"unit_id: {summary.get('unit_id')}",
        f"run_id: {summary.get('run_id')}",
        f"mode: {summary.get('mode')}",
        f"final_status: {summary.get('final_status')}",
        f"output_path: {summary.get('output_path')}",
        f"accepted_count: {summary.get('accepted_count')}",
        f"accepted_with_caveat_count: {summary.get('accepted_with_caveat_count')}",
        f"rejected_count: {summary.get('rejected_count')}",
        f"abstained_count: {summary.get('abstained_count')}",
        f"model_calls: {summary.get('model_calls')}",
        f"input_tokens: {summary.get('input_tokens')}",
        f"output_tokens: {summary.get('output_tokens')}",
        f"exit_code: {summary.get('exit_code')}",
    ]
    return "\n".join(lines)


def runner_metadata(context: RunnerContext) -> dict[str, Any]:
    return {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "runner_run_id": context.run_id,
        "unit_id": context.config.unit_id,
        "asset_record_key": context.config.asset_record_key,
        "configuration_hash": sha256_file(context.config_path),
        "selected_batch_hash": sha256_file(context.config.selected_batch_path),
        "source_registry_hash": sha256_file(context.config.source_registry_path),
        "dictionary_hash": sha256_file(context.config.dictionary_artifact_path),
        "frozen_internal_contract_version": context.config.frozen_internal_contract_version,
        "frozen_customer_contract_version": context.config.frozen_customer_contract_version,
        "max_model_retries": context.config.max_model_retries,
    }


def read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(data)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def safe_read_jsonl(path: Path, conflicts: list[str]) -> list[dict[str, Any]]:
    try:
        return read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        conflicts.append(f"corrupted JSONL artifact {path.name}: {exc}")
        return []


def load_batch_result(context: RunnerContext) -> dict[str, Any]:
    directory = batch_output_dir(context)
    return {
        "selected_batch": read_json_list(directory / "selected_batch.json"),
        "dry_run_validation": read_json_object(directory / "dry_run_validation.json"),
        "model_requests": read_jsonl(directory / "model_requests.jsonl"),
        "raw_model_responses": read_jsonl(directory / "raw_model_responses.jsonl"),
        "extraction_results_raw": read_json_list(directory / "extraction_results_raw.json"),
        "extraction_results_validated": read_json_list(
            directory / "extraction_results_validated.json"
        ),
        "final_adjudication": read_json_list(directory / "final_adjudication.json"),
        "execution_summary": read_json_object(directory / "execution_summary.json"),
        "input_provenance": read_json_object(directory / "input_provenance.json"),
    }


def load_adjudication_result(context: RunnerContext) -> dict[str, Any]:
    directory = adjudication_output_dir(context)
    return {
        "adjudication_summary": read_json_object(directory / "adjudication_summary.json"),
        "target_review": read_json_list(directory / "target_review.json"),
        "repaired_results": read_json_list(directory / "repaired_results.json"),
        "final_adjudication": read_json_list(directory / "final_adjudication.json"),
        "repair_audit": read_json_list(directory / "repair_audit.json"),
        "model_retry_requests": read_jsonl(directory / "model_retry_requests.jsonl"),
        "model_retry_responses": read_jsonl(directory / "model_retry_responses.jsonl"),
    }


def load_handoff_result(context: RunnerContext) -> dict[str, Any]:
    directory = handoff_output_dir(context)
    return {
        "internal_handoff": read_json_list(directory / "internal_handoff.json"),
        "customer_candidate_handoff": read_json_list(
            directory / "customer_candidate_handoff.json"
        ),
        "export_validation": read_json_object(directory / "export_validation.json"),
        "promotion_summary": read_json_object(directory / "promotion_summary.json"),
        "non_promotion_summary": read_json_object(directory / "non_promotion_summary.json"),
        "mapping_gaps": read_json_list(directory / "mapping_gaps.json"),
        "export_provenance": read_json_object(directory / "export_provenance.json"),
        "execution_summary": read_json_object(directory / "execution_summary.json"),
    }


def load_existing_extraction_result(context: RunnerContext) -> dict[str, Any]:
    return {
        "batch": load_batch_result(context),
        "adjudication": load_adjudication_result(context),
        "handoff": load_handoff_result(context),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(jsonable(value), ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()


def current_branch(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def batch_output_dir(context: RunnerContext) -> Path:
    return context.run_dir / "expanded_bounded_extraction_batch_v1"


def adjudication_output_dir(context: RunnerContext) -> Path:
    return context.run_dir / "expanded_bounded_extraction_adjudication_v1"


def handoff_output_dir(context: RunnerContext) -> Path:
    return context.run_dir / "expanded_downstream_handoff_exporter_v1"


def input_artifacts(context: RunnerContext) -> dict[str, str]:
    config = context.config
    return {
        "config": str(context.config_path),
        "selected_batch": str(config.selected_batch_path),
        "source_registry": str(config.source_registry_path),
        "cache_root": str(config.cache_root),
        "dictionary_artifact": str(config.dictionary_artifact_path),
    }


def output_artifacts_for_stage(context: RunnerContext, number: int) -> dict[str, str]:
    if number <= 3:
        return {"run_dir": str(context.run_dir)}
    if number == 4:
        return {"batch_output_dir": str(batch_output_dir(context))}
    if number == 5:
        return {"adjudication_output_dir": str(adjudication_output_dir(context))}
    if number in {6, 7, 8}:
        return {"handoff_output_dir": str(handoff_output_dir(context))}
    return {"run_dir": str(context.run_dir)}


def prohibited_metrics_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    metrics = zero_prohibited_invocations()
    for key in PROHIBITED_INVOCATION_KEYS:
        if int(payload.get(key, 0)) != 0:
            raise RuntimeError(f"prohibited stage invocation detected: {key}")
    return metrics


def zero_prohibited_invocations() -> dict[str, int]:
    return {key: 0 for key in PROHIBITED_INVOCATION_KEYS}


def exit_code_for_runtime_error(message: str) -> int:
    if "adjudication" in message.lower():
        return EXIT_ADJUDICATION_FAILURE
    if "handoff" in message.lower() or "export" in message.lower():
        return EXIT_EXPORT_FAILURE
    if "contract" in message.lower():
        return EXIT_CONTRACT_VALIDATION_FAILURE
    return EXIT_EXTRACTION_PROVIDER_FAILURE


def stage_path(stage: dict[str, Any]) -> str:
    if "stage_number" not in stage:
        return "unmaterialized_failure_stage.json"
    return f"stage_{int(stage['stage_number']):02d}_{stage['stage_name']}.json"


def write_log(context: RunnerContext, event: str, payload: dict[str, Any]) -> None:
    row = {"timestamp": utc_now(), "event": event, **payload}
    with (context.run_dir / "run_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items() if key != "config"}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value
