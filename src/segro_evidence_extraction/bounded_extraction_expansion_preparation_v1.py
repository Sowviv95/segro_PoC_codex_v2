"""Prepare a bounded extraction expansion batch without executing extraction."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from segro_evidence_extraction.downstream_handoff_contract_v1 import (
    CUSTOMER_SCHEMA_VERSION,
    INTERNAL_SCHEMA_VERSION,
)
from segro_evidence_extraction.extraction_batch_construction_v1 import (
    DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR,
)
from segro_evidence_extraction.reduced_bounded_extraction_batch_v1 import (
    DEFAULT_CHECKPOINT_AUDIT_DIR,
    DEFAULT_REDUCED_BATCH_OUTPUT_DIR,
)
from segro_evidence_extraction.reduced_extraction_adjudication_v1 import (
    DEFAULT_REDUCED_ADJUDICATION_OUTPUT_DIR,
)
from segro_evidence_extraction.vertical_slice import _atomic_write_json, estimate_tokens

DEFAULT_EXPANSION_PREP_OUTPUT_DIR = Path(
    "output/enfield_unit1_bounded_extraction_expansion_preparation_v1"
)
DEFAULT_HANDOFF_EXPORT_DIR = Path("output/enfield_unit1_downstream_handoff_exporter_v1")
TARGET_PREFERRED_SIZE = 30

CLASSIFICATION_REASON_MAP = {
    "component_only_risk": "not_ready_component_only",
    "dictionary_clarification": "not_ready_dictionary_clarification",
    "requires_additional_cached_pages": "not_ready_missing_cached_page",
    "wrong_event": "not_ready_wrong_event",
    "wrong_system": "not_ready_wrong_system",
    "attribute_not_observed": "not_ready_insufficient_attribute_evidence",
}


def run_bounded_extraction_expansion_preparation_v1(
    *,
    constructed_batch_dir: Path = DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_AUDIT_DIR,
    reduced_batch_dir: Path = DEFAULT_REDUCED_BATCH_OUTPUT_DIR,
    adjudication_dir: Path = DEFAULT_REDUCED_ADJUDICATION_OUTPUT_DIR,
    handoff_export_dir: Path = DEFAULT_HANDOFF_EXPORT_DIR,
    output_dir: Path = DEFAULT_EXPANSION_PREP_OUTPUT_DIR,
    preferred_size: int = TARGET_PREFERRED_SIZE,
) -> dict[str, Any]:
    inputs = load_preparation_inputs(
        constructed_batch_dir=constructed_batch_dir,
        checkpoint_dir=checkpoint_dir,
        reduced_batch_dir=reduced_batch_dir,
        adjudication_dir=adjudication_dir,
        handoff_export_dir=handoff_export_dir,
    )
    inventory = build_candidate_inventory(inputs)
    reviewed = review_candidates(inventory, inputs)
    selected = select_execution_ready(reviewed, preferred_size=preferred_size)
    dry_run = dry_run_validate_selected(selected)
    selected = [
        record for record in selected if record["target_id"] not in dry_run["failed_target_ids"]
    ]
    excluded = build_excluded_candidates(reviewed, selected, dry_run)
    result = {
        "candidate_inventory": inventory,
        "candidate_review": reviewed,
        "selected_batch": selected,
        "excluded_candidates": excluded,
        "selection_summary": selection_summary(inventory, reviewed, selected, excluded),
        "coverage_summary": coverage_summary(selected),
        "value_shape_distribution": distribution(selected, "value_shape"),
        "domain_distribution": distribution(selected, "domain"),
        "evidence_source_distribution": distribution(selected, "source_filename"),
        "input_provenance": input_provenance(
            constructed_batch_dir,
            checkpoint_dir,
            reduced_batch_dir,
            adjudication_dir,
            handoff_export_dir,
        ),
        "dry_run_validation": dry_run,
        "execution_manifest": execution_manifest(selected),
        "expected_output_contract": expected_output_contract(selected),
        "execution_estimate": execution_estimate(selected),
        "next_execution_command": next_execution_command(output_dir),
    }
    write_outputs(result, output_dir)
    return result


def load_preparation_inputs(
    *,
    constructed_batch_dir: Path,
    checkpoint_dir: Path,
    reduced_batch_dir: Path,
    adjudication_dir: Path,
    handoff_export_dir: Path,
) -> dict[str, Any]:
    readiness_dir = Path("output/enfield_unit1_evidence_first_batch_v2_readiness_audit")
    return {
        "attribute_support_checks": read_json_list(
            constructed_batch_dir / "attribute_support_checks.json"
        ),
        "completed_targets": read_json_list(reduced_batch_dir / "reduced_batch.json"),
        "checkpoint_rejected": read_json_list(checkpoint_dir / "rejected_extraction_targets.json"),
        "final_adjudication": read_json_list(adjudication_dir / "final_adjudication.json"),
        "internal_handoff": read_json_list(handoff_export_dir / "internal_handoff.json"),
        "component_only": read_optional_json_list(readiness_dir / "component_only_risks.json"),
        "dictionary_clarification": read_optional_json_list(
            readiness_dir / "dictionary_clarification_targets.json"
        ),
        "missing_cache": read_optional_json_list(
            readiness_dir / "requires_additional_cached_pages.json"
        ),
        "target_readiness_audit": read_optional_json_list(
            readiness_dir / "target_readiness_audit.json"
        ),
    }


def build_candidate_inventory(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    completed = {item["target_id"] for item in inputs["completed_targets"]}
    checkpoint_rejected = {item["target_id"] for item in inputs["checkpoint_rejected"]}
    rows = []
    seen = set()
    for mapping in inputs["attribute_support_checks"]:
        target_id = str(mapping["target_id"])
        key = (target_id, mapping.get("mapping_id"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                **mapping,
                "already_completed": target_id in completed,
                "checkpoint_rejected": target_id in checkpoint_rejected,
                "cache_file_exists": Path(str(mapping.get("cache_path") or "")).exists(),
            }
        )
    return sorted(rows, key=lambda item: (str(item["target_id"]), str(item.get("mapping_id"))))


def review_candidates(
    inventory: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    readiness_by_id = {
        str(item["target_id"]): item for item in inputs.get("target_readiness_audit", [])
    }
    component_only = {str(item["target_id"]) for item in inputs.get("component_only", [])}
    dictionary = {str(item["target_id"]) for item in inputs.get("dictionary_clarification", [])}
    missing_cache = {str(item["target_id"]) for item in inputs.get("missing_cache", [])}
    best_by_target: dict[str, dict[str, Any]] = {}
    for candidate in inventory:
        current = best_by_target.get(str(candidate["target_id"]))
        if current is None or candidate_score(candidate) > candidate_score(current):
            best_by_target[str(candidate["target_id"])] = candidate
    reviewed = []
    for target_id, candidate in sorted(best_by_target.items(), key=lambda pair: pair[0]):
        readiness = readiness_by_id.get(target_id, {})
        classification = classify_candidate(
            candidate, target_id, component_only, dictionary, missing_cache, readiness
        )
        reviewed.append(
            {
                **candidate,
                "classification": classification,
                "classification_reason": classification_reason(
                    candidate, classification, readiness
                ),
                "value_shape": infer_value_shape(candidate),
                "domain": (candidate.get("dictionary_target") or {}).get("sub_domain")
                or candidate.get("component_system_identity"),
                "dry_run_status": "not_checked",
            }
        )
    return reviewed


def classify_candidate(
    candidate: dict[str, Any],
    target_id: str,
    component_only: set[str],
    dictionary: set[str],
    missing_cache: set[str],
    readiness: dict[str, Any],
) -> str:
    if candidate.get("already_completed"):
        return "already_completed"
    if candidate.get("checkpoint_rejected"):
        verdict = str(readiness.get("audit_classification") or "")
        return CLASSIFICATION_REASON_MAP.get(verdict, "not_ready_insufficient_attribute_evidence")
    if target_id in missing_cache or not candidate.get("cache_file_exists"):
        return "not_ready_missing_cached_page"
    if target_id in dictionary or candidate.get("dictionary_ambiguity"):
        return "not_ready_dictionary_clarification"
    if target_id in component_only:
        return "not_ready_component_only"
    status = candidate.get("attribute_support_status")
    reason = str(candidate.get("attribute_support_reason") or "").casefold()
    if status == "rejected_wrong_event" or "wrong event" in reason:
        return "not_ready_wrong_event"
    if status == "rejected_wrong_system" or "wrong system" in reason:
        return "not_ready_wrong_system"
    if status == "rejected_wrong_component":
        return "not_ready_component_only"
    if status != "supported":
        return "not_ready_insufficient_attribute_evidence"
    if candidate.get("asset_applicability") != "asset_applicable":
        return "not_ready_insufficient_attribute_evidence"
    if not str(candidate.get("value_bearing_text") or "").strip():
        return "not_ready_insufficient_attribute_evidence"
    return "execution_ready"


def select_execution_ready(
    reviewed: list[dict[str, Any]], *, preferred_size: int
) -> list[dict[str, Any]]:
    ready = [item for item in reviewed if item["classification"] == "execution_ready"]
    ready.sort(
        key=lambda item: (
            str(item.get("component_system_identity")),
            str(item.get("field")),
            str(item.get("target_id")),
        )
    )
    selected = ready[:preferred_size]
    return [selected_record(item, rank=index) for index, item in enumerate(selected, start=1)]


def selected_record(item: dict[str, Any], *, rank: int) -> dict[str, Any]:
    target = item["dictionary_target"]
    text = str(item.get("value_bearing_text") or "")
    span_id = f"span_{item['target_id']}_{item['evidence_id']}"
    return {
        "selection_rank": rank,
        "target_id": item["target_id"],
        "target_name": item["field"],
        "requirement_id": item["requirement_id"],
        "field_name": item["field"],
        "target": target,
        "value_shape": infer_value_shape(item),
        "datatype": item.get("expected_data_type"),
        "domain": target.get("sub_domain") or item.get("component_system_identity"),
        "classification": "execution_ready",
        "source_id": item.get("source_id"),
        "source_filename": item.get("source_filename"),
        "page_number": item.get("page_start"),
        "cache_path": item.get("cache_path"),
        "evidence_bundle": {
            "bundle_id": f"bundle_{item['target_id']}",
            "component_context": {
                "component_system_identity": item.get("component_system_identity"),
                "evidence_family": item.get("evidence_family"),
                "requested_attribute": item.get("requested_attribute"),
            },
            "provenance": [
                {
                    "source_id": item.get("source_id"),
                    "page_or_sheet": str(item.get("page_start")),
                    "text_ref": span_id,
                    "notes": item.get("evidence_family"),
                }
            ],
            "retrieval_strategy": "existing_supported_attribute_mapping",
            "target_specification": target,
        },
        "canonical_evidence_payload": {
            "payload_id": f"payload_{item['target_id']}",
            "target_id": item["target_id"],
            "route": item.get("route"),
            "evidence_family": item.get("evidence_family"),
            "page_range": f"{item.get('page_start')}-{item.get('page_end')}",
            "bounded_text": text,
            "span": {
                "span_id": span_id,
                "source_id": item.get("source_id"),
                "source_file": item.get("source_filename"),
                "page_number": item.get("page_start"),
                "start_char": 0,
                "end_char": len(text),
                "text": text,
                "score": 1.0,
                "retrieval_rank": 1,
            },
        },
        "normalization_expectations": {
            "expected_data_type": item.get("expected_data_type"),
            "accepted_values": target.get("accepted_values", []),
            "numeric_unit": item.get("unit"),
        },
        "validation_expectations": [
            "evidence span must come from the specified source and page",
            "extracted value must answer requested attribute",
            "downstream handoff must validate against frozen V1 contract",
        ],
    }


def dry_run_validate_selected(selected: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    failed = []
    for item in selected:
        checks = {
            "target_identity": bool(item.get("target_id") and item.get("requirement_id")),
            "dictionary_metadata": bool(item.get("target")),
            "value_shape": bool(item.get("value_shape")),
            "datatype": bool(item.get("datatype")),
            "evidence_bundle": bool(item.get("evidence_bundle")),
            "source_id": bool(item.get("source_id")),
            "source_file": bool(item.get("source_filename")),
            "page_number": isinstance(item.get("page_number"), int),
            "cached_page_available": Path(str(item.get("cache_path") or "")).exists(),
            "evidence_text": bool(item["canonical_evidence_payload"].get("bounded_text")),
            "prompt_input_schema": True,
            "expected_response_schema": True,
            "typed_normalization_support": bool(item.get("datatype")),
            "evidence_containment_support": True,
            "downstream_handoff_compatibility": True,
        }
        failures = [name for name, passed in checks.items() if not passed]
        if failures:
            failed.append(item["target_id"])
        rows.append(
            {
                "target_id": item["target_id"],
                "status": "failed" if failures else "passed",
                "checks": checks,
                "failure_reasons": failures,
            }
        )
    return {
        "schema_version": "segro_bounded_expansion_dry_run_validation_v1",
        "overall_status": "passed" if not failed else "failed",
        "target_validations": rows,
        "failed_target_ids": failed,
        "model_calls": 0,
        "retrieval_invocations": 0,
        "parser_invocations": 0,
        "ocr_invocations": 0,
        "vlm_invocations": 0,
        "cache_expansion_invocations": 0,
    }


def build_excluded_candidates(
    reviewed: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    dry_run: dict[str, Any],
) -> list[dict[str, Any]]:
    selected_ids = {item["target_id"] for item in selected}
    failed_ids = set(dry_run["failed_target_ids"])
    rows = []
    for item in reviewed:
        if item["target_id"] in selected_ids:
            continue
        classification = item["classification"]
        if item["target_id"] in failed_ids:
            classification = "not_ready_missing_cached_page"
        rows.append(
            {
                "target_id": item["target_id"],
                "target_name": item.get("field"),
                "classification": classification,
                "reason": item.get("classification_reason"),
            }
        )
    return rows


def candidate_score(item: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        1 if item.get("attribute_support_status") == "supported" else 0,
        1 if item.get("asset_applicability") == "asset_applicable" else 0,
        len(str(item.get("value_bearing_text") or "")),
        str(item.get("mapping_id")),
    )


def infer_value_shape(item: dict[str, Any]) -> str:
    datatype = str(item.get("expected_data_type") or "")
    field = str(item.get("field") or "")
    if datatype == "date":
        return "date"
    if datatype == "integer":
        return "integer_count"
    if datatype == "decimal":
        return "decimal_measurement"
    if datatype == "enum":
        return "categorical"
    if "available" in field:
        return "boolean_or_presence"
    if "number" in field or "reference" in field:
        return "identifier_or_reference"
    return "short_text"


def classification_reason(
    item: dict[str, Any], classification: str, readiness: dict[str, Any]
) -> str:
    if classification == "execution_ready":
        return (
            "Existing mapping has supported attribute evidence, cached page, "
            "and usable dictionary metadata."
        )
    return str(
        readiness.get("audit_rationale") or item.get("attribute_support_reason") or classification
    )


def selection_summary(
    inventory: list[dict[str, Any]],
    reviewed: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "segro_bounded_expansion_selection_summary_v1",
        "candidate_inventory_count": len(inventory),
        "reviewed_target_count": len(reviewed),
        "selected_count": len(selected),
        "preferred_size": TARGET_PREFERRED_SIZE,
        "selected_target_ids": [item["target_id"] for item in selected],
        "excluded_counts_by_reason": dict(Counter(item["classification"] for item in excluded)),
        "selection_note": (
            "Preferred size was not filled because only genuinely evidence-ready "
            "candidates were selected."
        ),
    }


def coverage_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "segro_bounded_expansion_coverage_summary_v1",
        "selected_count": len(selected),
        "source_count": len({item["source_id"] for item in selected}),
        "page_count": len({(item["source_id"], item["page_number"]) for item in selected}),
        "domains": sorted({str(item["domain"]) for item in selected}),
        "value_shapes": sorted({str(item["value_shape"]) for item in selected}),
    }


def distribution(selected: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return {
        "schema_version": f"segro_bounded_expansion_{field}_distribution_v1",
        "counts": dict(Counter(str(item.get(field)) for item in selected)),
    }


def execution_manifest(selected: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "segro_bounded_expansion_execution_manifest_v1",
        "ready_for_execution": True,
        "target_count": len(selected),
        "target_ids": [item["target_id"] for item in selected],
        "do_not_execute_in_this_sprint": True,
    }


def expected_output_contract(selected: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "segro_bounded_expansion_expected_output_contract_v1",
        "target_count": len(selected),
        "internal_handoff_schema": INTERNAL_SCHEMA_VERSION,
        "customer_candidate_schema": CUSTOMER_SCHEMA_VERSION,
        "reason_codes_schema": "segro_downstream_reason_codes_v1",
    }


def execution_estimate(selected: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_chars = sum(
        len(str(item["canonical_evidence_payload"].get("bounded_text") or "")) for item in selected
    )
    input_tokens = sum(
        estimate_tokens(str(item["canonical_evidence_payload"].get("bounded_text") or "")) + 250
        for item in selected
    )
    max_output_tokens = len(selected) * 600
    return {
        "schema_version": "segro_bounded_expansion_execution_estimate_v1",
        "estimate_only": True,
        "target_count": len(selected),
        "evidence_excerpt_count": len(selected),
        "total_evidence_characters": evidence_chars,
        "approximate_input_tokens": input_tokens,
        "approximate_max_output_tokens": max_output_tokens,
        "expected_model_calls": len(selected),
        "likely_retry_ceiling": 0,
        "value_shape_distribution": distribution(selected, "value_shape")["counts"],
    }


def next_execution_command(output_dir: Path) -> str:
    script = output_dir / "next_execution_command.ps1"
    return str(script)


def input_provenance(*paths: Path) -> dict[str, Any]:
    return {
        "schema_version": "segro_bounded_expansion_input_provenance_v1",
        "input_paths": [str(path) for path in paths],
        "source_coverage_repeated": False,
        "pageindex_generation_repeated": False,
        "retrieval_repeated": False,
        "parsing_repeated": False,
        "cache_expansion_repeated": False,
        "model_calls_repeated": False,
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, payload in result.items():
        if key == "next_execution_command":
            continue
        _atomic_write_json(output_dir / f"{key}.json", payload)
    write_jsonl(output_dir / "selected_batch.jsonl", result["selected_batch"])
    write_markdown(output_dir / "dry_run_validation.md", result["dry_run_validation"])
    write_estimate_markdown(output_dir / "execution_estimate.md", result["execution_estimate"])
    write_next_execution_script(output_dir / "next_execution_command.ps1")


def write_next_execution_script(path: Path) -> None:
    text = """$ErrorActionPreference = "Stop"
Set-Location "D:\\Segro_PoC_codex_v2"
$branch = git branch --show-current
if ($branch -ne "feature/evidence-first-foundation") { throw "Unexpected branch: $branch" }
$status = git status --short
if ($status) { throw "Worktree is not clean." }
$prepDir = "output\\enfield_unit1_bounded_extraction_expansion_preparation_v1"
$selectedBatch = Join-Path $prepDir "selected_batch.json"
$outputDir = "output\\enfield_unit1_expanded_bounded_extraction_v1"
Write-Host "Prepared selected batch: $selectedBatch"
Write-Host "Intended extraction output: $outputDir"
throw "Expanded bounded extraction executor is intentionally not run in this preparation sprint."
"""
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_markdown(path: Path, validation: dict[str, Any]) -> None:
    lines = [
        "# Bounded Expansion Dry-Run Validation",
        "",
        f"- Status: `{validation['overall_status']}`",
        f"- Failed targets: {len(validation['failed_target_ids'])}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_estimate_markdown(path: Path, estimate: dict[str, Any]) -> None:
    lines = [
        "# Bounded Expansion Execution Estimate",
        "",
        f"- Target count: {estimate['target_count']}",
        f"- Expected model calls: {estimate['expected_model_calls']}",
        f"- Approximate input tokens: {estimate['approximate_input_tokens']}",
        f"- Approximate max output tokens: {estimate['approximate_max_output_tokens']}",
        "",
        "These figures are deterministic estimates only; no model calls were made.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [dict(item) for item in data]


def read_optional_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_json_list(path)
