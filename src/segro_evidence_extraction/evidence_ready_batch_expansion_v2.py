"""Prepare an evidence-ready scale-trial batch from existing artifacts only."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from segro_evidence_extraction.bounded_extraction_expansion_preparation_v1 import (
    distribution,
    dry_run_validate_selected,
    execution_estimate,
    write_jsonl,
    write_markdown,
)
from segro_evidence_extraction.expanded_bounded_extraction_batch_v1 import (
    CHECKPOINT_REJECTED_TARGET_IDS,
)
from segro_evidence_extraction.parsing.service import load_source_registry
from segro_evidence_extraction.vertical_slice import _atomic_write_json

DEFAULT_EVIDENCE_READY_BATCH_V2_OUTPUT_DIR = Path(
    "output/enfield_unit1_evidence_ready_batch_expansion_v2"
)
DEFAULT_SCALE_TRIAL_CONFIG_PATH = Path("config/enfield_unit1_scale_trial_runner_v1.json")
READINESS_AUDIT_DIR = Path("output/enfield_unit1_evidence_first_batch_v2_readiness_audit")
FALSE_POSITIVE_AUDIT_DIR = Path("output/enfield_unit1_evidence_first_batch_v2_false_positive_audit")
CACHE_EXPANSION_DIR = Path("output/enfield_unit1_evidence_first_batch_v2_cache_expansion_v1")
SOURCE_MANIFEST = Path("output/sprint3_source_ingestion/source_pack_manifest.json")
PAGE_CACHE_ROOT = Path("output/enfield_unit1_evidence_first_vertical_slice_v1/page_cache")

EXECUTED_TARGET_IDS = {
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
    "trg_b16ea18d75c226c0",
    "trg_2ab51d5b0cc8b48d",
    "trg_199c5560cea7955c",
}


def run_evidence_ready_batch_expansion_v2(
    *,
    output_dir: Path = DEFAULT_EVIDENCE_READY_BATCH_V2_OUTPUT_DIR,
    config_path: Path = DEFAULT_SCALE_TRIAL_CONFIG_PATH,
    preferred_min: int = 10,
    preferred_max: int = 30,
) -> dict[str, Any]:
    registry = source_registry_by_file(SOURCE_MANIFEST)
    candidates = candidate_inventory(registry)
    reviewed = [review_candidate(candidate) for candidate in candidates]
    selected = selected_records(reviewed, preferred_max=preferred_max)
    dry_run = dry_run_validate_selected(selected)
    selected = [
        record for record in selected if record["target_id"] not in dry_run["failed_target_ids"]
    ]
    excluded = excluded_candidates(reviewed, selected, dry_run)
    result = {
        "candidate_inventory": candidates,
        "candidate_review": reviewed,
        "selected_batch": selected,
        "excluded_candidates": excluded,
        "selection_summary": selection_summary(
            candidates,
            reviewed,
            selected,
            excluded,
            preferred_min=preferred_min,
            preferred_max=preferred_max,
        ),
        "coverage_summary": coverage_summary(selected),
        "value_shape_distribution": distribution(selected, "value_shape"),
        "domain_distribution": distribution(selected, "domain"),
        "evidence_source_distribution": distribution(selected, "source_filename"),
        "dry_run_validation": dry_run,
        "execution_estimate": execution_estimate(selected),
        "input_provenance": input_provenance(),
    }
    write_outputs(result, output_dir)
    write_scale_trial_config(config_path, output_dir / "selected_batch.json")
    return {**result, "scale_trial_config_path": str(config_path)}


def candidate_inventory(registry: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    readiness = read_json_list(READINESS_AUDIT_DIR / "target_readiness_audit.json")
    cache_candidates = {
        str(item["target_id"]): item
        for item in read_json_list(READINESS_AUDIT_DIR / "cached_evidence_candidates.json")
    }
    rows = []
    for item in readiness:
        source_file = str(item.get("best_cached_source") or item.get("audit_source_file") or "")
        source = registry.get(source_file, {})
        page_number = int(item.get("best_cached_page") or 0)
        cache_path = cache_path_for(source.get("source_id"), page_number)
        cached = cache_candidates.get(str(item["target_id"]), {})
        rows.append(
            {
                **item,
                "source_id": source.get("source_id"),
                "source_path": source.get("source_path"),
                "source_filename": source_file,
                "page_number": page_number,
                "cache_path": str(cache_path) if cache_path else None,
                "cache_file_exists": bool(cache_path and cache_path.exists()),
                "value_bearing_text": cached.get("excerpt") or item.get("best_cached_excerpt"),
            }
        )
    return sorted(rows, key=lambda row: (int(row.get("selection_rank") or 9999), row["target_id"]))


def review_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    target_id = str(candidate["target_id"])
    false_positive = false_positive_classification(target_id)
    classification = "execution_ready"
    reason = "Confirmed execution-ready in readiness audit with cached local evidence."
    if target_id in EXECUTED_TARGET_IDS:
        classification = "already_completed"
        reason = "Target was already executed in reduced or expanded bounded batches."
    elif target_id in CHECKPOINT_REJECTED_TARGET_IDS:
        classification = "wrong_system"
        reason = "Target was previously checkpoint-rejected."
    elif false_positive == "wrong_event_or_attribute":
        classification = "wrong_event"
        reason = "False-positive audit found the value belongs to the wrong event or attribute."
    elif false_positive == "component_only":
        classification = "component_only"
        reason = "False-positive audit found component evidence without the requested value."
    elif false_positive == "wrong_system":
        classification = "wrong_system"
        reason = "False-positive audit found the evidence belongs to the wrong system."
    elif not candidate.get("cache_file_exists"):
        classification = "missing_cached_evidence"
        reason = "Evidence page is not present in the existing parsed-page cache."
    elif candidate.get("audit_classification") == "dictionary_clarification":
        classification = "dictionary_clarification"
        reason = "Existing readiness audit requires dictionary clarification."
    elif candidate.get("audit_classification") != "execution_ready":
        classification = "insufficient_attribute_evidence"
        reason = str(candidate.get("audit_rationale") or "Requested value not supported.")
    elif not str(candidate.get("value_bearing_text") or "").strip():
        classification = "insufficient_attribute_evidence"
        reason = "No value-bearing evidence text is available in existing artifacts."
    return {
        **candidate,
        "classification": classification,
        "classification_reason": reason,
        "target_name": candidate.get("field_name"),
    }


def selected_records(
    reviewed: list[dict[str, Any]],
    *,
    preferred_max: int,
) -> list[dict[str, Any]]:
    ready = [item for item in reviewed if item["classification"] == "execution_ready"]
    rows = []
    for rank, item in enumerate(ready[:preferred_max], start=1):
        rows.append(selected_record(item, rank=rank))
    return rows


def selected_record(item: dict[str, Any], *, rank: int) -> dict[str, Any]:
    target = target_specification(item)
    text = str(item.get("value_bearing_text") or "")
    span_id = f"span_{item['target_id']}_scale_v2"
    return {
        "selection_rank": rank,
        "target_id": item["target_id"],
        "target_name": item["field_name"],
        "requirement_id": target["requirement_id"],
        "field_name": item["field_name"],
        "target": target,
        "value_shape": item.get("value_shape"),
        "datatype": item.get("datatype"),
        "domain": item.get("domain"),
        "classification": "execution_ready",
        "source_id": item.get("source_id"),
        "source_filename": item.get("source_filename"),
        "page_number": item.get("page_number"),
        "cache_path": item.get("cache_path"),
        "evidence_bundle": {
            "bundle_id": f"bundle_{item['target_id']}",
            "component_context": {
                "component_system_identity": component_identity(item),
                "evidence_family": item.get("readiness_class"),
                "requested_attribute": item.get("field_name"),
            },
            "provenance": [
                {
                    "source_id": item.get("source_id"),
                    "page_or_sheet": str(item.get("page_number")),
                    "text_ref": span_id,
                    "notes": item.get("audit_rationale"),
                }
            ],
            "retrieval_strategy": "existing_readiness_audit_cached_evidence",
            "target_specification": target,
        },
        "canonical_evidence_payload": {
            "payload_id": f"payload_{item['target_id']}",
            "target_id": item["target_id"],
            "route": item.get("expected_extraction_route"),
            "evidence_family": item.get("readiness_class"),
            "page_range": f"{item.get('page_number')}-{item.get('page_number')}",
            "bounded_text": text,
            "span": {
                "span_id": span_id,
                "source_id": item.get("source_id"),
                "source_file": item.get("source_filename"),
                "page_number": item.get("page_number"),
                "start_char": 0,
                "end_char": len(text),
                "text": text,
                "score": 1.0,
                "retrieval_rank": rank,
            },
        },
        "normalization_expectations": {
            "expected_data_type": item.get("datatype"),
            "accepted_values": [],
            "numeric_unit": item.get("unit"),
        },
    }


def target_specification(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_row_id": item["target_id"],
        "requirement_id": item.get("definition", "").split(" - ")[0] or item["target_id"],
        "requirement_text": item.get("definition") or item.get("field_name"),
        "expected_field": item.get("field_name"),
        "expected_data_type": item.get("datatype"),
        "unit": item.get("unit"),
        "accepted_values": [],
        "cardinality": "single",
        "component_type": None,
        "component_subtype": None,
        "source_guidance": item.get("source_guidance"),
        "likely_evidence_types": ["text"],
        "metadata": {
            "field_label": item.get("field_name"),
            "requirement_name": item.get("definition"),
        },
        "source_dictionary_provenance": {
            "dictionary_id": "segro_extraction_template",
            "row_number": item.get("dictionary_row"),
        },
        "sub_domain": item.get("domain"),
    }


def excluded_candidates(
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
            classification = "missing_cached_evidence"
        rows.append(
            {
                "target_id": item["target_id"],
                "target_name": item.get("field_name"),
                "classification": classification,
                "reason": item.get("classification_reason"),
                "source_file": item.get("source_filename"),
                "page_number": item.get("page_number"),
            }
        )
    return rows


def selection_summary(
    candidates: list[dict[str, Any]],
    reviewed: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    *,
    preferred_min: int,
    preferred_max: int,
) -> dict[str, Any]:
    return {
        "schema_version": "segro_evidence_ready_batch_expansion_v2_selection_summary",
        "candidate_inventory_count": len(candidates),
        "reviewed_target_count": len(reviewed),
        "selected_count": len(selected),
        "preferred_min": preferred_min,
        "preferred_max": preferred_max,
        "selected_target_ids": [item["target_id"] for item in selected],
        "excluded_counts_by_reason": dict(Counter(item["classification"] for item in excluded)),
        "selection_note": (
            "No targets were selected because all remaining confirmed-ready candidates were "
            "excluded by prior false-positive evidence review or other strict readiness rules."
            if not selected
            else "Selected only targets supported by existing cached evidence."
        ),
    }


def coverage_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "segro_evidence_ready_batch_expansion_v2_coverage_summary",
        "selected_count": len(selected),
        "source_count": len({item["source_id"] for item in selected}),
        "page_count": len({(item["source_id"], item["page_number"]) for item in selected}),
        "domains": sorted({str(item["domain"]) for item in selected}),
        "value_shapes": sorted({str(item["value_shape"]) for item in selected}),
    }


def false_positive_classification(target_id: str) -> str | None:
    for path in [
        FALSE_POSITIVE_AUDIT_DIR / "false_positive_trace.json",
        FALSE_POSITIVE_AUDIT_DIR / "target_false_positive_audit.json",
    ]:
        for row in read_optional_json_list(path):
            if str(row.get("target_id")) == target_id:
                return str(
                    row.get("false_positive_classification")
                    or row.get("audit_classification")
                    or ""
                )
    return None


def source_registry_by_file(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    for source in load_source_registry(path):
        rows[source.logical_path] = {
            "source_id": source.source_id,
            "source_path": source.original_path,
        }
    return rows


def cache_path_for(source_id: str | None, page_number: int) -> Path | None:
    if not source_id:
        return None
    matches = sorted((PAGE_CACHE_ROOT / source_id).glob(f"*/page_{page_number:06d}.json"))
    return matches[0] if matches else None


def component_identity(item: dict[str, Any]) -> str:
    terms = item.get("evidence_terms", {}).get("component_terms", [])
    return "_".join(str(term).casefold().replace(" ", "_") for term in terms) or str(
        item.get("domain") or "unknown"
    )


def input_provenance() -> dict[str, Any]:
    return {
        "schema_version": "segro_evidence_ready_batch_expansion_v2_input_provenance",
        "inputs": [
            str(READINESS_AUDIT_DIR),
            str(FALSE_POSITIVE_AUDIT_DIR),
            str(CACHE_EXPANSION_DIR),
            str(SOURCE_MANIFEST),
        ],
        "retrieval_repeated": False,
        "parsing_repeated": False,
        "ocr_repeated": False,
        "vlm_repeated": False,
        "cache_expansion_repeated": False,
        "source_coverage_repeated": False,
        "target_reselection_repeated": False,
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in result.items():
        _atomic_write_json(output_dir / f"{key}.json", value)
    write_jsonl(output_dir / "selected_batch.jsonl", result["selected_batch"])
    write_markdown(output_dir / "dry_run_validation.md", result["dry_run_validation"])


def write_scale_trial_config(config_path: Path, selected_batch_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        config_path,
        {
            "schema_version": "segro_unit_runner_v1",
            "config_version": "1.0.0",
            "unit_id": "enfield_unit1_scale_trial_v2",
            "asset_record_key": "enfield_unit1",
            "repository_root": ".",
            "selected_batch_path": str(selected_batch_path),
            "source_registry_path": str(SOURCE_MANIFEST),
            "cache_root": str(PAGE_CACHE_ROOT),
            "dictionary_artifact_path": (
                "output/enfield_unit1_extraction_batch_construction_v1/"
                "normalization_validation_plan.json"
            ),
            "output_root": "output/unit_runs",
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "execution_mode": "dry_run",
            "customer_caveat_policy": "internal_only",
            "max_model_retries": 1,
            "expected_branch": "feature/evidence-first-foundation",
            "frozen_internal_contract_version": "1.0.0",
            "frozen_customer_contract_version": "1.0.0",
        },
    )


def read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [dict(item) for item in data]


def read_optional_json_list(path: Path) -> list[dict[str, Any]]:
    return read_json_list(path) if path.exists() else []
