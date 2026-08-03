"""Adjudicate and deterministically repair the reduced bounded extraction run."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.reduced_bounded_extraction_batch_v1 import (
    DEFAULT_REDUCED_BATCH_OUTPUT_DIR,
    dictionary_valid,
    event_valid,
    evidence_bundle_record_from_request,
    final_decision,
    normalize_space,
    typed_value_valid,
)
from segro_evidence_extraction.vertical_slice import (
    ExtractionResult,
    _atomic_write_json,
    calibrate_extraction_value,
    infer_value_shape_assignment,
    materialize_selected_spans,
    parse_extraction_response,
)

DEFAULT_REDUCED_ADJUDICATION_OUTPUT_DIR = Path(
    "output/enfield_unit1_reduced_extraction_adjudication_v1"
)

REVIEW_TARGET_IDS = [
    "trg_33975e86ffde5524",
    "trg_7426d9722059de76",
    "trg_12690fb418279750",
    "trg_93d2e2b38d1de7b2",
    "trg_adb3dfa03b0c7cf3",
]

SHAPE_REPAIRS = {
    "unordered_or_unordered_list": "ordered_or_unordered_list",
}


def run_reduced_extraction_adjudication_v1(
    *,
    reduced_run_dir: Path = DEFAULT_REDUCED_BATCH_OUTPUT_DIR,
    output_dir: Path = DEFAULT_REDUCED_ADJUDICATION_OUTPUT_DIR,
) -> dict[str, Any]:
    inputs = load_adjudication_inputs(reduced_run_dir)
    reviewed = review_targets(inputs)
    final = build_adjudicated_final(inputs, reviewed)
    repaired = [item for item in reviewed if item["deterministic_repair_applied"]]
    summary = {
        "reviewed_target_ids": REVIEW_TARGET_IDS,
        "new_model_calls": 0,
        "deterministic_repairs": len(repaired),
        "decision_counts": dict(Counter(item["adjudicated_decision"] for item in final)),
        "abstentions_confirmed": [
            item["target_id"]
            for item in reviewed
            if item.get("substantive_outcome") == "confirmed_abstention"
        ],
    }
    result = {
        "adjudication_summary": summary,
        "target_review": reviewed,
        "repaired_results": repaired,
        "final_adjudication": final,
        "repair_audit": build_repair_audit(reviewed),
        "model_retry_requests": [],
        "model_retry_responses": [],
    }
    write_adjudication_outputs(result, output_dir)
    return result


def load_adjudication_inputs(reduced_run_dir: Path) -> dict[str, Any]:
    return {
        "reduced_batch": read_json_list(reduced_run_dir / "reduced_batch.json"),
        "original_final": read_json_list(reduced_run_dir / "final_adjudication.json"),
        "validated": read_json_list(reduced_run_dir / "extraction_results_validated.json"),
        "raw_responses": read_jsonl(reduced_run_dir / "raw_model_responses.jsonl"),
    }


def review_targets(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    batch_by_id = {item["target_id"]: item for item in inputs["reduced_batch"]}
    final_by_id = {item["target_id"]: item for item in inputs["original_final"]}
    validated_by_id = {item["target_row_id"]: item for item in inputs["validated"]}
    raw_by_id = {item["target_id"]: item for item in inputs["raw_responses"]}
    rows = []
    for target_id in REVIEW_TARGET_IDS:
        batch_item = batch_by_id[target_id]
        original = final_by_id[target_id]
        validated = validated_by_id[target_id]
        raw = raw_by_id[target_id]
        if original["final_decision"] == "abstained":
            rows.append(review_abstention(batch_item, original, validated))
        elif validated.get("evidence_containment") == "failed":
            rows.append(review_containment_repair(batch_item, original, validated, raw))
        elif response_has_repairable_schema(raw):
            rows.append(review_schema_repair(batch_item, original, validated, raw))
    return rows


def review_abstention(
    batch_item: dict[str, Any],
    original: dict[str, Any],
    validated: dict[str, Any],
) -> dict[str, Any]:
    audit = audit_abstention(batch_item)
    return {
        "target_id": original["target_id"],
        "target_name": original["target_name"],
        "original_decision": original["final_decision"],
        "adjudicated_decision": "abstained",
        "substantive_outcome": (
            "confirmed_abstention" if audit["confirmed"] else "unresolved_abstention"
        ),
        "deterministic_repair_applied": False,
        "model_retry_occurred": False,
        "decision_change_reason": None,
        "audit_reason": audit["reason"],
        "evidence_containment": validated["evidence_containment"],
        "event_validation": validated["event_validation"],
        "dictionary_validation": validated["dictionary_validation"],
        "typed_normalization": validated["typed_normalization"],
        "repaired_result": None,
    }


def audit_abstention(batch_item: dict[str, Any]) -> dict[str, Any]:
    target = batch_item["target"]
    expected_field = str(target.get("expected_field") or "").casefold()
    requirement = str(target.get("requirement_text") or "").casefold()
    evidence = str(batch_item["canonical_evidence_payload"]["bounded_text"])
    evidence_key = evidence_containment_key(evidence)
    accepted_values = [str(value) for value in target.get("accepted_values", [])]

    if accepted_values and not accepted_value_supported(accepted_values, evidence_key):
        return {
            "confirmed": True,
            "reason": (
                "Approved evidence does not contain a supported dictionary value for the "
                f"requested field. Accepted values: {', '.join(accepted_values)}."
            ),
        }
    if "manufacturer" in expected_field and not re.search(
        r"\b(manufacturer|manufactured by|make|made by|supplier)\b",
        evidence_key,
    ):
        return {
            "confirmed": True,
            "reason": (
                "Approved evidence names a system or party but does not label that name as the "
                "requested manufacturer value."
            ),
        }
    if "warehouse" in requirement and "except warehouse" in evidence_key:
        return {
            "confirmed": True,
            "reason": (
                "Approved evidence explicitly excludes Warehouse from the commissioned areas, "
                "so it does not clearly support the requested warehouse attribute."
            ),
        }
    return {
        "confirmed": False,
        "reason": "Approved evidence was insufficiently specific to overturn the abstention.",
    }


def accepted_value_supported(accepted_values: list[str], evidence_key: str) -> bool:
    for value in accepted_values:
        for candidate in re.split(r"\. maps from |,|/", value, flags=re.IGNORECASE):
            candidate_key = evidence_containment_key(candidate)
            if candidate_key and candidate_key in evidence_key:
                return True
    return False


def review_containment_repair(
    batch_item: dict[str, Any],
    original: dict[str, Any],
    validated: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    quote = raw_response_json(raw).get("value_bearing_quote")
    evidence_text = str(batch_item["canonical_evidence_payload"]["bounded_text"])
    safe = normalized_evidence_contains(evidence_text, str(quote or ""))
    repaired = dict(validated)
    issues = [
        issue
        for issue in repaired.get("validation_issues", [])
        if issue != "cited evidence quote/span is not contained in approved evidence bundle"
    ]
    if safe:
        repaired["evidence_containment"] = "passed"
        repaired["validation_issues"] = issues
    decision = "accepted" if safe and not issues else "rejected"
    return {
        "target_id": original["target_id"],
        "target_name": original["target_name"],
        "original_decision": original["final_decision"],
        "adjudicated_decision": decision,
        "substantive_outcome": "mechanical_containment_repaired" if safe else "rejected",
        "deterministic_repair_applied": safe,
        "model_retry_occurred": False,
        "decision_change_reason": (
            "Normalized containment accepted PDF text-layout artifact: '5 No' vs '5No'."
            if safe
            else None
        ),
        "raw_provider_quote": quote,
        "approved_evidence_fragment": "5No Dock Levellers",
        "evidence_containment": repaired["evidence_containment"],
        "event_validation": repaired["event_validation"],
        "dictionary_validation": repaired["dictionary_validation"],
        "typed_normalization": repaired["typed_normalization"],
        "repaired_result": repaired,
    }


def review_schema_repair(
    batch_item: dict[str, Any],
    original: dict[str, Any],
    validated: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    repaired_payload, repair_reason = repair_response_payload(raw_response_json(raw))
    target = TargetSpecification.model_validate(batch_item["target"])
    assignment = infer_value_shape_assignment(target)
    bundle = evidence_bundle_record_from_request(batch_item["constructed_request"])
    parsed = parse_extraction_response(
        target=target,
        bundle=bundle,
        provider="openai",
        model_name=str(raw["provider_response"].get("model") or "gpt-4o-mini"),
        response_text=json.dumps(repaired_payload, ensure_ascii=True, sort_keys=True),
    )
    parsed = materialize_selected_spans(parsed, bundle)
    parsed = calibrate_extraction_value(
        extraction=parsed,
        target=target,
        assignment=assignment,
        bundle=bundle,
    )
    validation = validate_repaired_extraction(parsed, batch_item, target, assignment)
    decision = final_decision(batch_item, parsed, validation)
    repaired = {
        **parsed.model_dump(mode="json"),
        "evidence_containment": validation["evidence_containment"],
        "typed_normalization": validation["typed_normalization"],
        "event_validation": validation["event_validation"],
        "dictionary_validation": validation["dictionary_validation"],
        "validation_issues": validation["issues"],
    }
    return {
        "target_id": original["target_id"],
        "target_name": original["target_name"],
        "original_decision": original["final_decision"],
        "adjudicated_decision": decision,
        "substantive_outcome": "schema_repaired_then_revalidated",
        "deterministic_repair_applied": True,
        "model_retry_occurred": False,
        "decision_change_reason": repair_reason,
        "original_schema_error": original.get("model_caveat"),
        "evidence_containment": repaired["evidence_containment"],
        "event_validation": repaired["event_validation"],
        "dictionary_validation": repaired["dictionary_validation"],
        "typed_normalization": repaired["typed_normalization"],
        "repaired_result": repaired,
    }


def validate_repaired_extraction(
    extraction: ExtractionResult,
    batch_item: dict[str, Any],
    target: TargetSpecification,
    assignment: Any,
) -> dict[str, Any]:
    issues: list[str] = []
    evidence_text = str(batch_item["canonical_evidence_payload"].get("bounded_text") or "")
    span = batch_item["canonical_evidence_payload"].get("span", {})
    quote = str(extraction.value_bearing_quote or extraction.supporting_evidence_excerpt or "")
    containment = extraction.status != "extracted" or (
        bool(quote) and normalized_evidence_contains(evidence_text, quote)
    )
    if not containment:
        issues.append("cited evidence quote/span is not contained in approved evidence bundle")
    typed = typed_value_valid(extraction, assignment)
    if not typed:
        issues.append("extracted value failed typed normalization")
    event = event_valid(extraction, batch_item)
    if not event:
        issues.append("extracted value belongs to the wrong event context")
    dictionary = dictionary_valid(extraction, target)
    if not dictionary:
        issues.append("extracted value is incompatible with dictionary metadata")
    source_page = extraction.status != "extracted" or (
        extraction.source_id == span.get("source_id")
        and extraction.page_number == span.get("page_number")
    )
    if not source_page:
        issues.append("extracted source/page is outside approved evidence bundle")
    return {
        "evidence_containment": "passed" if containment and source_page else "failed",
        "typed_normalization": "passed" if typed else "failed",
        "event_validation": "passed" if event else "failed",
        "dictionary_validation": "passed" if dictionary else "failed",
        "issues": issues,
    }


def repair_response_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    repaired = dict(payload)
    shape = repaired.get("proposed_value_shape")
    if shape in SHAPE_REPAIRS:
        repaired["proposed_value_shape"] = SHAPE_REPAIRS[str(shape)]
        return (
            repaired,
            f"Repaired proposed_value_shape literal {shape!r} to "
            f"{repaired['proposed_value_shape']!r}.",
        )
    return repaired, "No deterministic schema repair was available."


def response_has_repairable_schema(raw: dict[str, Any]) -> bool:
    return raw_response_json(raw).get("proposed_value_shape") in SHAPE_REPAIRS


def normalized_evidence_contains(evidence_text: str, quote: str) -> bool:
    if not quote.strip():
        return False
    evidence = evidence_containment_key(evidence_text)
    needle = evidence_containment_key(quote)
    return needle in evidence


def evidence_containment_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\u2022", " ")
    text = re.sub(r"[`\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(\d+)\s+No\b", r"\1No", text, flags=re.IGNORECASE)
    return normalize_space(text).casefold()


def raw_response_json(raw: dict[str, Any]) -> dict[str, Any]:
    content = raw["provider_response"]["choices"][0]["message"]["content"]
    payload = json.loads(str(content))
    if not isinstance(payload, dict):
        raise ValueError("Raw provider response content is not a JSON object.")
    return dict(payload)


def build_adjudicated_final(
    inputs: dict[str, Any],
    reviewed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    review_by_id = {item["target_id"]: item for item in reviewed}
    final = []
    for original in inputs["original_final"]:
        review = review_by_id.get(original["target_id"])
        if review is None:
            final.append(
                {
                    **original,
                    "original_decision": original["final_decision"],
                    "adjudicated_decision": original["final_decision"],
                    "deterministic_repair_applied": False,
                    "model_retry_occurred": False,
                    "decision_change_reason": None,
                }
            )
            continue
        repaired = review.get("repaired_result") or {}
        final.append(
            {
                **original,
                **final_value_updates(repaired),
                "original_decision": original["final_decision"],
                "adjudicated_decision": review["adjudicated_decision"],
                "final_decision": review["adjudicated_decision"],
                "deterministic_repair_applied": review["deterministic_repair_applied"],
                "model_retry_occurred": review["model_retry_occurred"],
                "decision_change_reason": review["decision_change_reason"],
                "evidence_containment": review["evidence_containment"],
                "event_validation": review["event_validation"],
                "dictionary_validation": review["dictionary_validation"],
                "typed_normalization": review["typed_normalization"],
            }
        )
    return final


def final_value_updates(repaired: dict[str, Any]) -> dict[str, Any]:
    if not repaired:
        return {}
    return {
        "raw_model_value": repaired.get("raw_model_value"),
        "evidence_value": repaired.get("evidence_value"),
        "normalized_value": repaired.get("normalized_value"),
        "display_value": repaired.get("display_value"),
        "source_id": repaired.get("source_id"),
        "source_file": repaired.get("source_file"),
        "page_number": repaired.get("page_number"),
        "confidence": repaired.get("confidence"),
        "model_caveat": repaired.get("ambiguity_or_caveat"),
        "validation_issues": repaired.get("validation_issues", []),
    }


def build_repair_audit(reviewed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "target_id": item["target_id"],
            "original_decision": item["original_decision"],
            "adjudicated_decision": item["adjudicated_decision"],
            "deterministic_repair_applied": item["deterministic_repair_applied"],
            "model_retry_occurred": item["model_retry_occurred"],
            "decision_change_reason": item["decision_change_reason"],
            "evidence_containment": item["evidence_containment"],
            "event_validation": item["event_validation"],
            "dictionary_validation": item["dictionary_validation"],
        }
        for item in reviewed
    ]


def write_adjudication_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "adjudication_summary.json",
        "target_review.json",
        "repaired_results.json",
        "final_adjudication.json",
        "repair_audit.json",
    ]:
        _atomic_write_json(output_dir / filename, result[filename.removesuffix(".json")])
    write_final_csv(result["final_adjudication"], output_dir / "final_adjudication.csv")
    write_jsonl(output_dir / "model_retry_requests.jsonl", result["model_retry_requests"])
    write_jsonl(output_dir / "model_retry_responses.jsonl", result["model_retry_responses"])


def write_final_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "target_id",
        "target_name",
        "original_decision",
        "adjudicated_decision",
        "checkpoint_status",
        "checkpoint_caveat",
        "display_value",
        "evidence_value",
        "normalized_value",
        "deterministic_repair_applied",
        "model_retry_occurred",
        "decision_change_reason",
        "evidence_containment",
        "event_validation",
        "dictionary_validation",
        "validation_issues",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")
    return [dict(item) for item in data]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)
