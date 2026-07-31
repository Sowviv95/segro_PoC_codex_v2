"""False-positive audit for Batch V2 evidence-ready targets."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from segro_evidence_extraction.batch_v2_evidence_readiness import (
    DEFAULT_READINESS_AUDIT_OUTPUT_DIR,
)
from segro_evidence_extraction.vertical_slice import _atomic_write_json

DEFAULT_FALSE_POSITIVE_OUTPUT_DIR = Path(
    "output/enfield_unit1_evidence_first_batch_v2_false_positive_audit"
)
EXPECTED_BATCH_V2_TARGET_COUNT = 75

FalsePositiveClass = Literal[
    "confirmed_execution_ready",
    "wrong_event_or_attribute",
    "component_only",
    "dictionary_ambiguous",
    "insufficient_cached_evidence",
]

CONFIRMED: FalsePositiveClass = "confirmed_execution_ready"


def run_batch_v2_false_positive_audit(
    *,
    readiness_dir: Path = DEFAULT_READINESS_AUDIT_OUTPUT_DIR,
    output_dir: Path = DEFAULT_FALSE_POSITIVE_OUTPUT_DIR,
    target_count: int = EXPECTED_BATCH_V2_TARGET_COUNT,
) -> dict[str, Any]:
    rows = load_false_positive_inputs(readiness_dir)
    result = build_false_positive_audit(rows, target_count=target_count)
    write_false_positive_outputs(result, output_dir)
    return result


def load_false_positive_inputs(readiness_dir: Path) -> list[dict[str, Any]]:
    path = readiness_dir / "corrected_selected_targets.json"
    if not path.exists():
        msg = f"Missing readiness input: {path}"
        raise FileNotFoundError(msg)
    rows = _as_list(_read_json(path), "corrected selected targets")
    return [row for row in rows if row.get("audit_classification") == "execution_ready"]


def build_false_positive_audit(
    rows: list[dict[str, Any]], *, target_count: int
) -> dict[str, Any]:
    duplicate_ids = {
        target_id
        for target_id, count in Counter(str(row["target_id"]) for row in rows).items()
        if count > 1
    }
    audited = [
        audit_false_positive(row, duplicate_target_id=str(row["target_id"]) in duplicate_ids)
        for row in rows
    ]
    confirmed = [row for row in audited if row["false_positive_classification"] == CONFIRMED]
    rejected = [row for row in audited if row["false_positive_classification"] != CONFIRMED]
    status = final_status(len(confirmed), target_count)
    metrics = {
        "audited_execution_ready_count": len(audited),
        "classification_counts": dict(
            Counter(str(row["false_positive_classification"]) for row in audited)
        ),
        "confirmed_count": len(confirmed),
        "rejected_count": len(rejected),
        "target_count_required": target_count,
        "additional_confirmed_targets_needed": max(0, target_count - len(confirmed)),
        "final_readiness_status": status,
        "extraction_approved": status != "blocked",
        "part1_count": sum(
            1
            for row in audited
            if row.get("audit_source_file") == "Building Manual - Part 1 General.pdf"
        ),
        "part1_classification_counts": dict(
            Counter(
                str(row["false_positive_classification"])
                for row in audited
                if row.get("audit_source_file") == "Building Manual - Part 1 General.pdf"
            )
        ),
        "date_target_findings": date_findings(audited),
    }
    return {
        "target_false_positive_audit": audited,
        "confirmed_execution_ready_targets": confirmed,
        "rejected_ready_targets": rejected,
        "wrong_event_or_attribute": by_class(audited, "wrong_event_or_attribute"),
        "component_only_targets": by_class(audited, "component_only"),
        "dictionary_ambiguous_targets": by_class(audited, "dictionary_ambiguous"),
        "insufficient_cached_evidence": by_class(audited, "insufficient_cached_evidence"),
        "false_positive_metrics": metrics,
        "false_positive_trace": build_trace(audited),
        "confirmed_batch_v2_source_plan": build_source_plan(confirmed),
        "confirmed_batch_v2_run_readiness": {
            "overall_status": status,
            "target_count_required": target_count,
            "confirmed_count": len(confirmed),
            "additional_confirmed_targets_needed": max(0, target_count - len(confirmed)),
            "extraction_approved": status != "blocked",
            "blockers": []
            if status != "blocked"
            else ["Fewer than 75 targets survived false-positive audit."],
        },
        "batch_v2_false_positive_review": build_review(audited),
        "batch_v2_cache_expansion_candidates": [
            cache_expansion_candidate(row) for row in rejected
        ],
    }


def audit_false_positive(
    row: dict[str, Any], *, duplicate_target_id: bool = False
) -> dict[str, Any]:
    field = str(row.get("field_name") or "")
    definition = str(row.get("definition") or "")
    value_shape = str(row.get("value_shape") or "")
    datatype = str(row.get("datatype") or "")
    excerpt = normalize_excerpt(str(row.get("best_cached_excerpt") or ""))
    lower = excerpt.lower()
    component_terms = component_terms_for(field)
    attribute_terms = attribute_terms_for(field, definition, value_shape)
    component_partial = any(term_match(term, lower) for term in component_terms)
    component_present = all(term_match(term, lower) for term in component_terms)
    attribute_present = any(term_match(term, lower) for term in attribute_terms)
    value_signal = value_signal_present(value_shape, datatype, field, lower, excerpt)
    classification, rationale = classify_false_positive(
        field=field,
        definition=definition,
        value_shape=value_shape,
        datatype=datatype,
        excerpt=lower,
        component_present=component_present,
        component_partial=component_partial,
        attribute_present=attribute_present,
        value_signal=value_signal,
        duplicate_target_id=duplicate_target_id,
    )
    return {
        **row,
        "false_positive_classification": classification,
        "false_positive_rationale": rationale,
        "requested_component_terms": component_terms,
        "requested_attribute_terms": attribute_terms,
        "fp_component_present": component_present,
        "fp_component_partial": component_partial,
        "fp_attribute_present": attribute_present,
        "fp_value_signal_present": value_signal,
        "fp_local_association": classification == CONFIRMED,
    }


def classify_false_positive(
    *,
    field: str,
    definition: str,
    value_shape: str,
    datatype: str,
    excerpt: str,
    component_present: bool,
    component_partial: bool,
    attribute_present: bool,
    value_signal: bool,
    duplicate_target_id: bool,
) -> tuple[FalsePositiveClass, str]:
    text = f"{field} {definition}".lower()
    if duplicate_target_id:
        return "dictionary_ambiguous", "Duplicate retained target ID prevents unique execution row."
    if dictionary_ambiguous(field, definition, datatype, value_shape):
        return (
            "dictionary_ambiguous",
            "Dictionary field semantics are ambiguous for this value shape.",
        )
    if value_shape == "date":
        if not value_signal:
            return "insufficient_cached_evidence", "No date value is present in the cached excerpt."
        if wrong_date_context(field, excerpt):
            return "wrong_event_or_attribute", "Date is tied to a different document/event."
        if attribute_present:
            return CONFIRMED, "Date is locally tied to the requested event."
        return "wrong_event_or_attribute", "Date lacks the requested event label."
    if "model_number" in field or "reference" in field:
        if drawing_or_generic_reference(excerpt):
            return "wrong_event_or_attribute", "Identifier is a drawing/generic reference."
        if component_present and attribute_present and value_signal:
            return CONFIRMED, "Identifier has component, label and plausible local value."
        return "wrong_event_or_attribute", "Identifier is not labelled for the requested field."
    if "manufacturer" in field:
        if component_present and attribute_present and value_signal:
            return CONFIRMED, "Manufacturer has component and manufacturer/make/supplier label."
        if component_present or component_partial:
            return "component_only", "Component appears without a manufacturer label/value."
        return "wrong_event_or_attribute", "Evidence does not identify a manufacturer."
    if value_shape in {"integer_count", "decimal_measurement"}:
        if component_present and attribute_present and value_signal:
            return CONFIRMED, "Numeric value is locally tied to requested component/attribute."
        if component_present or component_partial:
            return "component_only", "Component appears without the requested numeric attribute."
        return "wrong_event_or_attribute", "Numeric signal is not tied to the requested attribute."
    if value_shape in {"categorical", "descriptive_text", "short_text"}:
        if generic_or_compliance_excerpt(excerpt):
            return "wrong_event_or_attribute", "Excerpt is generic document/compliance wording."
        if component_present and attribute_present and value_signal:
            return CONFIRMED, "Component-specific categorical/description signal is local."
        if component_present or component_partial:
            return "component_only", "Component appears without requested characteristic."
        if any(term in text for term in ["status", "feature", "unit", "space"]):
            return "insufficient_cached_evidence", "Cached excerpt is too broad for this field."
        return "wrong_event_or_attribute", "Evidence does not answer requested descriptive field."
    return "insufficient_cached_evidence", "No exact requested-attribute support was found."


def component_terms_for(field: str) -> list[str]:
    stop = {
        "model",
        "number",
        "reference",
        "manufacturer",
        "name",
        "date",
        "count",
        "value",
        "unit",
        "description",
        "type",
        "installation",
        "operation",
        "mode",
        "component",
    }
    return [token for token in field.lower().split("_") if token and token not in stop][:5]


def attribute_terms_for(field: str, definition: str, value_shape: str) -> list[str]:
    text = f"{field} {definition}".lower()
    terms: list[str] = []
    if "construction_date" in field:
        terms.extend(["build year", "construction date", "completion"])
    elif "installation_date" in field:
        terms.extend(["installation date", "installed", "commissioned"])
    elif "model_number" in field:
        terms.extend(["model", "model number", "model no", "type"])
    elif "manufacturer" in field:
        terms.extend(["manufacturer", "supplier"])
    elif "count" in field or value_shape == "integer_count":
        terms.extend(["count", "number of", "no.", "qty", "quantity"])
    elif value_shape == "decimal_measurement":
        terms.extend(["area", "rating", "capacity", "height", "width", "depth"])
    elif value_shape in {"categorical", "descriptive_text", "short_text"}:
        terms.extend(["type", "description", "status", "name", "rating", "material"])
    if "certificate" in text:
        terms.append("certificate")
    return sorted(set(terms))


def value_signal_present(
    value_shape: str, datatype: str, field: str, lower: str, excerpt: str
) -> bool:
    if value_shape == "date":
        return bool(re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", excerpt))
    if "model_number" in field or "reference" in field:
        return bool(re.search(r"\b[A-Z]{1,6}[-/]?[A-Z0-9]{2,}(?:[-/][A-Z0-9]{2,})*\b", excerpt))
    if "manufacturer" in field:
        return any(term_match(term, lower) for term in ["manufacturer", "supplier"])
    if value_shape == "integer_count":
        return bool(re.search(r"\b\d+\s?(?:no\.?|nr|qty)\b", lower))
    if value_shape == "decimal_measurement":
        return bool(re.search(r"\b\d+(?:\.\d+)?\s?(?:m2|m²|m|kw|kva|sqm|%)\b", lower))
    if value_shape in {"categorical", "descriptive_text", "short_text"}:
        return not generic_or_compliance_excerpt(lower) and len(lower.split()) >= 5
    return bool(datatype)


def wrong_date_context(field: str, excerpt: str) -> bool:
    if "construction_date" in field:
        return any(term in excerpt for term in ["methodology", "waste management", "ref:", "issue"])
    if "installation_date" in field:
        return any(
            term in excerpt
            for term in ["approved inspectors date", "planning permission", "my ref", "dated"]
        )
    return False


def drawing_or_generic_reference(excerpt: str) -> bool:
    return any(
        term in excerpt
        for term in ["drawing no", "planning permission ref", "job reference", "my ref:"]
    )


def generic_or_compliance_excerpt(excerpt: str) -> bool:
    return any(
        term in excerpt
        for term in [
            "health and safety file",
            "does not accept any liability",
            "construction work",
            "planning permission",
            "local planning authority",
            "approved inspectors",
            "fuse links",
            "replacement fuse",
            "building manual",
        ]
    )


def dictionary_ambiguous(field: str, definition: str, datatype: str, value_shape: str) -> bool:
    text = f"{field} {definition}".lower()
    if "thermal_value" in field and datatype == "decimal" and value_shape == "categorical":
        return True
    if field.endswith("_count") and value_shape == "categorical":
        return True
    return any(term in text for term in ["unclear", "tbc", "maps from"])


def term_match(term: str, excerpt: str) -> bool:
    term = term.lower().strip()
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9]+", term):
        return bool(re.search(rf"\b{re.escape(term)}s?\b", excerpt))
    return term in excerpt


def by_class(rows: list[dict[str, Any]], classification: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["false_positive_classification"] == classification]


def date_findings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "target_id": row["target_id"],
            "field_name": row["field_name"],
            "classification": row["false_positive_classification"],
            "rationale": row["false_positive_rationale"],
            "source": row["audit_source_file"],
            "page": row["best_cached_page"],
            "excerpt": row["best_cached_excerpt"],
        }
        for row in rows
        if row.get("value_shape") == "date" or str(row.get("field_name", "")).endswith("_date")
    ]


def cache_expansion_candidate(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("audit_source_file") or row.get("likely_source_file")
    page = row.get("best_cached_page")
    priority = (
        "high"
        if row["false_positive_classification"] == "insufficient_cached_evidence"
        else "medium"
    )
    remain_deferred = row["false_positive_classification"] == "dictionary_ambiguous"
    return {
        "target_id": row["target_id"],
        "field_name": row["field_name"],
        "likely_source": source,
        "likely_hierarchy_section": row.get("likely_hierarchy_section"),
        "recommended_page_range": page_range(page),
        "reason_additional_pages_may_help": cache_expansion_reason(row),
        "priority": "low" if remain_deferred else priority,
        "remain_deferred_even_after_expansion": remain_deferred,
    }


def cache_expansion_reason(row: dict[str, Any]) -> str:
    classification = row["false_positive_classification"]
    if classification == "dictionary_ambiguous":
        return "Dictionary clarification is required before cache expansion is useful."
    if classification == "component_only":
        return "Adjacent or more specific pages may contain the missing requested attribute."
    if classification == "wrong_event_or_attribute":
        return (
            "Current cached excerpt answers a different attribute; nearby source pages "
            "may not help."
        )
    return "Current cached pages do not contain enough local requested-attribute evidence."


def page_range(page: Any) -> str:
    if not isinstance(page, int):
        return "unknown existing source section"
    start = max(1, page - 2)
    end = page + 2
    return f"{start}-{end}"


def build_source_plan(confirmed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "selection_rank": index,
            "target_id": row["target_id"],
            "field_name": row["field_name"],
            "source_document": row["audit_source_file"],
            "page": row["best_cached_page"],
            "evidence_excerpt": row["best_cached_excerpt"],
            "classification": row["false_positive_classification"],
        }
        for index, row in enumerate(confirmed, start=1)
    ]


def build_trace(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "target_id": row["target_id"],
            "field_name": row["field_name"],
            "input_readiness_classification": row.get("audit_classification"),
            "false_positive_classification": row["false_positive_classification"],
            "rationale": row["false_positive_rationale"],
            "source": row["audit_source_file"],
            "page": row["best_cached_page"],
        }
        for row in rows
    ]


def build_review(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "target ID": row["target_id"],
            "field name": row["field_name"],
            "definition": row["definition"],
            "datatype": row["datatype"],
            "unit": row["unit"],
            "value shape": row["value_shape"],
            "source": row["audit_source_file"],
            "page": row["best_cached_page"],
            "excerpt": row["best_cached_excerpt"],
            "classification": row["false_positive_classification"],
            "rationale": row["false_positive_rationale"],
        }
        for row in rows
    ]


def final_status(confirmed_count: int, target_count: int) -> str:
    if confirmed_count < target_count:
        return "blocked"
    return "ready_with_caveats" if confirmed_count > target_count else "ready"


def write_false_positive_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_outputs = {
        "target_false_positive_audit.json": result["target_false_positive_audit"],
        "confirmed_execution_ready_targets.json": result["confirmed_execution_ready_targets"],
        "rejected_ready_targets.json": result["rejected_ready_targets"],
        "wrong_event_or_attribute.json": result["wrong_event_or_attribute"],
        "component_only_targets.json": result["component_only_targets"],
        "dictionary_ambiguous_targets.json": result["dictionary_ambiguous_targets"],
        "insufficient_cached_evidence.json": result["insufficient_cached_evidence"],
        "false_positive_metrics.json": result["false_positive_metrics"],
        "false_positive_trace.json": result["false_positive_trace"],
        "confirmed_batch_v2_source_plan.json": result["confirmed_batch_v2_source_plan"],
        "confirmed_batch_v2_run_readiness.json": result["confirmed_batch_v2_run_readiness"],
        "batch_v2_cache_expansion_candidates.json": result[
            "batch_v2_cache_expansion_candidates"
        ],
    }
    for filename, payload in json_outputs.items():
        _atomic_write_json(output_dir / filename, payload)
    write_csv(
        output_dir / "target_false_positive_audit.csv",
        result["target_false_positive_audit"],
        audit_fields(),
    )
    write_csv(
        output_dir / "confirmed_execution_ready_targets.csv",
        result["confirmed_execution_ready_targets"],
        audit_fields(),
    )
    write_csv(
        output_dir / "batch_v2_false_positive_review.csv",
        result["batch_v2_false_positive_review"],
        review_fields(),
    )
    (output_dir / "batch_v2_false_positive_summary.md").write_text(
        summary_markdown(result), encoding="utf-8"
    )


def audit_fields() -> list[str]:
    return [
        "corrected_selection_rank",
        "target_id",
        "field_name",
        "definition",
        "datatype",
        "unit",
        "value_shape",
        "audit_source_file",
        "best_cached_page",
        "best_cached_excerpt",
        "false_positive_classification",
        "false_positive_rationale",
        "fp_component_present",
        "fp_attribute_present",
        "fp_value_signal_present",
        "fp_local_association",
    ]


def review_fields() -> list[str]:
    return [
        "target ID",
        "field name",
        "definition",
        "datatype",
        "unit",
        "value shape",
        "source",
        "page",
        "excerpt",
        "classification",
        "rationale",
    ]


def summary_markdown(result: dict[str, Any]) -> str:
    metrics = result["false_positive_metrics"]
    lines = [
        "# Batch V2 False-Positive Audit",
        "",
        f"- Audited retained execution-ready targets: {metrics['audited_execution_ready_count']}",
        f"- Confirmed targets: {metrics['confirmed_count']}",
        f"- Rejected targets: {metrics['rejected_count']}",
        f"- Classification counts: {metrics['classification_counts']}",
        f"- Final readiness status: {metrics['final_readiness_status']}",
        f"- Extraction approved: {metrics['extraction_approved']}",
        "",
        "This audit did not run extraction, parsing, OCR, VLM, or LLM calls.",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def normalize_excerpt(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _as_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        msg = f"Expected list for {label}"
        raise ValueError(msg)
    return [dict(item) for item in value]
