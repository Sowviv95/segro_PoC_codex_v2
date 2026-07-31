"""Deterministic Batch V2 target-selection readiness pack."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.target_semantics import derive_target_intent
from segro_evidence_extraction.vertical_slice import _atomic_write_json

DEFAULT_DICTIONARY_JSONL = Path("output/sprint2_dictionary_validation/normalized_targets.jsonl")
DEFAULT_BATCH_V1_DIR = Path("output/enfield_unit1_evidence_first_batch_v1")
DEFAULT_ADJUDICATION_DIR = Path("output/enfield_unit1_evidence_first_batch_v1_final_adjudication")
DEFAULT_SOURCE_MANIFEST = Path("output/sprint3_source_ingestion/source_pack_manifest.json")
DEFAULT_HIERARCHY_PATH = Path(
    "output/enfield_unit1_evidence_first_batch_v1_retrieval_diagnostic/revised_hierarchy.json"
)
DEFAULT_BATCH_V2_SELECTION_OUTPUT_DIR = Path(
    "output/enfield_unit1_evidence_first_batch_v2_selection"
)
EXPECTED_BATCH_V2_TARGET_COUNT = 75

ReadyClass = Literal[
    "ready_text_extraction",
    "ready_structured_table_extraction",
    "ready_identifier_or_date_extraction",
    "ready_measurement_extraction",
    "defer_visual",
    "defer_missing_source",
    "defer_dictionary_clarification",
    "defer_low_evidence_readiness",
    "exclude_prior_batch",
]

READY_CLASSES = {
    "ready_text_extraction",
    "ready_structured_table_extraction",
    "ready_identifier_or_date_extraction",
    "ready_measurement_extraction",
}

SCORE_FIELDS = [
    "evidence_source_readiness",
    "attribute_locality",
    "dictionary_clarity",
    "value_shape_reliability",
    "batch_v1_family_evidence",
    "boundedness",
]


class BatchV2SelectionError(ValueError):
    """Raised when a defensible Batch V2 selection cannot be produced."""


def run_batch_v2_selection_pack(
    *,
    dictionary_jsonl: Path = DEFAULT_DICTIONARY_JSONL,
    batch_v1_dir: Path = DEFAULT_BATCH_V1_DIR,
    adjudication_dir: Path = DEFAULT_ADJUDICATION_DIR,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    hierarchy_path: Path = DEFAULT_HIERARCHY_PATH,
    output_dir: Path = DEFAULT_BATCH_V2_SELECTION_OUTPUT_DIR,
    target_count: int = EXPECTED_BATCH_V2_TARGET_COUNT,
) -> dict[str, Any]:
    inputs = load_selection_inputs(
        dictionary_jsonl=dictionary_jsonl,
        batch_v1_dir=batch_v1_dir,
        adjudication_dir=adjudication_dir,
        source_manifest=source_manifest,
        hierarchy_path=hierarchy_path,
    )
    result = build_batch_v2_selection(inputs, target_count=target_count)
    write_selection_pack(result, output_dir)
    return result


def load_selection_inputs(
    *,
    dictionary_jsonl: Path,
    batch_v1_dir: Path,
    adjudication_dir: Path,
    source_manifest: Path,
    hierarchy_path: Path,
) -> dict[str, Any]:
    required = [
        dictionary_jsonl,
        batch_v1_dir / "selected_targets.json",
        adjudication_dir / "final_target_dispositions.json",
        adjudication_dir / "final_metrics.json",
        adjudication_dir / "manual_review_queue.json",
        adjudication_dir / "unsupported_targets.json",
        adjudication_dir / "ambiguous_targets.json",
        adjudication_dir / "rejected_prior_extractions.json",
        adjudication_dir / "evidence_audit.json",
        adjudication_dir / "disposition_trace.json",
        adjudication_dir / "lessons_for_next_batch.md",
        source_manifest,
        hierarchy_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        msg = f"Batch V2 selection inputs missing: {missing}"
        raise FileNotFoundError(msg)
    dictionary_targets = [
        TargetSpecification.model_validate(json.loads(line))
        for line in dictionary_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    batch_v1_selected_raw = _read_json(batch_v1_dir / "selected_targets.json")
    batch_v1_ids = {
        str(item["target"]["target_row_id"])
        for item in _as_list(batch_v1_selected_raw, "Batch V1 selected targets")
    }
    final_dispositions = _as_list(
        _read_json(adjudication_dir / "final_target_dispositions.json"),
        "final adjudication dispositions",
    )
    manual_review_ids = {
        str(item["target_id"])
        for item in _as_list(
            _read_json(adjudication_dir / "manual_review_queue.json"),
            "manual review queue",
        )
    }
    source_manifest_data = _read_json(source_manifest)
    hierarchy_data = _read_json(hierarchy_path)
    source_by_id = source_index(source_manifest_data)
    hierarchy_index_data = hierarchy_index(hierarchy_data, source_by_id)
    return {
        "dictionary_jsonl": str(dictionary_jsonl),
        "batch_v1_dir": str(batch_v1_dir),
        "adjudication_dir": str(adjudication_dir),
        "source_manifest": str(source_manifest),
        "hierarchy_path": str(hierarchy_path),
        "dictionary_targets": dictionary_targets,
        "batch_v1_ids": batch_v1_ids,
        "manual_review_ids": manual_review_ids,
        "final_dispositions": final_dispositions,
        "final_metrics": _read_json(adjudication_dir / "final_metrics.json"),
        "unsupported_targets": _as_list(
            _read_json(adjudication_dir / "unsupported_targets.json"),
            "unsupported targets",
        ),
        "ambiguous_targets": _as_list(
            _read_json(adjudication_dir / "ambiguous_targets.json"),
            "ambiguous targets",
        ),
        "rejected_prior_extractions": _as_list(
            _read_json(adjudication_dir / "rejected_prior_extractions.json"),
            "rejected prior extractions",
        ),
        "evidence_audit": _as_list(
            _read_json(adjudication_dir / "evidence_audit.json"),
            "evidence audit",
        ),
        "disposition_trace": _as_list(
            _read_json(adjudication_dir / "disposition_trace.json"),
            "disposition trace",
        ),
        "lessons_for_next_batch": (adjudication_dir / "lessons_for_next_batch.md").read_text(
            encoding="utf-8"
        ),
        "source_by_id": source_by_id,
        "hierarchy_index": hierarchy_index_data,
    }


def build_batch_v2_selection(inputs: dict[str, Any], *, target_count: int) -> dict[str, Any]:
    family_outcomes = build_family_outcomes(inputs["final_dispositions"])
    scored = [
        score_target(
            target=target,
            inputs=inputs,
            family_outcomes=family_outcomes,
        )
        for target in inputs["dictionary_targets"]
    ]
    scored = sorted(scored, key=candidate_sort_key)
    selected = select_candidates(scored, target_count=target_count)
    if len(selected) < target_count:
        status = "blocked"
    else:
        status = readiness_status(selected)
    selected = [
        {**candidate, "selection_rank": index}
        for index, candidate in enumerate(selected, start=1)
    ]
    candidate_scores = mark_candidate_selection(scored, selected)
    excluded = [
        item
        for item in candidate_scores
        if item["readiness_class"] == "exclude_prior_batch"
        or str(item["selection_status"]).startswith("excluded")
    ]
    deferred_visual = [
        item for item in candidate_scores if item["readiness_class"] == "defer_visual"
    ]
    deferred_missing = [
        item for item in candidate_scores if item["readiness_class"] == "defer_missing_source"
    ]
    clarification = [
        item
        for item in candidate_scores
        if item["readiness_class"] == "defer_dictionary_clarification"
    ]
    family_analysis = build_family_analysis(candidate_scores, family_outcomes)
    source_plan = build_source_plan(selected)
    trace = build_selection_trace(selected)
    metrics = build_selection_metrics(
        inputs=inputs,
        candidate_scores=candidate_scores,
        selected=selected,
        status=status,
    )
    readiness = {
        "overall_status": status,
        "target_count_requested": target_count,
        "selected_count": len(selected),
        "additional_ready_targets_needed": max(0, target_count - len(selected)),
        "blockers": readiness_blockers(metrics, selected),
        "caveats": readiness_caveats(metrics, selected),
    }
    review = build_review_rows(selected, candidate_scores)
    return {
        "selected_targets": selected,
        "candidate_scores": candidate_scores,
        "excluded_candidates": excluded,
        "deferred_visual_targets": deferred_visual,
        "deferred_missing_source_targets": deferred_missing,
        "dictionary_clarification_queue": clarification,
        "target_family_analysis": family_analysis,
        "selection_metrics": metrics,
        "selection_trace": trace,
        "batch_v2_source_plan": source_plan,
        "batch_v2_run_readiness": readiness,
        "batch_v2_review": review,
    }


def score_target(
    *,
    target: TargetSpecification,
    inputs: dict[str, Any],
    family_outcomes: dict[str, Counter[str]],
) -> dict[str, Any]:
    intent = derive_target_intent(target)
    prior_ids: set[str] = inputs["batch_v1_ids"]
    manual_ids: set[str] = inputs["manual_review_ids"]
    family = target_family(target)
    value_shape = selection_value_shape(target)
    likely_source_file, source_reason = likely_source(target, value_shape)
    hierarchy_section = likely_hierarchy_section(target, inputs["hierarchy_index"])
    expected_form = expected_evidence_form(target, value_shape)
    classification, deferral_reason = classify_readiness(
        target=target,
        target_id=target.target_row_id,
        prior_ids=prior_ids,
        manual_ids=manual_ids,
        value_shape=value_shape,
        likely_source_file=likely_source_file,
        family_outcomes=family_outcomes.get(family, Counter()),
    )
    scores = score_components(
        target=target,
        value_shape=value_shape,
        readiness_class=classification,
        likely_source_file=likely_source_file,
        hierarchy_section=hierarchy_section,
        family_outcomes=family_outcomes.get(family, Counter()),
    )
    total = sum(scores.values())
    selected_ready = classification in READY_CLASSES
    return {
        "selection_rank": None,
        "target_id": target.target_row_id,
        "dictionary_row": target.source_dictionary_provenance.row_number
        if target.source_dictionary_provenance
        else None,
        "domain": str(target.metadata.get("domain") or target.sub_domain),
        "sub_domain": target.sub_domain,
        "field_name": target.expected_field,
        "definition": target.requirement_text,
        "datatype": str(target.expected_data_type),
        "unit": target.unit,
        "value_shape": value_shape,
        "source_guidance": target.source_guidance,
        "likely_source_file": likely_source_file,
        "likely_hierarchy_section": hierarchy_section.get("title"),
        "likely_hierarchy_node": hierarchy_section.get("node_id"),
        "readiness_class": classification,
        "evidence_source_readiness": scores["evidence_source_readiness"],
        "attribute_locality": scores["attribute_locality"],
        "dictionary_clarity": scores["dictionary_clarity"],
        "value_shape_reliability": scores["value_shape_reliability"],
        "batch_v1_family_evidence": scores["batch_v1_family_evidence"],
        "boundedness": scores["boundedness"],
        "total_score": total,
        "selection_status": "candidate_ready" if selected_ready else "deferred",
        "selection_rationale": selection_rationale(
            target, value_shape, likely_source_file, source_reason, scores
        ),
        "known_risks": known_risks(target, classification, family_outcomes.get(family, Counter())),
        "expected_extraction_route": extraction_route(classification),
        "expected_evidence_form": expected_form,
        "expected_retrieval_query": f"{intent.primary_component} {intent.requested_attribute}",
        "family_key": family,
        "family_analogue_outcomes": dict(family_outcomes.get(family, Counter())),
        "exclusion_or_deferral_reason": deferral_reason,
        "rules_applied": rules_applied(target, classification, scores),
    }


def classify_readiness(
    *,
    target: TargetSpecification,
    target_id: str,
    prior_ids: set[str],
    manual_ids: set[str],
    value_shape: str,
    likely_source_file: str,
    family_outcomes: Counter[str],
) -> tuple[ReadyClass, str]:
    text = target_text(target)
    if target_id in prior_ids:
        return "exclude_prior_batch", "Target was already included in Batch V1."
    likely_evidence = {str(item) for item in target.likely_evidence_types}
    if (
        target_id in manual_ids
        or "drawing" in likely_evidence
        or any(term in text for term in ["drawing", "symbol", "layout"])
    ):
        return "defer_visual", "Likely visual/drawing evidence route; excluded from text batch."
    if any(term in text for term in ["mri", "oah", "lease", "tenant", "utility bill"]):
        return "defer_missing_source", "Likely requires unavailable operational/source system."
    if dictionary_ambiguous(target) or family_outcomes["dictionary_target_ambiguous"] >= 2:
        return (
            "defer_dictionary_clarification",
            "Dictionary semantics or family analogue ambiguous.",
        )
    if target.unit and value_shape in {"descriptive_text", "short_text"}:
        return (
            "defer_dictionary_clarification",
            "Dictionary unit conflicts with narrative value shape.",
        )
    if not likely_source_file:
        return (
            "defer_low_evidence_readiness",
            "No likely source document can be mapped from metadata.",
        )
    if family_outcomes["component_present_attribute_absent"] >= 2:
        return "defer_low_evidence_readiness", "Batch V1 analogue showed component-only evidence."
    if family_outcomes["unsupported_in_supplied_sources"] >= 3:
        return "defer_low_evidence_readiness", "Batch V1 analogue was repeatedly unsupported."
    if value_shape in {"date", "identifier_or_reference"}:
        return "ready_identifier_or_date_extraction", "Explicit date/reference style target."
    if value_shape == "decimal_measurement":
        return "ready_measurement_extraction", "Measurement target with likely source."
    if "table" in expected_evidence_form(target, value_shape):
        return "ready_structured_table_extraction", "Likely table or schedule target."
    return "ready_text_extraction", "Ready text evidence target."


def score_components(
    *,
    target: TargetSpecification,
    value_shape: str,
    readiness_class: str,
    likely_source_file: str,
    hierarchy_section: dict[str, Any],
    family_outcomes: Counter[str],
) -> dict[str, int]:
    ready = readiness_class in READY_CLASSES
    evidence = 10 if likely_source_file else 0
    if ready:
        evidence += 8
    if hierarchy_section:
        evidence += 5
    if any(kind in likely_source_file.lower() for kind in ["commissioning", "certificate"]):
        evidence += 2
    locality = 6
    text = target_text(target)
    if any(term in text for term in ["date", "reference", "certificate", "count", "number"]):
        locality += 7
    if any(term in text for term in ["schedule", "commissioning", "manufacturer", "model"]):
        locality += 4
    if family_outcomes["component_present_attribute_absent"]:
        locality -= 5
    clarity = 15
    if dictionary_ambiguous(target):
        clarity -= 8
    if target.unit and value_shape in {"descriptive_text", "short_text"}:
        clarity -= 6
    if len(target.requirement_text) < 20:
        clarity -= 3
    shape_score = {
        "integer_count": 14,
        "date": 14,
        "identifier_or_reference": 14,
        "decimal_measurement": 12,
        "categorical": 10,
        "short_text": 10,
        "descriptive_text": 8,
        "boolean_or_presence": 8,
        "ordered_or_unordered_list": 7,
    }.get(value_shape, 4)
    family_score = 8
    family_score += family_outcomes["accepted"] * 4
    family_score += family_outcomes["accepted_with_dictionary_caveat"] * 2
    family_score -= family_outcomes["invalid_prior_extraction"] * 2
    family_score -= family_outcomes["component_present_attribute_absent"] * 3
    family_score -= family_outcomes["dictionary_target_ambiguous"] * 3
    family_score -= family_outcomes["unsupported_in_supplied_sources"]
    bounded = 5
    if ready:
        bounded += 3
    if value_shape in {"date", "identifier_or_reference", "integer_count"}:
        bounded += 2
    return {
        "evidence_source_readiness": clamp(evidence, 0, 25),
        "attribute_locality": clamp(locality, 0, 20),
        "dictionary_clarity": clamp(clarity, 0, 15),
        "value_shape_reliability": clamp(shape_score, 0, 15),
        "batch_v1_family_evidence": clamp(family_score, 0, 15),
        "boundedness": clamp(bounded, 0, 10),
    }


def select_candidates(
    candidates: list[dict[str, Any]], *, target_count: int
) -> list[dict[str, Any]]:
    ready = [
        item
        for item in candidates
        if item["readiness_class"] in READY_CLASSES and int(item["total_score"]) >= 55
    ]
    selected: list[dict[str, Any]] = []
    domain_counts: Counter[str] = Counter()
    sub_domain_counts: Counter[str] = Counter()
    max_domain = max(1, int(target_count * 0.30))
    max_sub_domain = max(1, int(target_count * 0.25))
    for candidate in ready:
        domain = str(candidate["domain"])
        sub_domain = str(candidate["sub_domain"])
        if domain_counts[domain] >= max_domain:
            continue
        if sub_domain_counts[sub_domain] >= max_sub_domain:
            continue
        selected.append(candidate)
        domain_counts[domain] += 1
        sub_domain_counts[sub_domain] += 1
        if len(selected) == target_count:
            return selected
    for candidate in ready:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) == target_count:
            return selected
    return selected


def mark_candidate_selection(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_ids = {str(item["target_id"]) for item in selected}
    rank_by_id = {str(item["target_id"]): item["selection_rank"] for item in selected}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        target_id = str(candidate["target_id"])
        row = dict(candidate)
        if target_id in selected_ids:
            row["selection_status"] = "selected"
            row["selection_rank"] = rank_by_id[target_id]
        elif row["readiness_class"] == "exclude_prior_batch":
            row["selection_status"] = "excluded_prior_batch"
        elif row["readiness_class"] in READY_CLASSES:
            row["selection_status"] = "ready_not_selected"
        else:
            row["selection_status"] = "deferred"
        rows.append(row)
    return rows


def build_source_plan(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "selection_rank": item["selection_rank"],
            "target_id": item["target_id"],
            "field_name": item["field_name"],
            "likely_source_document": item["likely_source_file"],
            "likely_section": item["likely_hierarchy_section"],
            "existing_hierarchy_node": item["likely_hierarchy_node"],
            "existing_hierarchy_coverage_known": bool(item["likely_hierarchy_node"]),
            "expected_retrieval_query": item["expected_retrieval_query"],
            "adjacent_page_expansion_might_be_needed": item["value_shape"]
            in {"descriptive_text", "decimal_measurement"},
            "maximum_recommended_evidence_window": 3
            if item["readiness_class"] == "ready_identifier_or_date_extraction"
            else 5,
            "expected_extraction_route": item["expected_extraction_route"],
            "expected_evidence_form": item["expected_evidence_form"],
        }
        for item in selected
    ]


def build_selection_trace(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "selection_rank": item["selection_rank"],
            "target_id": item["target_id"],
            "rules_applied": item["rules_applied"],
            "score_contributions": {field: item[field] for field in SCORE_FIELDS},
            "total_score": item["total_score"],
            "family_analogue_used": item["family_analogue_outcomes"],
            "diversity_adjustment": "selected_under_domain_and_subdomain_caps",
            "final_selection_rank": item["selection_rank"],
        }
        for item in selected
    ]


def build_selection_metrics(
    *,
    inputs: dict[str, Any],
    candidate_scores: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    readiness_counts = Counter(str(item["readiness_class"]) for item in candidate_scores)
    selected_counts = Counter(str(item["readiness_class"]) for item in selected)
    selected_scores = [int(item["total_score"]) for item in selected]
    return {
        "total_dictionary_targets": len(inputs["dictionary_targets"]),
        "prior_batch_v1_targets_excluded": readiness_counts["exclude_prior_batch"],
        "remaining_candidates": (
            len(inputs["dictionary_targets"]) - readiness_counts["exclude_prior_batch"]
        ),
        "candidates_scored": len(candidate_scores),
        "ready_candidates": sum(readiness_counts[item] for item in READY_CLASSES),
        "deferred_visual": readiness_counts["defer_visual"],
        "deferred_missing_source": readiness_counts["defer_missing_source"],
        "dictionary_clarification": readiness_counts["defer_dictionary_clarification"],
        "low_readiness_deferrals": readiness_counts["defer_low_evidence_readiness"],
        "selected_count": len(selected),
        "overall_status": status,
        "selection_by_domain": dict(Counter(str(item["domain"]) for item in selected)),
        "selection_by_sub_domain": dict(Counter(str(item["sub_domain"]) for item in selected)),
        "selection_by_value_shape": dict(Counter(str(item["value_shape"]) for item in selected)),
        "selection_by_datatype": dict(Counter(str(item["datatype"]) for item in selected)),
        "selection_by_source_document": dict(
            Counter(str(item["likely_source_file"]) for item in selected)
        ),
        "selection_by_readiness_class": dict(selected_counts),
        "score_distribution": {
            "min": min(selected_scores) if selected_scores else None,
            "max": max(selected_scores) if selected_scores else None,
            "average": round(sum(selected_scores) / len(selected_scores), 2)
            if selected_scores
            else None,
        },
        "top_exclusion_reasons": dict(
            Counter(str(item["exclusion_or_deferral_reason"]) for item in candidate_scores)
            .most_common(10)
        ),
        "diversity_control_results": diversity_results(selected),
    }


def build_family_analysis(
    candidate_scores: list[dict[str, Any]],
    family_outcomes: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidate_scores:
        grouped[str(item["family_key"])].append(item)
    rows: list[dict[str, Any]] = []
    for family, items in grouped.items():
        readiness = Counter(str(item["readiness_class"]) for item in items)
        selected = sum(1 for item in items if item["selection_status"] == "selected")
        first = sorted(items, key=candidate_sort_key)[0]
        rows.append(
            {
                "family_key": family,
                "domain": first["domain"],
                "sub_domain": first["sub_domain"],
                "value_shape": first["value_shape"],
                "candidate_count": len(items),
                "selected_count": selected,
                "readiness_distribution": dict(readiness),
                "likely_evidence_source": first["likely_source_file"],
                "batch_v1_analogue_outcomes": dict(family_outcomes.get(family, Counter())),
                "recommended_route": family_route(items),
                "exclusion_rationale": family_exclusion(items),
            }
        )
    return sorted(rows, key=lambda item: (str(item["family_key"])))


def build_review_rows(
    selected: list[dict[str, Any]],
    candidate_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    top_deferred = [
        item
        for item in candidate_scores
        if item["selection_status"] != "selected"
        and item["readiness_class"] != "exclude_prior_batch"
    ][:75]
    rows = [review_row(item, selected=True) for item in selected]
    rows.extend(review_row(item, selected=False) for item in top_deferred)
    return rows


def review_row(item: dict[str, Any], *, selected: bool) -> dict[str, Any]:
    return {
        "selection rank": item.get("selection_rank") or "",
        "target ID": item["target_id"],
        "dictionary row": item["dictionary_row"],
        "domain": item["domain"],
        "sub-domain": item["sub_domain"],
        "field name": item["field_name"],
        "definition": item["definition"],
        "datatype": item["datatype"],
        "unit": item["unit"],
        "value shape": item["value_shape"],
        "source guidance": item["source_guidance"],
        "likely source file": item["likely_source_file"],
        "likely hierarchy section": item["likely_hierarchy_section"],
        "readiness class": item["readiness_class"],
        "evidence-source readiness": item["evidence_source_readiness"],
        "attribute locality": item["attribute_locality"],
        "dictionary clarity": item["dictionary_clarity"],
        "value-shape reliability": item["value_shape_reliability"],
        "Batch V1 family evidence": item["batch_v1_family_evidence"],
        "boundedness": item["boundedness"],
        "total score": item["total_score"],
        "selected": selected,
        "selection rationale": item["selection_rationale"],
        "known risks": "; ".join(item["known_risks"]),
        "expected evidence form": item["expected_evidence_form"],
        "expected extraction route": item["expected_extraction_route"],
        "exclusion or deferral reason": item["exclusion_or_deferral_reason"],
    }


def write_selection_pack(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_outputs = {
        "selected_targets.json": result["selected_targets"],
        "candidate_scores.json": result["candidate_scores"],
        "excluded_candidates.json": result["excluded_candidates"],
        "deferred_visual_targets.json": result["deferred_visual_targets"],
        "deferred_missing_source_targets.json": result["deferred_missing_source_targets"],
        "dictionary_clarification_queue.json": result["dictionary_clarification_queue"],
        "target_family_analysis.json": result["target_family_analysis"],
        "selection_metrics.json": result["selection_metrics"],
        "selection_trace.json": result["selection_trace"],
        "batch_v2_source_plan.json": result["batch_v2_source_plan"],
        "batch_v2_run_readiness.json": result["batch_v2_run_readiness"],
    }
    for filename, payload in json_outputs.items():
        _atomic_write_json(output_dir / filename, payload)
    write_csv(output_dir / "selected_targets.csv", result["selected_targets"], selected_fields())
    write_csv(output_dir / "candidate_scores.csv", result["candidate_scores"], candidate_fields())
    write_csv(
        output_dir / "target_family_analysis.csv",
        result["target_family_analysis"],
        family_fields(),
    )
    write_csv(output_dir / "batch_v2_review.csv", result["batch_v2_review"], review_fields())
    (output_dir / "batch_v2_selection_summary.md").write_text(
        selection_summary_markdown(result),
        encoding="utf-8",
    )
    (output_dir / "batch_v2_execution_recommendations.md").write_text(
        execution_recommendations_markdown(result),
        encoding="utf-8",
    )


def selected_fields() -> list[str]:
    return [
        "selection_rank",
        "target_id",
        "dictionary_row",
        "domain",
        "sub_domain",
        "field_name",
        "definition",
        "datatype",
        "unit",
        "value_shape",
        "source_guidance",
        "likely_source_file",
        "likely_hierarchy_section",
        "readiness_class",
        *SCORE_FIELDS,
        "total_score",
        "selection_rationale",
        "known_risks",
        "expected_extraction_route",
        "expected_evidence_form",
    ]


def candidate_fields() -> list[str]:
    return [
        *selected_fields(),
        "selection_status",
        "exclusion_or_deferral_reason",
        "family_key",
    ]


def family_fields() -> list[str]:
    return [
        "family_key",
        "domain",
        "sub_domain",
        "value_shape",
        "candidate_count",
        "selected_count",
        "readiness_distribution",
        "likely_evidence_source",
        "batch_v1_analogue_outcomes",
        "recommended_route",
        "exclusion_rationale",
    ]


def review_fields() -> list[str]:
    return [
        "selection rank",
        "target ID",
        "dictionary row",
        "domain",
        "sub-domain",
        "field name",
        "definition",
        "datatype",
        "unit",
        "value shape",
        "source guidance",
        "likely source file",
        "likely hierarchy section",
        "readiness class",
        "evidence-source readiness",
        "attribute locality",
        "dictionary clarity",
        "value-shape reliability",
        "Batch V1 family evidence",
        "boundedness",
        "total score",
        "selected",
        "selection rationale",
        "known risks",
        "expected evidence form",
        "expected extraction route",
        "exclusion or deferral reason",
    ]


def source_index(source_manifest_data: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in source_manifest_data.get("registered_sources", []):
        result[str(item["source_id"])] = item
    for item in source_manifest_data.get("discovered_files", []):
        source_id = item.get("source_id")
        if source_id:
            result[str(source_id)] = item
    return result


def hierarchy_index(
    hierarchy_data: Any,
    source_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    nodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in hierarchy_data.get("nodes", []):
        if node.get("node_type") not in {"section", "page", "text_block"}:
            continue
        source = source_by_id.get(str(node.get("source_id")), {})
        logical = (
            source.get("logical_path")
            or source.get("relative_path")
            or source.get("file_name")
        )
        nodes[str(node.get("source_id"))].append(
            {
                "node_id": node.get("node_id"),
                "title": node.get("title") or node.get("text_summary") or "",
                "text_summary": node.get("text_summary") or "",
                "source_file": logical or node.get("source_id"),
                "page_start": node.get("page_start"),
                "page_end": node.get("page_end"),
            }
        )
    return dict(nodes)


def likely_source(target: TargetSpecification, value_shape: str) -> tuple[str, str]:
    text = target_text(target)
    if "commission" in text:
        return "Rolec EV - Commissioning Document.pdf", "Commissioning keyword in target."
    if any(term in text for term in ["certificate", "expiry", "issue date", "assessment"]):
        return "Building Manual - Part 3 Building Services.pdf", "Certificate/statutory keyword."
    if any(term in text for term in ["planning", "approved use", "consent"]):
        return "Building Manual - Part 1 General.pdf", "Planning keyword."
    if any(term in text for term in ["roof", "wall", "floor", "cladding", "dock", "door"]):
        return "Building Manual - Part 1 General.pdf", "Building fabric/component keyword."
    if value_shape in {"date", "identifier_or_reference"}:
        return (
            "Building Manual - Part 1 General.pdf",
            "Date/reference target likely in general manual.",
        )
    if target.sub_domain in {"Size", "Property", "Location"}:
        return (
            "Building Manual - Part 1 General.pdf",
            "Property/size metadata likely in general manual.",
        )
    return "Building Manual - Part 1 General.pdf", "Default bounded manual source."


def likely_hierarchy_section(
    target: TargetSpecification,
    hierarchy: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    terms = [part for part in target.expected_field.lower().split("_") if len(part) > 3]
    best: dict[str, Any] = {}
    best_score = 0
    for nodes in hierarchy.values():
        for node in nodes:
            haystack = f"{node.get('title', '')} {node.get('text_summary', '')}".lower()
            score = sum(1 for term in terms if term in haystack)
            if score > best_score:
                best_score = score
                best = node
    return best if best_score > 0 else {}


def build_family_outcomes(final_dispositions: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    for item in final_dispositions:
        key = (
            f"{item.get('sub_domain')}|{item.get('value_shape')}|"
            f"{family_slug(str(item.get('field_name')))}"
        )
        outcomes[key][str(item.get("final_disposition"))] += 1
    return outcomes


def target_family(target: TargetSpecification) -> str:
    value_shape = selection_value_shape(target)
    return f"{target.sub_domain}|{value_shape}|{family_slug(target.expected_field)}"


def selection_value_shape(target: TargetSpecification) -> str:
    """Use dictionary field labels to avoid routing identifiers as counts."""

    base_shape = derive_target_intent(target).value_shape_family
    label_text = " ".join(
        [
            target.expected_field,
            str(target.metadata.get("field_label") or ""),
        ]
    ).lower()
    if "model number" in label_text or "model_number" in label_text or "reference" in label_text:
        return "identifier_or_reference"
    if any(term in label_text for term in ["manufacturer", "model name", "model_name"]):
        return "short_text"
    return base_shape


def family_slug(field_name: str) -> str:
    tokens = [token for token in field_name.lower().split("_") if token]
    suffixes = {
        "date",
        "count",
        "description",
        "type",
        "name",
        "manufacturer",
        "model",
        "number",
        "reference",
        "capacity",
        "value",
    }
    kept = [token for token in tokens if token not in suffixes]
    return "_".join(kept[:2] or tokens[:1])


def expected_evidence_form(target: TargetSpecification, value_shape: str) -> str:
    text = target_text(target)
    forms: list[str] = []
    if any(term in text for term in ["certificate", "issue", "expiry"]):
        forms.append("certificate field")
    if any(term in text for term in ["schedule", "table", "commissioning"]):
        forms.append("table or schedule row")
    if value_shape in {"integer_count", "decimal_measurement"}:
        forms.append("numeric statement with local component")
    if value_shape in {"date", "identifier_or_reference"}:
        forms.append("labelled date/reference field")
    if not forms:
        forms.append("labelled narrative clause")
    return "; ".join(forms)


def extraction_route(readiness_class: str) -> str:
    if readiness_class == "ready_structured_table_extraction":
        return "text/table evidence bundle then bounded extraction"
    if readiness_class == "ready_identifier_or_date_extraction":
        return "identifier/date evidence bundle then bounded extraction"
    if readiness_class == "ready_measurement_extraction":
        return "measurement evidence bundle then bounded extraction"
    if readiness_class == "ready_text_extraction":
        return "text evidence bundle then bounded extraction"
    return "deferred from text extraction batch"


def dictionary_ambiguous(target: TargetSpecification) -> bool:
    text = target_text(target)
    ambiguous_terms = ["tbc", "unknown", "to be confirmed", "maps from", "unclear"]
    if target.requirement_text.strip().lower().endswith(" - x"):
        return True
    label_text = " ".join(
        [
            target.expected_field,
            str(target.metadata.get("field_label") or ""),
        ]
    ).lower()
    definition = target.requirement_text.lower()
    datatype = str(target.expected_data_type).lower()
    descriptive_label_terms = ["manufacturer", "model", "reference", "name"]
    count_definition_terms = ["number of", "count", "no."]
    if (
        any(term in label_text for term in descriptive_label_terms)
        and datatype in {"integer", "decimal"}
        and any(term in definition for term in count_definition_terms)
    ):
        return True
    tokens = set(text.replace("-", " ").replace("/", " ").split())
    return "x" in tokens or any(term in text for term in ambiguous_terms)


def known_risks(
    target: TargetSpecification,
    readiness_class: str,
    family_outcomes: Counter[str],
) -> list[str]:
    risks: list[str] = []
    if readiness_class not in READY_CLASSES:
        risks.append(readiness_class)
    if family_outcomes["component_present_attribute_absent"]:
        risks.append("Batch V1 analogue had component-only evidence")
    if family_outcomes["dictionary_target_ambiguous"]:
        risks.append("Batch V1 analogue had dictionary ambiguity")
    if target.unit and selection_value_shape(target) in {
        "descriptive_text",
        "short_text",
    }:
        risks.append("unit may not apply to narrative value")
    return risks


def selection_rationale(
    target: TargetSpecification,
    value_shape: str,
    source_file: str,
    source_reason: str,
    scores: dict[str, int],
) -> str:
    return (
        f"{value_shape} target mapped to {source_file or 'no source'}; {source_reason} "
        f"Score contributions={scores}."
    )


def rules_applied(
    target: TargetSpecification,
    readiness_class: str,
    scores: dict[str, int],
) -> list[str]:
    return [
        f"readiness={readiness_class}",
        f"value_shape={selection_value_shape(target)}",
        f"datatype={target.expected_data_type}",
        f"scores={scores}",
    ]


def readiness_status(selected: list[dict[str, Any]]) -> str:
    if not selected:
        return "blocked"
    risks = diversity_results(selected)
    if risks["domain_cap_exception"] or risks["sub_domain_cap_exception"]:
        return "ready_with_caveats"
    return "ready_with_caveats"


def readiness_blockers(metrics: dict[str, Any], selected: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    if len(selected) < EXPECTED_BATCH_V2_TARGET_COUNT:
        blockers.append("Fewer than 75 ready candidates selected.")
    if any(not item["likely_source_file"] for item in selected):
        blockers.append("At least one selected target lacks source mapping.")
    return blockers


def readiness_caveats(metrics: dict[str, Any], selected: list[dict[str, Any]]) -> list[str]:
    caveats = ["Selection is metadata-readiness only; execution must still validate evidence."]
    diversity = metrics.get("diversity_control_results", {})
    if diversity.get("domain_cap_exception"):
        caveats.append("Domain concentration exceeds guidance.")
    if diversity.get("sub_domain_cap_exception"):
        caveats.append("Sub-domain concentration exceeds guidance.")
    return caveats


def diversity_results(selected: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(selected) or 1
    domain_counts = Counter(str(item["domain"]) for item in selected)
    sub_counts = Counter(str(item["sub_domain"]) for item in selected)
    value_shapes = Counter(str(item["value_shape"]) for item in selected)
    return {
        "max_domain_share": round(max(domain_counts.values(), default=0) / count, 3),
        "max_sub_domain_share": round(max(sub_counts.values(), default=0) / count, 3),
        "value_shape_family_count": len(value_shapes),
        "domain_cap_exception": any(value / count > 0.30 for value in domain_counts.values()),
        "sub_domain_cap_exception": any(value / count > 0.25 for value in sub_counts.values()),
        "value_shape_control_met": len(value_shapes) >= 4,
    }


def family_route(items: list[dict[str, Any]]) -> str:
    readiness = Counter(str(item["readiness_class"]) for item in items)
    if sum(readiness[item] for item in READY_CLASSES):
        return "candidate_for_text_batch"
    return str(readiness.most_common(1)[0][0]) if readiness else "unknown"


def family_exclusion(items: list[dict[str, Any]]) -> str:
    reasons = Counter(str(item["exclusion_or_deferral_reason"]) for item in items)
    return str(reasons.most_common(1)[0][0]) if reasons else ""


def candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(item["total_score"]),
        str(item["readiness_class"]) not in READY_CLASSES,
        int(item["dictionary_row"] or 999999),
        str(item["target_id"]),
    )


def target_text(target: TargetSpecification) -> str:
    return " ".join(
        [
            target.expected_field,
            target.requirement_text,
            target.source_guidance or "",
            target.component_type or "",
            target.component_subtype or "",
            target.sub_domain,
        ]
    ).lower()


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def selection_summary_markdown(result: dict[str, Any]) -> str:
    metrics = result["selection_metrics"]
    lines = [
        "# Batch V2 Target Selection Strategy",
        "",
        f"- Total dictionary targets: {metrics['total_dictionary_targets']}",
        f"- Prior Batch V1 excluded: {metrics['prior_batch_v1_targets_excluded']}",
        f"- Ready candidates: {metrics['ready_candidates']}",
        f"- Selected targets: {metrics['selected_count']}",
        f"- Overall readiness: {metrics['overall_status']}",
        f"- Selection by domain: {metrics['selection_by_domain']}",
        f"- Selection by value shape: {metrics['selection_by_value_shape']}",
        "",
        "## Scoring Weights",
        "",
        "- Evidence-source readiness: 0-25",
        "- Attribute locality: 0-20",
        "- Dictionary clarity: 0-15",
        "- Value-shape reliability: 0-15",
        "- Batch V1 family evidence: 0-15",
        "- Cost and boundedness: 0-10",
    ]
    return "\n".join(lines) + "\n"


def execution_recommendations_markdown(result: dict[str, Any]) -> str:
    metrics = result["selection_metrics"]
    routes = metrics["selection_by_readiness_class"]
    lines = [
        "# Batch V2 Execution Recommendations",
        "",
        "- Do not execute extraction until this selection pack is reviewed.",
        "- Start with identifier/date and count targets, then measurements, then narrative text.",
        "- Keep visual/deferred targets outside the text batch.",
        f"- Expected routes: {routes}",
        "- Use existing hierarchy/cache metadata first; do not parse new pages during "
        "selection review.",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


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
