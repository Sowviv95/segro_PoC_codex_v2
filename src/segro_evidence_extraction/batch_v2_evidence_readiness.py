"""Evidence-readiness audit for the proposed Batch V2 selection."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from segro_evidence_extraction.batch_selection_strategy import (
    DEFAULT_BATCH_V2_SELECTION_OUTPUT_DIR,
    READY_CLASSES,
    SCORE_FIELDS,
    candidate_sort_key,
)
from segro_evidence_extraction.vertical_slice import _atomic_write_json

DEFAULT_PAGE_CACHE_ROOT = Path("output/enfield_unit1_evidence_first_vertical_slice_v1/page_cache")
DEFAULT_ADJUDICATION_DIR = Path("output/enfield_unit1_evidence_first_batch_v1_final_adjudication")
DEFAULT_READINESS_AUDIT_OUTPUT_DIR = Path(
    "output/enfield_unit1_evidence_first_batch_v2_readiness_audit"
)
EXPECTED_BATCH_V2_TARGET_COUNT = 75

AuditClass = Literal[
    "execution_ready",
    "ready_with_evidence_caveat",
    "wrong_source_mapping",
    "component_only_risk",
    "attribute_not_observed",
    "requires_additional_cached_pages",
    "dictionary_clarification",
    "defer_visual",
]

PASSING_CLASSES = {"execution_ready", "ready_with_evidence_caveat"}
NON_EVIDENCE_CLASSES = {
    "exclude_prior_batch",
    "defer_visual",
    "defer_missing_source",
    "defer_dictionary_clarification",
    "defer_low_evidence_readiness",
}


class BatchV2ReadinessError(ValueError):
    """Raised when readiness audit inputs are inconsistent."""


def run_batch_v2_readiness_audit(
    *,
    selection_dir: Path = DEFAULT_BATCH_V2_SELECTION_OUTPUT_DIR,
    page_cache_root: Path = DEFAULT_PAGE_CACHE_ROOT,
    adjudication_dir: Path = DEFAULT_ADJUDICATION_DIR,
    output_dir: Path = DEFAULT_READINESS_AUDIT_OUTPUT_DIR,
    target_count: int = EXPECTED_BATCH_V2_TARGET_COUNT,
) -> dict[str, Any]:
    inputs = load_readiness_inputs(
        selection_dir=selection_dir,
        page_cache_root=page_cache_root,
        adjudication_dir=adjudication_dir,
    )
    result = build_readiness_audit(inputs, target_count=target_count)
    write_readiness_outputs(result, output_dir)
    return result


def load_readiness_inputs(
    *,
    selection_dir: Path,
    page_cache_root: Path,
    adjudication_dir: Path,
) -> dict[str, Any]:
    required = [
        selection_dir / "selected_targets.json",
        selection_dir / "candidate_scores.json",
        selection_dir / "selection_metrics.json",
        selection_dir / "batch_v2_source_plan.json",
        selection_dir / "batch_v2_run_readiness.json",
        adjudication_dir / "lessons_for_next_batch.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        msg = f"Batch V2 readiness audit inputs missing: {missing}"
        raise FileNotFoundError(msg)
    pages = load_cached_pages(page_cache_root)
    return {
        "selection_dir": str(selection_dir),
        "page_cache_root": str(page_cache_root),
        "adjudication_dir": str(adjudication_dir),
        "selected_targets": _read_json(selection_dir / "selected_targets.json"),
        "candidate_scores": _read_json(selection_dir / "candidate_scores.json"),
        "selection_metrics": _read_json(selection_dir / "selection_metrics.json"),
        "source_plan": _read_json(selection_dir / "batch_v2_source_plan.json"),
        "prior_lessons": (adjudication_dir / "lessons_for_next_batch.md").read_text(
            encoding="utf-8"
        ),
        "cached_pages": pages,
    }


def build_readiness_audit(inputs: dict[str, Any], *, target_count: int) -> dict[str, Any]:
    selected = _as_list(inputs["selected_targets"], "selected targets")
    candidates = _as_list(inputs["candidate_scores"], "candidate scores")
    if len({str(item["target_id"]) for item in selected}) != len(selected):
        raise BatchV2ReadinessError("Selected targets contain duplicate target IDs.")

    selected_ids = {str(item["target_id"]) for item in selected}
    original_audit = [
        audit_target(item, inputs, audit_stage="original_selection") for item in selected
    ]
    confirmed = [
        item for item in original_audit if item["audit_classification"] in PASSING_CLASSES
    ]
    removed = [
        item for item in original_audit if item["audit_classification"] not in PASSING_CLASSES
    ]

    replacements: list[dict[str, Any]] = []
    corrected_ids = {str(item["target_id"]) for item in confirmed}
    replacement_audit: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=candidate_sort_key):
        target_id = str(candidate["target_id"])
        if target_id in selected_ids or target_id in corrected_ids:
            continue
        if candidate.get("readiness_class") not in READY_CLASSES:
            continue
        audited = audit_target(candidate, inputs, audit_stage="replacement_candidate")
        replacement_audit.append(audited)
        if audited["audit_classification"] in PASSING_CLASSES:
            replacements.append(audited)
            corrected_ids.add(target_id)
            if len(confirmed) + len(replacements) >= target_count:
                break

    corrected = confirmed + replacements
    corrected = [
        {**item, "corrected_selection_rank": index}
        for index, item in enumerate(corrected[:target_count], start=1)
    ]
    status = "ready_with_caveats" if len(corrected) == target_count else "blocked"
    all_audited = original_audit + replacement_audit
    metrics = build_metrics(
        inputs=inputs,
        original_audit=original_audit,
        replacement_audit=replacement_audit,
        corrected=corrected,
        status=status,
        target_count=target_count,
    )
    return {
        "target_readiness_audit": original_audit,
        "confirmed_execution_ready_targets": confirmed,
        "corrected_selected_targets": corrected,
        "removed_selected_targets": removed,
        "replacement_target_audit": replacement_audit,
        "source_mapping_corrections": [
            item for item in all_audited if item["audit_classification"] == "wrong_source_mapping"
        ],
        "component_only_risks": [
            item for item in all_audited if item["audit_classification"] == "component_only_risk"
        ],
        "attribute_not_observed": [
            item for item in all_audited if item["audit_classification"] == "attribute_not_observed"
        ],
        "requires_additional_cached_pages": [
            item
            for item in all_audited
            if item["audit_classification"] == "requires_additional_cached_pages"
        ],
        "deferred_visual_targets": [
            item for item in all_audited if item["audit_classification"] == "defer_visual"
        ],
        "dictionary_clarification_targets": [
            item
            for item in all_audited
            if item["audit_classification"] == "dictionary_clarification"
        ],
        "model_manufacturer_audit": model_manufacturer_summary(original_audit),
        "part1_concentration_audit": part1_summary(original_audit),
        "cached_evidence_candidates": [
            cached_evidence_row(item)
            for item in all_audited
            if item.get("best_cached_excerpt")
        ],
        "readiness_metrics": metrics,
        "readiness_trace": build_trace(original_audit, replacement_audit, corrected),
        "corrected_batch_v2_source_plan": build_source_plan(corrected),
        "corrected_batch_v2_run_readiness": {
            "overall_status": status,
            "target_count_requested": target_count,
            "corrected_selected_count": len(corrected),
            "additional_ready_targets_needed": max(0, target_count - len(corrected)),
            "batch_v2_extraction_approved": status != "blocked",
            "caveats": [
                "Evidence audit is lexical over existing cached pages only.",
                "Bounded extraction must still validate canonical spans and typed values.",
            ],
            "blockers": []
            if status != "blocked"
            else ["Fewer than 75 evidence-ready targets found in cached pages."],
        },
        "batch_v2_readiness_review": build_review(original_audit, replacement_audit, corrected),
    }


def audit_target(
    target: dict[str, Any],
    inputs: dict[str, Any],
    *,
    audit_stage: str,
) -> dict[str, Any]:
    field_name = str(target["field_name"])
    value_shape = str(target["value_shape"])
    proposed_source = str(target.get("likely_source_file") or "")
    pages_by_source: dict[str, list[dict[str, Any]]] = inputs["cached_pages"]
    query = build_query_terms(field_name, str(target.get("definition") or ""), value_shape)
    source_pages = pages_by_source.get(proposed_source, [])
    source_has_cache = bool(source_pages)
    source_evidence = find_best_signal(source_pages, query, value_shape)
    all_signals = {
        source: find_best_signal(pages, query, value_shape)
        for source, pages in pages_by_source.items()
    }
    better_source, better_signal = best_alternate_source(
        proposed_source, source_evidence, all_signals
    )
    classification, rationale = classify_audit(
        target=target,
        query=query,
        source_has_cache=source_has_cache,
        source_signal=source_evidence,
        better_source=better_source,
        better_signal=better_signal,
    )
    winning_signal = (
        better_signal
        if classification == "wrong_source_mapping" and better_signal is not None
        else source_evidence
    )
    return {
        **target,
        "audit_stage": audit_stage,
        "proposed_source_file": proposed_source,
        "audit_source_file": better_source or proposed_source,
        "source_mapping_confirmed": not better_source,
        "component_observed": source_evidence["component_observed"],
        "requested_attribute_observed": source_evidence["attribute_observed"],
        "plausible_value_signal_observed": source_evidence["value_observed"],
        "local_association_observed": source_evidence["local_association"],
        "best_cached_page": winning_signal.get("page_number"),
        "best_cached_excerpt": winning_signal.get("excerpt"),
        "best_cached_source": winning_signal.get("source_file"),
        "better_existing_source_file": better_source,
        "audit_classification": classification,
        "audit_rationale": rationale,
        "evidence_terms": query,
        "source_cache_page_count": len(source_pages),
    }


def classify_audit(
    *,
    target: dict[str, Any],
    query: dict[str, list[str]],
    source_has_cache: bool,
    source_signal: dict[str, Any],
    better_source: str | None,
    better_signal: dict[str, Any] | None,
) -> tuple[AuditClass, str]:
    readiness_class = str(target.get("readiness_class") or "")
    text = f"{target.get('field_name', '')} {target.get('definition', '')}".lower()
    if readiness_class == "defer_visual" or any(
        term in text for term in ["drawing", "symbol", "layout", "roof plant"]
    ):
        return "defer_visual", "Target likely requires drawing or visual evidence."
    if readiness_class == "defer_dictionary_clarification" or dictionary_conflict(target):
        return "dictionary_clarification", "Dictionary metadata prevents a clean evidence test."
    if better_source and better_signal:
        return (
            "wrong_source_mapping",
            f"Proposed source weaker than cached signal in {better_source}.",
        )
    if not source_has_cache:
        return (
            "requires_additional_cached_pages",
            "Proposed source has no cached pages available for readiness audit.",
        )
    if source_signal["local_association"]:
        if source_signal["score"] >= 8:
            return "execution_ready", "Component, attribute and plausible value co-occur locally."
        return (
            "ready_with_evidence_caveat",
            "Local value signal exists but should be verified with bounded retrieval.",
        )
    if source_signal["component_observed"] and not source_signal["attribute_observed"]:
        return (
            "component_only_risk",
            "Component appears in cached text but requested attribute was not observed locally.",
        )
    if source_signal["component_observed"] or source_signal["attribute_observed"]:
        return (
            "attribute_not_observed",
            "Cached text contains partial evidence but no local requested-attribute value.",
        )
    return (
        "requires_additional_cached_pages",
        "No relevant signal was observed in currently cached pages for the proposed source.",
    )


def build_query_terms(field_name: str, definition: str, value_shape: str) -> dict[str, list[str]]:
    tokens = [token for token in re.split(r"[_\W]+", field_name.lower()) if len(token) > 2]
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
        "source",
        "system",
        "component",
        "infra",
        "infrastructure",
        "area",
        "space",
    }
    components = [token for token in tokens if token not in stop][:4]
    attributes = attribute_terms(field_name, definition, value_shape)
    value_patterns = value_patterns_for(value_shape, field_name)
    return {
        "component_terms": components,
        "attribute_terms": attributes,
        "value_patterns": value_patterns,
        "field_terms": [field_name.lower()],
    }


def attribute_terms(field_name: str, definition: str, value_shape: str) -> list[str]:
    text = f"{field_name} {definition}".lower()
    terms: list[str] = []
    if "model" in text:
        terms.extend(["model", "model no", "model number", "type", "reference"])
    if "manufacturer" in text:
        terms.extend(["manufacturer", "make", "supplier"])
    if "reference" in text:
        terms.extend(["reference", "ref", "certificate no", "serial"])
    if value_shape == "date":
        terms.extend(["date", "issue", "expiry", "certificate"])
    if value_shape == "integer_count":
        terms.extend(["no.", "number of", "count", "qty", "quantity"])
    if value_shape == "decimal_measurement":
        terms.extend(["area", "size", "rating", "capacity", "height", "width", "depth"])
    if value_shape in {"descriptive_text", "short_text", "categorical"}:
        terms.extend(["description", "type", "rating", "material", "construction"])
    return sorted(set(terms))


def value_patterns_for(value_shape: str, field_name: str) -> list[str]:
    if value_shape == "date":
        return [r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", r"\b\d{4}-\d{2}-\d{2}\b"]
    if value_shape == "integer_count":
        return [r"\b\d+\s?(?:no\.?|nr|qty)?\b"]
    if value_shape == "decimal_measurement":
        return [r"\b\d+(?:\.\d+)?\s?(?:m2|m²|m|kw|kva|sqm|%)\b"]
    if value_shape == "identifier_or_reference" or "model" in field_name:
        return [r"\b[A-Z]{1,6}[-/]?[A-Z0-9]{2,}(?:[-/][A-Z0-9]{2,})*\b"]
    return [r"\b[A-Z][A-Za-z0-9&./-]{2,}\b"]


def find_best_signal(
    pages: list[dict[str, Any]],
    query: dict[str, list[str]],
    value_shape: str,
) -> dict[str, Any]:
    best = empty_signal()
    for page in pages:
        text = str(page.get("extracted_text") or "")
        for excerpt in candidate_excerpts(text):
            signal = score_excerpt(excerpt, query, value_shape)
            if signal["score"] > best["score"]:
                best = {
                    **signal,
                    "excerpt": trim_excerpt(excerpt),
                    "page_number": page.get("page_number"),
                    "source_file": page.get("source_file"),
                }
    return best


def score_excerpt(
    excerpt: str,
    query: dict[str, list[str]],
    value_shape: str,
) -> dict[str, Any]:
    lower = excerpt.lower()
    words = set(re.findall(r"[a-z0-9]+", lower))
    component_hits = [
        term for term in query["component_terms"] if term_matches(term, lower, words)
    ]
    attribute_hits = [
        term for term in query["attribute_terms"] if term_matches(term, lower, words)
    ]
    value_hits = [
        pattern
        for pattern in query["value_patterns"]
        if re.search(pattern, excerpt, flags=re.IGNORECASE)
    ]
    component = bool(component_hits) if query["component_terms"] else True
    attribute = bool(attribute_hits)
    value = bool(value_hits) and value_allowed(value_shape, lower, attribute)
    field_name = query.get("field_terms", [""])[0]
    if "model" in field_name and not (
        term_matches("model", lower, words) or term_matches("type", lower, words)
    ):
        value = False
    if "manufacturer" in field_name and not any(
        term_matches(term, lower, words) for term in ["manufacturer", "make", "supplier"]
    ):
        value = False
    score = len(component_hits) * 2 + len(attribute_hits) * 2 + (4 if value else 0)
    local = component and attribute and value
    return {
        "score": score,
        "component_observed": component,
        "attribute_observed": attribute,
        "value_observed": value,
        "local_association": local,
        "component_hits": component_hits,
        "attribute_hits": attribute_hits,
    }


def value_allowed(value_shape: str, lower_excerpt: str, attribute: bool) -> bool:
    if value_shape in {"identifier_or_reference", "integer_count", "decimal_measurement"}:
        return attribute
    if "manufacturer" in lower_excerpt:
        return any(term in lower_excerpt for term in ["manufacturer", "make", "supplier"])
    return True


def term_matches(term: str, lower_excerpt: str, words: set[str]) -> bool:
    if not term.strip():
        return False
    if re.fullmatch(r"[a-z0-9]+", term):
        return term in words or f"{term}s" in words
    return term in lower_excerpt


def best_alternate_source(
    proposed_source: str,
    source_signal: dict[str, Any],
    all_signals: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    best_source = None
    best_signal: dict[str, Any] | None = None
    for source, signal in all_signals.items():
        if source == proposed_source:
            continue
        if not signal["local_association"]:
            continue
        if signal["score"] >= source_signal["score"] + 3:
            if best_signal is None or signal["score"] > best_signal["score"]:
                best_source = source
                best_signal = signal
    return best_source, best_signal


def candidate_excerpts(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    excerpts: list[str] = []
    for index, _line in enumerate(lines):
        window = " ".join(lines[max(0, index - 1) : index + 2])
        excerpts.append(window)
    if not excerpts and text.strip():
        excerpts.append(text.strip()[:600])
    return excerpts


def load_cached_pages(page_cache_root: Path) -> dict[str, list[dict[str, Any]]]:
    pages_by_source: dict[str, list[dict[str, Any]]] = {}
    if not page_cache_root.exists():
        return pages_by_source
    for path in sorted(page_cache_root.rglob("page_*.json")):
        data = _read_json(path)
        source_file = Path(str(data.get("source_path") or data.get("source_id"))).name
        row = {
            "source_file": source_file,
            "page_number": data.get("page_number"),
            "extracted_text": data.get("extracted_text") or "",
            "cache_path": str(path),
        }
        pages_by_source.setdefault(source_file, []).append(row)
    for pages in pages_by_source.values():
        pages.sort(key=lambda item: int(item.get("page_number") or 0))
    return pages_by_source


def dictionary_conflict(target: dict[str, Any]) -> bool:
    field_name = str(target.get("field_name", "")).lower()
    text = f"{field_name} {target.get('definition', '')}".lower()
    if str(target.get("unit") or "") and str(target.get("value_shape")) in {
        "short_text",
        "descriptive_text",
    }:
        return True
    if field_name.endswith("_unit") and str(target.get("datatype")) in {"decimal", "integer"}:
        return True
    return any(term in text for term in ["unclear", "tbc", "maps from"])


def build_metrics(
    *,
    inputs: dict[str, Any],
    original_audit: list[dict[str, Any]],
    replacement_audit: list[dict[str, Any]],
    corrected: list[dict[str, Any]],
    status: str,
    target_count: int,
) -> dict[str, Any]:
    original_counts = Counter(str(item["audit_classification"]) for item in original_audit)
    corrected_counts = Counter(str(item["audit_classification"]) for item in corrected)
    return {
        "original_selected_count": len(original_audit),
        "original_audit_by_classification": dict(original_counts),
        "replacement_candidates_audited": len(replacement_audit),
        "replacements_accepted": sum(
            1 for item in corrected if item.get("audit_stage") == "replacement_candidate"
        ),
        "corrected_selected_count": len(corrected),
        "target_count_requested": target_count,
        "additional_ready_targets_needed": max(0, target_count - len(corrected)),
        "overall_status": status,
        "batch_v2_extraction_approved": status != "blocked",
        "corrected_by_classification": dict(corrected_counts),
        "corrected_by_domain": dict(Counter(str(item["domain"]) for item in corrected)),
        "corrected_by_value_shape": dict(Counter(str(item["value_shape"]) for item in corrected)),
        "corrected_by_source": dict(Counter(str(item["audit_source_file"]) for item in corrected)),
        "part1_target_count": sum(
            1
            for item in original_audit
            if item.get("proposed_source_file") == "Building Manual - Part 1 General.pdf"
        ),
        "model_manufacturer_reference_count": len(
            [item for item in original_audit if model_manufacturer_reference(item)]
        ),
        "cache_location": inputs["page_cache_root"],
        "cached_source_count": len(inputs["cached_pages"]),
        "cached_page_count": sum(len(pages) for pages in inputs["cached_pages"].values()),
    }


def model_manufacturer_summary(original_audit: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [item for item in original_audit if model_manufacturer_reference(item)]
    counts = Counter(str(item["audit_classification"]) for item in rows)
    unsupported = sum(
        counts[item]
        for item in [
            "attribute_not_observed",
            "requires_additional_cached_pages",
            "dictionary_clarification",
        ]
    )
    return {
        "target_count": len(rows),
        "execution_ready": counts["execution_ready"],
        "ready_with_evidence_caveat": counts["ready_with_evidence_caveat"],
        "component_only": counts["component_only_risk"],
        "wrong_source": counts["wrong_source_mapping"],
        "unsupported": unsupported,
        "classification_counts": dict(counts),
        "targets": rows,
    }


def part1_summary(original_audit: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        item
        for item in original_audit
        if item.get("proposed_source_file") == "Building Manual - Part 1 General.pdf"
    ]
    return {
        "part1_mapped_count": len(rows),
        "part1_mapping_confirmed": sum(
            1 for item in rows if item["audit_classification"] in PASSING_CLASSES
        ),
        "mapping_corrected_to_another_source": sum(
            1 for item in rows if item["audit_classification"] == "wrong_source_mapping"
        ),
        "component_only": sum(
            1 for item in rows if item["audit_classification"] == "component_only_risk"
        ),
        "attribute_not_observed": sum(
            1 for item in rows if item["audit_classification"] == "attribute_not_observed"
        ),
        "requires_uncached_pages": sum(
            1
            for item in rows
            if item["audit_classification"] == "requires_additional_cached_pages"
        ),
        "visual_only": sum(1 for item in rows if item["audit_classification"] == "defer_visual"),
        "dictionary_clarification": sum(
            1 for item in rows if item["audit_classification"] == "dictionary_clarification"
        ),
        "classification_counts": dict(Counter(str(item["audit_classification"]) for item in rows)),
    }


def model_manufacturer_reference(item: dict[str, Any]) -> bool:
    text = f"{item.get('field_name', '')} {item.get('definition', '')}".lower()
    return any(term in text for term in ["model", "manufacturer", "reference"])


def build_source_plan(corrected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "corrected_selection_rank": item["corrected_selection_rank"],
            "target_id": item["target_id"],
            "field_name": item["field_name"],
            "source_document": item["audit_source_file"],
            "cached_page": item["best_cached_page"],
            "evidence_excerpt": item["best_cached_excerpt"],
            "expected_retrieval_query": item.get("expected_retrieval_query"),
            "expected_extraction_route": item.get("expected_extraction_route"),
            "maximum_recommended_evidence_window": 3
            if item.get("audit_classification") == "execution_ready"
            else 5,
            "readiness_classification": item["audit_classification"],
        }
        for item in corrected
    ]


def build_trace(
    original_audit: list[dict[str, Any]],
    replacement_audit: list[dict[str, Any]],
    corrected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    corrected_ids = {str(item["target_id"]) for item in corrected}
    rows: list[dict[str, Any]] = []
    for item in original_audit + replacement_audit:
        rows.append(
            {
                "target_id": item["target_id"],
                "audit_stage": item["audit_stage"],
                "original_selection_rank": item.get("selection_rank"),
                "audit_classification": item["audit_classification"],
                "retained_in_corrected_selection": str(item["target_id"]) in corrected_ids,
                "rationale": item["audit_rationale"],
                "source": item["audit_source_file"],
                "page": item["best_cached_page"],
            }
        )
    return rows


def build_review(
    original_audit: list[dict[str, Any]],
    replacement_audit: list[dict[str, Any]],
    corrected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    corrected_ids = {str(item["target_id"]) for item in corrected}
    rows: list[dict[str, Any]] = []
    for item in original_audit + replacement_audit[:75]:
        rows.append(
            {
                "target ID": item["target_id"],
                "field name": item["field_name"],
                "domain": item["domain"],
                "value shape": item["value_shape"],
                "proposed source": item["proposed_source_file"],
                "audit source": item["audit_source_file"],
                "component observed": item["component_observed"],
                "attribute observed": item["requested_attribute_observed"],
                "value signal observed": item["plausible_value_signal_observed"],
                "local association": item["local_association_observed"],
                "audit classification": item["audit_classification"],
                "corrected selected": str(item["target_id"]) in corrected_ids,
                "page": item["best_cached_page"],
                "excerpt": item["best_cached_excerpt"],
                "rationale": item["audit_rationale"],
            }
        )
    return rows


def cached_evidence_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_id": item["target_id"],
        "field_name": item["field_name"],
        "source_file": item["best_cached_source"],
        "page": item["best_cached_page"],
        "excerpt": item["best_cached_excerpt"],
        "classification": item["audit_classification"],
    }


def write_readiness_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_outputs = {
        "target_readiness_audit.json": result["target_readiness_audit"],
        "confirmed_execution_ready_targets.json": result["confirmed_execution_ready_targets"],
        "corrected_selected_targets.json": result["corrected_selected_targets"],
        "removed_selected_targets.json": result["removed_selected_targets"],
        "replacement_target_audit.json": result["replacement_target_audit"],
        "source_mapping_corrections.json": result["source_mapping_corrections"],
        "component_only_risks.json": result["component_only_risks"],
        "attribute_not_observed.json": result["attribute_not_observed"],
        "requires_additional_cached_pages.json": result["requires_additional_cached_pages"],
        "deferred_visual_targets.json": result["deferred_visual_targets"],
        "dictionary_clarification_targets.json": result["dictionary_clarification_targets"],
        "model_manufacturer_audit.json": result["model_manufacturer_audit"],
        "part1_concentration_audit.json": result["part1_concentration_audit"],
        "cached_evidence_candidates.json": result["cached_evidence_candidates"],
        "readiness_metrics.json": result["readiness_metrics"],
        "readiness_trace.json": result["readiness_trace"],
        "corrected_batch_v2_source_plan.json": result["corrected_batch_v2_source_plan"],
        "corrected_batch_v2_run_readiness.json": result["corrected_batch_v2_run_readiness"],
    }
    for filename, payload in json_outputs.items():
        _atomic_write_json(output_dir / filename, payload)
    write_csv(
        output_dir / "target_readiness_audit.csv",
        result["target_readiness_audit"],
        audit_fields(),
    )
    write_csv(
        output_dir / "confirmed_execution_ready_targets.csv",
        result["confirmed_execution_ready_targets"],
        audit_fields(),
    )
    write_csv(
        output_dir / "corrected_selected_targets.csv",
        result["corrected_selected_targets"],
        ["corrected_selection_rank", *audit_fields()],
    )
    write_csv(
        output_dir / "batch_v2_readiness_review.csv",
        result["batch_v2_readiness_review"],
        review_fields(),
    )
    (output_dir / "batch_v2_readiness_summary.md").write_text(
        summary_markdown(result), encoding="utf-8"
    )
    (output_dir / "batch_v2_execution_gate.md").write_text(
        gate_markdown(result), encoding="utf-8"
    )


def audit_fields() -> list[str]:
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
        "proposed_source_file",
        "audit_source_file",
        "audit_classification",
        "component_observed",
        "requested_attribute_observed",
        "plausible_value_signal_observed",
        "local_association_observed",
        "best_cached_page",
        "best_cached_excerpt",
        "audit_rationale",
        *SCORE_FIELDS,
        "total_score",
    ]


def review_fields() -> list[str]:
    return [
        "target ID",
        "field name",
        "domain",
        "value shape",
        "proposed source",
        "audit source",
        "component observed",
        "attribute observed",
        "value signal observed",
        "local association",
        "audit classification",
        "corrected selected",
        "page",
        "excerpt",
        "rationale",
    ]


def summary_markdown(result: dict[str, Any]) -> str:
    metrics = result["readiness_metrics"]
    part1 = result["part1_concentration_audit"]
    model = result["model_manufacturer_audit"]
    lines = [
        "# Batch V2 Evidence-Readiness Audit",
        "",
        f"- Original selected targets audited: {metrics['original_selected_count']}",
        f"- Corrected selected targets: {metrics['corrected_selected_count']}",
        f"- Overall status: {metrics['overall_status']}",
        f"- Additional ready targets needed: {metrics['additional_ready_targets_needed']}",
        f"- Original audit distribution: {metrics['original_audit_by_classification']}",
        f"- Part 1 mapped targets: {part1['part1_mapped_count']}",
        f"- Part 1 classification counts: {part1['classification_counts']}",
        f"- Model/manufacturer/reference targets: {model['target_count']}",
        f"- Model/manufacturer/reference counts: {model['classification_counts']}",
        "",
        "This audit used only existing cached parsed pages and metadata. It did not parse pages, "
        "retrieve new evidence, run extraction, or call hosted models.",
    ]
    return "\n".join(lines) + "\n"


def gate_markdown(result: dict[str, Any]) -> str:
    readiness = result["corrected_batch_v2_run_readiness"]
    approved = readiness["batch_v2_extraction_approved"]
    lines = [
        "# Batch V2 Execution Gate",
        "",
        f"- Status: {readiness['overall_status']}",
        f"- Extraction approved: {approved}",
        f"- Corrected selected targets: {readiness['corrected_selected_count']}",
        f"- Additional ready targets needed: {readiness['additional_ready_targets_needed']}",
    ]
    if not approved:
        lines.append("- Do not run Batch V2 extraction until evidence readiness improves.")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def empty_signal() -> dict[str, Any]:
    return {
        "score": 0,
        "component_observed": False,
        "attribute_observed": False,
        "value_observed": False,
        "local_association": False,
        "component_hits": [],
        "attribute_hits": [],
        "excerpt": "",
        "page_number": None,
        "source_file": None,
    }


def trim_excerpt(text: str, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


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
