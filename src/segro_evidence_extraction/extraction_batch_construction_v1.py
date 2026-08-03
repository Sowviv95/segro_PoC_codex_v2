"""Evidence-backed extraction batch construction from validated cached evidence."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from segro_evidence_extraction.evidence_gap_refinement_v1 import DEFAULT_EVIDENCE_GAP_OUTPUT_DIR
from segro_evidence_extraction.models.common import ProvenanceRef
from segro_evidence_extraction.models.evidence_bundle import (
    EvidenceBundle,
    RetrievalScore,
    TextEvidenceUnit,
)
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.source_coverage_pageindex import (
    DEFAULT_PAGE_CACHE_ROOT,
    load_canonical_cached_pages,
    write_csv,
)
from segro_evidence_extraction.vertical_slice import (
    EvidenceSpan,
    RetrievalScoreBreakdown,
    RetrievedEvidence,
    _atomic_write_json,
)

DEFAULT_NORMALIZED_TARGETS_JSONL = Path(
    "output/sprint2_dictionary_validation/normalized_targets.jsonl"
)
DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR = Path(
    "output/enfield_unit1_extraction_batch_construction_v1"
)

NON_VISUAL_ROUTES = {"text", "table", "certificate", "drawing_text"}
FAMILY_ORDER = [
    "commissioning_results",
    "statutory_compliance",
    "installation_details",
    "dates_certificates",
    "identifiers_references",
    "manufacturer_model",
    "dimensions_capacities",
    "materials_finishes",
    "counts",
    "component_description",
]
SECTION_FILES = [
    ("commissioning_evidence_sections.json", "commissioning_results"),
    ("statutory_evidence_sections.json", "statutory_compliance"),
    ("installation_evidence_sections.json", "installation_details"),
]


def run_extraction_batch_construction_v1(
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_GAP_OUTPUT_DIR,
    dictionary_jsonl: Path = DEFAULT_NORMALIZED_TARGETS_JSONL,
    cache_root: Path = DEFAULT_PAGE_CACHE_ROOT,
    output_dir: Path = DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR,
    max_targets: int = 30,
) -> dict[str, Any]:
    """Build a small executable batch from observed value-bearing evidence only."""

    started = time.perf_counter()
    max_targets = min(max_targets, 30)
    targets = load_targets(dictionary_jsonl)
    cached_pages = load_canonical_cached_pages(cache_root)
    observed = build_observed_value_evidence(evidence_dir, cached_pages)
    mappings = build_evidence_to_target_mappings(observed, targets)
    support_checks = check_attribute_support(mappings)
    selected, rejected, visual = select_extraction_targets(
        support_checks, max_targets=max_targets
    )
    evidence_payloads = build_canonical_evidence_payloads(selected)
    requests = build_bounded_extraction_requests(selected, evidence_payloads)
    validation_plan = build_normalization_validation_plan(selected)
    metrics = build_batch_metrics(
        observed=observed,
        mappings=mappings,
        support_checks=support_checks,
        selected=selected,
        rejected=rejected,
        visual=visual,
        requests=requests,
        max_targets=max_targets,
        runtime_ms=(time.perf_counter() - started) * 1000,
    )
    result = {
        "observed_value_evidence": observed,
        "evidence_to_target_mappings": mappings,
        "attribute_support_checks": support_checks,
        "selected_extraction_targets": selected,
        "selected_targets_by_source": group_counts(selected, "source_filename"),
        "selected_targets_by_family": group_counts(selected, "evidence_family"),
        "selected_targets_by_route": group_counts(selected, "route"),
        "bounded_extraction_requests": requests,
        "canonical_evidence_payloads": evidence_payloads,
        "normalization_validation_plan": validation_plan,
        "deferred_visual_candidates": visual,
        "rejected_mapped_candidates": rejected,
        "batch_metrics": metrics,
        "batch_trace": build_batch_trace(observed, mappings, support_checks, selected, rejected),
        "batch_review": build_batch_review(selected),
    }
    write_batch_outputs(result, output_dir)
    return result


def load_targets(path: Path) -> list[TargetSpecification]:
    return [
        TargetSpecification.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_observed_value_evidence(
    evidence_dir: Path, cached_pages: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    text_by_key = cached_page_text_lookup(cached_pages)
    cache_by_key = cached_page_path_lookup(cached_pages)
    rows: list[dict[str, Any]] = []
    for filename, family in SECTION_FILES:
        for section in read_json_list(evidence_dir / filename):
            source_id = str(section["source_id"])
            page = int(section["page_start"])
            page_end = int(section.get("page_end") or page)
            text = text_by_key.get((source_id, page), "")
            page_type = first_page_type(section)
            route = infer_route(str(section.get("recommended_route") or "text"), page_type)
            signals = observed_value_signals(text)
            value_bearing = is_value_bearing(signals, family)
            rows.append(
                observed_record(
                    row_number=len(rows) + 1,
                    section=section,
                    family=family,
                    page=page,
                    page_end=page_end,
                    page_type=page_type,
                    route=route,
                    text=text,
                    signals=signals,
                    value_bearing=value_bearing,
                    cache_path=cache_by_key.get((source_id, page), ""),
                    evidence_page_exists=(source_id, page) in text_by_key,
                )
            )
            if family != "commissioning_results" and signals["has_commissioning_event"]:
                derived_value_bearing = is_value_bearing(signals, "commissioning_results")
                rows.append(
                    observed_record(
                        row_number=len(rows) + 1,
                        section=section,
                        family="commissioning_results",
                        page=page,
                        page_end=page_end,
                        page_type=page_type,
                        route=route,
                        text=text,
                        signals=signals,
                        value_bearing=derived_value_bearing,
                        cache_path=cache_by_key.get((source_id, page), ""),
                        evidence_page_exists=(source_id, page) in text_by_key,
                    )
                )
    return sorted(rows, key=evidence_sort_key)


def observed_record(
    *,
    row_number: int,
    section: dict[str, Any],
    family: str,
    page: int,
    page_end: int,
    page_type: str,
    route: str,
    text: str,
    signals: dict[str, Any],
    value_bearing: bool,
    cache_path: str,
    evidence_page_exists: bool,
) -> dict[str, Any]:
    return {
        "evidence_id": f"obs_{row_number:04d}",
        "source_id": str(section["source_id"]),
        "source_filename": str(section.get("source_filename") or ""),
        "page_start": page,
        "page_end": page_end,
        "page_type": page_type,
        "route": route,
        "section": str(section.get("section") or ""),
        "evidence_family": family,
        "component_system_identity": infer_component_identity(text),
        "observed_attributes": sorted(infer_observed_attributes(text, family)),
        "value_bearing_text": bounded_excerpt(text),
        "text_character_count": len(text),
        "cache_path": cache_path,
        "evidence_page_exists": evidence_page_exists,
        "asset_applicability": asset_applicability(text),
        "value_signals": signals,
        "value_bearing": value_bearing,
        "value_bearing_reason": value_bearing_reason(signals, value_bearing),
    }


def cached_page_text_lookup(
    cached_pages: dict[str, list[dict[str, Any]]]
) -> dict[tuple[str, int], str]:
    return {
        (source_id, int(page["page_number"])): str(page.get("extracted_text") or "")
        for source_id, pages in cached_pages.items()
        for page in pages
    }


def cached_page_path_lookup(
    cached_pages: dict[str, list[dict[str, Any]]]
) -> dict[tuple[str, int], str]:
    return {
        (source_id, int(page["page_number"])): str(page.get("cache_path") or "")
        for source_id, pages in cached_pages.items()
        for page in pages
    }


def observed_value_signals(text: str) -> dict[str, Any]:
    lower = normalize_text(text)
    return {
        "has_date": bool(re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", lower)),
        "has_reference": has_any(
            lower, ["ref", "reference", "certificate no", "cert no", "certificate number"]
        ),
        "has_number": bool(
            re.search(r"\b\d+(?:\.\d+)?\s?(mm|m2|m3|kw|kwp|m3/s|l/s|%)?\b", lower)
        ),
        "has_result": has_any(
            lower,
            [
                "pass",
                "passed",
                "satisfactory",
                "complete",
                "completed",
                "complies",
                "approved",
                "commissioned",
            ],
        ),
        "has_commissioning_event": has_any(
            lower, ["commissioning certificate", "certificate of commissioning", "commissioned on"]
        ),
        "has_statutory_event": has_any(
            lower,
            [
                "planning granted",
                "approved",
                "certificate of conformity",
                "installation certificate",
                "certificate number",
                "certificate of installation",
            ],
        ),
        "has_identity": component_keys(lower) != {"asset"},
        "has_attribute": bool(infer_observed_attributes(text, "")),
        "heading_only": len(lower) < 160,
        "maintenance_only": is_maintenance_only(lower),
        "visual_only": False,
    }


def is_value_bearing(signals: dict[str, Any], family: str) -> bool:
    if signals["heading_only"] or signals["maintenance_only"]:
        return False
    if family == "commissioning_results":
        return bool(
            signals["has_identity"]
            and signals["has_attribute"]
            and signals["has_commissioning_event"]
            and (signals["has_date"] or signals["has_reference"])
            and signals["has_result"]
        )
    if family == "statutory_compliance":
        return bool(
            signals["has_identity"]
            and signals["has_statutory_event"]
            and (signals["has_date"] or signals["has_reference"])
            and signals["has_result"]
        )
    return bool(signals["has_identity"] and signals["has_attribute"])


def build_evidence_to_target_mappings(
    evidence_rows: list[dict[str, Any]], targets: list[TargetSpecification]
) -> list[dict[str, Any]]:
    target_index = target_lookup(targets)
    rows: list[dict[str, Any]] = []
    for evidence in evidence_rows:
        if not evidence["value_bearing"]:
            continue
        seen_targets: set[str] = set()
        for key in mapping_keys_for_evidence(evidence):
            for target in target_index.get(key, []):
                if target.target_row_id in seen_targets:
                    continue
                seen_targets.add(target.target_row_id)
                rows.append(mapping_row(evidence, target, key))
    return sorted(rows, key=mapping_sort_key)


def target_lookup(targets: list[TargetSpecification]) -> dict[str, list[TargetSpecification]]:
    lookup: dict[str, list[TargetSpecification]] = {}
    for target in targets:
        field = target.expected_field.lower()
        attributes = attribute_keys(field)
        if not attributes:
            continue
        for component in target_component_keys(field):
            for attribute in attributes:
                lookup.setdefault(f"{component}:{attribute}", []).append(target)
    return lookup


def mapping_keys_for_evidence(evidence: dict[str, Any]) -> list[str]:
    source_text = f"{evidence['component_system_identity']} {evidence['value_bearing_text']}"
    return sorted(
        {
            f"{component}:{attribute}"
            for component in component_keys(source_text)
            for attribute in evidence["observed_attributes"]
        }
    )


def mapping_row(evidence: dict[str, Any], target: TargetSpecification, key: str) -> dict[str, Any]:
    component, attribute = key.split(":", 1)
    return {
        "mapping_id": f"map_{evidence['evidence_id']}_{target.target_row_id}",
        "evidence_id": evidence["evidence_id"],
        "target_id": target.target_row_id,
        "requirement_id": target.requirement_id,
        "requirement": target.requirement_text,
        "field": target.expected_field,
        "expected_data_type": str(target.expected_data_type),
        "unit": target.unit,
        "component_system_identity": component,
        "requested_attribute": attribute,
        "source_id": evidence["source_id"],
        "source_filename": evidence["source_filename"],
        "page_start": evidence["page_start"],
        "page_end": evidence["page_end"],
        "page_type": evidence["page_type"],
        "route": evidence["route"],
        "section": evidence["section"],
        "evidence_family": evidence["evidence_family"],
        "value_bearing_text": evidence["value_bearing_text"],
        "asset_applicability": evidence["asset_applicability"],
        "cache_path": evidence["cache_path"],
        "evidence_page_exists": evidence["evidence_page_exists"],
        "dictionary_target": target.model_dump(mode="json"),
    }


def check_attribute_support(mappings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mapping in mappings:
        text = normalize_text(mapping["value_bearing_text"])
        status, reason = support_status(mapping, text)
        rows.append(
            {
                **mapping,
                "attribute_support_status": status,
                "attribute_support_reason": reason,
                "executable_route": str(mapping["route"]) in NON_VISUAL_ROUTES,
                "dictionary_ambiguity": dictionary_ambiguous(
                    str(mapping["field"]), str(mapping["requested_attribute"])
                ),
            }
        )
    return rows


def support_status(mapping: dict[str, Any], text: str) -> tuple[str, str]:
    route = str(mapping["route"])
    attribute = str(mapping["requested_attribute"])
    component = str(mapping["component_system_identity"])
    if route == "visual":
        return "defer_visual", "visual-only evidence excluded"
    if not mapping.get("evidence_page_exists"):
        return "rejected_missing_cache", "evidence reference is not present in cache"
    if len(text) < 160:
        return "rejected_heading_only", "evidence is only a heading or sparse label"
    if "maintenance" in text and "certificate" not in text and "commission" not in text:
        return "rejected_maintenance_only", "maintenance-only context"
    if is_index_or_contents_page(text):
        return "rejected_heading_only", "contents or index entry only"
    if is_generic_regulatory_guidance(text):
        return "rejected_wrong_event", "generic regulatory guidance, not completed asset evidence"
    if str(mapping["field"]).startswith("pv_certification") and "certificate" not in text:
        return "rejected_wrong_event", "PV certification target requires certificate evidence"
    if is_unrelated_product_datasheet(mapping, text):
        return "rejected_wrong_component", "generic product datasheet for another component family"
    if not component_supported_by_text(component, text):
        return "rejected_wrong_component", "evidence belongs to another system or event"
    if attribute in {"date", "installation_date", "certificate_issue_date"} and not re.search(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text
    ):
        return "rejected_attribute_absent", "date requested but no date-like value present"
    if attribute in {"manufacturer", "model_name", "model_number"} and not has_any(
        text,
        [
            "manufacturer",
            "model",
            "supplier",
            "ideal standard",
            "clymac",
            "cp electronics",
            "apex",
            "rolec",
        ],
    ):
        return "rejected_attribute_absent", "manufacturer/model requested but absent"
    if attribute in {"model_name", "model_number"} and not has_any(
        text, ["model", "system type", "type:"]
    ):
        return "rejected_attribute_absent", "model requested but no model label present"
    if attribute == "manufacturer" and not manufacturer_signal_present(component, text):
        return (
            "rejected_attribute_absent",
            "manufacturer requested but only contractor/context found",
        )
    if attribute == "count" and not (
        has_any(text, ["no.", "number", "quantity", "qty"])
        or bool(re.search(r"\b\d+\s*no\b", text))
    ):
        return "rejected_attribute_absent", "count requested but no count-like value present"
    if attribute == "capacity" and not has_any(
        text, ["capacity", "rating", "kw", "kwp", "l/s", "m3/s"]
    ):
        return "rejected_attribute_absent", "capacity requested but no capacity-like value present"
    if attribute == "dimensions" and not has_any(
        text,
        ["length", "width", "diameter", "mm"],
    ):
        return (
            "rejected_attribute_absent",
            "dimension requested but no dimension-like value present",
        )
    if attribute == "material_type" and not has_any(text, ["material", "finish", "cladding"]):
        return "rejected_attribute_absent", "material requested but material signal absent"
    if attribute in {"component_description", "description"} and not has_any(
        text, ["material", "finish", "type", "description", "system", "cladding", "roof", "wall"]
    ):
        return "rejected_attribute_absent", "description/material evidence absent"
    if attribute in {"certificate_available", "certificate_type", "certificate_name_or_number"}:
        if "certificate" not in text:
            return "rejected_attribute_absent", "certificate attribute requested but absent"
    if attribute == "installation_type" and not has_any(
        text, ["installed", "installation", "mounted", "fixed", "configuration", "type"]
    ):
        return "rejected_attribute_absent", "installation detail requested but absent"
    return "supported", "component and requested attribute co-occur with value-bearing context"


def select_extraction_targets(
    checks: list[dict[str, Any]], *, max_targets: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    visual: list[dict[str, Any]] = []
    used_targets: set[str] = set()
    used_pairs: set[tuple[str, str]] = set()
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for row in sorted(checks, key=selection_sort_key):
        if row["route"] == "visual":
            visual.append({**row, "defer_reason": "visual-only route"})
            continue
        reject_reason = selection_reject_reason(
            row, used_targets, used_pairs, source_counts, family_counts
        )
        if reject_reason:
            rejected.append({**row, "rejection_reason": reject_reason})
            continue
        selected_row = {
            **row,
            "selection_rank": len(selected) + 1,
            "validation_expectations": validation_expectations(row),
        }
        selected.append(selected_row)
        used_targets.add(str(row["target_id"]))
        used_pairs.add((str(row["target_id"]), str(row["evidence_id"])))
        source_counts[str(row["source_id"])] += 1
        family_counts[str(row["evidence_family"])] += 1
        if len(selected) >= max_targets:
            break
    selected_pairs = {(row["target_id"], row["evidence_id"]) for row in selected}
    rejected_keys = {(row["target_id"], row["evidence_id"]) for row in rejected}
    visual_keys = {(row["target_id"], row["evidence_id"]) for row in visual}
    for row in checks:
        key = (row["target_id"], row["evidence_id"])
        if key not in selected_pairs and key not in rejected_keys and key not in visual_keys:
            rejected.append({**row, "rejection_reason": "not selected within balanced cap"})
    return selected, sorted(rejected, key=mapping_sort_key), sorted(visual, key=mapping_sort_key)


def selection_reject_reason(
    row: dict[str, Any],
    used_targets: set[str],
    used_pairs: set[tuple[str, str]],
    source_counts: Counter[str],
    family_counts: Counter[str],
) -> str:
    if row["attribute_support_status"] != "supported":
        return str(row["attribute_support_reason"])
    if not row["executable_route"]:
        return "non-executable route"
    if row["dictionary_ambiguity"]:
        return "dictionary target ambiguous"
    if row["target_id"] in used_targets:
        return "duplicate target"
    if (row["target_id"], row["evidence_id"]) in used_pairs:
        return "duplicate target/evidence pair"
    if source_counts[str(row["source_id"])] >= 8:
        return "source concentration cap"
    if family_counts[str(row["evidence_family"])] >= 8:
        return "family concentration cap"
    return ""


def build_canonical_evidence_payloads(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for row in selected:
        text = str(row["value_bearing_text"])
        span_id = f"span_{row['target_id']}_{row['evidence_id']}"
        payloads.append(
            {
                "payload_id": f"payload_{row['target_id']}",
                "target_id": row["target_id"],
                "span": EvidenceSpan(
                    span_id=span_id,
                    source_id=row["source_id"],
                    source_file=row["source_filename"],
                    page_number=int(row["page_start"]),
                    hierarchy_node_id=f"evidence_section:{row['evidence_id']}",
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    retrieval_rank=1,
                    score=1.0,
                ).model_dump(mode="json"),
                "bounded_text": text,
                "page_range": f"{row['page_start']}-{row['page_end']}",
                "route": row["route"],
                "evidence_family": row["evidence_family"],
            }
        )
    return payloads


def build_bounded_extraction_requests(
    selected: list[dict[str, Any]], payloads: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    payload_by_target = {item["target_id"]: item for item in payloads}
    requests: list[dict[str, Any]] = []
    for row in selected:
        target = TargetSpecification.model_validate(row["dictionary_target"])
        payload = payload_by_target[row["target_id"]]
        bundle = EvidenceBundle(
            bundle_id=f"bundle_{row['target_id']}",
            target_specification=target,
            candidate_document_ids=[row["source_id"]],
            candidate_section_ids=[f"evidence_section:{row['evidence_id']}"],
            selected_pages_or_sheets=[str(row["page_start"])],
            text_evidence_units=[
                TextEvidenceUnit(
                    unit_id=f"text_{row['evidence_id']}",
                    node_id=f"evidence_section:{row['evidence_id']}",
                    text_ref=payload["span"]["span_id"],
                    excerpt=payload["bounded_text"],
                    page_or_sheet=str(row["page_start"]),
                    relevance_reason=row["attribute_support_reason"],
                )
            ],
            component_context={
                "component_system_identity": row["component_system_identity"],
                "requested_attribute": row["requested_attribute"],
                "evidence_family": row["evidence_family"],
            },
            retrieval_scores=[
                RetrievalScore(
                    item_id=str(row["evidence_id"]),
                    score=1.0,
                    strategy="evidence_backed_batch_construction_v1",
                    reason="validated observed value-bearing evidence",
                )
            ],
            retrieval_strategy="observed_value_evidence_to_compatible_dictionary_field",
            selection_reasons=[row["attribute_support_reason"]],
            provenance=[
                ProvenanceRef(
                    source_id=row["source_id"],
                    node_id=f"evidence_section:{row['evidence_id']}",
                    page_or_sheet=str(row["page_start"]),
                    text_ref=payload["span"]["span_id"],
                    notes=row["evidence_family"],
                )
            ],
            warnings=[],
        )
        retrieved = RetrievedEvidence(
            target_row_id=row["target_id"],
            rank=1,
            node_id=f"evidence_section:{row['evidence_id']}",
            source_id=row["source_id"],
            source_file=row["source_filename"],
            page_start=int(row["page_start"]),
            page_end=int(row["page_end"]),
            score=1.0,
            score_components=RetrievalScoreBreakdown(final_score=1.0),
            matched_terms=[row["component_system_identity"], row["requested_attribute"]],
            hierarchy_path=[str(row["section"])],
            excerpt=payload["bounded_text"],
        )
        requests.append(
            {
                "request_id": f"extract_{row['target_id']}",
                "target_id": row["target_id"],
                "target": target.model_dump(mode="json"),
                "evidence_bundle": bundle.model_dump(mode="json"),
                "retrieved_evidence": retrieved.model_dump(mode="json"),
                "route": row["route"],
                "normalization_expectations": normalization_expectations(row),
                "validation_expectations": row["validation_expectations"],
            }
        )
    return requests


def build_normalization_validation_plan(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "target_id": row["target_id"],
            "expected_data_type": row["expected_data_type"],
            "unit": row["unit"],
            "normalization_expectations": normalization_expectations(row),
            "validation_expectations": row["validation_expectations"],
        }
        for row in selected
    ]


def build_batch_metrics(**items: Any) -> dict[str, Any]:
    selected = items["selected"]
    requests = items["requests"]
    target_ids = [row["target_id"] for row in selected]
    pair_ids = [(row["target_id"], row["evidence_id"]) for row in selected]
    route_ok = all(row["route"] in NON_VISUAL_ROUTES for row in selected)
    pages_ok = all(bool(row.get("cache_path")) for row in selected)
    contracts_ok = len(requests) == len(selected)
    return {
        "observed_evidence_records": len(items["observed"]),
        "value_bearing_evidence_records": sum(
            1 for row in items["observed"] if row["value_bearing"]
        ),
        "evidence_to_target_mappings": len(items["mappings"]),
        "attribute_support_checks": len(items["support_checks"]),
        "selected_target_count": len(selected),
        "max_target_cap": items["max_targets"],
        "unique_target_ids": len(target_ids) == len(set(target_ids)),
        "duplicate_target_evidence_pairs": len(pair_ids) != len(set(pair_ids)),
        "all_evidence_pages_exist": pages_ok,
        "all_routes_executable_non_visual": route_ok,
        "selected_targets_by_source": group_counts(selected, "source_filename"),
        "selected_targets_by_family": group_counts(selected, "evidence_family"),
        "selected_targets_by_route": group_counts(selected, "route"),
        "rejected_mapped_candidates": len(items["rejected"]),
        "deferred_visual_candidates": len(items["visual"]),
        "contract_validation_status": "valid" if contracts_ok else "invalid",
        "batch_ready_to_run": bool(selected)
        and len(selected) <= items["max_targets"]
        and len(target_ids) == len(set(target_ids))
        and route_ok
        and pages_ok
        and contracts_ok,
        "extraction_calls_made": 0,
        "llm_calls_made": 0,
        "ocr_calls_made": 0,
        "vlm_calls_made": 0,
        "external_api_calls_made": 0,
        "runtime_ms": round(items["runtime_ms"], 2),
    }


def validation_expectations(row: dict[str, Any]) -> list[str]:
    return [
        "evidence span must come from the specified source and page range",
        f"extracted value must answer requested attribute: {row['requested_attribute']}",
        f"value must be compatible with dictionary field: {row['field']}",
        "component/system identity must remain asset-applicable",
    ]


def normalization_expectations(row: dict[str, Any]) -> dict[str, Any]:
    dtype = str(row["expected_data_type"])
    return {
        "expected_data_type": dtype,
        "date_format": "ISO 8601 if date-like" if dtype == "date" else None,
        "numeric_unit": row["unit"],
        "accepted_values": row["dictionary_target"].get("accepted_values", []),
    }


def build_batch_trace(
    observed: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
    support: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"stage": "load_validated_evidence_sections", "rows": len(observed)},
        {"stage": "observed_value_evidence", "rows": len(observed)},
        {"stage": "evidence_to_target_mappings", "rows": len(mappings)},
        {"stage": "attribute_support_checks", "rows": len(support)},
        {"stage": "selected_extraction_targets", "rows": len(selected)},
        {"stage": "rejected_mapped_candidates", "rows": len(rejected)},
        {"stage": "extraction_model_calls", "rows": 0},
    ]


def build_batch_review(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "selection_rank": row["selection_rank"],
            "target_id": row["target_id"],
            "field": row["field"],
            "source_filename": row["source_filename"],
            "page_range": f"{row['page_start']}-{row['page_end']}",
            "route": row["route"],
            "evidence_family": row["evidence_family"],
            "review_note": "ready for bounded extraction request",
        }
        for row in selected
    ]


def write_batch_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_outputs = {
        "observed_value_evidence.json": result["observed_value_evidence"],
        "evidence_to_target_mappings.json": result["evidence_to_target_mappings"],
        "attribute_support_checks.json": result["attribute_support_checks"],
        "selected_extraction_targets.json": result["selected_extraction_targets"],
        "selected_targets_by_source.json": result["selected_targets_by_source"],
        "selected_targets_by_family.json": result["selected_targets_by_family"],
        "selected_targets_by_route.json": result["selected_targets_by_route"],
        "bounded_extraction_requests.json": result["bounded_extraction_requests"],
        "canonical_evidence_payloads.json": result["canonical_evidence_payloads"],
        "normalization_validation_plan.json": result["normalization_validation_plan"],
        "deferred_visual_candidates.json": result["deferred_visual_candidates"],
        "rejected_mapped_candidates.json": result["rejected_mapped_candidates"],
        "batch_metrics.json": result["batch_metrics"],
        "batch_trace.json": result["batch_trace"],
    }
    for filename, payload in json_outputs.items():
        _atomic_write_json(output_dir / filename, payload)
    for filename, rows in {
        "observed_value_evidence.csv": result["observed_value_evidence"],
        "evidence_to_target_mappings.csv": result["evidence_to_target_mappings"],
        "attribute_support_checks.csv": result["attribute_support_checks"],
        "selected_extraction_targets.csv": result["selected_extraction_targets"],
        "batch_review.csv": result["batch_review"],
    }.items():
        write_csv(output_dir / filename, rows)
    (output_dir / "batch_summary.md").write_text(batch_summary(result), encoding="utf-8")
    (output_dir / "next_extraction_run.md").write_text(
        next_extraction_run(result), encoding="utf-8"
    )


def batch_summary(result: dict[str, Any]) -> str:
    metrics = result["batch_metrics"]
    return (
        "# Evidence-Backed Extraction Batch Construction V1\n\n"
        f"- Evidence sections inspected: {metrics['observed_evidence_records']}\n"
        f"- Value-bearing evidence records: {metrics['value_bearing_evidence_records']}\n"
        f"- Evidence-to-target mappings: {metrics['evidence_to_target_mappings']}\n"
        f"- Selected targets: {metrics['selected_target_count']}\n"
        f"- Ready to run: {metrics['batch_ready_to_run']}\n"
        "- Extraction calls made: 0\n"
    )


def next_extraction_run(result: dict[str, Any]) -> str:
    count = result["batch_metrics"]["selected_target_count"]
    return (
        "# Next Extraction Run\n\n"
        f"- Prepared bounded extraction requests: {count}\n"
        "- Recommended next sprint: run the prepared bounded extraction requests in "
        "`bounded_extraction_requests.json` without repeating target selection.\n"
        "- Existing extraction workflows should be extended to consume "
        "`output/enfield_unit1_extraction_batch_construction_v1/"
        "bounded_extraction_requests.json` directly.\n"
    )


def infer_observed_attributes(text: str, family: str) -> set[str]:
    lower = normalize_text(text)
    attrs: set[str] = set()
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", lower):
        attrs.update({"date", "installation_date", "certificate_issue_date"})
    if has_any(lower, ["certificate no", "cert no", "certificate number", "reference", "ref"]):
        attrs.update({"certificate_name_or_number", "model_number"})
    if has_any(lower, ["certificate", "commissioning certificate"]):
        attrs.update({"certificate_available", "certificate_type", "component_description"})
    if has_any(
        lower,
        ["manufacturer", "supplier", "ideal standard", "clymac", "cp electronics", "apex"],
    ):
        attrs.add("manufacturer")
    if has_any(lower, ["model", "orbit", "contour", "doc m", "system type"]):
        attrs.update({"model_name", "model_number"})
    if has_any(
        lower, ["length", "width", "diameter", "capacity", "kw", "kwp", "m3/s", "mm", "l/s"]
    ):
        attrs.update({"capacity", "dimensions"})
    if has_any(lower, ["no.", "number", "quantity", "qty"]):
        attrs.add("count")
    if has_any(lower, ["material", "finish", "cladding", "roof", "wall", "floor"]):
        attrs.update({"material_type", "description", "component_description"})
    if has_any(lower, ["installed", "installation", "mounted", "fixed", "configuration", "type"]):
        attrs.add("installation_type")
    if family == "statutory_compliance":
        attrs.update({"certificate_available", "certificate_type", "certificate_name_or_number"})
    return attrs


def infer_component_identity(text: str) -> str:
    lower = normalize_text(text)
    checks = [
        ("fire_safety", ["fire detection", "fire alarm", "fire safety"]),
        ("electrical_safety", ["electrical", "niceic"]),
        ("mansafe_roof_horizontal_lifeline", ["horizontal lifeline", "lifeline"]),
        ("internal_luminaire", ["lighting", "luminaire", "cp electronics"]),
        ("pv_other", ["photovoltaic", " pv ", "solar"]),
        ("external_cladding", ["cladding", "wall"]),
        ("roof_construction", ["roof"]),
        ("floor_construction", ["floor slab", "concrete floor", "floor"]),
        ("loading_door_overhead", ["loading door", "sectional door"]),
        ("dock_leveller", ["dock leveller", "dock"]),
        ("water_meter", ["water meter"]),
        ("elec_meter", ["electric meter", "meter"]),
    ]
    for name, terms in checks:
        if has_any(lower, terms):
            return name
    if "certificate" in lower:
        return "certificate"
    return "asset"


def component_keys(value: str) -> set[str]:
    lower = normalize_text(value)
    keys: set[str] = set()
    for component in [
        "fire_safety",
        "electrical_safety",
        "mansafe_roof_horizontal_lifeline",
        "internal_luminaire",
        "lighting_busbar",
        "pv_other",
        "external_cladding",
        "roof_construction",
        "floor_construction",
        "wall_construction",
        "loading_door_overhead",
        "dock_leveller",
        "water_meter",
        "elec_meter",
    ]:
        if component in lower or component.replace("_", " ") in lower:
            keys.add(component)
    if "fire" in lower and ("alarm" in lower or "certificate" in lower):
        keys.add("fire_safety")
    if "electrical" in lower and "certificate" in lower:
        keys.add("electrical_safety")
    if "lifeline" in lower:
        keys.add("mansafe_roof_horizontal_lifeline")
    if "lighting" in lower or "luminaire" in lower or "cp electronics" in lower:
        keys.add("internal_luminaire")
    if "cladding" in lower:
        keys.add("external_cladding")
    if "roof" in lower:
        keys.add("roof_construction")
    if "floor" in lower:
        keys.add("floor_construction")
    return keys or {"asset"}


def target_component_keys(field: str) -> set[str]:
    lower = normalize_text(field)
    prefixes = [
        "fire_safety",
        "electrical_safety",
        "mansafe_roof_horizontal_lifeline",
        "internal_luminaire",
        "lighting_busbar",
        "pv_certification",
        "pv_other",
        "external_cladding",
        "roof_construction",
        "floor_construction",
        "wall_construction",
        "loading_door_overhead",
        "dock_leveller",
        "water_meter",
        "elec_meter",
    ]
    keys: set[str] = set()
    for prefix in prefixes:
        if lower.startswith(prefix):
            keys.add("pv_other" if prefix == "pv_certification" else prefix)
    return keys


def attribute_keys(field: str) -> set[str]:
    lower = field.lower()
    attrs = {
        "certificate_available": ["certificate_available"],
        "certificate_type": ["certificate_type"],
        "certificate_name_or_number": ["certificate_name_or_number"],
        "certificate_issue_date": ["certificate_issue_date", "issue_date"],
        "installation_date": ["installation_date"],
        "installation_type": ["installation_type"],
        "manufacturer": ["manufacturer"],
        "model_name": ["model_name"],
        "model_number": ["model_number"],
        "capacity": ["capacity", "rating"],
        "count": ["count"],
        "component_description": ["component_description", "description"],
        "description": ["description"],
        "material_type": ["material_type"],
        "dimensions": ["width", "height", "length", "diameter", "area"],
    }
    return {key for key, terms in attrs.items() if any(term in lower for term in terms)}


def component_supported_by_text(component: str, text: str) -> bool:
    terms_by_component = {
        "fire_safety": ["fire", "alarm", "detection"],
        "electrical_safety": ["electrical installation certificate", "niceic"],
        "mansafe_roof_horizontal_lifeline": ["lifeline", "mansafe", "fall protection"],
        "internal_luminaire": ["lighting", "luminaire", "cp electronics"],
        "lighting_busbar": ["lighting", "busbar"],
        "pv_other": ["photovoltaic", "solar", " pv "],
        "external_cladding": ["cladding"],
        "roof_construction": ["roof construction", "profiled metal clad roof", "rooflights"],
        "floor_construction": ["floor slab", "concrete floor"],
        "wall_construction": ["wall", "cladding"],
        "loading_door_overhead": ["loading door", "sectional door"],
        "dock_leveller": ["dock leveller", "dock"],
        "water_meter": ["water meter"],
        "elec_meter": ["electric meter", "meter"],
    }
    return has_any(text, terms_by_component.get(component, [component.replace("_", " ")]))


def dictionary_ambiguous(field: str, attribute: str) -> bool:
    lower = field.lower()
    if attribute in {"capacity", "manufacturer", "model_name", "model_number"} and "dock_" in lower:
        return True
    if "other" in lower and attribute in {"description", "component_description"}:
        return True
    return False


def asset_applicability(text: str) -> str:
    lower = normalize_text(text)
    if "unit 1" in lower or "enfield" in lower or "east duck lees" in lower:
        return "asset_applicable"
    return "asset_applicability_not_explicit"


def first_page_type(section: dict[str, Any]) -> str:
    primary = section.get("primary_page_types")
    if isinstance(primary, dict) and primary:
        return str(next(iter(primary)))
    return "unknown"


def normalize_route(route: str) -> str:
    if route in {"certificate", "table", "visual", "drawing_text", "text"}:
        return route
    if route == "drawing text":
        return "drawing_text"
    return "text"


def infer_route(route: str, page_type: str) -> str:
    if page_type == "certificate":
        return "certificate"
    if page_type in {"structured_table", "schedule"}:
        return "table"
    return normalize_route(route)


def value_bearing_reason(signals: dict[str, Any], value_bearing: bool) -> str:
    if value_bearing:
        return "explicit value-bearing signals"
    if signals["heading_only"]:
        return "heading or sparse label only"
    if signals["maintenance_only"]:
        return "maintenance-only context"
    return "component-only or requested attribute absent"


def bounded_excerpt(text: str, max_chars: int = 1200) -> str:
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def evidence_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    family = str(item["evidence_family"])
    return (
        FAMILY_ORDER.index(family) if family in FAMILY_ORDER else 99,
        str(item["source_filename"]),
        int(item["page_start"]),
        str(item["evidence_id"]),
    )


def mapping_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    family = str(item["evidence_family"])
    return (
        FAMILY_ORDER.index(family) if family in FAMILY_ORDER else 99,
        str(item["source_filename"]),
        int(item["page_start"]),
        str(item["target_id"]),
    )


def selection_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    route_rank = {"certificate": 0, "table": 1, "text": 2, "drawing_text": 3}
    attr_rank = {
        "certificate_name_or_number": 0,
        "certificate_issue_date": 1,
        "certificate_type": 2,
        "certificate_available": 3,
        "installation_type": 4,
        "manufacturer": 5,
        "model_name": 6,
        "capacity": 7,
        "dimensions": 8,
        "count": 9,
        "component_description": 10,
    }
    return (
        FAMILY_ORDER.index(str(row["evidence_family"]))
        if str(row["evidence_family"]) in FAMILY_ORDER
        else 99,
        route_rank.get(str(row["route"]), 9),
        attr_rank.get(str(row["requested_attribute"]), 99),
        str(row["source_filename"]),
        int(row["page_start"]),
        str(row["target_id"]),
    )


def group_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(item) for item in payload] if isinstance(payload, list) else []


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def is_maintenance_only(lower: str) -> bool:
    early = lower[:350]
    if "regular maintenance" in early or lower.startswith("building and office maintenance"):
        return True
    return has_any(lower, ["coshh", "safety data sheet"]) and not has_any(
        lower, ["certificate", "commission", "installed", "approved", "test"]
    )


def is_index_or_contents_page(text: str) -> bool:
    return has_any(
        text,
        [
            "certificates from the following companies are included",
            "building manual d - commissioning / test certificates",
            "drawing 30803-",
        ],
    )


def is_generic_regulatory_guidance(text: str) -> bool:
    return "building regulations requirements" in text and has_any(
        text, ["must satisfy", "will also meet", "approved document"]
    )


def is_unrelated_product_datasheet(mapping: dict[str, Any], text: str) -> bool:
    component = str(mapping["component_system_identity"])
    generic_product_page = "product data sheet" in text or "product installation guide" in text
    if not generic_product_page:
        return False
    return component in {
        "external_cladding",
        "roof_construction",
        "floor_construction",
        "wall_construction",
    }


def manufacturer_signal_present(component: str, text: str) -> bool:
    if "access & cleaning strategy" in text:
        return False
    if "manufacturer" in text or "supplier" in text:
        return True
    known_vendor_terms = {
        "internal_luminaire": ["cp electronics"],
        "fire_safety": ["clymac"],
        "mansafe_roof_horizontal_lifeline": ["apex"],
        "pv_other": ["rolec"],
    }
    return has_any(text, known_vendor_terms.get(component, []))
