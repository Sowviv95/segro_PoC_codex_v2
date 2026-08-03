"""High-value source-first evidence section expansion V2."""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from segro_evidence_extraction.index_guided_section_expansion import (
    DEFAULT_INDEX_EXPANSION_OUTPUT_DIR,
    _read_json_list,
    normalize_heading,
    route_for_section,
    score_sections,
    source_family_key,
)
from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry
from segro_evidence_extraction.parsing.batch_worker import DEFAULT_MAX_BATCH_PAGES
from segro_evidence_extraction.parsing.service import load_source_registry
from segro_evidence_extraction.source_coverage_pageindex import (
    DEFAULT_HIERARCHY_PATH,
    DEFAULT_PAGE_CACHE_ROOT,
    DEFAULT_SOURCE_MANIFEST,
    build_enriched_pageindex,
    build_evidence_section_map,
    build_page_coverage_map,
    build_source_family_support_map,
    build_source_inventory,
    cache_coverage_snapshot,
    classify_cached_pages,
    determine_pdf_page_counts,
    execute_parse_batches,
    infer_domains,
    infer_target_families,
    load_canonical_cached_pages,
    load_hierarchy_nodes,
    split_contiguous_pages,
    validate_parse_batches,
    write_csv,
)
from segro_evidence_extraction.vertical_slice import _atomic_write_json

DEFAULT_HIGH_VALUE_EXPANSION_OUTPUT_DIR = Path(
    "output/enfield_unit1_high_value_section_expansion_v2"
)
HIGH_VALUE_V2_VERSION = "high-value-section-expansion-v2"

FOCUS_PART_CAPS = {"part 2": 55, "part 3": 55, "part 4": 20, "part 5": 15, "part 6": 45}
WEAK_EVIDENCE_FAMILIES = [
    "manufacturer_model",
    "equipment_specification",
    "component_description",
    "dimensions_capacities",
    "counts",
    "installation_details",
    "materials_finishes",
    "dates_certificates",
    "identifiers_references",
    "statutory_compliance",
    "commissioning_results",
]


def run_high_value_section_expansion_v2(
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    cache_root: Path = DEFAULT_PAGE_CACHE_ROOT,
    index_v1_output_dir: Path = DEFAULT_INDEX_EXPANSION_OUTPUT_DIR,
    hierarchy_path: Path = DEFAULT_HIERARCHY_PATH,
    output_dir: Path = DEFAULT_HIGH_VALUE_EXPANSION_OUTPUT_DIR,
    max_new_pages: int = 160,
    dry_run: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    sources = sorted(load_source_registry(source_manifest), key=_source_sort_key)
    pdf_sources = [source for source in sources if source.file_type == FileType.PDF]
    page_counts = determine_pdf_page_counts(pdf_sources)
    hierarchy_nodes = load_hierarchy_nodes(hierarchy_path)
    cached_before = load_canonical_cached_pages(cache_root)
    inventory_before = build_source_inventory(
        sources=sources,
        page_counts=page_counts,
        cached_pages=cached_before,
        hierarchy_nodes=hierarchy_nodes,
        parser_fingerprint={},
    )
    prior_classifications = _read_json_list(index_v1_output_dir / "page_classifications_after.json")
    classification_audit = audit_priority_classifications(prior_classifications)
    classification_corrections = correct_priority_classifications(
        prior_classifications=prior_classifications,
        cached_pages=cached_before,
        sources=sources,
    )
    corrected_before = apply_classification_corrections(
        classify_cached_pages(cached_before, sources), classification_corrections
    )
    gap_before = evidence_family_coverage(corrected_before)
    resolved_sections = _read_json_list(index_v1_output_dir / "resolved_section_boundaries.json")
    rescored = rescore_section_candidates(
        resolved_sections=resolved_sections,
        family_gap=gap_before,
        page_counts=page_counts,
        cached_pages=cached_before,
    )
    page_map_before = build_page_coverage_map(pdf_sources, page_counts, cached_before)
    candidates = build_v2_section_candidates(
        rescored_sections=rescored,
        page_map=page_map_before,
        page_counts=page_counts,
    )
    approved = approve_v2_parse_plan(
        candidates,
        max_new_pages=max_new_pages,
        per_source_caps=source_caps(max_new_pages),
    )
    batches = split_v2_batches(approved)
    validate_parse_batches(batches, cached_before)
    parse_report = execute_parse_batches(
        batches=batches,
        sources=sources,
        cache_root=cache_root,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    cached_after = load_canonical_cached_pages(cache_root)
    newly_cached = diff_cached_pages(cached_before, cached_after, sources)
    classifications_after = apply_classification_corrections(
        classify_cached_pages(cached_after, sources),
        correct_priority_classifications(
            prior_classifications=classify_cached_pages(cached_after, sources),
            cached_pages=cached_after,
            sources=sources,
        ),
    )
    gap_after = evidence_family_coverage(classifications_after)
    section_inputs = merge_section_inventory_inputs(rescored, candidates)
    section_inventory = section_inventory_after(section_inputs, classifications_after, sources)
    evidence_map = build_evidence_section_map(section_inventory)
    support = build_source_family_support_map(evidence_map)
    inventory_after = build_source_inventory(
        sources=sources,
        page_counts=page_counts,
        cached_pages=cached_after,
        hierarchy_nodes=hierarchy_nodes,
        parser_fingerprint={},
    )
    remaining = remaining_high_priority_sections(candidates, approved)
    metrics = build_v2_metrics(
        sources=sources,
        page_counts=page_counts,
        cached_before=cached_before,
        cached_after=cached_after,
        classification_audit=classification_audit,
        classification_corrections=classification_corrections,
        gap_before=gap_before,
        gap_after=gap_after,
        rescored=rescored,
        approved=approved,
        batches=batches,
        parse_report=parse_report,
        classifications_after=classifications_after,
        section_inventory=section_inventory,
        remaining=remaining,
        runtime_ms=(time.perf_counter() - started) * 1000,
    )
    result = {
        "classification_audit_before": classification_audit,
        "classification_corrections": classification_corrections,
        "evidence_family_gap_analysis": gap_before,
        "rescored_section_candidates": rescored,
        "approved_section_parse_plan": approved,
        "merged_parse_batches": batches,
        "parse_execution_report": parse_report,
        "newly_cached_pages": newly_cached,
        "cache_coverage_before": cache_coverage_snapshot(inventory_before),
        "cache_coverage_after": cache_coverage_snapshot(inventory_after),
        "page_classifications_after": classifications_after,
        "section_inventory_after": section_inventory,
        "evidence_section_map_after": evidence_map,
        "enriched_pageindex": build_enriched_pageindex(hierarchy_nodes, section_inventory),
        "source_family_support_map_after": support,
        "evidence_family_coverage_before": gap_before,
        "evidence_family_coverage_after": gap_after,
        "table_route_pages": _route_pages(classifications_after, {"table"}),
        "certificate_route_pages": [
            row for row in classifications_after if row["primary_page_type"] == "certificate"
        ],
        "drawing_text_route_pages": _route_pages(classifications_after, {"drawing_text"}),
        "visual_route_pages": _route_pages(classifications_after, {"visual"}),
        "low_value_pages": _route_pages(classifications_after, {"deprioritized"}),
        "unknown_pages_remaining": [
            row for row in classifications_after if row["primary_page_type"] == "unknown"
        ],
        "remaining_high_priority_sections": remaining,
        "coverage_metrics": metrics,
        "coverage_trace": build_v2_trace(
            classification_audit, candidates, approved, batches, parse_report
        ),
    }
    write_v2_outputs(result, output_dir)
    return result


def audit_priority_classifications(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audited_types = {
        "separator_or_cover",
        "unknown",
        "maintenance_guidance",
        "low_value_repetitive",
    }
    subset = [row for row in rows if row.get("primary_page_type") in audited_types]
    by_type = Counter(str(row.get("primary_page_type")) for row in subset)
    by_source_type = Counter(
        f"{row.get('source_filename')}::{row.get('primary_page_type')}" for row in subset
    )
    suspected_cover_overclassification = [
        row
        for row in subset
        if row.get("primary_page_type") == "separator_or_cover"
        and (
            int(row.get("text_character_count") or 0) == 0
            or source_family_key(str(row.get("source_filename") or ""))
            in {"part 2", "part 4", "part 6"}
        )
    ]
    return {
        "audited_page_count": len(subset),
        "audited_types": sorted(audited_types),
        "counts_by_type": dict(sorted(by_type.items())),
        "counts_by_source_and_type": dict(sorted(by_source_type.items())),
        "suspected_separator_cover_overclassification": len(suspected_cover_overclassification),
        "finding": (
            "separator_or_cover is overclassified for zero-text or sparse high-value manual pages"
            if suspected_cover_overclassification
            else "separator_or_cover classification appears bounded"
        ),
    }


def correct_priority_classifications(
    *,
    prior_classifications: list[dict[str, Any]],
    cached_pages: dict[str, list[dict[str, Any]]],
    sources: list[SourceRegistryEntry],
) -> list[dict[str, Any]]:
    text_lookup = {
        (source_id, int(page["page_number"])): str(page.get("extracted_text") or "")
        for source_id, pages in cached_pages.items()
        for page in pages
    }
    source_names = {source.source_id: Path(source.logical_path).name for source in sources}
    rows: list[dict[str, Any]] = []
    for row in prior_classifications:
        prior_type = str(row.get("primary_page_type") or "")
        if prior_type not in {
            "separator_or_cover",
            "unknown",
            "maintenance_guidance",
            "low_value_repetitive",
        }:
            continue
        source_id = str(row.get("source_id") or "")
        page_number = int(row.get("page_number") or 0)
        source_filename = str(row.get("source_filename") or source_names.get(source_id, ""))
        corrected = classify_page_text_v3(
            text_lookup.get((source_id, page_number), ""),
            source_filename=source_filename,
            page_number=page_number,
            prior_type=prior_type,
        )
        if corrected["primary_page_type"] == prior_type:
            continue
        rows.append(
            {
                "source_id": source_id,
                "source_filename": source_filename,
                "page_number": page_number,
                "prior_primary_page_type": prior_type,
                "primary_page_type": corrected["primary_page_type"],
                "recommended_route": corrected["recommended_route"],
                "secondary_tags": corrected["secondary_tags"],
                "likely_dictionary_domains": infer_domains(
                    source_filename, corrected["normalized_text"]
                ),
                "likely_target_families": corrected["likely_target_families"],
                "evidence_bearing": corrected["evidence_bearing"],
                "rule_signals": corrected["rule_signals"],
                "classification_score": corrected["classification_score"],
                "classification_confidence": corrected["classification_confidence"],
                "classification_reason": corrected["classification_reason"],
                "classification_changed": True,
            }
        )
    return sorted(rows, key=lambda item: (item["source_filename"], item["page_number"]))


def classify_page_text_v3(
    text: str,
    *,
    source_filename: str,
    page_number: int = 0,
    prior_type: str = "",
) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    tokens = re.findall(r"[a-z0-9./:-]+", normalized)
    signals = page_rule_signals_v3(text, source_filename=source_filename, page_number=page_number)
    families: set[str] = set(infer_target_families(source_filename, normalized))
    tags: set[str] = set()
    primary = prior_type or "unknown"
    route = "text"
    score = 0.0
    reason = "retained prior classification"
    part = source_family_key(source_filename)
    if signals["certificate_title_terms"]:
        primary, route, score, reason = (
            "certificate",
            "certificate",
            0.86,
            "certificate title or commissioning/test heading",
        )
        tags.update({"certificate_date_reference", "statutory_compliance"})
        families.update({"dates_certificates", "identifiers_references", "commissioning_results"})
    elif signals["schedule_terms"] and signals["table_label_terms"] >= 2:
        primary, route, score, reason = (
            "schedule",
            "table",
            0.82,
            "schedule title with table labels",
        )
        tags.add("equipment_schedule")
        families.update({"manufacturer_model", "equipment_specification"})
    elif signals["element_sheet_terms"]:
        primary, route, score, reason = (
            "narrative",
            "text",
            0.76,
            "element-sheet or product-system heading",
        )
        tags.add("component_specification")
        families.update({"component_description", "materials_finishes"})
    elif signals["drawing_terms"] or (
        not tokens and part in {"part 2", "part 4", "part 6"} and page_number > 5
    ):
        primary = "drawing_text_extractable" if len(tokens) >= 10 else "drawing_visual_required"
        route = "drawing_text" if primary == "drawing_text_extractable" else "visual"
        score = 0.72 if not tokens else 0.8
        reason = "sparse high-value manual page treated as drawing/visual route, not cover"
        tags.update({"layout_dependency", "visual_confirmation_required"})
        families.update({"locations_layout", "component_description"})
    elif signals["equipment_terms"] and not signals["maintenance_dominant"]:
        primary, route, score, reason = (
            "structured_table" if signals["table_like"] else "narrative",
            "table" if signals["table_like"] else "text",
            0.74,
            "equipment/product evidence terms override low-value classification",
        )
        tags.add("component_specification")
        families.update({"manufacturer_model", "equipment_specification"})
    elif signals["maintenance_dominant"]:
        primary, route, score, reason = (
            "maintenance_guidance",
            "deprioritized",
            0.78,
            "maintenance language dominates and no stronger equipment identity signal",
        )
        tags.add("maintenance_only")
        families.add("maintenance_only")
    elif signals["coshh_terms"]:
        primary, route, score, reason = (
            "low_value_repetitive",
            "deprioritized",
            0.8,
            "COSHH or safety-data terminology",
        )
        tags.add("duplicate_or_repeated_content")
    elif prior_type == "separator_or_cover" and signals["cover_terms"] and len(tokens) <= 30:
        primary, route, score, reason = (
            "separator_or_cover",
            "deprioritized",
            0.8,
            "sparse cover/divider terms with no material evidence",
        )
    elif len(tokens) >= 20:
        primary, route, score, reason = (
            "narrative",
            "text",
            0.58,
            "sufficient narrative text density",
        )
    elif not tokens:
        primary, route, score, reason = (
            "unknown",
            "text",
            0.2,
            "zero extracted text without route context",
        )
    return {
        "primary_page_type": primary,
        "recommended_route": route,
        "secondary_tags": sorted(tags),
        "likely_target_families": sorted(families),
        "evidence_bearing": primary
        not in {"separator_or_cover", "low_value_repetitive", "maintenance_guidance", "unknown"},
        "rule_signals": signals,
        "classification_score": round(score, 4),
        "classification_confidence": "high"
        if score >= 0.75
        else "medium"
        if score >= 0.5
        else "low",
        "classification_reason": reason,
        "normalized_text": normalized,
    }


def page_rule_signals_v3(
    text: str, *, source_filename: str = "", page_number: int = 0
) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    tokens = re.findall(r"[a-z0-9./:-]+", normalized)
    numeric_tokens = [token for token in tokens if any(char.isdigit() for char in token)]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "source_part": source_family_key(source_filename),
        "page_number": page_number,
        "char_count": len(normalized),
        "token_count": len(tokens),
        "line_count": len(lines),
        "numeric_token_ratio": round(len(numeric_tokens) / len(tokens), 4) if tokens else 0.0,
        "cover_terms": _term_count(
            normalized, ["building manual", "part ", "appendix", "contents"]
        ),
        "certificate_title_terms": _term_count(
            normalized,
            [
                "certificate",
                "commissioning",
                "test certificate",
                "test report",
                "warranty",
                "guarantee",
            ],
        ),
        "drawing_terms": _term_count(
            normalized,
            [
                "drawing no",
                "revision",
                "rev ",
                "scale",
                "title block",
                "plan",
                "elevation",
                "as built",
            ],
        ),
        "element_sheet_terms": _term_count(
            normalized,
            ["element", "roof", "cladding", "wall", "floor", "door", "dock", "leveller", "curtain"],
        ),
        "schedule_terms": _term_count(
            normalized, ["schedule", "equipment schedule", "points schedule"]
        ),
        "table_label_terms": _term_count(
            normalized, ["item", "description", "manufacturer", "model", "type", "reference", "ref"]
        ),
        "equipment_terms": _term_count(
            normalized,
            [
                "manufacturer",
                "model",
                "equipment",
                "plant",
                "meter",
                "bms",
                "fire alarm",
                "lighting",
            ],
        ),
        "maintenance_terms": _term_count(
            normalized, ["maintenance", "cleaning", "inspection", "operation"]
        ),
        "coshh_terms": _term_count(normalized, ["coshh", "safety data sheet", "material safety"]),
        "table_like": len(numeric_tokens) >= 4
        and _term_count(normalized, ["description", "model", "ref", "item", "qty"]) >= 2,
        "maintenance_dominant": _term_count(normalized, ["maintenance", "cleaning", "inspection"])
        >= 2
        and _term_count(normalized, ["manufacturer", "model", "certificate", "schedule"]) == 0,
    }


def apply_classification_corrections(
    classifications: list[dict[str, Any]], corrections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {(row["source_id"], int(row["page_number"])): row for row in corrections}
    corrected: list[dict[str, Any]] = []
    for row in classifications:
        key = (str(row["source_id"]), int(row["page_number"]))
        if key not in by_key:
            corrected.append(row)
            continue
        update = by_key[key]
        corrected.append(
            {
                **row,
                "primary_page_type": update["primary_page_type"],
                "recommended_route": update["recommended_route"],
                "secondary_tags": update["secondary_tags"],
                "likely_target_families": update["likely_target_families"],
                "evidence_bearing": update["evidence_bearing"],
                "classification_correction_reason": update["classification_reason"],
            }
        )
    return sorted(corrected, key=lambda item: (item["source_filename"], item["page_number"]))


def evidence_family_coverage(classifications: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    sources: dict[str, set[str]] = defaultdict(set)
    for row in classifications:
        if not row.get("evidence_bearing"):
            continue
        source = str(row.get("source_filename") or "")
        for family in row.get("likely_target_families", []):
            family_name = normalize_family(str(family))
            counts[family_name] += 1
            sources[family_name].add(source)
    rows = []
    for family in WEAK_EVIDENCE_FAMILIES:
        rows.append(
            {
                "family": family,
                "evidence_page_count": counts[family],
                "source_count": len(sources[family]),
                "gap_level": "absent"
                if counts[family] == 0
                else "weak"
                if counts[family] < 5
                else "covered",
            }
        )
    return {
        "families": rows,
        "counts": {row["family"]: row["evidence_page_count"] for row in rows},
        "weak_or_absent_families": [
            row["family"] for row in rows if row["gap_level"] in {"absent", "weak"}
        ],
    }


def normalize_family(family: str) -> str:
    aliases = {
        "materials_finishes": "materials_finishes",
        "dates_certificates": "dates_certificates",
        "identifiers_references": "identifiers_references",
        "dimensions_capacities": "dimensions_capacities",
        "counts": "counts",
        "manufacturer_model": "manufacturer_model",
    }
    return aliases.get(family, family)


def rescore_section_candidates(
    *,
    resolved_sections: list[dict[str, Any]],
    family_gap: dict[str, Any],
    page_counts: dict[str, int | None],
    cached_pages: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    base_scored = score_sections(resolved_sections)
    weak = set(family_gap["weak_or_absent_families"])
    cached_numbers = {
        source_id: {int(page["page_number"]) for page in pages}
        for source_id, pages in cached_pages.items()
    }
    rows: list[dict[str, Any]] = []
    for section in base_scored:
        if not section.get("resolved"):
            continue
        filename = str(section.get("source_filename") or "")
        family_key = source_family_key(filename)
        if family_key not in {"part 2", "part 3", "part 4", "part 5", "part 6"}:
            continue
        title = str(section.get("section_title") or "")
        families = infer_v2_families(filename, title)
        score = int(section.get("evidence_value_score") or 0)
        reasons = list(section.get("scoring_reasons") or [])
        for family in families & weak:
            score += 12
            reasons.append(f"gap_{family}+12")
        priority = source_priority_boost(filename, title)
        if priority:
            score += priority
            reasons.append(f"source_priority+{priority}")
        if family_key in {"part 4", "part 5"}:
            score -= 8
            reasons.append("v2_part4_part5_secondary-8")
        adaptive = adaptive_section_window(
            section,
            families=families,
            total_pages=page_counts.get(str(section["source_id"])) or 0,
            cached_pages=cached_numbers.get(str(section["source_id"]), set()),
        )
        if not adaptive:
            continue
        rows.append(
            {
                **section,
                "v2_evidence_value_score": score,
                "v2_scoring_reasons": reasons,
                "expected_evidence_families": sorted(families),
                "adaptive_window_pages": adaptive,
                "adaptive_window_page_count": len(adaptive),
                "adaptive_window_rationale": adaptive_window_rationale(families, title),
                "recommended_route": route_for_section(title),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            -int(item["v2_evidence_value_score"]),
            item["source_filename"],
            item["adaptive_window_pages"][0],
        ),
    )


def infer_v2_families(source_filename: str, title: str) -> set[str]:
    lower = f"{source_filename} {title}".lower()
    families = set(infer_target_families(source_filename, lower))
    if any(
        term in lower
        for term in ["manufacturer", "model", "supplier", "product", "plant", "equipment"]
    ):
        families.update({"manufacturer_model", "equipment_specification"})
    if any(
        term in lower
        for term in [
            "roof",
            "cladding",
            "wall",
            "floor",
            "door",
            "dock",
            "leveller",
            "barrier",
            "gate",
        ]
    ):
        families.update({"component_description", "materials_finishes"})
    if any(
        term in lower for term in ["load", "capacity", "rating", "dimension", "slab", "structural"]
    ):
        families.add("dimensions_capacities")
    if any(
        term in lower
        for term in ["certificate", "commission", "test", "completion", "warranty", "guarantee"]
    ):
        families.update({"dates_certificates", "identifiers_references", "commissioning_results"})
    if any(term in lower for term in ["fire", "building control", "statutory", "hazard"]):
        families.add("statutory_compliance")
    if any(term in lower for term in ["door", "dock", "meter", "charger", "parking", "bollard"]):
        families.add("counts")
    if any(term in lower for term in ["installation", "installed", "operation"]):
        families.add("installation_details")
    families.discard("section_discovery")
    families.discard("maintenance_only")
    return families or {"component_description"}


def source_priority_boost(source_filename: str, title: str) -> int:
    lower = f"{source_filename} {title}".lower()
    if "part 2" in lower and any(
        term in lower
        for term in [
            "roof",
            "roof lights",
            "cladding",
            "wall",
            "floor",
            "loading",
            "dock",
            "door",
            "finish",
            "structural",
        ]
    ):
        return 18
    if "part 3" in lower and any(
        term in lower
        for term in [
            "equipment",
            "schedule",
            "plant",
            "fire",
            "lighting",
            "meter",
            "bms",
            "commission",
            "photovoltaic",
        ]
    ):
        return 18
    if "part 6" in lower and any(
        term in lower
        for term in [
            "certificate",
            "commission",
            "test",
            "warranty",
            "guarantee",
            "bms",
            "photovoltaic",
        ]
    ):
        return 20
    if "part 4" in lower and any(
        term in lower for term in ["drainage", "bollard", "gate", "cycle", "barrier", "retaining"]
    ):
        return 8
    if "part 5" in lower and any(
        term in lower for term in ["structural", "load", "fire", "roof access", "hazard"]
    ):
        return 8
    return 0


def adaptive_section_window(
    section: dict[str, Any],
    *,
    families: set[str],
    total_pages: int,
    cached_pages: set[int],
) -> list[int]:
    if total_pages <= 0:
        return []
    start = int(section.get("pdf_page_start") or 0)
    end = int(section.get("pdf_page_end") or start)
    if start <= 0:
        return []
    route = route_for_section(str(section.get("section_title") or ""))
    window_size = adaptive_window_size(
        route=route, families=families, section_page_count=max(end - start + 1, 1)
    )
    section_probe_end = min(total_pages, max(end, start + window_size - 1))
    pages = [
        page for page in range(start, section_probe_end + 1) if page not in cached_pages
    ]
    if pages:
        return pages[:window_size]
    probe_start = min(total_pages, end + 1)
    return [
        page
        for page in range(probe_start, min(total_pages, probe_start + window_size + 20) + 1)
        if page not in cached_pages
    ][:window_size]


def adaptive_window_size(*, route: str, families: set[str], section_page_count: int) -> int:
    if route == "certificate":
        return 3
    if "commissioning_results" in families:
        return 8
    if "equipment_specification" in families or "manufacturer_model" in families:
        return 8
    if "component_description" in families or "materials_finishes" in families:
        return min(max(section_page_count, 3), 5)
    return min(max(section_page_count, 3), 6)


def adaptive_window_rationale(families: set[str], title: str) -> str:
    if "commissioning_results" in families:
        return "commissioning block: enough pages to capture identity, date/reference and result"
    if "equipment_specification" in families or "manufacturer_model" in families:
        return "equipment/schedule section: bounded window to capture identity and specification"
    if "dates_certificates" in families:
        return "certificate start: capture title, date, reference and system context"
    return f"section evidence sample for {title}"


def build_v2_section_candidates(
    *,
    rescored_sections: list[dict[str, Any]],
    page_map: list[dict[str, Any]],
    page_counts: dict[str, int | None],
) -> list[dict[str, Any]]:
    uncached = {
        (str(row["source_id"]), int(row["page_number"]))
        for row in page_map
        if row["cache_status"] == "uncached"
    }
    used_candidate_pages: set[tuple[str, int]] = set()
    next_probe_by_source: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for section in rescored_sections:
        if int(section["v2_evidence_value_score"]) < 15:
            continue
        source_id = str(section["source_id"])
        window_size = int(section["adaptive_window_page_count"])
        pages = [
            page
            for page in section["adaptive_window_pages"]
            if (source_id, int(page)) in uncached
            and (source_id, int(page)) not in used_candidate_pages
        ]
        if not pages:
            probe = max(
                next_probe_by_source.get(source_id, 0), int(section.get("pdf_page_end") or 0) + 1
            )
            pages = []
            total = page_counts.get(source_id) or 0
            while probe <= total and len(pages) < window_size:
                key = (source_id, probe)
                if key in uncached and key not in used_candidate_pages:
                    pages.append(probe)
                probe += 1
        if not pages:
            continue
        if max(pages) > (page_counts.get(source_id) or 0):
            continue
        for page in pages:
            used_candidate_pages.add((source_id, int(page)))
        next_probe_by_source[source_id] = max(pages) + 1
        rows.append(
            {
                **section,
                "candidate_id": f"v2_sec_{len(rows) + 1:04d}",
                "missing_pages": pages,
                "missing_page_count": len(pages),
                "candidate_decision": "candidate",
                "selection_basis": "v2_gap_rescored_adaptive_section_window",
            }
        )
    rows.extend(
        build_frontier_continuation_candidates(
            page_map=page_map,
            page_counts=page_counts,
            used_candidate_pages=used_candidate_pages,
            start_index=len(rows) + 1,
        )
    )
    return sorted(
        rows,
        key=lambda item: (
            -int(item["v2_evidence_value_score"]),
            item["source_filename"],
            item["missing_pages"][0],
        ),
    )


def build_frontier_continuation_candidates(
    *,
    page_map: list[dict[str, Any]],
    page_counts: dict[str, int | None],
    used_candidate_pages: set[tuple[str, int]],
    start_index: int,
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in page_map:
        by_source[str(row["source_id"])].append(row)
    rows: list[dict[str, Any]] = []
    for source_id, pages in sorted(by_source.items()):
        filename = str(pages[0]["source_filename"])
        source_key = source_family_key(filename)
        templates = frontier_templates(source_key)
        if not templates:
            continue
        cached_numbers = sorted(
            int(row["page_number"]) for row in pages if row["cache_status"] == "cached"
        )
        probe = (max(cached_numbers) + 1) if cached_numbers else 1
        total = page_counts.get(source_id) or 0
        for template in templates:
            window: list[int] = []
            while probe <= total and len(window) < int(template["window_size"]):
                key = (source_id, probe)
                if key not in used_candidate_pages and any(
                    row["cache_status"] == "uncached" and int(row["page_number"]) == probe
                    for row in pages
                ):
                    window.append(probe)
                    used_candidate_pages.add(key)
                probe += 1
            if not window:
                continue
            rows.append(
                {
                    "candidate_id": f"v2_sec_{start_index + len(rows):04d}",
                    "entry_id": f"frontier_{source_id}_{len(rows) + 1:03d}",
                    "source_id": source_id,
                    "source_filename": filename,
                    "section_label": template["section_label"],
                    "section_title": template["section_title"],
                    "normalized_section_title": normalize_heading(str(template["section_title"])),
                    "parent_section": "",
                    "pdf_page_start": window[0],
                    "pdf_page_end": window[-1],
                    "page_count": len(window),
                    "resolved": True,
                    "resolution_method": "frontier_continuation_from_cached_index_order",
                    "resolution_confidence": 0.35,
                    "range_already_cached": False,
                    "cached_pages_in_range": [],
                    "uncached_pages_in_range": window,
                    "bounded_expansion_required": True,
                    "v2_evidence_value_score": template["score"],
                    "v2_scoring_reasons": template["reasons"],
                    "expected_evidence_families": template["families"],
                    "adaptive_window_pages": window,
                    "adaptive_window_page_count": len(window),
                    "adaptive_window_rationale": template["rationale"],
                    "recommended_route": template["route"],
                    "missing_pages": window,
                    "missing_page_count": len(window),
                    "candidate_decision": "candidate",
                    "selection_basis": "v2_frontier_continuation_high_value_source_family",
                }
            )
    return rows


def frontier_templates(source_key: str) -> list[dict[str, Any]]:
    if source_key == "part 2":
        return [
            _frontier_template(
                "2.3",
                "ROOF AND ROOFLIGHTS CONTINUATION",
                8,
                ["component_description", "materials_finishes", "dimensions_capacities"],
                "text",
                46,
            ),
            _frontier_template(
                "2.4",
                "CLADDING AND WALL SYSTEMS",
                8,
                ["component_description", "materials_finishes", "manufacturer_model"],
                "text",
                44,
            ),
            _frontier_template(
                "2.5",
                "DOORS AND LOADING DOORS",
                8,
                ["component_description", "counts", "dimensions_capacities"],
                "text",
                44,
            ),
            _frontier_template(
                "2.6",
                "DOCK EQUIPMENT AND LEVELLERS",
                8,
                ["component_description", "counts", "equipment_specification"],
                "text",
                42,
            ),
            _frontier_template(
                "2.7",
                "WINDOWS CURTAIN WALLING FINISHES",
                8,
                ["component_description", "materials_finishes", "manufacturer_model"],
                "text",
                40,
            ),
        ]
    if source_key == "part 3":
        return [
            _frontier_template(
                "3.2",
                "MECHANICAL PLANT AND EQUIPMENT SCHEDULES",
                10,
                ["manufacturer_model", "equipment_specification", "dimensions_capacities"],
                "table",
                48,
            ),
            _frontier_template(
                "3.3",
                "ELECTRICAL PLANT METERS AND CONTROLS",
                10,
                ["manufacturer_model", "equipment_specification", "identifiers_references"],
                "table",
                46,
            ),
            _frontier_template(
                "3.4",
                "FIRE ALARM DETECTION AND TESTING",
                8,
                ["commissioning_results", "dates_certificates", "statutory_compliance"],
                "certificate",
                46,
            ),
            _frontier_template(
                "3.5",
                "LIGHTING BMS AND CONTROLS",
                8,
                ["manufacturer_model", "equipment_specification", "installation_details"],
                "table",
                42,
            ),
            _frontier_template(
                "3.6",
                "PHOTOVOLTAIC AND SERVICE CAPACITY",
                8,
                ["dimensions_capacities", "commissioning_results", "manufacturer_model"],
                "table",
                40,
            ),
        ]
    if source_key == "part 6":
        return [
            _frontier_template(
                "D.1",
                "FIRE ALARM COMMISSIONING CERTIFICATES",
                6,
                ["commissioning_results", "dates_certificates", "identifiers_references"],
                "certificate",
                52,
            ),
            _frontier_template(
                "D.2",
                "ELECTRICAL TEST CERTIFICATES",
                6,
                ["commissioning_results", "dates_certificates", "statutory_compliance"],
                "certificate",
                50,
            ),
            _frontier_template(
                "D.3",
                "MECHANICAL COMMISSIONING CERTIFICATES",
                6,
                ["commissioning_results", "manufacturer_model", "equipment_specification"],
                "certificate",
                48,
            ),
            _frontier_template(
                "D.4",
                "BMS POINTS AND AIR CONDITIONING SCHEDULES",
                8,
                ["manufacturer_model", "equipment_specification", "identifiers_references"],
                "table",
                46,
            ),
            _frontier_template(
                "D.5",
                "ROOFING CLADDING AND DRAINAGE GUARANTEES",
                6,
                ["dates_certificates", "materials_finishes", "identifiers_references"],
                "certificate",
                44,
            ),
        ]
    return []


def _frontier_template(
    label: str,
    title: str,
    window_size: int,
    families: list[str],
    route: str,
    score: int,
) -> dict[str, Any]:
    return {
        "section_label": label,
        "section_title": title,
        "window_size": window_size,
        "families": families,
        "route": route,
        "score": score,
        "reasons": [
            "frontier_continuation_from_cached_index_order",
            "v2_evidence_family_gap_priority",
        ],
        "rationale": f"{title}: adaptive frontier sample for {', '.join(families)}",
    }


def approve_v2_parse_plan(
    candidates: list[dict[str, Any]],
    *,
    max_new_pages: int,
    per_source_caps: dict[str, int],
) -> list[dict[str, Any]]:
    approved: list[dict[str, Any]] = []
    used_pages: set[tuple[str, int]] = set()
    used_families: Counter[str] = Counter()
    source_usage: Counter[str] = Counter()
    total_used = 0
    ordered = breadth_first_candidates(candidates)
    for candidate in ordered:
        source_id = str(candidate["source_id"])
        source_key = source_family_key(str(candidate["source_filename"]))
        if source_key not in per_source_caps or total_used >= max_new_pages:
            continue
        source_remaining = per_source_caps[source_key] - source_usage[source_id]
        if source_remaining <= 0:
            continue
        family_pages = adaptive_family_page_limit(candidate, used_families)
        pages = [
            int(page)
            for page in candidate["missing_pages"]
            if (source_id, int(page)) not in used_pages
        ][: min(source_remaining, max_new_pages - total_used, family_pages)]
        if not pages:
            continue
        for page in pages:
            used_pages.add((source_id, page))
        for family in candidate["expected_evidence_families"]:
            used_families[str(family)] += len(pages)
        source_usage[source_id] += len(pages)
        total_used += len(pages)
        for start, end in split_contiguous_pages(pages, DEFAULT_MAX_BATCH_PAGES):
            approved.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "source_id": source_id,
                    "source_filename": candidate["source_filename"],
                    "section_label": candidate["section_label"],
                    "section_title": candidate["section_title"],
                    "page_start": start,
                    "page_end": end,
                    "page_count": end - start + 1,
                    "approved_pages": list(range(start, end + 1)),
                    "expected_evidence_families": candidate["expected_evidence_families"],
                    "v2_evidence_value_score": candidate["v2_evidence_value_score"],
                    "recommended_route": candidate["recommended_route"],
                    "approval_rationale": candidate["adaptive_window_rationale"],
                }
            )
    return sorted(approved, key=lambda item: (item["source_filename"], item["page_start"]))


def breadth_first_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_source[source_family_key(str(candidate["source_filename"]))].append(candidate)
    for values in by_source.values():
        values.sort(
            key=lambda item: (-int(item["v2_evidence_value_score"]), item["missing_pages"][0])
        )
    order = ["part 2", "part 3", "part 6", "part 4", "part 5"]
    output: list[dict[str, Any]] = []
    while any(by_source.values()):
        for key in order:
            if by_source[key]:
                output.append(by_source[key].pop(0))
    return output


def adaptive_family_page_limit(candidate: dict[str, Any], used_families: Counter[str]) -> int:
    families = [str(family) for family in candidate["expected_evidence_families"]]
    if any(used_families[family] < 12 for family in families):
        return DEFAULT_MAX_BATCH_PAGES
    return max(3, min(5, len(candidate["missing_pages"])))


def split_v2_batches(approved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages_by_source: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in approved:
        for page in item["approved_pages"]:
            pages_by_source[str(item["source_id"])][int(page)].append(item)
    batches: list[dict[str, Any]] = []
    for source_id, page_items in sorted(pages_by_source.items()):
        pages = sorted(page_items)
        for start, end in split_contiguous_pages(pages, DEFAULT_MAX_BATCH_PAGES):
            first = page_items[start][0]
            batches.append(
                {
                    "batch_id": f"hv2_{len(batches) + 1:04d}",
                    "source_id": source_id,
                    "source_filename": first["source_filename"],
                    "page_start": start,
                    "page_end": end,
                    "page_count": end - start + 1,
                    "worker_batch_page_limit": DEFAULT_MAX_BATCH_PAGES,
                    "parse_required": True,
                    "candidate_ids": sorted(
                        {
                            item["candidate_id"]
                            for page in range(start, end + 1)
                            for item in page_items.get(page, [])
                        }
                    ),
                    "expected_evidence_families": sorted(
                        {
                            family
                            for page in range(start, end + 1)
                            for item in page_items.get(page, [])
                            for family in item["expected_evidence_families"]
                        }
                    ),
                }
            )
    return sorted(batches, key=lambda item: (item["source_filename"], item["page_start"]))


def section_inventory_after(
    rescored_sections: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    sources: list[SourceRegistryEntry],
) -> list[dict[str, Any]]:
    by_page = {(str(row["source_id"]), int(row["page_number"])): row for row in classifications}
    source_names = {source.source_id: Path(source.logical_path).name for source in sources}
    rows: list[dict[str, Any]] = []
    for section in rescored_sections:
        pages = [
            by_page[(str(section["source_id"]), page)]
            for page in range(int(section["pdf_page_start"]), int(section["pdf_page_end"]) + 1)
            if (str(section["source_id"]), page) in by_page
        ]
        sampled_pages = [
            by_page[(str(section["source_id"]), page)]
            for page in section.get("adaptive_window_pages", [])
            if (str(section["source_id"]), int(page)) in by_page
        ]
        all_pages = pages or sampled_pages
        page_types = Counter(str(page["primary_page_type"]) for page in all_pages)
        routes = Counter(str(page["recommended_route"]) for page in all_pages)
        title = str(section.get("section_title") or "")
        rows.append(
            {
                "section_id": (
                    f"{section['source_id']}_{section['section_label']}_"
                    f"{section['pdf_page_start']}"
                ),
                "source_id": section["source_id"],
                "source_filename": section.get("source_filename")
                or source_names.get(str(section["source_id"]), ""),
                "part": part_from_filename(str(section.get("source_filename") or "")),
                "parent_section": section.get("parent_section") or "",
                "section": title,
                "subsection": section.get("section_label") or "",
                "page_start": section.get("pdf_page_start"),
                "page_end": section.get("pdf_page_end"),
                "cached_page_coverage": round(
                    len(all_pages) / max(int(section.get("page_count") or 1), 1), 4
                ),
                "primary_page_types": dict(sorted(page_types.items())),
                "secondary_evidence_tags": sorted(
                    {tag for page in all_pages for tag in page.get("secondary_tags", [])}
                ),
                "likely_dictionary_domains": infer_domains(
                    str(section.get("source_filename") or ""), title
                ),
                "likely_target_families": section.get("expected_evidence_families")
                or sorted(infer_v2_families(str(section.get("source_filename") or ""), title)),
                "recommended_route": routes.most_common(1)[0][0]
                if routes
                else section.get("recommended_route", "text"),
                "evidence_value_score": section.get(
                    "v2_evidence_value_score", section.get("evidence_value_score", 0)
                ),
                "section_resolution_confidence": section.get("resolution_confidence", 0),
                "evidence_bearing": bool(all_pages)
                or int(section.get("v2_evidence_value_score") or 0) >= 30,
            }
        )
    return sorted(
        rows, key=lambda item: (item["source_filename"], item["page_start"], item["section"])
    )


def merge_section_inventory_inputs(
    rescored_sections: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for section in rescored_sections:
        key = str(section.get("entry_id") or section.get("candidate_id") or id(section))
        merged[key] = section
    for candidate in candidates:
        key = str(candidate.get("entry_id") or candidate.get("candidate_id") or id(candidate))
        merged[key] = {**merged.get(key, {}), **candidate}
    return sorted(
        merged.values(),
        key=lambda item: (
            str(item.get("source_filename") or ""),
            int(item.get("pdf_page_start") or 0),
            str(item.get("section_title") or ""),
        ),
    )


def build_v2_metrics(
    *,
    sources: list[SourceRegistryEntry],
    page_counts: dict[str, int | None],
    cached_before: dict[str, list[dict[str, Any]]],
    cached_after: dict[str, list[dict[str, Any]]],
    classification_audit: dict[str, Any],
    classification_corrections: list[dict[str, Any]],
    gap_before: dict[str, Any],
    gap_after: dict[str, Any],
    rescored: list[dict[str, Any]],
    approved: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    parse_report: dict[str, Any],
    classifications_after: list[dict[str, Any]],
    section_inventory: list[dict[str, Any]],
    remaining: list[dict[str, Any]],
    runtime_ms: float,
) -> dict[str, Any]:
    total_pages = sum(count or 0 for count in page_counts.values())
    before = sum(len(pages) for pages in cached_before.values())
    after = sum(len(pages) for pages in cached_after.values())
    page_types = Counter(str(row["primary_page_type"]) for row in classifications_after)
    routes = Counter(str(row["recommended_route"]) for row in classifications_after)
    readiness = readiness_status_v2(
        section_inventory=section_inventory,
        gap_after=gap_after,
        page_types=page_types,
        newly_parsed=parse_report["new_pages_parsed"],
        separator_corrections=sum(
            1
            for row in classification_corrections
            if row["prior_primary_page_type"] == "separator_or_cover"
        ),
    )
    return {
        "total_registered_pdfs": sum(1 for source in sources if source.file_type == FileType.PDF),
        "total_pdf_pages": total_pages,
        "cached_pages_before": before,
        "cached_pages_after": after,
        "newly_parsed_pages": parse_report["new_pages_parsed"],
        "coverage_before_percentage": round(before / total_pages * 100, 2) if total_pages else 0.0,
        "coverage_after_percentage": round(after / total_pages * 100, 2) if total_pages else 0.0,
        "coverage_change_by_source": coverage_change_by_source(
            cached_before, cached_after, sources, page_counts
        ),
        "classification_audit_finding": classification_audit["finding"],
        "separator_cover_pages_before": classification_audit["counts_by_type"].get(
            "separator_or_cover", 0
        ),
        "separator_cover_corrections": sum(
            1
            for row in classification_corrections
            if row["prior_primary_page_type"] == "separator_or_cover"
        ),
        "unknown_corrections": sum(
            1 for row in classification_corrections if row["prior_primary_page_type"] == "unknown"
        ),
        "classification_corrections": len(classification_corrections),
        "evidence_family_gaps_before": gap_before["weak_or_absent_families"],
        "evidence_family_gaps_after": gap_after["weak_or_absent_families"],
        "rescored_candidates": len(rescored),
        "approved_sections": len({row["candidate_id"] for row in approved}),
        "planned_pages": sum(int(row["page_count"]) for row in approved),
        "parser_batches": len(batches),
        "parser_batches_run": parse_report["parser_batches_run"],
        "parser_runtime_ms": parse_report["runtime_ms"],
        "runtime_ms": round(runtime_ms, 2),
        "failures": parse_report["parse_failure_count"],
        "timeouts_or_restarts": parse_report["timeout_or_restart_count"],
        "page_type_distribution_after": dict(sorted(page_types.items())),
        "high_value_sections_added": len(
            [
                section
                for section in section_inventory
                if int(section.get("evidence_value_score") or 0) >= 30
            ]
        ),
        "evidence_family_coverage_before": gap_before["counts"],
        "evidence_family_coverage_after": gap_after["counts"],
        "table_routes": routes.get("table", 0),
        "certificate_routes": routes.get("certificate", 0) + page_types.get("certificate", 0),
        "drawing_text_routes": routes.get("drawing_text", 0),
        "visual_routes": routes.get("visual", 0),
        "remaining_high_priority_sections": len(remaining),
        "unknown_pages_remaining": page_types.get("unknown", 0),
        "target_selection_readiness": readiness,
    }


def readiness_status_v2(
    *,
    section_inventory: list[dict[str, Any]],
    gap_after: dict[str, Any],
    page_types: Counter[str],
    newly_parsed: int,
    separator_corrections: int,
) -> str:
    high_value_by_part = Counter(
        str(section.get("part") or "")
        for section in section_inventory
        if int(section.get("evidence_value_score") or 0) >= 30 and section.get("evidence_bearing")
    )
    key_families = {
        "manufacturer_model",
        "equipment_specification",
        "dimensions_capacities",
        "dates_certificates",
        "commissioning_results",
    }
    remaining_gaps = set(gap_after["weak_or_absent_families"])
    covered_key = key_families - remaining_gaps
    blocking_gaps = {
        "commissioning_results",
        "statutory_compliance",
        "installation_details",
    }
    if (
        not (remaining_gaps & blocking_gaps)
        and high_value_by_part["Part 2"] >= 3
        and high_value_by_part["Part 3"] >= 3
        and high_value_by_part["Part 6"] >= 2
        and len(covered_key) >= 4
        and page_types.get("unknown", 0) <= 20
        and separator_corrections > 0
        and newly_parsed > 0
    ):
        return "ready for evidence-backed target-family selection"
    if (
        newly_parsed > 0
        and high_value_by_part["Part 2"]
        and high_value_by_part["Part 3"]
        and high_value_by_part["Part 6"]
    ):
        return "partially ready"
    return "not ready"


def remaining_high_priority_sections(
    candidates: list[dict[str, Any]], approved: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    approved_ids = {str(row["candidate_id"]) for row in approved}
    return [row for row in candidates if str(row["candidate_id"]) not in approved_ids][:50]


def source_caps(max_new_pages: int) -> dict[str, int]:
    scale = max_new_pages / 160 if max_new_pages else 0
    return {key: max(1, int(value * scale)) for key, value in FOCUS_PART_CAPS.items()}


def diff_cached_pages(
    before: dict[str, list[dict[str, Any]]],
    after: dict[str, list[dict[str, Any]]],
    sources: list[SourceRegistryEntry],
) -> list[dict[str, Any]]:
    names = {source.source_id: Path(source.logical_path).name for source in sources}
    before_keys = {
        (source_id, int(page["page_number"]))
        for source_id, pages in before.items()
        for page in pages
    }
    rows = []
    for source_id, pages in after.items():
        for page in pages:
            key = (source_id, int(page["page_number"]))
            if key not in before_keys:
                rows.append(
                    {
                        "source_id": source_id,
                        "source_filename": names.get(source_id, ""),
                        "page_number": int(page["page_number"]),
                        "cache_path": page["cache_path"],
                    }
                )
    return sorted(rows, key=lambda item: (item["source_filename"], item["page_number"]))


def coverage_change_by_source(
    before: dict[str, list[dict[str, Any]]],
    after: dict[str, list[dict[str, Any]]],
    sources: list[SourceRegistryEntry],
    page_counts: dict[str, int | None],
) -> list[dict[str, Any]]:
    rows = []
    for source in sources:
        if source.file_type != FileType.PDF:
            continue
        total = page_counts.get(source.source_id) or 0
        before_count = len(before.get(source.source_id, []))
        after_count = len(after.get(source.source_id, []))
        rows.append(
            {
                "source_id": source.source_id,
                "source_filename": Path(source.logical_path).name,
                "cached_before": before_count,
                "cached_after": after_count,
                "new_pages": after_count - before_count,
                "coverage_before_percentage": round(before_count / total * 100, 2)
                if total
                else 0.0,
                "coverage_after_percentage": round(after_count / total * 100, 2) if total else 0.0,
            }
        )
    return rows


def write_v2_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_outputs = {
        "classification_audit_before.json": result["classification_audit_before"],
        "classification_corrections.json": result["classification_corrections"],
        "evidence_family_gap_analysis.json": result["evidence_family_gap_analysis"],
        "rescored_section_candidates.json": result["rescored_section_candidates"],
        "approved_section_parse_plan.json": result["approved_section_parse_plan"],
        "merged_parse_batches.json": result["merged_parse_batches"],
        "parse_execution_report.json": result["parse_execution_report"],
        "newly_cached_pages.json": result["newly_cached_pages"],
        "cache_coverage_before.json": result["cache_coverage_before"],
        "cache_coverage_after.json": result["cache_coverage_after"],
        "page_classifications_after.json": result["page_classifications_after"],
        "section_inventory_after.json": result["section_inventory_after"],
        "evidence_section_map_after.json": result["evidence_section_map_after"],
        "enriched_pageindex.json": result["enriched_pageindex"],
        "source_family_support_map_after.json": result["source_family_support_map_after"],
        "evidence_family_coverage_before.json": result["evidence_family_coverage_before"],
        "evidence_family_coverage_after.json": result["evidence_family_coverage_after"],
        "table_route_pages.json": result["table_route_pages"],
        "certificate_route_pages.json": result["certificate_route_pages"],
        "drawing_text_route_pages.json": result["drawing_text_route_pages"],
        "visual_route_pages.json": result["visual_route_pages"],
        "low_value_pages.json": result["low_value_pages"],
        "unknown_pages_remaining.json": result["unknown_pages_remaining"],
        "remaining_high_priority_sections.json": result["remaining_high_priority_sections"],
        "coverage_metrics.json": result["coverage_metrics"],
        "coverage_trace.json": result["coverage_trace"],
    }
    for filename, payload in json_outputs.items():
        _atomic_write_json(output_dir / filename, payload)
    csv_outputs = {
        "classification_corrections.csv": result["classification_corrections"],
        "rescored_section_candidates.csv": result["rescored_section_candidates"],
        "approved_section_parse_plan.csv": result["approved_section_parse_plan"],
        "section_inventory_after.csv": result["section_inventory_after"],
        "evidence_section_map_after.csv": result["evidence_section_map_after"],
    }
    for filename, rows in csv_outputs.items():
        write_csv(output_dir / filename, rows)
    (output_dir / "source_coverage_summary.md").write_text(
        summary_markdown(result), encoding="utf-8"
    )
    (output_dir / "next_target_selection_readiness.md").write_text(
        readiness_markdown(result["coverage_metrics"]), encoding="utf-8"
    )


def summary_markdown(result: dict[str, Any]) -> str:
    metrics = result["coverage_metrics"]
    return (
        "# High-Value Evidence Section Expansion V2\n\n"
        f"- Classification audit: {metrics['classification_audit_finding']}\n"
        f"- Cached before: {metrics['cached_pages_before']}\n"
        f"- Newly parsed pages: {metrics['newly_parsed_pages']}\n"
        f"- Cached after: {metrics['cached_pages_after']}\n"
        f"- Coverage after: {metrics['coverage_after_percentage']}%\n"
        f"- Separator/cover corrections: {metrics['separator_cover_corrections']}\n"
        f"- Unknown pages remaining: {metrics['unknown_pages_remaining']}\n"
        f"- Readiness: {metrics['target_selection_readiness']}\n"
    )


def readiness_markdown(metrics: dict[str, Any]) -> str:
    return (
        "# Next Target Selection Readiness\n\n"
        f"- Status: {metrics['target_selection_readiness']}\n"
        f"- Approved sections: {metrics['approved_sections']}\n"
        f"- Newly parsed pages: {metrics['newly_parsed_pages']}\n"
        f"- Remaining high-priority sections: {metrics['remaining_high_priority_sections']}\n"
        "- Evidence-family gaps after: "
        f"{json.dumps(metrics['evidence_family_gaps_after'], sort_keys=True)}\n"
    )


def build_v2_trace(
    classification_audit: dict[str, Any],
    candidates: list[dict[str, Any]],
    approved: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    parse_report: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {"stage": "classification_audit", "summary": classification_audit},
        {"stage": "rescored_candidate_plan", "rows": len(candidates)},
        {"stage": "approved_section_parse_plan", "rows": len(approved)},
        {"stage": "merged_parse_batches", "rows": len(batches)},
        {"stage": "parse_execution", "summary": parse_report},
    ]


def _route_pages(rows: list[dict[str, Any]], routes: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if row["recommended_route"] in routes]


def _term_count(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if term in text)


def part_from_filename(filename: str) -> str:
    match = re.search(r"part\s+(\d+)", filename.lower())
    return f"Part {match.group(1)}" if match else ""


def _source_sort_key(source: SourceRegistryEntry) -> tuple[str, str]:
    return (Path(source.logical_path).name.lower(), source.source_id)
