"""Index-guided source-first evidence section expansion."""

from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry
from segro_evidence_extraction.parsing.batch_worker import DEFAULT_MAX_BATCH_PAGES
from segro_evidence_extraction.parsing.service import load_source_registry
from segro_evidence_extraction.source_coverage_pageindex import (
    DEFAULT_HIERARCHY_PATH,
    DEFAULT_PAGE_CACHE_ROOT,
    DEFAULT_SOURCE_COVERAGE_OUTPUT_DIR,
    DEFAULT_SOURCE_MANIFEST,
    build_enriched_pageindex,
    build_evidence_section_map,
    build_page_coverage_map,
    build_source_family_support_map,
    build_source_inventory,
    cache_coverage_snapshot,
    classify_cached_pages,
    classify_page_text,
    csv_value,
    determine_pdf_page_counts,
    execute_parse_batches,
    infer_domains,
    infer_target_families,
    load_canonical_cached_pages,
    load_hierarchy_nodes,
    split_contiguous_pages,
    validate_parse_batches,
)
from segro_evidence_extraction.vertical_slice import _atomic_write_json

DEFAULT_INDEX_EXPANSION_OUTPUT_DIR = Path("output/enfield_unit1_index_guided_section_expansion_v1")
INDEX_GUIDED_VERSION = "index-guided-section-expansion-v1"

MAJOR_SOURCE_ALLOCATION = {
    "part 2": 38,
    "part 3": 38,
    "part 4": 18,
    "part 5": 14,
    "part 6": 28,
}


def run_index_guided_section_expansion_v1(
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    cache_root: Path = DEFAULT_PAGE_CACHE_ROOT,
    v1_output_dir: Path = DEFAULT_SOURCE_COVERAGE_OUTPUT_DIR,
    hierarchy_path: Path = DEFAULT_HIERARCHY_PATH,
    output_dir: Path = DEFAULT_INDEX_EXPANSION_OUTPUT_DIR,
    max_new_pages: int = 120,
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
    page_map_before = build_page_coverage_map(pdf_sources, page_counts, cached_before)
    v1_audit = audit_v1_artifacts(
        v1_output_dir=v1_output_dir,
        sources=sources,
        page_counts=page_counts,
        cached_pages=cached_before,
    )
    baseline = build_baseline_reconciliation(v1_audit, inventory_before)
    prior_classifications = _read_json_list(v1_output_dir / "page_classifications.json")
    reclassification = reclassify_unknown_pages(prior_classifications, cached_before, sources)
    classifications_before = merge_reclassifications(prior_classifications, reclassification)
    index_entries = extract_contents_index_entries(
        classifications_before, cached_before, hierarchy_nodes
    )
    resolved_sections = resolve_section_boundaries(
        index_entries=index_entries,
        hierarchy_nodes=hierarchy_nodes,
        page_counts=page_counts,
        cached_pages=cached_before,
        sources=sources,
    )
    scores = score_sections(resolved_sections)
    candidates = build_uncached_section_candidates(
        scored_sections=scores,
        page_map=page_map_before,
        page_counts=page_counts,
    )
    approved = approve_section_parse_plan(
        candidates,
        max_new_pages=max_new_pages,
        per_source_caps=source_caps(max_new_pages),
    )
    batches = split_approved_section_batches(approved)
    validate_parse_batches(batches, cached_before)
    parse_report = execute_parse_batches(
        batches=batches,
        sources=sources,
        cache_root=cache_root,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    cached_after = load_canonical_cached_pages(cache_root)
    newly_cached_pages = diff_cached_pages(cached_before, cached_after, sources)
    raw_classifications_after = classify_cached_pages(cached_after, sources)
    classifications_after = merge_reclassifications(raw_classifications_after, reclassification)
    sections_after = section_inventory_from_resolved_sections(
        resolved_sections,
        classifications_after,
        sources,
    )
    evidence_map_after = build_evidence_section_map(sections_after)
    source_support_after = build_source_family_support_map(evidence_map_after)
    inventory_after = build_source_inventory(
        sources=sources,
        page_counts=page_counts,
        cached_pages=cached_after,
        hierarchy_nodes=hierarchy_nodes,
        parser_fingerprint={},
    )
    metrics = build_index_expansion_metrics(
        sources=sources,
        page_counts=page_counts,
        baseline=baseline,
        cached_before=cached_before,
        cached_after=cached_after,
        reclassification=reclassification,
        index_entries=index_entries,
        resolved_sections=resolved_sections,
        candidates=candidates,
        approved=approved,
        parse_report=parse_report,
        classifications_after=classifications_after,
        sections_after=sections_after,
        runtime_ms=(time.perf_counter() - started) * 1000,
    )
    result = {
        "v1_artifact_audit": v1_audit,
        "coverage_baseline_reconciliation": baseline,
        "unknown_page_reclassification": reclassification,
        "contents_index_entries": index_entries,
        "resolved_section_boundaries": resolved_sections,
        "section_priority_scores": scores,
        "uncached_section_candidates": candidates,
        "approved_section_parse_plan": approved,
        "merged_parse_batches": batches,
        "parse_execution_report": parse_report,
        "cache_coverage_before": cache_coverage_snapshot(inventory_before),
        "cache_coverage_after": cache_coverage_snapshot(inventory_after),
        "newly_cached_pages": newly_cached_pages,
        "page_classifications_after": classifications_after,
        "section_inventory_after": sections_after,
        "evidence_section_map_after": evidence_map_after,
        "enriched_pageindex": build_enriched_pageindex(hierarchy_nodes, sections_after),
        "source_family_support_map_after": source_support_after,
        "table_route_pages": _route_pages(classifications_after, {"table"}),
        "certificate_route_pages": [
            row for row in classifications_after if row["primary_page_type"] == "certificate"
        ],
        "visual_route_pages": _route_pages(classifications_after, {"visual"}),
        "low_value_pages": _route_pages(classifications_after, {"deprioritized"}),
        "unknown_pages_remaining": [
            row for row in classifications_after if row["primary_page_type"] == "unknown"
        ],
        "coverage_metrics": metrics,
        "coverage_trace": build_coverage_trace(
            v1_audit=v1_audit,
            reclassification=reclassification,
            index_entries=index_entries,
            resolved_sections=resolved_sections,
            candidates=candidates,
            approved=approved,
            batches=batches,
            parse_report=parse_report,
        ),
        "source_coverage_review": build_source_review(
            inventory_before, inventory_after, sections_after
        ),
    }
    write_index_expansion_outputs(result, output_dir)
    return result


def audit_v1_artifacts(
    *,
    v1_output_dir: Path,
    sources: list[SourceRegistryEntry],
    page_counts: dict[str, int | None],
    cached_pages: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    source_ids = {source.source_id for source in sources}
    duplicate_keys = [
        f"{source_id}:{page_number}"
        for source_id, pages in cached_pages.items()
        for page_number, count in Counter(int(page["page_number"]) for page in pages).items()
        if count > 1
    ]
    orphan_cached_pages = [
        {"source_id": source_id, "page_number": page["page_number"]}
        for source_id, pages in cached_pages.items()
        for page in pages
        if source_id not in source_ids
    ]
    classifications = _read_json_list(v1_output_dir / "page_classifications.json")
    sections = _read_json_list(v1_output_dir / "section_inventory.json")
    route_files = [
        "table_route_pages.json",
        "certificate_route_pages.json",
        "visual_route_pages.json",
        "low_value_pages.json",
    ]
    cached_key_set = {
        (source_id, int(page["page_number"]))
        for source_id, pages in cached_pages.items()
        for page in pages
    }
    invalid_route_refs: list[dict[str, Any]] = []
    for filename in route_files:
        for row in _read_json_list(v1_output_dir / filename):
            key = (str(row.get("source_id") or ""), int(row.get("page_number") or 0))
            if key not in cached_key_set:
                invalid_route_refs.append(
                    {"route_file": filename, "source_page": f"{key[0]}:{key[1]}"}
                )
    out_of_bounds_sections = [
        {
            "section_id": section.get("section_id"),
            "source_id": section.get("source_id"),
            "page_start": section.get("page_start"),
            "page_end": section.get("page_end"),
        }
        for section in sections
        if not _range_within_bounds(section, page_counts)
    ]
    page_type_counts = Counter(str(row.get("primary_page_type")) for row in classifications)
    metrics = _read_json_dict(v1_output_dir / "coverage_metrics.json")
    metric_counts = dict(metrics.get("page_type_distribution", {}))
    parse_report = _read_json_dict(v1_output_dir / "parse_execution_report.json")
    cached_total = sum(len(pages) for pages in cached_pages.values())
    total_pages = sum(count or 0 for count in page_counts.values())
    issues = []
    if orphan_cached_pages:
        issues.append("orphan_cached_pages")
    if duplicate_keys:
        issues.append("duplicate_cached_pages")
    if out_of_bounds_sections:
        issues.append("section_out_of_bounds")
    if invalid_route_refs:
        issues.append("invalid_route_references")
    if dict(sorted(page_type_counts.items())) != dict(sorted(metric_counts.items())):
        issues.append("page_type_metric_mismatch")
    return {
        "v1_output_dir": str(v1_output_dir),
        "status": "pass" if not issues else "warning",
        "issues": issues,
        "registered_source_count": len(sources),
        "registered_pdf_count": sum(1 for source in sources if source.file_type == FileType.PDF),
        "total_pdf_pages": total_pages,
        "cached_pages_current": cached_total,
        "uncached_pages_current": max(total_pages - cached_total, 0),
        "orphan_cached_pages": orphan_cached_pages,
        "duplicate_cached_page_records": duplicate_keys,
        "out_of_bounds_sections": out_of_bounds_sections,
        "invalid_route_references": invalid_route_refs,
        "page_type_counts_from_classifications": dict(sorted(page_type_counts.items())),
        "page_type_counts_from_metrics": metric_counts,
        "section_rows_are_section_counts_not_page_counts": True,
        "parse_execution_reconciles_with_current_cache": parse_report.get("parse_failure_count", 0)
        == 0,
        "v1_parse_execution_report": parse_report,
    }


def build_baseline_reconciliation(
    v1_audit: dict[str, Any],
    current_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    current_cached = sum(int(row["cached_page_count"]) for row in current_inventory)
    return {
        "historical_baseline_before_source_coverage_v1": {
            "status": "historical baseline unavailable",
            "reason": (
                "V1 artifacts were regenerated after an earlier run cached additional pages; "
                "the original pre-sprint cache count cannot be reconstructed from "
                "canonical artifacts."
            ),
        },
        "current_cache_before_index_expansion_v1": {
            "cached_pages": current_cached,
            "source_count": len(current_inventory),
        },
        "v1_artifact_audit_status": v1_audit["status"],
    }


def reclassify_unknown_pages(
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
        if row.get("primary_page_type") != "unknown":
            continue
        source_id = str(row["source_id"])
        page_number = int(row["page_number"])
        text = text_lookup.get((source_id, page_number), "")
        enhanced = classify_page_text_v2(text, source_filename=source_names.get(source_id, ""))
        changed = enhanced["primary_page_type"] != "unknown"
        rows.append(
            {
                "source_id": source_id,
                "source_filename": row.get("source_filename") or source_names.get(source_id, ""),
                "page_number": page_number,
                "prior_primary_page_type": "unknown",
                "primary_page_type": enhanced["primary_page_type"],
                "secondary_tags": enhanced["secondary_tags"],
                "recommended_route": enhanced["recommended_route"],
                "rule_signals": enhanced["rule_signals"],
                "classification_score": enhanced["classification_score"],
                "classification_confidence": enhanced["classification_confidence"],
                "classification_reason": enhanced["classification_reason"],
                "classification_changed": changed,
            }
        )
    return sorted(rows, key=lambda item: (item["source_filename"], item["page_number"]))


def classify_page_text_v2(text: str, *, source_filename: str = "") -> dict[str, Any]:
    base = classify_page_text(text, source_filename=source_filename)
    signals = page_rule_signals(text)
    score = 0.0
    reasons: list[str] = []
    primary = str(base["primary_page_type"])
    route = str(base["recommended_route"])
    tags = set(str(tag) for tag in base["secondary_tags"])
    if primary != "unknown":
        score += 0.65
        reasons.append(f"V1 classifier matched {primary}")
    if signals["empty_page"]:
        return _classification_v2(
            "separator_or_cover",
            "deprioritized",
            tags | {"duplicate_or_repeated_content"},
            signals,
            0.72,
            "No extracted text; retained as non-evidence parser/visual gap.",
        )
    if signals["certificate_terms"] >= 1 or signals["commissioning_terms"] >= 2:
        score += 0.3
        primary, route = "certificate", "certificate"
        tags.update({"certificate_date_reference", "commissioning_result"})
        reasons.append("certificate or commissioning terms")
    elif signals["schedule_terms"] >= 2 and signals["table_label_terms"] >= 2:
        score += 0.28
        primary, route = "schedule", "table"
        tags.add("equipment_schedule")
        reasons.append("schedule heading with table labels")
    elif signals["drawing_terms"] >= 2 or signals["revision_terms"] >= 2:
        score += 0.24
        primary = (
            "drawing_text_extractable"
            if signals["token_count"] >= 12
            else "drawing_visual_required"
        )
        route = "drawing_text" if primary == "drawing_text_extractable" else "visual"
        tags.add("drawing_symbol_dependency")
        reasons.append("drawing title/revision terms")
    elif signals["maintenance_terms"] >= 2:
        score += 0.25
        primary, route = "maintenance_guidance", "deprioritized"
        tags.add("maintenance_only")
        reasons.append("maintenance guidance terms")
    elif signals["coshh_terms"] >= 1:
        score += 0.3
        primary, route = "low_value_repetitive", "deprioritized"
        tags.add("duplicate_or_repeated_content")
        reasons.append("COSHH or safety-data terms")
    elif signals["table_like"] and signals["token_count"] >= 10:
        score += 0.22
        primary, route = "structured_table", "table"
        reasons.append("table-like numeric/label pattern")
    elif signals["heading_density"] >= 0.45 and signals["token_count"] >= 8:
        score += 0.18
        primary, route = "narrative", "text"
        reasons.append("heading-dense narrative page")
    confidence = min(score, 0.95)
    if confidence < 0.2 and primary == "unknown":
        reasons.append("insufficient deterministic signals")
    return _classification_v2(primary, route, tags, signals, confidence, "; ".join(reasons))


def page_rule_signals(text: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    tokens = re.findall(r"[a-z0-9./:-]+", normalized)
    numeric_tokens = [token for token in tokens if any(char.isdigit() for char in token)]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    uppercase_lines = sum(
        1 for line in lines if line.upper() == line and any(c.isalpha() for c in line)
    )
    return {
        "empty_page": len(normalized) == 0,
        "char_count": len(normalized),
        "token_count": len(tokens),
        "line_count": len(lines),
        "numeric_token_ratio": round(len(numeric_tokens) / len(tokens), 4) if tokens else 0.0,
        "heading_density": round(uppercase_lines / len(lines), 4) if lines else 0.0,
        "repeated_short_lines": sum(1 for line in lines if len(line) <= 12),
        "table_like": _contains_any(normalized, ["qty", "item", "description", "model", "ref"])
        and len(numeric_tokens) >= 3,
        "certificate_terms": _term_count(
            normalized, ["certificate", "certification", "test report"]
        ),
        "commissioning_terms": _term_count(normalized, ["commission", "commissioning", "test"]),
        "schedule_terms": _term_count(
            normalized, ["schedule", "equipment schedule", "points schedule"]
        ),
        "drawing_terms": _term_count(normalized, ["drawing", "scale", "plan", "elevation"]),
        "revision_terms": _term_count(normalized, ["revision", "rev", "drawing no"]),
        "table_label_terms": _term_count(
            normalized, ["item", "description", "manufacturer", "model", "ref"]
        ),
        "maintenance_terms": _term_count(normalized, ["maintenance", "inspection", "cleaning"]),
        "coshh_terms": _term_count(normalized, ["coshh", "safety data sheet", "material safety"]),
    }


def extract_contents_index_entries(
    classifications: list[dict[str, Any]],
    cached_pages: dict[str, list[dict[str, Any]]],
    hierarchy_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    text_lookup = {
        (source_id, int(page["page_number"])): str(page.get("extracted_text") or "")
        for source_id, pages in cached_pages.items()
        for page in pages
    }
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None]] = set()
    for page in classifications:
        if page.get("primary_page_type") != "index_or_contents":
            continue
        source_id = str(page["source_id"])
        for entry in parse_contents_entries(
            text_lookup.get((source_id, int(page["page_number"])), ""),
            source_id=source_id,
            source_filename=str(page.get("source_filename") or ""),
            index_page=int(page["page_number"]),
        ):
            key = (source_id, entry["section_label"], entry.get("printed_page"))
            if key not in seen:
                seen.add(key)
                entries.append(entry)
    for node in hierarchy_nodes:
        title = str(node.get("title") or node.get("text_summary") or "")
        source_id = str(node.get("source_id") or "")
        if not source_id or not looks_like_section_title(title):
            continue
        label, section_title = split_section_label_title(title)
        key = (source_id, label, int(node.get("page_start") or 0))
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "entry_id": f"hier_{source_id}_{len(entries) + 1:04d}",
                "source_id": source_id,
                "source_filename": "",
                "index_page": None,
                "section_label": label,
                "section_title": section_title,
                "normalized_section_title": normalize_heading(section_title),
                "parent_section": parent_section_label(label),
                "printed_page": None,
                "candidate_pdf_page": int(node.get("page_start") or 0) or None,
                "entry_type": classify_entry_type(title),
                "extraction_method": "existing_hierarchy_node",
            }
        )
    return sorted(
        entries, key=lambda item: (item["source_id"], item["section_label"], item["entry_id"])
    )


def parse_contents_entries(
    text: str,
    *,
    source_id: str,
    source_filename: str,
    index_page: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    index = 0
    while index < len(lines):
        line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        combined = f"{line} {next_line}".strip()
        match = re.match(r"^([A-Z]|\d+(?:\.\d+){0,4})\s+(.+?)(?:\s+(\d{1,4}))?$", combined)
        if match and _valid_section_label(match.group(1)):
            label = match.group(1)
            title = re.sub(r"\s+\d{1,4}$", "", match.group(2)).strip(" .:-")
            if title and len(title) > 2:
                entries.append(
                    _contents_entry(
                        source_id=source_id,
                        source_filename=source_filename,
                        index_page=index_page,
                        label=label,
                        title=title,
                        printed_page=int(match.group(3)) if match.group(3) else None,
                        ordinal=len(entries) + 1,
                    )
                )
                index += 2
                continue
        inline = re.finditer(r"\b([A-Z]|\d+(?:\.\d+){1,4})\s+([A-Z][A-Z0-9 &/(),.-]{3,})", line)
        for match_inline in inline:
            label = match_inline.group(1)
            title = match_inline.group(2).strip(" .:-")
            if _valid_section_label(label):
                entries.append(
                    _contents_entry(
                        source_id=source_id,
                        source_filename=source_filename,
                        index_page=index_page,
                        label=label,
                        title=title,
                        printed_page=None,
                        ordinal=len(entries) + 1,
                    )
                )
        index += 1
    return entries


def resolve_section_boundaries(
    *,
    index_entries: list[dict[str, Any]],
    hierarchy_nodes: list[dict[str, Any]],
    page_counts: dict[str, int | None],
    cached_pages: dict[str, list[dict[str, Any]]],
    sources: list[SourceRegistryEntry],
) -> list[dict[str, Any]]:
    source_names = {source.source_id: Path(source.logical_path).name for source in sources}
    nodes_by_source = _section_nodes_by_source(hierarchy_nodes)
    source_ordinals = entry_ordinals_by_source(index_entries)
    index_anchor_by_source = {
        source_id: max(
            [
                int(entry.get("index_page") or 0)
                for entry in index_entries
                if entry["source_id"] == source_id
            ]
            or [0]
        )
        for source_id in {str(entry["source_id"]) for entry in index_entries}
    }
    cached_numbers = {
        source_id: {int(page["page_number"]) for page in pages}
        for source_id, pages in cached_pages.items()
    }
    resolved: list[dict[str, Any]] = []
    for entry in index_entries:
        source_id = str(entry["source_id"])
        total = page_counts.get(source_id) or 0
        node = best_matching_node(entry, nodes_by_source.get(source_id, []))
        start = (
            int(node.get("page_start") or 0) if node else int(entry.get("candidate_pdf_page") or 0)
        )
        if start <= 0 and entry.get("printed_page"):
            start = resolve_printed_page_to_pdf_page(entry, cached_numbers.get(source_id, set()))
        fallback_used = False
        if start <= 0 and entry.get("extraction_method") == "contents_page_text":
            ordinal = source_ordinals.get(str(entry["entry_id"]), 0)
            anchor = index_anchor_by_source.get(source_id, 0)
            start = min(max(anchor + 1 + (ordinal * 5), 1), total)
            fallback_used = True
        if start <= 0 or start > total:
            resolved.append(
                _unresolved_section(entry, source_names.get(source_id, ""), "no_start_page")
            )
            continue
        end = (
            int(node.get("page_end") or start)
            if node
            else min(start + (4 if fallback_used else 14), total)
        )
        end = max(start, min(end, total, start + 14))
        wanted = set(range(start, end + 1))
        cached = cached_numbers.get(source_id, set())
        resolved.append(
            {
                **entry,
                "source_filename": source_names.get(source_id, entry.get("source_filename", "")),
                "pdf_page_start": start,
                "pdf_page_end": end,
                "page_count": end - start + 1,
                "resolution_method": "hierarchy_title_match"
                if node
                else "index_sequence_after_contents"
                if fallback_used
                else "index_sequence_or_printed_page",
                "resolution_confidence": 0.9 if node else 0.45 if fallback_used else 0.55,
                "range_already_cached": wanted <= cached,
                "cached_pages_in_range": sorted(wanted & cached),
                "uncached_pages_in_range": sorted(wanted - cached),
                "bounded_expansion_required": bool(wanted - cached),
                "resolved": True,
                "unresolved_reason": "",
            }
        )
    return sorted(
        resolved,
        key=lambda item: (
            str(item["source_filename"]),
            int(item.get("pdf_page_start") or 999999),
            str(item["section_label"]),
        ),
    )


def score_sections(resolved_sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section in resolved_sections:
        title = f"{section.get('section_label', '')} {section.get('section_title', '')}".lower()
        score = 0
        reasons: list[str] = []
        dimensions = {
            "certificate_likelihood": _score_terms(title, ["certificate", "test", "commission"]),
            "schedule_likelihood": _score_terms(title, ["schedule", "points", "equipment"]),
            "manufacturer_model_likelihood": _score_terms(
                title, ["manufacturer", "model", "supplier", "product"]
            ),
            "date_reference_likelihood": _score_terms(
                title, ["certificate", "completion", "guarantee"]
            ),
            "dimension_capacity_likelihood": _score_terms(
                title, ["load", "capacity", "rating", "dimension"]
            ),
            "count_likelihood": _score_terms(
                title, ["door", "dock", "parking", "charger", "meter"]
            ),
            "component_description_likelihood": _score_terms(title, HIGH_VALUE_TERMS),
            "statutory_evidence_likelihood": _score_terms(
                title,
                ["fire", "building control", "strategy", "hazard", "structural", "roof access"],
            ),
            "duplication_risk": _score_terms(title, ["appendix", "literature"]),
            "maintenance_only_risk": _score_terms(title, ["maintenance", "manual", "cleaning"]),
            "coshh_safety_data_risk": _score_terms(title, ["coshh", "safety data"]),
            "visual_only_risk": _score_terms(title, ["drawing", "plan", "elevation"]),
        }
        boosts = [
            "certificate_likelihood",
            "schedule_likelihood",
            "manufacturer_model_likelihood",
            "date_reference_likelihood",
            "dimension_capacity_likelihood",
            "count_likelihood",
            "component_description_likelihood",
            "statutory_evidence_likelihood",
        ]
        for key in boosts:
            if dimensions[key]:
                score += dimensions[key] * 10
                reasons.append(f"{key}+{dimensions[key] * 10}")
        for key in ["maintenance_only_risk", "coshh_safety_data_risk", "visual_only_risk"]:
            if dimensions[key]:
                score -= dimensions[key] * 8
                reasons.append(f"{key}-{dimensions[key] * 8}")
        page_cost = len(section.get("uncached_pages_in_range", []))
        if page_cost:
            score -= min(page_cost, 10)
            reasons.append(f"page_cost-{min(page_cost, 10)}")
        if section.get("range_already_cached"):
            score -= 5
            reasons.append("already_cached-5")
        if section.get("resolved"):
            score += 10
            reasons.append("resolved+10")
        rows.append(
            {
                **section,
                "evidence_value_score": score,
                "scoring_dimensions": dimensions,
                "scoring_reasons": reasons,
                "recommended_route": route_for_section(title),
                "likely_dictionary_domains": infer_domains(
                    str(section.get("source_filename") or ""), title
                ),
                "likely_target_families": infer_target_families(
                    str(section.get("source_filename") or ""), title
                ),
            }
        )
    return sorted(
        rows, key=lambda item: (-int(item["evidence_value_score"]), str(item["source_filename"]))
    )


def build_uncached_section_candidates(
    *,
    scored_sections: list[dict[str, Any]],
    page_map: list[dict[str, Any]],
    page_counts: dict[str, int | None],
) -> list[dict[str, Any]]:
    uncached = {
        (str(row["source_id"]), int(row["page_number"]))
        for row in page_map
        if row["cache_status"] == "uncached"
    }
    rows: list[dict[str, Any]] = []
    for section in scored_sections:
        if not section.get("resolved"):
            continue
        missing = [
            page
            for page in range(int(section["pdf_page_start"]), int(section["pdf_page_end"]) + 1)
            if (str(section["source_id"]), page) in uncached
        ]
        expansion_basis = "within_resolved_range"
        if not missing and int(section["evidence_value_score"]) >= 15:
            missing = adjacent_uncached_pages(section, uncached, page_counts)
            expansion_basis = "adjacent_to_cached_section_anchor"
        if not missing:
            continue
        if int(section["evidence_value_score"]) < 12:
            continue
        rows.append(
            {
                **section,
                "candidate_id": f"sec_{len(rows) + 1:04d}",
                "missing_pages": missing,
                "missing_page_count": len(missing),
                "within_source_bounds": max(missing)
                <= (page_counts.get(str(section["source_id"])) or 0),
                "candidate_decision": "candidate",
                "expansion_basis": expansion_basis,
            }
        )
    return sorted(
        rows, key=lambda item: (-int(item["evidence_value_score"]), item["source_filename"])
    )


def adjacent_uncached_pages(
    section: dict[str, Any],
    uncached: set[tuple[str, int]],
    page_counts: dict[str, int | None],
) -> list[int]:
    source_id = str(section["source_id"])
    total = page_counts.get(source_id) or 0
    start = int(section["pdf_page_end"]) + 1
    pages = [
        page
        for page in range(start, min(total, start + 80) + 1)
        if (source_id, page) in uncached
    ]
    return pages[:5]


def approve_section_parse_plan(
    candidates: list[dict[str, Any]],
    *,
    max_new_pages: int,
    per_source_caps: dict[str, int],
) -> list[dict[str, Any]]:
    approved: list[dict[str, Any]] = []
    used_pages: set[tuple[str, int]] = set()
    source_usage: Counter[str] = Counter()
    total_used = 0
    for candidate in candidates:
        source_id = str(candidate["source_id"])
        family_key = source_family_key(str(candidate["source_filename"]))
        if family_key not in per_source_caps:
            continue
        source_cap = per_source_caps[family_key]
        if source_usage[source_id] >= source_cap or total_used >= max_new_pages:
            continue
        remaining_source = source_cap - source_usage[source_id]
        remaining_total = max_new_pages - total_used
        selected = [
            page for page in candidate["missing_pages"] if (source_id, int(page)) not in used_pages
        ][: min(remaining_source, remaining_total, DEFAULT_MAX_BATCH_PAGES)]
        if not selected:
            continue
        for page in selected:
            used_pages.add((source_id, int(page)))
        source_usage[source_id] += len(selected)
        total_used += len(selected)
        for start, end in split_contiguous_pages(selected, DEFAULT_MAX_BATCH_PAGES):
            approved.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "source_id": source_id,
                    "source_filename": candidate["source_filename"],
                    "section_label": candidate["section_label"],
                    "section_title": candidate["section_title"],
                    "pdf_page_start": candidate["pdf_page_start"],
                    "pdf_page_end": candidate["pdf_page_end"],
                    "page_start": start,
                    "page_end": end,
                    "page_count": end - start + 1,
                    "approved_pages": list(range(start, end + 1)),
                    "evidence_value_score": candidate["evidence_value_score"],
                    "recommended_route": candidate.get("recommended_route")
                    or route_for_section(str(candidate["section_title"])),
                    "approval_rationale": (
                        "Index-guided high-value section within source and total caps."
                    ),
                }
            )
    return sorted(approved, key=lambda item: (item["source_filename"], item["page_start"]))


def split_approved_section_batches(approved: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                    "batch_id": f"idx_{len(batches) + 1:04d}",
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
                }
            )
    return sorted(batches, key=lambda item: (item["source_filename"], item["page_start"]))


def section_inventory_from_resolved_sections(
    resolved_sections: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    sources: list[SourceRegistryEntry],
) -> list[dict[str, Any]]:
    by_key = {(str(page["source_id"]), int(page["page_number"])): page for page in classifications}
    source_names = {source.source_id: Path(source.logical_path).name for source in sources}
    rows: list[dict[str, Any]] = []
    for section in resolved_sections:
        if not section.get("resolved"):
            continue
        pages = [
            by_key[(str(section["source_id"]), page)]
            for page in range(int(section["pdf_page_start"]), int(section["pdf_page_end"]) + 1)
            if (str(section["source_id"]), page) in by_key
        ]
        page_types = Counter(str(page["primary_page_type"]) for page in pages)
        routes = Counter(str(page["recommended_route"]) for page in pages)
        title = str(section["section_title"])
        route = route_for_section(title)
        if routes:
            route = routes.most_common(1)[0][0]
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
                "section": section["section_title"],
                "subsection": section["section_label"],
                "element_sheet": title if "element" in title.lower() else "",
                "certificate_type": title if "certificate" in title.lower() else "",
                "schedule_type": title if "schedule" in title.lower() else "",
                "drawing_title": title if route in {"drawing_text", "visual"} else "",
                "page_start": section["pdf_page_start"],
                "page_end": section["pdf_page_end"],
                "source_page_labels": [],
                "cached_page_coverage": round(len(pages) / int(section["page_count"]), 4)
                if int(section["page_count"])
                else 0,
                "primary_page_types": dict(sorted(page_types.items())),
                "secondary_evidence_tags": sorted(
                    {tag for page in pages for tag in page["secondary_tags"]}
                ),
                "likely_dictionary_domains": section.get("likely_dictionary_domains")
                or infer_domains(str(section.get("source_filename") or ""), title),
                "likely_target_families": section.get("likely_target_families")
                or infer_target_families(str(section.get("source_filename") or ""), title),
                "recommended_route": route,
                "evidence_value_score": section.get("evidence_value_score", 0),
                "section_resolution_confidence": section.get("resolution_confidence", 0),
                "evidence_bearing": bool(pages)
                or int(section.get("evidence_value_score", 0)) >= 20,
            }
        )
    return sorted(
        rows, key=lambda item: (item["source_filename"], item["page_start"], item["section"])
    )


def build_index_expansion_metrics(
    *,
    sources: list[SourceRegistryEntry],
    page_counts: dict[str, int | None],
    baseline: dict[str, Any],
    cached_before: dict[str, list[dict[str, Any]]],
    cached_after: dict[str, list[dict[str, Any]]],
    reclassification: list[dict[str, Any]],
    index_entries: list[dict[str, Any]],
    resolved_sections: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    approved: list[dict[str, Any]],
    parse_report: dict[str, Any],
    classifications_after: list[dict[str, Any]],
    sections_after: list[dict[str, Any]],
    runtime_ms: float,
) -> dict[str, Any]:
    total_pages = sum(count or 0 for count in page_counts.values())
    before = sum(len(pages) for pages in cached_before.values())
    after = sum(len(pages) for pages in cached_after.values())
    page_types = Counter(str(row["primary_page_type"]) for row in classifications_after)
    routes = Counter(str(row["recommended_route"]) for row in classifications_after)
    unresolved = [section for section in resolved_sections if not section.get("resolved")]
    high_value_remaining = [
        candidate
        for candidate in candidates
        if candidate["candidate_id"] not in {item["candidate_id"] for item in approved}
    ][:25]
    readiness = readiness_status(
        sections_after=sections_after,
        page_types=page_types,
        after_cached=after,
        total_pages=total_pages,
        approved=approved,
    )
    return {
        "total_registered_pdfs": sum(1 for source in sources if source.file_type == FileType.PDF),
        "total_pdf_pages": total_pages,
        "historical_baseline_availability": baseline[
            "historical_baseline_before_source_coverage_v1"
        ]["status"],
        "current_cached_pages_before_this_sprint": before,
        "newly_parsed_pages": parse_report["new_pages_parsed"],
        "cached_pages_after_this_sprint": after,
        "coverage_before_percentage": round(before / total_pages * 100, 2) if total_pages else 0.0,
        "coverage_after_percentage": round(after / total_pages * 100, 2) if total_pages else 0.0,
        "coverage_change_by_source": coverage_change_by_source(
            cached_before, cached_after, sources, page_counts
        ),
        "number_of_v1_unknown_pages": len(reclassification),
        "number_of_unknown_pages_reclassified": sum(
            1 for row in reclassification if row["classification_changed"]
        ),
        "unknown_pages_remaining": page_types.get("unknown", 0),
        "detected_index_entries": len(index_entries),
        "resolved_section_boundaries": sum(
            1 for section in resolved_sections if section.get("resolved")
        ),
        "unresolved_index_entries": len(unresolved),
        "high_value_candidate_sections": len(candidates),
        "approved_sections": len({item["candidate_id"] for item in approved}),
        "parser_batches": parse_report["parser_batches_run"],
        "failures": parse_report["parse_failure_count"],
        "timeouts_or_restarts": parse_report["timeout_or_restart_count"],
        "page_type_distribution_after_expansion": dict(sorted(page_types.items())),
        "evidence_bearing_sections_by_source": dict(
            Counter(
                str(section["source_filename"])
                for section in sections_after
                if section["evidence_bearing"]
            )
        ),
        "table_routes": routes.get("table", 0),
        "certificate_routes": routes.get("certificate", 0) + page_types.get("certificate", 0),
        "visual_routes": routes.get("visual", 0),
        "low_value_sections": sum(
            1 for section in sections_after if section["recommended_route"] == "deprioritized"
        ),
        "target_family_support_coverage": sorted(
            {family for section in sections_after for family in section["likely_target_families"]}
        ),
        "remaining_high_priority_uncached_sections": high_value_remaining,
        "target_selection_readiness": readiness,
        "runtime_ms": round(runtime_ms, 2),
    }


def write_index_expansion_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_outputs = {
        "v1_artifact_audit.json": result["v1_artifact_audit"],
        "coverage_baseline_reconciliation.json": result["coverage_baseline_reconciliation"],
        "unknown_page_reclassification.json": result["unknown_page_reclassification"],
        "contents_index_entries.json": result["contents_index_entries"],
        "resolved_section_boundaries.json": result["resolved_section_boundaries"],
        "section_priority_scores.json": result["section_priority_scores"],
        "uncached_section_candidates.json": result["uncached_section_candidates"],
        "approved_section_parse_plan.json": result["approved_section_parse_plan"],
        "merged_parse_batches.json": result["merged_parse_batches"],
        "parse_execution_report.json": result["parse_execution_report"],
        "cache_coverage_before.json": result["cache_coverage_before"],
        "cache_coverage_after.json": result["cache_coverage_after"],
        "newly_cached_pages.json": result["newly_cached_pages"],
        "page_classifications_after.json": result["page_classifications_after"],
        "section_inventory_after.json": result["section_inventory_after"],
        "evidence_section_map_after.json": result["evidence_section_map_after"],
        "enriched_pageindex.json": result["enriched_pageindex"],
        "source_family_support_map_after.json": result["source_family_support_map_after"],
        "table_route_pages.json": result["table_route_pages"],
        "certificate_route_pages.json": result["certificate_route_pages"],
        "visual_route_pages.json": result["visual_route_pages"],
        "low_value_pages.json": result["low_value_pages"],
        "unknown_pages_remaining.json": result["unknown_pages_remaining"],
        "coverage_metrics.json": result["coverage_metrics"],
        "coverage_trace.json": result["coverage_trace"],
    }
    for filename, payload in json_outputs.items():
        _atomic_write_json(output_dir / filename, payload)
    csv_outputs = {
        "unknown_page_reclassification.csv": result["unknown_page_reclassification"],
        "contents_index_entries.csv": result["contents_index_entries"],
        "resolved_section_boundaries.csv": result["resolved_section_boundaries"],
        "section_priority_scores.csv": result["section_priority_scores"],
        "uncached_section_candidates.csv": result["uncached_section_candidates"],
        "approved_section_parse_plan.csv": result["approved_section_parse_plan"],
        "section_inventory_after.csv": result["section_inventory_after"],
        "evidence_section_map_after.csv": result["evidence_section_map_after"],
        "source_coverage_review.csv": result["source_coverage_review"],
    }
    for filename, rows in csv_outputs.items():
        write_csv(output_dir / filename, rows)
    (output_dir / "source_coverage_summary.md").write_text(
        summary_markdown(result), encoding="utf-8"
    )
    (output_dir / "next_target_selection_readiness.md").write_text(
        readiness_markdown(result["coverage_metrics"]), encoding="utf-8"
    )


HIGH_VALUE_TERMS = [
    "roof",
    "cladding",
    "wall",
    "floor",
    "door",
    "loading",
    "dock",
    "leveller",
    "fire alarm",
    "lighting",
    "meter",
    "bms",
    "photovoltaic",
    "drainage",
    "bollard",
    "gate",
    "cycle",
    "barrier",
    "retaining",
    "structural",
    "load",
    "guarantee",
]


def merge_reclassifications(
    prior: list[dict[str, Any]], reclassification: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {(row["source_id"], int(row["page_number"])): row for row in reclassification}
    merged = []
    for row in prior:
        key = (row["source_id"], int(row["page_number"]))
        if key in by_key and by_key[key]["classification_changed"]:
            update = by_key[key]
            merged.append(
                {
                    **row,
                    "primary_page_type": update["primary_page_type"],
                    "secondary_tags": update["secondary_tags"],
                    "recommended_route": update["recommended_route"],
                }
            )
        else:
            merged.append(row)
    return merged


def entry_ordinals_by_source(entries: list[dict[str, Any]]) -> dict[str, int]:
    counters: Counter[str] = Counter()
    ordinals: dict[str, int] = {}
    for entry in entries:
        source_id = str(entry["source_id"])
        counters[source_id] += 1
        ordinals[str(entry["entry_id"])] = counters[source_id]
    return ordinals


def normalize_heading(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def split_section_label_title(value: str) -> tuple[str, str]:
    cleaned = re.sub(r"\s+", " ", value).strip()
    element = re.match(r"^element\s*:\s*(\d+(?:\.\d+)*)\s+(.+)$", cleaned, re.I)
    if element:
        return element.group(1), element.group(2).strip()
    match = re.match(r"^([A-Z]|\d+(?:\.\d+){0,4})\s*[-:]?\s+(.+)$", cleaned)
    if match:
        return match.group(1), match.group(2).strip()
    embedded = re.search(r"\b([A-Z])\s*-\s*([A-Z][A-Z0-9 /&(),.-]{5,})", cleaned)
    if embedded:
        return embedded.group(1), embedded.group(2).strip()
    return "", cleaned


def looks_like_section_title(value: str) -> bool:
    label, title = split_section_label_title(value)
    return bool(label and title and not title.lower().startswith("page "))


def parent_section_label(label: str) -> str:
    if "." in label:
        return ".".join(label.split(".")[:-1])
    return ""


def classify_entry_type(title: str) -> str:
    lower = title.lower()
    if "certificate" in lower or "test" in lower or "commission" in lower:
        return "certificate_or_commissioning"
    if "schedule" in lower:
        return "schedule"
    if any(term in lower for term in ["drawing", "plan", "elevation"]):
        return "drawing_appendix"
    if any(term in lower for term in ["element", "roof", "door", "wall", "floor"]):
        return "element_or_component_section"
    return "section"


def resolve_printed_page_to_pdf_page(entry: dict[str, Any], cached_pages: set[int]) -> int:
    printed = int(entry.get("printed_page") or 0)
    if printed in cached_pages:
        return printed
    for offset in range(-5, 6):
        if printed + offset in cached_pages:
            return printed + offset
    return printed


def infer_section_end(
    entry: dict[str, Any], entries: list[dict[str, Any]], total_pages: int
) -> int:
    start = int(entry.get("candidate_pdf_page") or entry.get("printed_page") or 0)
    if start <= 0:
        return 0
    sibling_starts = sorted(
        int(item.get("candidate_pdf_page") or item.get("printed_page") or 0)
        for item in entries
        if item["source_id"] == entry["source_id"]
        and int(item.get("candidate_pdf_page") or item.get("printed_page") or 0) > start
    )
    return min((sibling_starts[0] - 1) if sibling_starts else start + 4, total_pages)


def best_matching_node(entry: dict[str, Any], nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    label = str(entry.get("section_label") or "")
    title_norm = str(entry.get("normalized_section_title") or "")
    best: dict[str, Any] | None = None
    best_score = 0
    for node in nodes:
        node_title = str(node.get("title") or node.get("text_summary") or "")
        node_label, node_section = split_section_label_title(node_title)
        node_norm = normalize_heading(node_section or node_title)
        score = 0
        if label and label == node_label:
            score += 5
        if title_norm and (title_norm in node_norm or node_norm in title_norm):
            score += 4
        if score > best_score:
            best_score = score
            best = node
    return best if best_score >= 5 else None


def _section_nodes_by_source(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        source_id = str(node.get("source_id") or "")
        if source_id and looks_like_section_title(
            str(node.get("title") or node.get("text_summary") or "")
        ):
            grouped[source_id].append(node)
    return grouped


def _unresolved_section(entry: dict[str, Any], source_filename: str, reason: str) -> dict[str, Any]:
    return {
        **entry,
        "source_filename": source_filename,
        "pdf_page_start": None,
        "pdf_page_end": None,
        "page_count": 0,
        "resolution_method": "unresolved",
        "resolution_confidence": 0.0,
        "range_already_cached": False,
        "cached_pages_in_range": [],
        "uncached_pages_in_range": [],
        "bounded_expansion_required": False,
        "resolved": False,
        "unresolved_reason": reason,
    }


def _contents_entry(
    *,
    source_id: str,
    source_filename: str,
    index_page: int,
    label: str,
    title: str,
    printed_page: int | None,
    ordinal: int,
) -> dict[str, Any]:
    return {
        "entry_id": f"idx_{source_id}_{index_page:04d}_{ordinal:03d}",
        "source_id": source_id,
        "source_filename": source_filename,
        "index_page": index_page,
        "section_label": label,
        "section_title": title,
        "normalized_section_title": normalize_heading(title),
        "parent_section": parent_section_label(label),
        "printed_page": printed_page,
        "candidate_pdf_page": printed_page,
        "entry_type": classify_entry_type(title),
        "extraction_method": "contents_page_text",
    }


def _valid_section_label(label: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]|\d+(?:\.\d+){0,4}", label))


def _classification_v2(
    primary: str,
    route: str,
    tags: set[str],
    signals: dict[str, Any],
    confidence: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "primary_page_type": primary,
        "secondary_tags": sorted(tags),
        "recommended_route": route,
        "rule_signals": signals,
        "classification_score": round(confidence, 4),
        "classification_confidence": "high"
        if confidence >= 0.7
        else "medium"
        if confidence >= 0.4
        else "low",
        "classification_reason": reason,
    }


def route_for_section(title: str) -> str:
    lower = title.lower()
    if "certificate" in lower or "commission" in lower or "test" in lower:
        return "certificate"
    if "schedule" in lower or "points" in lower:
        return "table"
    if any(term in lower for term in ["drawing", "plan", "elevation"]):
        return "drawing_text"
    if any(term in lower for term in ["maintenance", "coshh", "safety data"]):
        return "deprioritized"
    return "text"


def source_caps(max_new_pages: int) -> dict[str, int]:
    scale = max_new_pages / 120 if max_new_pages else 0
    return {key: max(1, int(value * scale)) for key, value in MAJOR_SOURCE_ALLOCATION.items()}


def source_family_key(filename: str) -> str:
    lower = filename.lower()
    for key in MAJOR_SOURCE_ALLOCATION:
        if key in lower:
            return key
    return "other"


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


def readiness_status(
    *,
    sections_after: list[dict[str, Any]],
    page_types: Counter[str],
    after_cached: int,
    total_pages: int,
    approved: list[dict[str, Any]],
) -> str:
    high_value_sections = [
        section
        for section in sections_after
        if int(section.get("evidence_value_score") or 0) >= 20 and section["evidence_bearing"]
    ]
    source_count = len({section["source_id"] for section in high_value_sections})
    useful_types = page_types.get("schedule", 0) + page_types.get("certificate", 0)
    unknown_ratio = page_types.get("unknown", 0) / after_cached if after_cached else 1.0
    if source_count >= 4 and useful_types >= 20 and unknown_ratio < 0.35 and approved:
        return "ready for evidence-backed target-family selection"
    if high_value_sections and total_pages and after_cached / total_pages > 0.04:
        return "partially ready"
    return "not ready"


def build_coverage_trace(**items: Any) -> list[dict[str, Any]]:
    return [{"stage": key, "summary": _trace_summary(value)} for key, value in items.items()]


def _trace_summary(value: Any) -> Any:
    if isinstance(value, list):
        return {"rows": len(value)}
    if isinstance(value, dict):
        return {
            key: value[key]
            for key in sorted(value)
            if key in {"status", "new_pages_parsed", "parser_batches_run", "parse_failure_count"}
        }
    return value


def build_source_review(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    before_by_id = {row["source_id"]: row for row in before}
    section_counts = Counter(
        str(section["source_id"]) for section in sections if section["evidence_bearing"]
    )
    return [
        {
            "source_id": row["source_id"],
            "source_filename": row["source_filename"],
            "cached_before": before_by_id[row["source_id"]]["cached_page_count"],
            "cached_after": row["cached_page_count"],
            "cache_percentage_after": row["cache_percentage"],
            "evidence_bearing_sections": section_counts[str(row["source_id"])],
            "review_note": "needs further section-guided parsing"
            if row["source_type"] == "pdf" and float(row["cache_percentage"]) < 10.0
            else "sufficient sampled coverage for now",
        }
        for row in after
    ]


def part_from_filename(filename: str) -> str:
    match = re.search(r"part\s+(\d+)", filename.lower())
    return f"Part {match.group(1)}" if match else ""


def summary_markdown(result: dict[str, Any]) -> str:
    metrics = result["coverage_metrics"]
    return (
        "# Index-Guided Evidence Section Expansion V1\n\n"
        f"- V1 artifact audit: {result['v1_artifact_audit']['status']}\n"
        f"- Historical baseline: {metrics['historical_baseline_availability']}\n"
        f"- Cached before: {metrics['current_cached_pages_before_this_sprint']}\n"
        f"- Newly parsed pages: {metrics['newly_parsed_pages']}\n"
        f"- Cached after: {metrics['cached_pages_after_this_sprint']}\n"
        f"- Coverage after: {metrics['coverage_after_percentage']}%\n"
        f"- Index entries detected: {metrics['detected_index_entries']}\n"
        f"- Section boundaries resolved: {metrics['resolved_section_boundaries']}\n"
        f"- Unknown pages remaining: {metrics['unknown_pages_remaining']}\n"
        f"- Readiness: {metrics['target_selection_readiness']}\n"
    )


def readiness_markdown(metrics: dict[str, Any]) -> str:
    return (
        "# Next Target Selection Readiness\n\n"
        f"- Status: {metrics['target_selection_readiness']}\n"
        f"- High-value candidate sections: {metrics['high_value_candidate_sections']}\n"
        f"- Approved sections: {metrics['approved_sections']}\n"
        f"- Remaining high-priority uncached sections: "
        f"{len(metrics['remaining_high_priority_uncached_sections'])}\n"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def _route_pages(rows: list[dict[str, Any]], routes: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if row["recommended_route"] in routes]


def _range_within_bounds(section: dict[str, Any], page_counts: dict[str, int | None]) -> bool:
    source_id = str(section.get("source_id") or "")
    total = page_counts.get(source_id)
    start = int(section.get("page_start") or section.get("pdf_page_start") or 0)
    end = int(section.get("page_end") or section.get("pdf_page_end") or start)
    return bool(total is None or (start >= 1 and end <= total and end >= start))


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _term_count(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if term in text)


def _score_terms(text: str, terms: list[str]) -> int:
    return min(3, _term_count(text, terms))


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(item) for item in payload] if isinstance(payload, list) else []


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _source_sort_key(source: SourceRegistryEntry) -> tuple[str, str]:
    return (Path(source.logical_path).name.lower(), source.source_id)
