"""Commissioning, statutory and installation evidence-gap refinement."""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from segro_evidence_extraction.high_value_section_expansion_v2 import (
    DEFAULT_HIGH_VALUE_EXPANSION_OUTPUT_DIR,
    apply_classification_corrections,
    classify_page_text_v3,
    coverage_change_by_source,
    diff_cached_pages,
)
from segro_evidence_extraction.index_guided_section_expansion import _read_json_list
from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry
from segro_evidence_extraction.parsing.batch_worker import DEFAULT_MAX_BATCH_PAGES
from segro_evidence_extraction.parsing.service import load_source_registry
from segro_evidence_extraction.source_coverage_pageindex import (
    DEFAULT_HIERARCHY_PATH,
    DEFAULT_PAGE_CACHE_ROOT,
    DEFAULT_SOURCE_MANIFEST,
    build_enriched_pageindex,
    build_source_family_support_map,
    build_source_inventory,
    cache_coverage_snapshot,
    classify_cached_pages,
    determine_pdf_page_counts,
    execute_parse_batches,
    infer_domains,
    load_canonical_cached_pages,
    load_hierarchy_nodes,
    split_contiguous_pages,
    validate_parse_batches,
    write_csv,
)
from segro_evidence_extraction.vertical_slice import _atomic_write_json

DEFAULT_EVIDENCE_GAP_OUTPUT_DIR = Path("output/enfield_unit1_evidence_gap_refinement_v1")
DEFAULT_GAP_SOURCE_CONFIG_PATH = Path("config/enfield_unit1_evidence_gap_source_config_v1.json")
GAP_FAMILIES = ["commissioning_results", "statutory_compliance", "installation_details"]
Family = Literal["commissioning_results", "statutory_compliance", "installation_details"]


def run_evidence_gap_refinement_v1(
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    cache_root: Path = DEFAULT_PAGE_CACHE_ROOT,
    v2_output_dir: Path = DEFAULT_HIGH_VALUE_EXPANSION_OUTPUT_DIR,
    hierarchy_path: Path = DEFAULT_HIERARCHY_PATH,
    source_config_path: Path = DEFAULT_GAP_SOURCE_CONFIG_PATH,
    output_dir: Path = DEFAULT_EVIDENCE_GAP_OUTPUT_DIR,
    max_new_pages: int = 60,
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
    provenance = reconcile_cache_provenance(v2_output_dir, cached_before)
    base_classifications = classify_cached_pages(cached_before, sources)
    corrected_before = apply_classification_corrections(
        base_classifications,
        current_classification_corrections(base_classifications, cached_before, sources),
    )
    baseline = build_evidence_gap_baseline(corrected_before, cached_before)
    candidates = build_gap_section_candidates(
        v2_output_dir=v2_output_dir,
        baseline=baseline,
        cached_pages=cached_before,
        page_counts=page_counts,
        source_frontier_templates=load_source_frontier_templates(source_config_path),
    )
    approved = approve_gap_parse_plan(candidates, max_new_pages=max_new_pages)
    batches = split_gap_batches(approved)
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
    after_raw = classify_cached_pages(cached_after, sources)
    classifications_after = apply_classification_corrections(
        after_raw,
        current_classification_corrections(after_raw, cached_after, sources),
    )
    family_sections = validate_family_sections(classifications_after, cached_after)
    evidence_map = evidence_map_from_family_sections(family_sections)
    inventory_after = build_source_inventory(
        sources=sources,
        page_counts=page_counts,
        cached_pages=cached_after,
        hierarchy_nodes=hierarchy_nodes,
        parser_fingerprint={},
    )
    validation = build_family_validation(family_sections)
    remaining = [family for family, row in validation.items() if row["status"] != "valid"]
    metrics = build_gap_metrics(
        sources=sources,
        page_counts=page_counts,
        cached_before=cached_before,
        cached_after=cached_after,
        provenance=provenance,
        baseline=baseline,
        candidates=candidates,
        approved=approved,
        batches=batches,
        parse_report=parse_report,
        family_sections=family_sections,
        validation=validation,
        remaining=remaining,
        classifications_after=classifications_after,
        runtime_ms=(time.perf_counter() - started) * 1000,
    )
    result = {
        "cache_provenance_reconciliation": provenance,
        "evidence_gap_baseline": baseline,
        "gap_section_candidates": candidates,
        "approved_gap_parse_plan": approved,
        "merged_parse_batches": batches,
        "parse_execution_report": parse_report,
        "newly_cached_pages": newly_cached,
        "cache_coverage_before": cache_coverage_snapshot(inventory_before),
        "cache_coverage_after": cache_coverage_snapshot(inventory_after),
        "page_classifications_after": classifications_after,
        "evidence_section_map_after": evidence_map,
        "enriched_pageindex": build_enriched_pageindex(hierarchy_nodes, evidence_map),
        "commissioning_evidence_sections": family_sections["commissioning_results"],
        "statutory_evidence_sections": family_sections["statutory_compliance"],
        "installation_evidence_sections": family_sections["installation_details"],
        "evidence_family_validation": validation,
        "source_family_support_map_after": build_source_family_support_map(evidence_map),
        "table_route_pages": _route_pages(classifications_after, {"table"}),
        "certificate_route_pages": [
            row for row in classifications_after if row["primary_page_type"] == "certificate"
        ],
        "drawing_text_route_pages": _route_pages(classifications_after, {"drawing_text"}),
        "visual_route_pages": _route_pages(classifications_after, {"visual"}),
        "remaining_evidence_gaps": remaining,
        "coverage_metrics": metrics,
        "coverage_trace": build_gap_trace(
            provenance, baseline, candidates, approved, batches, parse_report
        ),
    }
    write_gap_outputs(result, output_dir)
    return result


def reconcile_cache_provenance(
    v2_output_dir: Path, cached_pages: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    metrics = _read_json_dict(v2_output_dir / "coverage_metrics.json")
    v2_new = _read_json_list(v2_output_dir / "newly_cached_pages.json")
    current_cached = sum(len(pages) for pages in cached_pages.values())
    final_run_before = int(metrics.get("cached_pages_before") or 0)
    final_run_after = int(metrics.get("cached_pages_after") or 0)
    cumulative_v2_start = 291
    return {
        "current_cache_before_this_sprint": current_cached,
        "known_v2_cumulative_start_cache": cumulative_v2_start,
        "known_v2_cumulative_added_pages": max(current_cached - cumulative_v2_start, 0),
        "v2_final_artifact_run_cached_before": final_run_before,
        "v2_final_artifact_run_cached_after": final_run_after,
        "v2_final_artifact_run_new_pages": int(metrics.get("newly_parsed_pages") or 0),
        "v2_final_artifact_newly_cached_records": len(v2_new),
        "earlier_v2_run_added_pages_estimate": max(final_run_before - cumulative_v2_start, 0),
        "provenance_note": (
            "This sprint baseline is the current cache. Earlier V2 additions are "
            "preserved as prior provenance and are not counted as this sprint parsing."
        ),
    }


def current_classification_corrections(
    classifications: list[dict[str, Any]],
    cached_pages: dict[str, list[dict[str, Any]]],
    sources: list[SourceRegistryEntry],
) -> list[dict[str, Any]]:
    text_lookup = {
        (source_id, int(page["page_number"])): str(page.get("extracted_text") or "")
        for source_id, pages in cached_pages.items()
        for page in pages
    }
    source_names = {source.source_id: Path(source.logical_path).name for source in sources}
    corrections = []
    for row in classifications:
        source_id = str(row["source_id"])
        page_number = int(row["page_number"])
        prior = str(row["primary_page_type"])
        enhanced = classify_page_text_v3(
            text_lookup.get((source_id, page_number), ""),
            source_filename=str(row.get("source_filename") or source_names.get(source_id, "")),
            page_number=page_number,
            prior_type=prior,
        )
        if enhanced["primary_page_type"] == prior:
            continue
        corrections.append(
            {
                "source_id": source_id,
                "source_filename": row.get("source_filename") or source_names.get(source_id, ""),
                "page_number": page_number,
                "prior_primary_page_type": prior,
                "primary_page_type": enhanced["primary_page_type"],
                "recommended_route": enhanced["recommended_route"],
                "secondary_tags": enhanced["secondary_tags"],
                "likely_target_families": enhanced["likely_target_families"],
                "evidence_bearing": enhanced["evidence_bearing"],
                "classification_reason": enhanced["classification_reason"],
            }
        )
    return corrections


def build_evidence_gap_baseline(
    classifications: list[dict[str, Any]],
    cached_pages: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    family_sections = validate_family_sections(classifications, cached_pages)
    validation = build_family_validation(family_sections)
    return {
        "family_validation": validation,
        "families_requiring_parsing": [
            family for family, row in validation.items() if row["status"] != "valid"
        ],
        "cached_pages_before": sum(len(pages) for pages in cached_pages.values()),
    }


def validate_family_sections(
    classifications: list[dict[str, Any]],
    cached_pages: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    text_lookup = {
        (source_id, int(page["page_number"])): str(page.get("extracted_text") or "")
        for source_id, pages in cached_pages.items()
        for page in pages
    }
    rows: dict[str, list[dict[str, Any]]] = {family: [] for family in GAP_FAMILIES}
    for page in classifications:
        source_id = str(page["source_id"])
        page_number = int(page["page_number"])
        text = text_lookup.get((source_id, page_number), "")
        for family in GAP_FAMILIES:
            check = validate_family_text(family, text)
            if check["valid"]:
                rows[family].append(
                    {
                        "source_id": source_id,
                        "source_filename": page.get("source_filename", ""),
                        "page_start": page_number,
                        "page_end": page_number,
                        "section": family.replace("_", " ").title(),
                        "subsection": "",
                        "primary_page_types": {page["primary_page_type"]: 1},
                        "secondary_evidence_tags": page.get("secondary_tags", []),
                        "likely_dictionary_domains": infer_domains(
                            str(page.get("source_filename") or ""), text
                        ),
                        "likely_target_families": [family],
                        "recommended_route": page.get("recommended_route", "text"),
                        "evidence_value_score": check["score"],
                        "evidence_bearing": True,
                        "validation_signals": check["signals"],
                        "validation_reason": check["reason"],
                    }
                )
    return {
        family: sorted(items, key=lambda item: (item["source_filename"], item["page_start"]))
        for family, items in rows.items()
    }


def validate_family_text(family: str, text: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    signals = {
        "identity": _has(
            normalized,
            [
                "system",
                "equipment",
                "plant",
                "fire alarm",
                "bms",
                "door",
                "roof",
                "electrical",
                "mechanical",
                "meter",
            ],
        ),
        "commission_event": _has(normalized, ["commission", "test", "inspection", "certification"]),
        "date_or_reference": bool(re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", normalized))
        or _has(normalized, ["reference", "ref", "certificate no", "job no", "cert no"]),
        "result": _has(
            normalized,
            [
                "pass",
                "passed",
                "satisfactory",
                "complete",
                "completed",
                "result",
                "measured",
                "approved",
            ],
        ),
        "authority": _has(
            normalized,
            [
                "building control",
                "approved inspector",
                "authority",
                "regulations",
                "bs ",
                "certifier",
                "niceic",
                "certificate",
            ],
        ),
        "approved_status": _has(
            normalized,
            ["approved", "certified", "compliant", "completion", "satisfactory", "certificate"],
        ),
        "installation_detail": _has(
            normalized,
            [
                "installed",
                "installation",
                "fixed",
                "mounted",
                "configuration",
                "type",
                "location",
                "material",
                "constructed",
                "finish",
            ],
        ),
        "maintenance_only": _has(
            normalized, ["maintenance", "cleaning", "coshh", "safety data sheet"]
        )
        and not _has(normalized, ["certificate", "commission", "installed"]),
    }
    if signals["maintenance_only"]:
        return {
            "valid": False,
            "score": 0,
            "signals": signals,
            "reason": "maintenance-only evidence rejected",
        }
    if family == "commissioning_results":
        valid = (
            signals["identity"]
            and signals["commission_event"]
            and signals["date_or_reference"]
            and signals["result"]
        )
    elif family == "statutory_compliance":
        valid = signals["authority"] and signals["date_or_reference"] and signals["approved_status"]
    else:
        valid = (
            signals["identity"]
            and signals["installation_detail"]
            and not signals["maintenance_only"]
        )
    score = sum(1 for value in signals.values() if value) * 10
    reason = "valid deterministic signal combination" if valid else "heading or partial signal only"
    return {"valid": valid, "score": score, "signals": signals, "reason": reason}


def build_family_validation(family_sections: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    validation = {}
    for family in GAP_FAMILIES:
        sections = family_sections.get(family, [])
        source_count = len({row["source_id"] for row in sections})
        validation[family] = {
            "valid_section_count": len(sections),
            "source_count": source_count,
            "status": "valid" if len(sections) >= 2 and source_count >= 1 else "gap",
        }
    return validation


def build_gap_section_candidates(
    *,
    v2_output_dir: Path,
    baseline: dict[str, Any],
    cached_pages: dict[str, list[dict[str, Any]]],
    page_counts: dict[str, int | None],
    source_frontier_templates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    required = set(baseline["families_requiring_parsing"])
    if not required:
        return []
    v2_remaining = _read_json_list(v2_output_dir / "remaining_high_priority_sections.json")
    source_frontiers = {
        source_id: max(int(page["page_number"]) for page in pages)
        for source_id, pages in cached_pages.items()
        if pages
    }
    rows: list[dict[str, Any]] = []
    for candidate in v2_remaining:
        families = [
            family
            for family in candidate.get("expected_evidence_families", [])
            if family in required
        ]
        if not families:
            continue
        pages = [
            int(page)
            for page in candidate.get("missing_pages", [])
            if int(page)
            not in {
                int(cached["page_number"])
                for cached in cached_pages.get(str(candidate["source_id"]), [])
            }
        ]
        if not pages:
            continue
        rows.append(
            _gap_candidate(candidate, families, pages, len(rows) + 1, "v2_remaining_high_priority")
        )
    for row in frontier_gap_templates(
        required,
        source_frontiers,
        page_counts,
        source_frontier_templates or [],
    ):
        rows.append({**row, "candidate_id": f"gap_{len(rows) + 1:04d}"})
    return sorted(
        rows,
        key=lambda item: (
            -int(item["priority_score"]),
            item["source_filename"],
            item["page_start"],
        ),
    )


def frontier_gap_templates(
    required: set[str],
    source_frontiers: dict[str, int],
    page_counts: dict[str, int | None],
    source_templates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    templates = []
    for source in source_templates:
        source_id = str(source["source_id"])
        filename = str(source["source_filename"])
        section = str(source["section"])
        start = source_frontiers.get(source_id, 0) + 1
        total = page_counts.get(source_id) or 0
        if start <= 1 or start > total:
            continue
        trigger_families = {str(family) for family in source.get("trigger_families", [])}
        if not required & trigger_families:
            continue
        families = [
            str(family)
            for family in source.get("evidence_families", [])
            if str(family) in GAP_FAMILIES
        ]
        families = [
            family for family in families if family in required or family in trigger_families
        ]
        if not families:
            continue
        page_window_size = int(source.get("page_window_size") or 1)
        templates.append(
            _frontier_gap(
                source_id,
                filename,
                section,
                start,
                min(start + page_window_size - 1, total),
                families,
                str(source.get("recommended_route") or "text"),
                int(source.get("priority_score") or 50),
            )
        )
    return templates


def load_source_frontier_templates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected source frontier configuration object: {path}")
    rows = payload.get("source_frontier_templates", [])
    if not isinstance(rows, list):
        raise ValueError(f"Expected source_frontier_templates list: {path}")
    templates = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid source_frontier_templates[{index}] in {path}")
        required_keys = {
            "source_id",
            "source_filename",
            "section",
            "trigger_families",
            "evidence_families",
            "recommended_route",
            "priority_score",
            "page_window_size",
        }
        missing = sorted(required_keys - set(row))
        if missing:
            raise ValueError(f"Missing source frontier keys {missing} in {path}")
        templates.append(dict(row))
    return templates


def _frontier_gap(
    source_id: str,
    filename: str,
    section: str,
    start: int,
    end: int,
    families: list[str],
    route: str,
    score: int,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_filename": filename,
        "section": section,
        "section_label": "",
        "page_start": start,
        "page_end": end,
        "page_count": end - start + 1,
        "approved_pages": list(range(start, end + 1)),
        "evidence_families": families,
        "expected_evidence": expected_evidence_for(families),
        "recommended_route": route,
        "priority_score": score,
        "current_cache_insufficiency": (
            "family validation has fewer than two complete evidence sections"
        ),
        "selection_basis": "frontier_gap_template",
    }


def _gap_candidate(
    candidate: dict[str, Any], families: list[str], pages: list[int], index: int, basis: str
) -> dict[str, Any]:
    return {
        "candidate_id": f"gap_{index:04d}",
        "source_id": candidate["source_id"],
        "source_filename": candidate["source_filename"],
        "section": candidate["section_title"],
        "section_label": candidate.get("section_label", ""),
        "page_start": min(pages),
        "page_end": max(pages),
        "page_count": len(pages),
        "approved_pages": pages,
        "evidence_families": families,
        "expected_evidence": expected_evidence_for(families),
        "recommended_route": candidate.get("recommended_route", "text"),
        "priority_score": int(candidate.get("v2_evidence_value_score") or 0),
        "current_cache_insufficiency": "family validation remains incomplete in current cache",
        "selection_basis": basis,
    }


def approve_gap_parse_plan(
    candidates: list[dict[str, Any]], *, max_new_pages: int
) -> list[dict[str, Any]]:
    approved = []
    used_pages: set[tuple[str, int]] = set()
    family_usage: Counter[str] = Counter()
    total = 0
    for candidate in candidates:
        if total >= max_new_pages:
            break
        source_id = str(candidate["source_id"])
        family_limit = min(DEFAULT_MAX_BATCH_PAGES, max_new_pages - total)
        pages = [
            int(page)
            for page in candidate["approved_pages"]
            if (source_id, int(page)) not in used_pages
        ][:family_limit]
        if not pages:
            continue
        if all(family_usage[family] >= 20 for family in candidate["evidence_families"]):
            continue
        for page in pages:
            used_pages.add((source_id, page))
        for family in candidate["evidence_families"]:
            family_usage[family] += len(pages)
        total += len(pages)
        for start, end in split_contiguous_pages(pages, DEFAULT_MAX_BATCH_PAGES):
            approved.append(
                {
                    **candidate,
                    "page_start": start,
                    "page_end": end,
                    "page_count": end - start + 1,
                    "approved_pages": list(range(start, end + 1)),
                }
            )
    return sorted(approved, key=lambda item: (item["source_filename"], item["page_start"]))


def split_gap_batches(approved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages_by_source: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in approved:
        for page in item["approved_pages"]:
            pages_by_source[str(item["source_id"])][int(page)].append(item)
    batches: list[dict[str, Any]] = []
    for source_id, page_items in sorted(pages_by_source.items()):
        for start, end in split_contiguous_pages(sorted(page_items), DEFAULT_MAX_BATCH_PAGES):
            first = page_items[start][0]
            batches.append(
                {
                    "batch_id": f"gap_{len(batches) + 1:04d}",
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
                            for item in page_items[page]
                        }
                    ),
                    "evidence_families": sorted(
                        {
                            family
                            for page in range(start, end + 1)
                            for item in page_items[page]
                            for family in item["evidence_families"]
                        }
                    ),
                }
            )
    return sorted(batches, key=lambda item: (item["source_filename"], item["page_start"]))


def evidence_map_from_family_sections(
    family_sections: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for family, sections in family_sections.items():
        for index, section in enumerate(sections, start=1):
            rows.append({"section_id": f"{family}_{index:04d}", **section})
    return sorted(
        rows, key=lambda item: (item["source_filename"], item["page_start"], item["section"])
    )


def build_gap_metrics(**kwargs: Any) -> dict[str, Any]:
    sources = kwargs["sources"]
    page_counts = kwargs["page_counts"]
    cached_before = kwargs["cached_before"]
    cached_after = kwargs["cached_after"]
    parse_report = kwargs["parse_report"]
    validation = kwargs["validation"]
    classifications_after = kwargs["classifications_after"]
    total_pages = sum(count or 0 for count in page_counts.values())
    before = sum(len(pages) for pages in cached_before.values())
    after = sum(len(pages) for pages in cached_after.values())
    page_types = Counter(str(row["primary_page_type"]) for row in classifications_after)
    readiness = (
        "ready for evidence-backed target-family selection"
        if all(row["status"] == "valid" for row in validation.values())
        else "partially ready"
        if parse_report["new_pages_parsed"]
        or any(row["status"] == "valid" for row in validation.values())
        else "not ready"
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
        "approved_sections": len(kwargs["approved"]),
        "planned_pages": sum(int(row["page_count"]) for row in kwargs["approved"]),
        "parser_batches": len(kwargs["batches"]),
        "parser_batches_run": parse_report["parser_batches_run"],
        "parser_runtime_ms": parse_report["runtime_ms"],
        "failures": parse_report["parse_failure_count"],
        "timeouts_or_restarts": parse_report["timeout_or_restart_count"],
        "family_validation": validation,
        "remaining_evidence_gaps": kwargs["remaining"],
        "page_type_distribution_after": dict(sorted(page_types.items())),
        "target_selection_readiness": readiness,
        "runtime_ms": round(kwargs["runtime_ms"], 2),
        "cache_provenance": kwargs["provenance"],
    }


def write_gap_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, key in {
        "cache_provenance_reconciliation.json": "cache_provenance_reconciliation",
        "evidence_gap_baseline.json": "evidence_gap_baseline",
        "gap_section_candidates.json": "gap_section_candidates",
        "approved_gap_parse_plan.json": "approved_gap_parse_plan",
        "merged_parse_batches.json": "merged_parse_batches",
        "parse_execution_report.json": "parse_execution_report",
        "newly_cached_pages.json": "newly_cached_pages",
        "cache_coverage_before.json": "cache_coverage_before",
        "cache_coverage_after.json": "cache_coverage_after",
        "page_classifications_after.json": "page_classifications_after",
        "evidence_section_map_after.json": "evidence_section_map_after",
        "enriched_pageindex.json": "enriched_pageindex",
        "commissioning_evidence_sections.json": "commissioning_evidence_sections",
        "statutory_evidence_sections.json": "statutory_evidence_sections",
        "installation_evidence_sections.json": "installation_evidence_sections",
        "evidence_family_validation.json": "evidence_family_validation",
        "source_family_support_map_after.json": "source_family_support_map_after",
        "table_route_pages.json": "table_route_pages",
        "certificate_route_pages.json": "certificate_route_pages",
        "drawing_text_route_pages.json": "drawing_text_route_pages",
        "visual_route_pages.json": "visual_route_pages",
        "remaining_evidence_gaps.json": "remaining_evidence_gaps",
        "coverage_metrics.json": "coverage_metrics",
        "coverage_trace.json": "coverage_trace",
    }.items():
        _atomic_write_json(output_dir / filename, result[key])
    write_csv(output_dir / "gap_section_candidates.csv", result["gap_section_candidates"])
    write_csv(output_dir / "approved_gap_parse_plan.csv", result["approved_gap_parse_plan"])
    write_csv(output_dir / "evidence_section_map_after.csv", result["evidence_section_map_after"])
    (output_dir / "source_coverage_summary.md").write_text(
        summary_markdown(result), encoding="utf-8"
    )
    (output_dir / "next_target_selection_readiness.md").write_text(
        readiness_markdown(result), encoding="utf-8"
    )


def summary_markdown(result: dict[str, Any]) -> str:
    metrics = result["coverage_metrics"]
    return (
        "# Commissioning, Statutory and Installation Evidence Refinement V1\n\n"
        f"- Cached before: {metrics['cached_pages_before']}\n"
        f"- Newly parsed pages: {metrics['newly_parsed_pages']}\n"
        f"- Cached after: {metrics['cached_pages_after']}\n"
        f"- Remaining evidence gaps: {json.dumps(metrics['remaining_evidence_gaps'])}\n"
        f"- Readiness: {metrics['target_selection_readiness']}\n"
    )


def readiness_markdown(result: dict[str, Any]) -> str:
    metrics = result["coverage_metrics"]
    return (
        "# Next Target Selection Readiness\n\n"
        f"- Status: {metrics['target_selection_readiness']}\n"
        f"- Remaining evidence gaps: {json.dumps(metrics['remaining_evidence_gaps'])}\n"
        f"- Family validation: {json.dumps(metrics['family_validation'], sort_keys=True)}\n"
    )


def build_gap_trace(*items: Any) -> list[dict[str, Any]]:
    names = ["cache_provenance", "baseline", "candidates", "approved", "batches", "parse_report"]
    return [
        {"stage": name, "summary": _trace_summary(value)}
        for name, value in zip(names, items, strict=True)
    ]


def expected_evidence_for(families: list[str]) -> str:
    bits = []
    if "commissioning_results" in families:
        bits.append("system identity, commissioning/test event, date/reference and result")
    if "statutory_compliance" in families:
        bits.append("approval/certificate authority, date/reference and certified status")
    if "installation_details" in families:
        bits.append("installed component with configuration, method, type or location detail")
    return "; ".join(bits)


def _trace_summary(value: Any) -> Any:
    if isinstance(value, list):
        return {"rows": len(value)}
    if isinstance(value, dict):
        return {
            key: value[key]
            for key in sorted(value)
            if key
            in {
                "new_pages_parsed",
                "parser_batches_run",
                "parse_failure_count",
                "current_cache_before_this_sprint",
                "families_requiring_parsing",
            }
        }
    return value


def _route_pages(rows: list[dict[str, Any]], routes: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if row["recommended_route"] in routes]


def _has(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _source_sort_key(source: SourceRegistryEntry) -> tuple[str, str]:
    return (Path(source.logical_path).name.lower(), source.source_id)
