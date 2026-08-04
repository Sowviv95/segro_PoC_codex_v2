"""Source-first coverage inventory and PageIndex expansion workflow."""

from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import pymupdf

from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry
from segro_evidence_extraction.parsing.batch_worker import (
    DEFAULT_MAX_BATCH_PAGES,
    BatchRequest,
    BatchWorkerConfig,
)
from segro_evidence_extraction.parsing.cache import config_hash
from segro_evidence_extraction.parsing.models import ParsingConfig
from segro_evidence_extraction.parsing.page_cache import (
    CachedBatchParseResult,
    CachedBatchParsingService,
    CanonicalParsedPageCache,
)
from segro_evidence_extraction.parsing.pdf import PyMuPdfPageParser
from segro_evidence_extraction.parsing.service import load_source_registry
from segro_evidence_extraction.vertical_slice import _atomic_write_json

DEFAULT_SOURCE_MANIFEST = Path("output/sprint3_source_ingestion/source_pack_manifest.json")
DEFAULT_PAGE_CACHE_ROOT = Path("output/enfield_unit1_evidence_first_vertical_slice_v1/page_cache")
DEFAULT_HIERARCHY_PATH = Path(
    "output/enfield_unit1_evidence_first_batch_v1_retrieval_diagnostic/revised_hierarchy.json"
)
DEFAULT_SOURCE_COVERAGE_OUTPUT_DIR = Path("output/enfield_unit1_source_coverage_pageindex_v1")
SOURCE_COVERAGE_VERSION = "source-coverage-pageindex-v1"

PageType = Literal[
    "narrative",
    "structured_table",
    "certificate",
    "schedule",
    "drawing_text_extractable",
    "drawing_visual_required",
    "project_element_sheet",
    "product_datasheet",
    "manufacturer_literature",
    "safety_data",
    "supplier_contact",
    "residual_hazard_schedule",
    "emergency_contacts",
    "reference_only",
    "access_cleaning_guidance",
    "structural_report",
    "structural_calculation",
    "structural_drawing",
    "loading_schedule",
    "fire_strategy_drawing",
    "hazardous_material_statement",
    "appendix_index",
    "certificate_index",
    "commissioning_certificate",
    "test_certificate",
    "installation_completion_certificate",
    "laboratory_test_report",
    "equipment_schedule",
    "model_serial_schedule",
    "bms_points_schedule",
    "mechanical_test_sheet",
    "electrical_test_sheet",
    "fire_system_certificate",
    "pv_commissioning_record",
    "work_permit_template",
    "maintenance_guidance",
    "index_or_contents",
    "separator_or_cover",
    "statutory_planning_decision",
    "blank_unusable",
    "low_value_repetitive",
    "unknown",
]
Route = Literal["text", "table", "drawing_text", "visual", "deprioritized"]


def run_source_coverage_pageindex_v1(
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    cache_root: Path = DEFAULT_PAGE_CACHE_ROOT,
    hierarchy_path: Path = DEFAULT_HIERARCHY_PATH,
    output_dir: Path = DEFAULT_SOURCE_COVERAGE_OUTPUT_DIR,
    max_new_pages: int = 60,
    dry_run: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    sources = sorted(load_source_registry(source_manifest), key=_source_sort_key)
    pdf_sources = [source for source in sources if source.file_type == FileType.PDF]
    page_counts = determine_pdf_page_counts(pdf_sources)
    parser = PyMuPdfPageParser()
    parser_config = ParsingConfig()
    parser_fingerprint = {
        "parser_name": parser.parser_name,
        "parser_version": parser.parser_version,
        "parser_config_fingerprint": config_hash(parser_config),
    }
    cached_before = load_canonical_cached_pages(cache_root)
    hierarchy = load_hierarchy_nodes(hierarchy_path)
    inventory_before = build_source_inventory(
        sources=sources,
        page_counts=page_counts,
        cached_pages=cached_before,
        hierarchy_nodes=hierarchy,
        parser_fingerprint=parser_fingerprint,
    )
    page_map_before = build_page_coverage_map(pdf_sources, page_counts, cached_before)
    classifications_before = classify_cached_pages(cached_before, sources)
    sections_before = detect_sections(classifications_before, sources, hierarchy)
    candidates = build_uncached_page_candidates(
        inventory=inventory_before,
        page_map=page_map_before,
        sections=sections_before,
    )
    approved = approve_parse_plan(candidates, max_new_pages=max_new_pages)
    merged_batches = split_approved_parse_batches(approved)
    validate_parse_batches(merged_batches, cached_before)
    parse_report = execute_parse_batches(
        batches=merged_batches,
        sources=sources,
        cache_root=cache_root,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    cached_after = load_canonical_cached_pages(cache_root)
    inventory_after = build_source_inventory(
        sources=sources,
        page_counts=page_counts,
        cached_pages=cached_after,
        hierarchy_nodes=hierarchy,
        parser_fingerprint=parser_fingerprint,
    )
    classifications_after = classify_cached_pages(cached_after, sources)
    sections_after = detect_sections(classifications_after, sources, hierarchy)
    evidence_map = build_evidence_section_map(sections_after)
    enriched = build_enriched_pageindex(hierarchy, sections_after)
    support = build_source_family_support_map(evidence_map)
    metrics = build_coverage_metrics(
        sources=sources,
        page_counts=page_counts,
        inventory_before=inventory_before,
        inventory_after=inventory_after,
        classifications=classifications_after,
        sections=sections_after,
        parse_report=parse_report,
        runtime_ms=(time.perf_counter() - started) * 1000,
    )
    result = {
        "source_coverage_inventory": inventory_after,
        "source_coverage_inventory_before": inventory_before,
        "page_coverage_map": build_page_coverage_map(pdf_sources, page_counts, cached_after),
        "page_classifications": classifications_after,
        "section_inventory": sections_after,
        "evidence_section_map": evidence_map,
        "uncached_page_candidates": candidates,
        "approved_parse_plan": approved,
        "merged_parse_batches": merged_batches,
        "cache_coverage_before": cache_coverage_snapshot(inventory_before),
        "cache_coverage_after": cache_coverage_snapshot(inventory_after),
        "parse_execution_report": parse_report,
        "enriched_pageindex": enriched,
        "source_family_support_map": support,
        "coverage_metrics": metrics,
        "coverage_trace": build_coverage_trace(candidates, approved, merged_batches, parse_report),
        "source_coverage_review": build_review_rows(inventory_after, sections_after),
    }
    write_source_coverage_outputs(result, output_dir)
    return result


def determine_pdf_page_counts(sources: list[SourceRegistryEntry]) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for source in sources:
        count: int | None = None
        if count is None:
            try:
                with pymupdf.open(source.original_path) as document:  # type: ignore[no-untyped-call]
                    count = int(document.page_count)
            except Exception:  # noqa: BLE001 - inventory records unknown count
                count = None
        if count is None:
            count = source.page_count
        if count is None:
            raw_count = source.metadata.get("page_count")
            count = int(raw_count) if isinstance(raw_count, int) else None
        counts[source.source_id] = count
    return counts


def load_canonical_cached_pages(cache_root: Path) -> dict[str, list[dict[str, Any]]]:
    pages_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not cache_root.exists():
        return {}
    for path in sorted(cache_root.rglob("page_*.json")):
        data = _read_json(path)
        source_id = str(data.get("source_id") or "")
        if not source_id:
            continue
        pages_by_source[source_id].append(
            {
                "source_id": source_id,
                "source_path": str(data.get("source_path") or ""),
                "source_file": Path(str(data.get("source_path") or source_id)).name,
                "page_number": int(data.get("page_number") or 0),
                "extracted_text": str(data.get("extracted_text") or ""),
                "text_character_count": int(data.get("text_character_count") or 0),
                "parser_name": str(data.get("parser_name") or ""),
                "parser_version": str(data.get("parser_version") or ""),
                "parser_config_fingerprint": str(data.get("parser_config_fingerprint") or ""),
                "cache_path": str(path),
            }
        )
    return {
        source_id: sorted(pages, key=lambda item: int(item["page_number"]))
        for source_id, pages in sorted(pages_by_source.items())
    }


def build_source_inventory(
    *,
    sources: list[SourceRegistryEntry],
    page_counts: dict[str, int | None],
    cached_pages: dict[str, list[dict[str, Any]]],
    hierarchy_nodes: list[dict[str, Any]],
    parser_fingerprint: dict[str, str],
) -> list[dict[str, Any]]:
    hierarchy_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in hierarchy_nodes:
        source_id = str(node.get("source_id") or "")
        if source_id:
            hierarchy_by_source[source_id].append(node)
    rows: list[dict[str, Any]] = []
    for source in sorted(sources, key=_source_sort_key):
        total_pages = page_counts.get(source.source_id)
        cached_numbers = {
            int(page["page_number"]) for page in cached_pages.get(source.source_id, [])
        }
        cached_count = len(cached_numbers)
        uncached_count = max((total_pages or 0) - cached_count, 0) if total_pages else 0
        cache_percentage = round(cached_count / total_pages * 100, 2) if total_pages else 0.0
        nodes = hierarchy_by_source.get(source.source_id, [])
        row = {
            "source_id": source.source_id,
            "source_filename": Path(source.logical_path).name,
            "source_type": str(source.file_type),
            "classification": source.classification or "",
            "content_hash": source.file_hash,
            "file_size_bytes": source.size_bytes,
            "total_page_count": total_pages,
            "cached_page_count": cached_count,
            "uncached_page_count": uncached_count,
            "cache_percentage": cache_percentage,
            "parser_fingerprint": dict(parser_fingerprint),
            "known_hierarchy_nodes": len(nodes),
            "known_index_or_contents_pages": sorted(
                _node_pages(nodes, {"index", "contents", "content"})
            ),
            "narrative_sections": _count_node_titles(nodes, {"section", "description"}),
            "table_heavy_sections": _count_node_titles(nodes, {"schedule", "table"}),
            "certificate_sections": _count_node_titles(nodes, {"certificate", "test"}),
            "drawing_heavy_sections": _count_node_titles(nodes, {"drawing", "plan", "elevation"}),
            "image_heavy_sections": 0,
            "low_value_or_repetitive_sections": _count_node_titles(
                nodes, {"maintenance", "coshh", "safety data"}
            ),
            "likely_dictionary_domains": infer_domains(Path(source.logical_path).name, ""),
            "likely_target_families": infer_target_families(Path(source.logical_path).name, ""),
            "recommended_route": source_route(Path(source.logical_path).name),
            "parent_archive_source_id": source.parent_archive_source_id,
            "archive_member_path": source.archive_member_path,
        }
        rows.append(row)
    return rows


def build_page_coverage_map(
    sources: list[SourceRegistryEntry],
    page_counts: dict[str, int | None],
    cached_pages: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sorted(sources, key=_source_sort_key):
        total_pages = page_counts.get(source.source_id) or 0
        cached = {int(page["page_number"]) for page in cached_pages.get(source.source_id, [])}
        for page_number in range(1, total_pages + 1):
            rows.append(
                {
                    "source_id": source.source_id,
                    "source_filename": Path(source.logical_path).name,
                    "page_number": page_number,
                    "cache_status": "cached" if page_number in cached else "uncached",
                }
            )
    return rows


def classify_cached_pages(
    cached_pages: dict[str, list[dict[str, Any]]],
    sources: list[SourceRegistryEntry],
) -> list[dict[str, Any]]:
    names = {source.source_id: Path(source.logical_path).name for source in sources}
    rows: list[dict[str, Any]] = []
    for source_id, pages in sorted(cached_pages.items()):
        source_filename = names.get(source_id, "")
        for page in pages:
            classified = classify_page_text(
                str(page.get("extracted_text") or ""),
                source_filename=source_filename,
            )
            rows.append(
                {
                    "source_id": source_id,
                    "source_filename": source_filename or str(page.get("source_file") or ""),
                    "page_number": int(page["page_number"]),
                    "text_character_count": int(page.get("text_character_count") or 0),
                    **classified,
                    "cache_path": str(page.get("cache_path") or ""),
                }
            )
    return sorted(rows, key=lambda item: (item["source_filename"], item["page_number"]))


def classify_page_text(text: str, *, source_filename: str = "") -> dict[str, Any]:
    normalized = _normalize(text)
    tags: set[str] = set()
    primary: PageType = "unknown"
    route: Route = "text"
    if not normalized:
        primary, route = "blank_unusable", "deprioritized"
    elif _is_cover_or_separator(normalized):
        primary, route = "separator_or_cover", "deprioritized"
    if normalized and _is_part6_appendix_index(normalized):
        primary, route = "appendix_index", "deprioritized"
        tags.add("appendix_navigation")
    elif normalized and _is_index_or_contents(normalized):
        primary, route = "index_or_contents", "deprioritized"
    if normalized and _is_certificate_index(normalized):
        primary, route = "certificate_index", "deprioritized"
        tags.add("certificate_navigation")
    if normalized and _is_project_element_sheet(normalized):
        primary, route = "project_element_sheet", "text"
        tags.update({"project_specific", "installed_project_evidence", "component_specification"})
    if normalized and primary != "project_element_sheet" and _is_reference_only_page(normalized):
        primary, route = "reference_only", "deprioritized"
        tags.add("cross_reference_only")
    elif normalized and _is_residual_hazard_schedule(normalized):
        primary, route = "residual_hazard_schedule", "table"
        tags.update({"health_and_safety_only", "operational_safety_control"})
    elif normalized and _is_emergency_contacts(normalized):
        primary, route = "emergency_contacts", "text"
        tags.add("emergency_contact_details")
    elif normalized and _is_hazardous_material_statement(normalized):
        primary, route = "hazardous_material_statement", "text"
        tags.update({"health_and_safety_only", "project_specific"})
    elif normalized and _is_access_cleaning_guidance(normalized) and not _is_work_permit_template(normalized):
        primary, route = "access_cleaning_guidance", "text"
        tags.update({"operational_safety_control", "maintenance_access"})
    elif normalized and primary not in {"appendix_index", "certificate_index"} and _is_work_permit_template(normalized):
        primary, route = "work_permit_template", "deprioritized"
        tags.add("template_only")
    elif normalized and primary not in {"appendix_index", "certificate_index"} and _is_pv_commissioning_record(normalized):
        primary, route = "pv_commissioning_record", "table"
        tags.update({"commissioning_test_evidence", "project_specific", "unit_scoped"})
    elif normalized and primary not in {"appendix_index", "certificate_index"} and _is_bms_points_schedule(normalized):
        primary, route = "bms_points_schedule", "table"
        tags.update({"commissioning_test_evidence", "equipment_schedule", "unit_scoped"})
    elif normalized and primary not in {"appendix_index", "certificate_index"} and _is_air_conditioning_model_serial_schedule(normalized):
        primary, route = "model_serial_schedule", "table"
        tags.update({"commissioning_test_evidence", "manufacturer_model_table", "unit_scoped"})
    elif normalized and primary not in {"appendix_index", "certificate_index"} and _is_mechanical_test_sheet(normalized):
        primary, route = "mechanical_test_sheet", "table"
        tags.update({"commissioning_test_evidence", "mechanical_test_values", "unit_scoped"})
    elif normalized and primary not in {"appendix_index", "certificate_index"} and _is_water_laboratory_report(normalized):
        primary, route = "laboratory_test_report", "table"
        tags.update({"commissioning_test_evidence", "laboratory_result"})
    elif normalized and primary not in {"appendix_index", "certificate_index"} and _is_fire_system_certificate(normalized):
        primary, route = "fire_system_certificate", "text"
        tags.update({"commissioning_test_evidence", "certificate_date_reference", "unit_scoped"})
    elif normalized and primary not in {"appendix_index", "certificate_index"} and _is_commissioning_certificate(normalized):
        primary, route = "commissioning_certificate", "text"
        tags.update({"commissioning_test_evidence", "certificate_date_reference"})
    elif normalized and primary not in {"appendix_index", "certificate_index"} and _is_installation_completion_certificate(normalized):
        primary, route = "installation_completion_certificate", "text"
        tags.update({"commissioning_test_evidence", "certificate_date_reference"})
    elif normalized and _is_loading_schedule(normalized):
        primary, route = "loading_schedule", "table"
        tags.update({"structural_loading", "component_specification"})
    elif normalized and _is_structural_report(normalized):
        primary, route = "structural_report", "text"
        tags.update({"structural_loading", "component_specification", "project_specific"})
    elif normalized and _is_structural_calculation(normalized):
        primary, route = "structural_calculation", "table"
        tags.update({"structural_loading", "calculation_input"})
    if normalized and _is_safety_data(normalized):
        primary, route = "safety_data", "deprioritized"
        tags.update({"generic_reference_literature", "health_and_safety_only"})
    elif normalized and _is_generic_manufacturer_literature(normalized):
        primary, route = "manufacturer_literature", "deprioritized"
        tags.add("generic_reference_literature")
    elif normalized and _is_supplier_contact_only(normalized):
        primary, route = "supplier_contact", "deprioritized"
        tags.add("supplier_contact_only")
    if (
        normalized
        and primary not in {
            "project_element_sheet",
        "manufacturer_literature",
        "safety_data",
        "supplier_contact",
        "residual_hazard_schedule",
        "reference_only",
        "appendix_index",
        "certificate_index",
        "work_permit_template",
        "commissioning_certificate",
        "test_certificate",
        "installation_completion_certificate",
        "laboratory_test_report",
        "equipment_schedule",
        "model_serial_schedule",
        "bms_points_schedule",
        "mechanical_test_sheet",
        "electrical_test_sheet",
        "fire_system_certificate",
        "pv_commissioning_record",
        "access_cleaning_guidance",
        "hazardous_material_statement",
        "structural_report",
        "loading_schedule",
        }
        and _is_maintenance_only(normalized, source_filename)
    ):
        primary, route = "maintenance_guidance", "deprioritized"
        tags.add("maintenance_only")
    if normalized and primary not in {"safety_data"} and _is_low_value_repetitive(normalized):
        primary, route = "low_value_repetitive", "deprioritized"
        tags.add("duplicate_or_repeated_content")
    if normalized and _is_planning_decision(normalized):
        primary, route = "statutory_planning_decision", "text"
        tags.update({"planning_condition", "statutory_compliance", "required_or_approved_status"})
    elif (
        normalized
        and primary not in {
            "manufacturer_literature",
            "safety_data",
            "appendix_index",
            "certificate_index",
            "work_permit_template",
        }
        and _is_certificate(normalized)
    ):
        primary, route = "certificate", "text"
        tags.update({"certificate_date_reference", "statutory_compliance"})
    elif normalized and _is_schedule(normalized):
        primary, route = "schedule", "table"
        tags.add("equipment_schedule")
    elif normalized and _is_table(normalized):
        primary, route = "structured_table", "table"
    if normalized and primary not in {"project_element_sheet", "safety_data"} and _is_product_datasheet(normalized):
        primary = "product_datasheet"
        if _is_project_specific_context(normalized):
            tags.add("manufacturer_model_table")
        else:
            route = "deprioritized"
            tags.add("generic_reference_literature")
    if normalized and primary not in {
        "project_element_sheet",
        "manufacturer_literature",
        "safety_data",
        "supplier_contact",
        "residual_hazard_schedule",
        "emergency_contacts",
        "reference_only",
        "appendix_index",
        "certificate_index",
        "work_permit_template",
        "commissioning_certificate",
        "test_certificate",
        "installation_completion_certificate",
        "laboratory_test_report",
        "equipment_schedule",
        "model_serial_schedule",
        "bms_points_schedule",
        "mechanical_test_sheet",
        "electrical_test_sheet",
        "fire_system_certificate",
        "pv_commissioning_record",
        "access_cleaning_guidance",
        "structural_report",
        "loading_schedule",
        "hazardous_material_statement",
    } and _is_drawing(normalized):
        if _is_fire_strategy_drawing(normalized):
            primary, route = "fire_strategy_drawing", "drawing_text"
            tags.update({"fire_strategy", "drawing_symbol_dependency"})
        elif _is_structural_drawing(normalized):
            primary, route = "structural_drawing", "drawing_text"
            tags.update({"structural_loading", "drawing_symbol_dependency"})
        elif _drawing_text_extractable(normalized):
            primary, route = "drawing_text_extractable", "drawing_text"
        else:
            primary, route = "drawing_visual_required", "visual"
            tags.update({"layout_dependency", "visual_confirmation_required"})
    tags.update(_secondary_tags(normalized))
    if primary == "unknown" and len(normalized) > 120:
        primary = "narrative"
    if primary in {"structured_table", "schedule"}:
        route = "table"
    evidence_bearing = primary not in {
        "separator_or_cover",
        "index_or_contents",
        "blank_unusable",
        "low_value_repetitive",
        "maintenance_guidance",
        "manufacturer_literature",
        "safety_data",
        "supplier_contact",
        "residual_hazard_schedule",
        "reference_only",
        "appendix_index",
        "certificate_index",
        "work_permit_template",
        "unknown",
    }
    evidence_role = "direct evidence" if evidence_bearing else "navigation only"
    if primary == "statutory_planning_decision":
        evidence_role = "supporting/contextual evidence"
    elif primary == "blank_unusable":
        evidence_role = "unusable"
    elif primary in {"manufacturer_literature", "product_datasheet"} and not _is_project_specific_context(normalized):
        evidence_role = "generic reference literature"
    elif primary == "safety_data":
        evidence_role = "generic reference literature"
    elif primary == "supplier_contact":
        evidence_role = "supporting/contextual evidence"
    elif primary == "residual_hazard_schedule":
        evidence_role = "operational or safety guidance"
    elif primary in {"access_cleaning_guidance", "hazardous_material_statement"}:
        evidence_role = "supporting/contextual evidence"
    elif primary == "reference_only":
        evidence_role = "cross-reference only"
    elif primary in {"appendix_index", "certificate_index"}:
        evidence_role = "navigation only"
    elif primary == "work_permit_template":
        evidence_role = "template only"
    elif primary in {
        "commissioning_certificate",
        "test_certificate",
        "installation_completion_certificate",
        "laboratory_test_report",
        "equipment_schedule",
        "model_serial_schedule",
        "bms_points_schedule",
        "mechanical_test_sheet",
        "electrical_test_sheet",
        "fire_system_certificate",
        "pv_commissioning_record",
    }:
        evidence_role = "commissioning/test evidence"
    domains = infer_domains(source_filename, normalized)
    families = infer_target_families(source_filename, normalized)
    if primary in {"separator_or_cover", "index_or_contents", "appendix_index", "certificate_index"}:
        domains = ["general"]
        families = ["section_discovery"]
    elif primary == "blank_unusable":
        domains = ["unknown"]
        families = ["section_discovery"]
    elif primary in {"manufacturer_literature", "product_datasheet", "safety_data"} and not _is_project_specific_context(normalized):
        families = [
            family
            for family in families
            if family in {"maintenance_only", "materials_finishes", "identifiers_references"}
        ] or ["maintenance_only"]
    elif primary == "supplier_contact":
        families = ["identifiers_references"]
    elif primary in {"residual_hazard_schedule", "access_cleaning_guidance"}:
        families = ["maintenance_only", "locations_layout"]
    elif primary == "work_permit_template":
        families = ["maintenance_only", "section_discovery"]
    elif primary in {"pv_commissioning_record"}:
        families = ["quantities_counts", "identifiers_references", "commissioning_dates"]
    elif primary in {"model_serial_schedule", "bms_points_schedule"}:
        families = ["identifiers_references", "equipment_models"]
    elif primary in {"mechanical_test_sheet", "laboratory_test_report"}:
        families = ["performance_tests", "commissioning_dates"]
    elif primary in {"fire_system_certificate", "commissioning_certificate", "installation_completion_certificate"}:
        families = ["certificate_dates", "identifiers_references", "commissioning_dates"]
    elif primary == "emergency_contacts":
        families = ["identifiers_references"]
    elif primary in {"structural_report", "structural_drawing", "loading_schedule"}:
        domains = sorted(set(domains) | {"building_fabric"})
        families = sorted(set(families) | {"dimensions_capacities", "materials_finishes"})
    elif primary == "reference_only":
        families = ["section_discovery"]
    return {
        "primary_page_type": primary,
        "secondary_tags": sorted(tags),
        "recommended_route": route,
        "likely_dictionary_domains": domains,
        "likely_target_families": families,
        "evidence_bearing": evidence_bearing,
        "evidence_role": evidence_role,
    }


def detect_sections(
    page_classifications: list[dict[str, Any]],
    sources: list[SourceRegistryEntry],
    hierarchy_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in page_classifications:
        by_source[str(page["source_id"])].append(page)
    names = {source.source_id: Path(source.logical_path).name for source in sources}
    for source_id, pages in sorted(by_source.items(), key=lambda item: names.get(item[0], item[0])):
        sorted_pages = sorted(pages, key=lambda item: int(item["page_number"]))
        current: list[dict[str, Any]] = []
        current_title = "Cached pages"
        for page in sorted_pages:
            title = section_title_from_page(page)
            starts = bool(title) or not current
            if starts and current:
                rows.append(
                    _section_row(source_id, names.get(source_id, ""), current_title, current)
                )
                current = []
            if title:
                current_title = title
            current.append(page)
        if current:
            rows.append(_section_row(source_id, names.get(source_id, ""), current_title, current))
    rows.extend(hierarchy_sections_missing_from_cache(hierarchy_nodes, rows, names))
    return sorted(
        rows,
        key=lambda item: (item["source_filename"], item["page_start"], item["title"]),
    )


def section_title_from_page(page: dict[str, Any]) -> str:
    if page["primary_page_type"] == "index_or_contents":
        return "Index / contents"
    if page["primary_page_type"] == "statutory_planning_decision":
        return "Planning decision evidence"
    if page["primary_page_type"] == "certificate":
        return "Certificate / test evidence"
    if page["primary_page_type"] == "project_element_sheet":
        return "Project element sheet evidence"
    if page["primary_page_type"] in {"manufacturer_literature", "product_datasheet"}:
        return "Generic manufacturer literature"
    if page["primary_page_type"] == "safety_data":
        return "Safety / COSHH data"
    if page["primary_page_type"] == "residual_hazard_schedule":
        return "Residual hazard schedule"
    if page["primary_page_type"] == "emergency_contacts":
        return "Emergency contacts"
    if page["primary_page_type"] == "reference_only":
        return "Cross-reference only"
    if page["primary_page_type"] == "appendix_index":
        return "Part 6 appendix index"
    if page["primary_page_type"] == "certificate_index":
        return "Appendix D certificate index"
    if page["primary_page_type"] == "work_permit_template":
        return "Appendix E work permit template"
    if page["primary_page_type"] in {
        "commissioning_certificate",
        "test_certificate",
        "installation_completion_certificate",
        "fire_system_certificate",
        "pv_commissioning_record",
    }:
        return "Appendix D commissioning / test certificate"
    if page["primary_page_type"] in {
        "equipment_schedule",
        "model_serial_schedule",
        "bms_points_schedule",
        "mechanical_test_sheet",
        "electrical_test_sheet",
        "laboratory_test_report",
    }:
        return "Appendix D test / equipment schedule"
    if page["primary_page_type"] == "access_cleaning_guidance":
        return "Access and cleaning guidance"
    if page["primary_page_type"] in {"structural_report", "structural_calculation"}:
        return "Structural report / calculations"
    if page["primary_page_type"] in {"structural_drawing", "fire_strategy_drawing"}:
        return "Drawing evidence"
    if page["primary_page_type"] == "loading_schedule":
        return "Schedule / table evidence"
    if page["primary_page_type"] in {"schedule", "structured_table"}:
        return "Schedule / table evidence"
    if page["primary_page_type"].startswith("drawing"):
        return "Drawing evidence"
    return ""


def build_uncached_page_candidates(
    *,
    inventory: list[dict[str, Any]],
    page_map: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    section_lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in sections:
        section_lookup[str(section["source_id"])].append(section)
    uncached_by_source: dict[str, list[int]] = defaultdict(list)
    for row in page_map:
        if row["cache_status"] == "uncached":
            uncached_by_source[str(row["source_id"])].append(int(row["page_number"]))
    rows: list[dict[str, Any]] = []
    for source in inventory:
        source_id = str(source["source_id"])
        filename = str(source["source_filename"])
        total_pages = int(source["total_page_count"] or 0)
        cached_count = int(source["cached_page_count"])
        if total_pages == 0:
            continue
        priority_windows = prioritized_windows(filename, total_pages, cached_count)
        for index, (start, end, reason, priority, route) in enumerate(priority_windows, start=1):
            missing = [
                page for page in range(start, end + 1) if page in set(uncached_by_source[source_id])
            ]
            rows.append(
                {
                    "candidate_id": f"{source_id}_cand_{index:03d}",
                    "source_id": source_id,
                    "source_filename": filename,
                    "page_start": start,
                    "page_end": end,
                    "page_count": end - start + 1,
                    "missing_pages": missing,
                    "cached_pages_in_range": (end - start + 1) - len(missing),
                    "priority": priority,
                    "recommended_route": route,
                    "reason": reason,
                    "decision": "candidate" if missing else "already_sufficiently_represented",
                    "near_known_sections": [
                        item["title"]
                        for item in section_lookup[source_id]
                        if _range_near(item, start, end)
                    ][:5],
                }
            )
    return sorted(rows, key=lambda item: (item["source_filename"], item["page_start"]))


def prioritized_windows(
    filename: str,
    total_pages: int,
    cached_count: int,
) -> list[tuple[int, int, str, str, Route]]:
    lower = filename.lower()
    windows: list[tuple[int, int, str, str, Route]] = []
    intro_end = min(total_pages, 10)
    if cached_count < intro_end:
        windows.append(
            (
                1,
                intro_end,
                "Front matter usually contains index, contents and document section anchors.",
                "high",
                "text",
            )
        )
    if "part 6" in lower or "appendices" in lower or "commission" in lower or "ev" in lower:
        windows.append(
            (
                1,
                min(total_pages, 10),
                (
                    "Appendix/EV documents are likely to contain certificates, "
                    "schedules or commissioning evidence."
                ),
                "high",
                "table",
            )
        )
    elif "part 3" in lower:
        windows.append(
            (
                1,
                min(total_pages, 10),
                (
                    "Building services front matter is needed to route schedules, "
                    "certificates and equipment sections."
                ),
                "high",
                "table",
            )
        )
    elif "part 2" in lower:
        windows.append(
            (
                1,
                min(total_pages, 10),
                (
                    "Building fabric front matter is needed to route element sheets "
                    "and manufacturer/product sections."
                ),
                "high",
                "table",
            )
        )
    elif "part 4" in lower:
        windows.append(
            (
                1,
                min(total_pages, 10),
                (
                    "External works front matter is needed before separating "
                    "specifications from long maintenance/COSHH material."
                ),
                "medium",
                "text",
            )
        )
    elif "part 5" in lower:
        windows.append(
            (
                1,
                min(total_pages, 10),
                (
                    "Health and safety front matter may identify hazards, structural "
                    "principles and low-value safety-only sections."
                ),
                "medium",
                "text",
            )
        )
    return _dedupe_windows(windows)


def approve_parse_plan(
    candidates: list[dict[str, Any]],
    *,
    max_new_pages: int,
) -> list[dict[str, Any]]:
    approved: list[dict[str, Any]] = []
    used_pages: set[tuple[str, int]] = set()
    budget = max_new_pages
    for candidate in sorted(candidates, key=_candidate_sort_key):
        missing = [
            int(page)
            for page in candidate["missing_pages"]
            if (str(candidate["source_id"]), int(page)) not in used_pages
        ]
        if not missing:
            continue
        if candidate["priority"] not in {"high", "medium"}:
            continue
        selected = missing[:budget]
        if not selected:
            break
        for page in selected:
            used_pages.add((str(candidate["source_id"]), page))
        for start, end in split_contiguous_pages(selected, DEFAULT_MAX_BATCH_PAGES):
            approved.append(
                {
                    **candidate,
                    "page_start": start,
                    "page_end": end,
                    "page_count": end - start + 1,
                    "approved_pages": list(range(start, end + 1)),
                    "decision": "approved_for_parsing",
                    "approval_rationale": (
                        "Bounded source-first range; cached pages are excluded and worker batch "
                        "size is capped at 10 pages."
                    ),
                }
            )
        budget -= len(selected)
        if budget <= 0:
            break
    return sorted(approved, key=lambda item: (item["source_filename"], item["page_start"]))


def split_approved_parse_batches(approved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages_by_source: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for item in approved:
        for page in item["approved_pages"]:
            pages_by_source[str(item["source_id"])][int(page)] = item
    batches: list[dict[str, Any]] = []
    for source_id, page_items in sorted(pages_by_source.items()):
        pages = sorted(page_items)
        for start, end in split_contiguous_pages(pages, DEFAULT_MAX_BATCH_PAGES):
            first = page_items[start]
            batches.append(
                {
                    "source_id": source_id,
                    "source_filename": first["source_filename"],
                    "page_start": start,
                    "page_end": end,
                    "page_count": end - start + 1,
                    "worker_batch_page_limit": DEFAULT_MAX_BATCH_PAGES,
                    "parse_required": True,
                    "candidate_ids": sorted(
                        {
                            str(page_items[page]["candidate_id"])
                            for page in range(start, end + 1)
                            if page in page_items
                        }
                    ),
                }
            )
    return sorted(batches, key=lambda item: (item["source_filename"], item["page_start"]))


def validate_parse_batches(
    batches: list[dict[str, Any]],
    cached_pages: dict[str, list[dict[str, Any]]],
) -> None:
    seen: set[tuple[str, int]] = set()
    cached = {
        (source_id, int(page["page_number"]))
        for source_id, pages in cached_pages.items()
        for page in pages
    }
    for batch in batches:
        if int(batch["page_count"]) > DEFAULT_MAX_BATCH_PAGES:
            raise ValueError("Parse batch exceeds maximum 10-page worker limit")
        for page_number in range(int(batch["page_start"]), int(batch["page_end"]) + 1):
            key = (str(batch["source_id"]), page_number)
            if key in seen:
                raise ValueError(f"Duplicate parse request: {key}")
            if key in cached:
                raise ValueError(f"Already cached page was scheduled for parsing: {key}")
            seen.add(key)


def execute_parse_batches(
    *,
    batches: list[dict[str, Any]],
    sources: list[SourceRegistryEntry],
    cache_root: Path,
    output_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    source_by_id = {source.source_id: source for source in sources}
    parse_results: list[CachedBatchParseResult] = []
    if not dry_run:
        service = CachedBatchParsingService(
            cache=CanonicalParsedPageCache(cache_root),
            batch_config=BatchWorkerConfig(),
        )
        for batch in batches:
            source = source_by_id[str(batch["source_id"])]
            parse_results.append(
                service.parse(
                    BatchRequest(
                        source=source,
                        source_path=source.original_path,
                        output_dir=str(
                            output_dir
                            / "parser_workers"
                            / (
                                f"{source.source_id}_{int(batch['page_start']):04d}_"
                                f"{int(batch['page_end']):04d}"
                            )
                        ),
                        page_start=int(batch["page_start"]),
                        page_end=int(batch["page_end"]),
                    )
                )
            )
    failures: list[dict[str, Any]] = []
    for result in parse_results:
        for batch_result in result.batch_results:
            failures.extend(
                failure.model_dump(mode="json") for failure in batch_result.pages_failed
            )
    return {
        "dry_run": dry_run,
        "parser_batches_run": sum(result.worker_invocation_count for result in parse_results),
        "planned_parser_batches": len(batches),
        "new_pages_parsed": sum(len(result.pages_newly_parsed) for result in parse_results),
        "pages_loaded_from_cache": sum(
            len(result.pages_loaded_from_cache) for result in parse_results
        ),
        "parse_failures": failures,
        "parse_failure_count": len(failures),
        "timeout_or_restart_count": sum(result.restart_count for result in parse_results),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 2),
        "batch_ranges": [
            {
                "source_id": result.source_id,
                "requested_page_start": result.requested_page_start,
                "requested_page_end": result.requested_page_end,
                "pages_newly_parsed": result.pages_newly_parsed,
                "pages_loaded_from_cache": result.pages_loaded_from_cache,
                "missing_ranges_sent_to_workers": result.missing_ranges_sent_to_workers,
            }
            for result in parse_results
        ],
    }


def build_evidence_section_map(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **section,
            "supported_target_family_hints": section["likely_target_families"],
            "supported_domain_hints": section["likely_dictionary_domains"],
        }
        for section in sections
        if section["evidence_bearing"]
    ]


def build_enriched_pageindex(
    hierarchy_nodes: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "pageindex_version": SOURCE_COVERAGE_VERSION,
        "base_hierarchy_node_count": len(hierarchy_nodes),
        "base_hierarchy_nodes": hierarchy_nodes,
        "source_first_sections": sections,
    }


def build_source_family_support_map(evidence_map: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for section in evidence_map:
        source_id = str(section["source_id"])
        source_filename = str(section["source_filename"])
        key = (source_id, source_filename)
        entry = grouped.setdefault(
            key,
            {
                "source_id": source_id,
                "source_filename": source_filename,
                "likely_dictionary_domains": set(),
                "likely_target_families": set(),
                "recommended_routes": set(),
                "evidence_section_count": 0,
            },
        )
        entry["evidence_section_count"] += 1
        entry["likely_dictionary_domains"].update(section["likely_dictionary_domains"])
        entry["likely_target_families"].update(section["likely_target_families"])
        entry["recommended_routes"].add(section["recommended_route"])
    rows = []
    for entry in grouped.values():
        rows.append(
            {
                **entry,
                "likely_dictionary_domains": sorted(entry["likely_dictionary_domains"]),
                "likely_target_families": sorted(entry["likely_target_families"]),
                "recommended_routes": sorted(entry["recommended_routes"]),
            }
        )
    return sorted(rows, key=lambda item: item["source_filename"])


def build_coverage_metrics(
    *,
    sources: list[SourceRegistryEntry],
    page_counts: dict[str, int | None],
    inventory_before: list[dict[str, Any]],
    inventory_after: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    parse_report: dict[str, Any],
    runtime_ms: float,
) -> dict[str, Any]:
    pdf_source_count = sum(1 for source in sources if source.file_type == FileType.PDF)
    total_pages = sum(count or 0 for count in page_counts.values())
    before_cached = sum(int(row["cached_page_count"]) for row in inventory_before)
    after_cached = sum(int(row["cached_page_count"]) for row in inventory_after)
    page_types = Counter(str(row["primary_page_type"]) for row in classifications)
    routes = Counter(str(row["recommended_route"]) for row in classifications)
    poor = [
        row["source_filename"]
        for row in inventory_after
        if row["source_type"] == "pdf" and float(row["cache_percentage"]) < 10.0
    ]
    readiness = "not ready"
    if after_cached and len([section for section in sections if section["evidence_bearing"]]) >= 8:
        readiness = "partially ready"
    if not poor and routes.get("table", 0) and routes.get("text", 0):
        readiness = "ready for evidence-backed target-family selection"
    return {
        "total_registered_sources": len(sources),
        "total_pdf_sources": pdf_source_count,
        "total_pages_across_registered_pdfs": total_pages,
        "cached_pages_before": before_cached,
        "uncached_pages_before": max(total_pages - before_cached, 0),
        "new_pages_parsed": parse_report["new_pages_parsed"],
        "cached_pages_after": after_cached,
        "uncached_pages_after": max(total_pages - after_cached, 0),
        "overall_cache_coverage_percentage": round(after_cached / total_pages * 100, 2)
        if total_pages
        else 0.0,
        "coverage_percentage_by_source": {
            str(row["source_filename"]): row["cache_percentage"] for row in inventory_after
        },
        "page_type_distribution": dict(sorted(page_types.items())),
        "recommended_route_distribution": dict(sorted(routes.items())),
        "evidence_bearing_sections_by_source": dict(
            Counter(
                str(section["source_filename"])
                for section in sections
                if section["evidence_bearing"]
            )
        ),
        "likely_supported_domains": sorted(
            {domain for section in sections for domain in section["likely_dictionary_domains"]}
        ),
        "likely_supported_target_families": sorted(
            {family for section in sections for family in section["likely_target_families"]}
        ),
        "table_routes": routes.get("table", 0),
        "certificate_routes": page_types.get("certificate", 0),
        "visual_routes": routes.get("visual", 0),
        "low_value_sections": sum(
            1 for section in sections if section["recommended_route"] == "deprioritized"
        ),
        "sources_with_poor_coverage": poor,
        "sections_still_requiring_parsing": [
            row["source_filename"] for row in inventory_after if int(row["uncached_page_count"]) > 0
        ],
        "sections_requiring_future_visual_review": [
            section["title"] for section in sections if section["recommended_route"] == "visual"
        ],
        "target_selection_readiness": readiness,
        "source_coverage_sufficient_for_evidence_backed_target_selection": readiness
        == "ready for evidence-backed target-family selection",
        "runtime_ms": round(runtime_ms, 2),
    }


def cache_coverage_snapshot(inventory: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(row["total_page_count"] or 0) for row in inventory)
    cached = sum(int(row["cached_page_count"]) for row in inventory)
    return {
        "total_pages": total,
        "cached_pages": cached,
        "uncached_pages": max(total - cached, 0),
        "overall_cache_coverage_percentage": round(cached / total * 100, 2) if total else 0.0,
        "by_source": [
            {
                "source_id": row["source_id"],
                "source_filename": row["source_filename"],
                "total_page_count": row["total_page_count"],
                "cached_page_count": row["cached_page_count"],
                "uncached_page_count": row["uncached_page_count"],
                "cache_percentage": row["cache_percentage"],
            }
            for row in inventory
        ],
    }


def write_source_coverage_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_outputs = {
        "source_coverage_inventory.json": result["source_coverage_inventory"],
        "page_coverage_map.json": result["page_coverage_map"],
        "page_classifications.json": result["page_classifications"],
        "section_inventory.json": result["section_inventory"],
        "evidence_section_map.json": result["evidence_section_map"],
        "uncached_page_candidates.json": result["uncached_page_candidates"],
        "approved_parse_plan.json": result["approved_parse_plan"],
        "merged_parse_batches.json": result["merged_parse_batches"],
        "cache_coverage_before.json": result["cache_coverage_before"],
        "cache_coverage_after.json": result["cache_coverage_after"],
        "parse_execution_report.json": result["parse_execution_report"],
        "enriched_pageindex.json": result["enriched_pageindex"],
        "source_family_support_map.json": result["source_family_support_map"],
        "visual_route_pages.json": _route_pages(result["page_classifications"], "visual"),
        "table_route_pages.json": _route_pages(result["page_classifications"], "table"),
        "certificate_route_pages.json": [
            row
            for row in result["page_classifications"]
            if row["primary_page_type"] == "certificate"
        ],
        "low_value_pages.json": [
            row
            for row in result["page_classifications"]
            if row["recommended_route"] == "deprioritized"
        ],
        "coverage_metrics.json": result["coverage_metrics"],
        "coverage_trace.json": result["coverage_trace"],
    }
    for filename, payload in json_outputs.items():
        _atomic_write_json(output_dir / filename, payload)
    csv_specs = {
        "source_coverage_inventory.csv": result["source_coverage_inventory"],
        "page_coverage_map.csv": result["page_coverage_map"],
        "section_inventory.csv": result["section_inventory"],
        "evidence_section_map.csv": result["evidence_section_map"],
        "source_coverage_review.csv": result["source_coverage_review"],
    }
    for filename, rows in csv_specs.items():
        write_csv(output_dir / filename, rows)
    (output_dir / "source_coverage_summary.md").write_text(
        source_coverage_summary(result), encoding="utf-8"
    )
    (output_dir / "next_target_selection_readiness.md").write_text(
        readiness_markdown(result["coverage_metrics"]), encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def source_coverage_summary(result: dict[str, Any]) -> str:
    metrics = result["coverage_metrics"]
    lines = [
        "# Source Coverage and PageIndex Expansion V1",
        "",
        f"- Total registered sources: {metrics['total_registered_sources']}",
        f"- Total PDF sources: {metrics['total_pdf_sources']}",
        f"- Total PDF pages: {metrics['total_pages_across_registered_pdfs']}",
        f"- Cached pages before: {metrics['cached_pages_before']}",
        f"- New pages parsed: {metrics['new_pages_parsed']}",
        f"- Cached pages after: {metrics['cached_pages_after']}",
        f"- Overall cache coverage: {metrics['overall_cache_coverage_percentage']}%",
        f"- Readiness: {metrics['target_selection_readiness']}",
        "",
        "## Page Types",
    ]
    for name, count in sorted(metrics["page_type_distribution"].items()):
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Poor Coverage Sources"])
    for source in metrics["sources_with_poor_coverage"]:
        lines.append(f"- {source}")
    return "\n".join(lines) + "\n"


def readiness_markdown(metrics: dict[str, Any]) -> str:
    readiness = metrics["target_selection_readiness"]
    return (
        "# Next Target Selection Readiness\n\n"
        f"- Status: {readiness}\n"
        "- Evidence-bearing sections by source: "
        f"{json.dumps(metrics['evidence_bearing_sections_by_source'], sort_keys=True)}\n"
        "- Poor coverage sources: "
        f"{json.dumps(metrics['sources_with_poor_coverage'], sort_keys=True)}\n"
        "- Do not select another extraction batch unless the next sprint accepts this "
        "evidence map as sufficient.\n"
    )


def infer_domains(source_filename: str, text: str) -> list[str]:
    lower = f"{source_filename} {text}".lower()
    domains: set[str] = set()
    if "part 1" in lower or "planning" in lower or "building control" in lower:
        domains.update({"general", "statutory"})
    if "part 2" in lower or any(term in lower for term in ["roof", "wall", "door", "floor"]):
        domains.add("building_fabric")
    if "part 3" in lower or any(
        term in lower for term in ["mechanical", "electrical", "fire alarm", "meter"]
    ):
        domains.add("building_services")
    if "part 4" in lower or any(
        term in lower for term in ["landscaping", "drainage", "bollard", "gate"]
    ):
        domains.add("external_works")
    if "part 5" in lower or "health" in lower or "hazard" in lower:
        domains.add("health_and_safety")
    if "part 6" in lower or "appendix" in lower or "certificate" in lower or "commission" in lower:
        domains.add("appendices_certificates")
    if "ev" in lower or "charger" in lower or "rolec" in lower:
        domains.add("ev_charging")
    return sorted(domains) or ["unknown"]


def infer_target_families(source_filename: str, text: str) -> list[str]:
    lower = f"{source_filename} {text}".lower()
    families: set[str] = set()
    family_terms = {
        "manufacturer_model": ["manufacturer", "model", "supplier"],
        "dates_certificates": ["date", "certificate", "completion", "commission"],
        "counts": ["no.", "number", "count", "qty", "quantity"],
        "dimensions_capacities": [
            "dimension",
            "width",
            "height",
            "depth",
            "capacity",
            "rating",
            "load",
        ],
        "materials_finishes": ["material", "finish", "clad", "roof", "wall", "floor"],
        "identifiers_references": ["serial", "reference", "drawing no", "certificate no"],
        "locations_layout": ["location", "plan", "layout", "elevation"],
        "maintenance_only": ["maintenance", "coshh", "safety data sheet"],
    }
    for family, terms in family_terms.items():
        if any(term in lower for term in terms):
            families.add(family)
    return sorted(families) or ["section_discovery"]


def source_route(source_filename: str) -> Route:
    lower = source_filename.lower()
    if "drawing" in lower:
        return "drawing_text"
    if "ev" in lower or "part 3" in lower or "part 6" in lower:
        return "table"
    if "zip" in lower:
        return "deprioritized"
    return "text"


def split_contiguous_pages(pages: list[int], max_pages: int) -> list[tuple[int, int]]:
    if not pages:
        return []
    ranges: list[tuple[int, int]] = []
    start = pages[0]
    end = pages[0]
    for page in pages[1:]:
        if page == end + 1 and page - start + 1 <= max_pages:
            end = page
        else:
            ranges.append((start, end))
            start = page
            end = page
    ranges.append((start, end))
    return ranges


def _section_row(
    source_id: str,
    source_filename: str,
    title: str,
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    page_types = Counter(str(page["primary_page_type"]) for page in pages)
    routes = Counter(str(page["recommended_route"]) for page in pages)
    route = routes.most_common(1)[0][0] if routes else "text"
    tags = sorted({tag for page in pages for tag in page["secondary_tags"]})
    domains = sorted({domain for page in pages for domain in page["likely_dictionary_domains"]})
    families = sorted({family for page in pages for family in page["likely_target_families"]})
    return {
        "section_id": f"{source_id}_{min(int(page['page_number']) for page in pages):06d}",
        "source_id": source_id,
        "source_filename": source_filename,
        "title": title,
        "page_start": min(int(page["page_number"]) for page in pages),
        "page_end": max(int(page["page_number"]) for page in pages),
        "page_count": len(pages),
        "primary_page_types": dict(sorted(page_types.items())),
        "secondary_evidence_tags": tags,
        "likely_dictionary_domains": domains,
        "likely_target_families": families,
        "recommended_route": route,
        "evidence_bearing": any(bool(page["evidence_bearing"]) for page in pages),
    }


def hierarchy_sections_missing_from_cache(
    hierarchy_nodes: list[dict[str, Any]],
    existing_sections: list[dict[str, Any]],
    source_names: dict[str, str],
) -> list[dict[str, Any]]:
    existing = {
        (str(section["source_id"]), int(section["page_start"]), int(section["page_end"]))
        for section in existing_sections
    }
    rows: list[dict[str, Any]] = []
    for node in hierarchy_nodes:
        source_id = str(node.get("source_id") or "")
        start = int(node.get("page_start") or 0)
        end = int(node.get("page_end") or start)
        if not source_id or start <= 0 or (source_id, start, end) in existing:
            continue
        title = str(node.get("title") or node.get("text_summary") or "Known hierarchy node")
        classified = classify_page_text(title, source_filename=source_names.get(source_id, ""))
        rows.append(
            {
                "section_id": str(node.get("node_id") or f"{source_id}_{start:06d}"),
                "source_id": source_id,
                "source_filename": source_names.get(source_id, ""),
                "title": title,
                "page_start": start,
                "page_end": end,
                "page_count": max(end - start + 1, 1),
                "primary_page_types": {classified["primary_page_type"]: 1},
                "secondary_evidence_tags": classified["secondary_tags"],
                "likely_dictionary_domains": classified["likely_dictionary_domains"],
                "likely_target_families": classified["likely_target_families"],
                "recommended_route": classified["recommended_route"],
                "evidence_bearing": classified["evidence_bearing"],
            }
        )
    return rows


def load_hierarchy_nodes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = _read_json(path)
    nodes = payload.get("nodes") if isinstance(payload, dict) else []
    return [dict(node) for node in nodes] if isinstance(nodes, list) else []


def build_coverage_trace(
    candidates: list[dict[str, Any]],
    approved: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    parse_report: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {"stage": "candidate_plan", "rows": len(candidates)},
        {"stage": "approved_parse_plan", "rows": len(approved)},
        {"stage": "merged_parse_batches", "rows": len(batches)},
        {"stage": "parse_execution", **parse_report},
    ]


def build_review_rows(
    inventory: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    section_counts = Counter(
        str(section["source_id"]) for section in sections if section["evidence_bearing"]
    )
    return [
        {
            "source_id": row["source_id"],
            "source_filename": row["source_filename"],
            "total_page_count": row["total_page_count"],
            "cached_page_count": row["cached_page_count"],
            "cache_percentage": row["cache_percentage"],
            "evidence_bearing_section_count": section_counts[str(row["source_id"])],
            "recommended_route": row["recommended_route"],
            "review_note": "poor coverage"
            if float(row["cache_percentage"]) < 10.0
            else "covered sample present",
        }
        for row in inventory
    ]


def _secondary_tags(text: str) -> set[str]:
    tags: set[str] = set()
    checks = {
        "manufacturer_model_table": ["manufacturer", "model"],
        "component_specification": ["specification", "description", "type"],
        "commissioning_result": ["commission", "test result"],
        "dimensions_measurements": [" mm", " m2", "width", "height", "dimension"],
        "count_annotation": ["no.", "number", "qty", "quantity"],
        "planning_condition": ["planning condition", "planning approval"],
        "statutory_compliance": ["building control", "certificate", "regulation"],
        "structural_loading": ["load", "kn/m", "structural"],
        "material_finish": ["material", "finish", "clad"],
        "equipment_schedule": ["schedule", "equipment"],
        "health_and_safety_only": ["hazard", "health and safety"],
        "drawing_symbol_dependency": ["symbol", "legend"],
        "layout_dependency": ["layout", "plan", "elevation"],
        "visual_confirmation_required": ["refer to drawing", "see drawing"],
    }
    for tag, terms in checks.items():
        if any(term in text for term in terms):
            tags.add(tag)
    return tags


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_index_or_contents(text: str) -> bool:
    return bool(
        re.search(r"\b(contents|index)\b", text)
        and (
            len(re.findall(r"\b\d+(\.\d+)*\b", text)) >= 3
            or len(re.findall(r"\.{3,}|\s\d{1,4}\s", text)) >= 3
        )
    )


def _is_certificate(text: str) -> bool:
    if _is_generic_manufacturer_literature(text) or _is_safety_data(text):
        return False
    if any(term in text for term in ["gradation analysis test report", "concrete cube register"]):
        return False
    return any(
        term in text
        for term in [
            "certificate",
            "certification",
            "test certificate",
            "completion certificate",
            "test report",
        ]
    )


def _is_planning_decision(text: str) -> bool:
    decision_terms = [
        "planning granted",
        "local planning authority",
        "planning permission",
        "planning approval",
        "town and country planning act",
    ]
    condition_terms = [
        "shall be submitted",
        "shall be installed",
        "hereby approved",
        "prior to occupation",
        "prior to superstructure",
        "condition",
    ]
    return any(term in text for term in decision_terms) and any(
        term in text for term in condition_terms
    )


def _is_schedule(text: str) -> bool:
    return "schedule" in text and any(
        term in text for term in ["item", "description", "manufacturer", "model", "reference"]
    )


def _is_table(text: str) -> bool:
    numeric_cells = len(re.findall(r"\b\d+[\d.,]*\b", text))
    table_labels = sum(
        1 for term in ["item", "qty", "quantity", "ref", "description", "model"] if term in text
    )
    lab_schedule_terms = [
        "sieve size",
        "percent passing",
        "concrete cube register",
        "sample number",
        "date of test",
        "gradation analysis test report",
        "control limits",
        "complies",
    ]
    return numeric_cells >= 8 and (
        len(re.findall(r"\s{2,}|\t|\|", text)) >= 3
        or table_labels >= 3
        or sum(1 for term in lab_schedule_terms if term in text) >= 2
    )


def _is_residual_hazard_schedule(text: str) -> bool:
    return (
        "remaining identified" in text
        and "hazard" in text
        and "proposed control measure" in text
    )


def _is_emergency_contacts(text: str) -> bool:
    return (
        "emergency contacts" in text
        and any(term in text for term in ["emergency number", "gas leak", "supplier"])
    )


def _is_reference_only_page(text: str) -> bool:
    short_reference = len(text) < 360 and "refer to" in text
    return short_reference and any(
        term in text
        for term in [
            "overleaf",
            "part 3",
            "part 6",
            "m&e",
            "manual",
            "drawings",
            "calculation report",
        ]
    )


def _is_part6_appendix_index(text: str) -> bool:
    return "part 6 - index" in text and "appendices" in text


def _is_certificate_index(text: str) -> bool:
    return (
        "commissioning / test certificates" in text
        and "certificates from the following companies are included" in text
    )


def _is_work_permit_template(text: str) -> bool:
    return (
        "work permit" in text
        and "valid for day of issue only" in text
        and any(term in text for term in ["nature of work", "estimated time period"])
    )


def _is_fire_system_certificate(text: str) -> bool:
    return (
        any(term in text for term in ["fire detection", "fire alarm", "disabled refuge"])
        and "certificate" in text
        and any(term in text for term in ["commissioning", "bs5839", "bs 5839"])
    )


def _is_commissioning_certificate(text: str) -> bool:
    return "commissioning certificate" in text or "certificate of commissioning" in text


def _is_installation_completion_certificate(text: str) -> bool:
    return "certificate of installation" in text or "certificate of completion" in text


def _is_pv_commissioning_record(text: str) -> bool:
    return any(term in text for term in ["pv commissioning form", "solar pv certificate"]) and any(
        term in text for term in ["unit 1", "array module", "inverter", "photovoltaic"]
    )


def _is_bms_points_schedule(text: str) -> bool:
    return "points schedule" in text and any(
        term in text for term in ["trend iq", "outstation", "airtech controls", "bms"]
    )


def _is_air_conditioning_model_serial_schedule(text: str) -> bool:
    return any(term in text for term in ["model number", "unit serial number", "serial no"]) and any(
        term in text
        for term in ["outdoor unit", "indoor unit", "bc controller", "mitsubishi", "pury-"]
    )


def _is_mechanical_test_sheet(text: str) -> bool:
    return any(term in text for term in ["wc extract fan", "ahu supply fan", "indoor unit supply air"]) and any(
        term in text for term in ["measured volume", "design volume", "commissioning report"]
    )


def _is_water_laboratory_report(text: str) -> bool:
    return any(term in text for term in ["als environmental", "test report", "certificate of conformity"]) and any(
        term in text for term in ["sample date", "water", "disinfection", "legionella"]
    )


def _is_hazardous_material_statement(text: str) -> bool:
    return any(
        term in text
        for term in [
            "hazardous materials used in construction",
            "asbestos statement",
            "no asbestos containing products were specified",
            "no asbestos containing products were specified / used",
        ]
    )


def _is_access_cleaning_guidance(text: str) -> bool:
    access_terms = [
        "access and cleaning strategy",
        "roof access guidance",
        "roof access methods",
        "roof access / fall restraint system",
        "cat ladder",
        "roof hatch",
        "man-safe",
        "fall restraint",
        "mewp",
        "roof work permit",
    ]
    return any(term in text for term in access_terms)


def _is_structural_report(text: str) -> bool:
    return (
        ("segro park, unit 1" in text and "129896" in text)
        and any(term in text for term in ["structural material use", "design summary", "fairhurst"])
        and not _is_structural_calculation(text)
    )


def _is_structural_calculation(text: str) -> bool:
    return (
        "fairhurst" in text
        and "calcs for" in text
        and any(term in text for term in ["tedds calculation", "design shear", "load combination"])
    )


def _is_loading_schedule(text: str) -> bool:
    loading_terms = [
        "office slab",
        "warehouse ground slab specification",
        "live warehouse 50kn/m2",
        "designed for imposed load",
        "imposed loads",
        "load per pile",
        "construction loads",
    ]
    return any(term in text for term in loading_terms) and (
        _is_table(text) or any(term in text for term in ["kn/m", "imposed load", "load per pile"])
    )


def _is_fire_strategy_drawing(text: str) -> bool:
    return any(term in text for term in ["fd60", "fd30", "fire strategy", "bs 5839"])


def _is_structural_drawing(text: str) -> bool:
    return any(
        term in text
        for term in [
            "piling layout",
            "pile cap",
            "ground beam",
            "warehouse ground slab specification",
            "office slab",
            "foundation loads",
            "structural drawings",
            "fairhurst drawing",
        ]
    ) and any(term in text for term in ["drwg", "drawing", "notes:", "grid"])


def _is_drawing(text: str) -> bool:
    if _is_safety_data(text):
        return False
    explicit_drawing = any(
        term in text for term in ["drawing no", "dwg no", "title block", "as built drawing"]
    )
    scaled_plan = ("scale:" in text or "scale " in text) and any(
        term in text for term in ["plan", "elevation", "layout", "drawing"]
    )
    plan_or_elevation = any(term in text for term in ["floor plan", "roof plan", "site plan"])
    elevation_with_layout = "elevation" in text and any(
        term in text for term in ["grid", "drawing", "scale", "layout"]
    )
    return explicit_drawing or scaled_plan or plan_or_elevation or elevation_with_layout


def _drawing_text_extractable(text: str) -> bool:
    return (
        any(term in text for term in ["drawing no", "title", "scale", "revision"])
        and len(text) >= 60
    )


def _is_product_datasheet(text: str) -> bool:
    return any(
        term in text for term in ["technical data sheet", "product data", "datasheet", "data sheet"]
    ) and any(term in text for term in ["manufacturer", "model", "specification"])


def _is_project_element_sheet(text: str) -> bool:
    return bool(
        re.search(r"\belement\s*:\s*\d+\.\d+\.\d+\b", text)
        and "nature of installation" in text
        and "product description" in text
    )


def _is_project_specific_context(text: str) -> bool:
    project_terms = [
        "segro park",
        "enfield",
        "east duck lees lane",
        "unit 1",
        "project number",
        "p18-010",
        "new building",
    ]
    return any(term in text for term in project_terms) or _is_project_element_sheet(text)


def _is_safety_data(text: str) -> bool:
    safety_terms = [
        "safety data sheet",
        "material safety data",
        "coshh assessment",
        "hazards identification",
        "regulation (ec) no. 1907/2006",
        "regulation (ec) no 1907/2006",
        "first aid measures",
        "exposure controls/personal protection",
    ]
    return any(term in text for term in safety_terms)


def _is_generic_manufacturer_literature(text: str) -> bool:
    literature_terms = [
        "product data sheet",
        "technical data sheet",
        "declaration of performance",
        "certificate of approval",
        "product conformity certification",
        "paving maintenance & repair guide",
    ]
    if not any(term in text for term in literature_terms):
        return False
    project_terms = [
        "nature of installation",
        "work description",
        "scope of works",
        "site:",
        "segro park enfield",
        "new building",
    ]
    return not any(term in text for term in project_terms)


def _is_supplier_contact_only(text: str) -> bool:
    contact_terms = sum(
        1 for term in ["telephone", "tel:", "fax", "email", "supplier", "company"] if term in text
    )
    installation_terms = ["nature of installation", "product description", "work description"]
    return contact_terms >= 3 and not any(term in text for term in installation_terms)


def _is_maintenance_only(text: str, source_filename: str) -> bool:
    lower = f"{source_filename} {text}".lower()
    return "maintenance" in lower and not any(
        term in lower
        for term in [
            "certificate",
            "schedule",
            "commission",
            "roof access guidance",
            "access and cleaning strategy",
        ]
    )


def _is_low_value_repetitive(text: str) -> bool:
    return (
        any(term in text for term in ["safety data sheet", "coshh", "material safety data"])
        or text.count("©") >= 3
    )


def _is_cover_or_separator(text: str) -> bool:
    return len(text) < 160 and any(
        term in text for term in ["building manual", "part ", "appendix"]
    )


def _range_near(section: dict[str, Any], start: int, end: int) -> bool:
    return int(section["page_end"]) >= start - 2 and int(section["page_start"]) <= end + 2


def _dedupe_windows(
    windows: list[tuple[int, int, str, str, Route]],
) -> list[tuple[int, int, str, str, Route]]:
    seen: set[tuple[int, int]] = set()
    unique = []
    for item in windows:
        key = (item[0], item[1])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    return (
        priority_rank.get(str(item["priority"]), 9),
        str(item["source_filename"]),
        int(item["page_start"]),
    )


def _node_pages(nodes: list[dict[str, Any]], terms: set[str]) -> set[int]:
    pages = set()
    for node in nodes:
        title = str(node.get("title") or node.get("text_summary") or "").lower()
        if any(term in title for term in terms):
            start = int(node.get("page_start") or 0)
            if start:
                pages.add(start)
    return pages


def _count_node_titles(nodes: list[dict[str, Any]], terms: set[str]) -> int:
    return sum(
        1
        for node in nodes
        if any(
            term in str(node.get("title") or node.get("text_summary") or "").lower()
            for term in terms
        )
    )


def _route_pages(rows: list[dict[str, Any]], route: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["recommended_route"] == route]


def _source_sort_key(source: SourceRegistryEntry) -> tuple[str, str]:
    return (Path(source.logical_path).name.lower(), source.source_id)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)
