"""Targeted cache expansion for Batch V2 false-positive candidates."""

from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from segro_evidence_extraction.batch_v2_evidence_readiness import (
    DEFAULT_PAGE_CACHE_ROOT,
    load_cached_pages,
)
from segro_evidence_extraction.batch_v2_false_positive_audit import (
    DEFAULT_FALSE_POSITIVE_OUTPUT_DIR,
    audit_false_positive,
)
from segro_evidence_extraction.parsing.batch_worker import (
    DEFAULT_MAX_BATCH_PAGES,
    BatchRequest,
    BatchWorkerConfig,
)
from segro_evidence_extraction.parsing.page_cache import (
    CachedBatchParseResult,
    CachedBatchParsingService,
    CanonicalParsedPageCache,
)
from segro_evidence_extraction.parsing.service import load_source_registry
from segro_evidence_extraction.vertical_slice import _atomic_write_json

DEFAULT_SOURCE_MANIFEST = Path("output/sprint3_source_ingestion/source_pack_manifest.json")
DEFAULT_CACHE_EXPANSION_OUTPUT_DIR = Path(
    "output/enfield_unit1_evidence_first_batch_v2_cache_expansion_v1"
)
DEFAULT_SELECTION_DIR = Path("output/enfield_unit1_evidence_first_batch_v2_selection")
MAX_SELECTED_TARGETS = 30
MIN_SELECTED_TARGETS = 20


def run_batch_v2_cache_expansion_v1(
    *,
    false_positive_dir: Path = DEFAULT_FALSE_POSITIVE_OUTPUT_DIR,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    cache_root: Path = DEFAULT_PAGE_CACHE_ROOT,
    output_dir: Path = DEFAULT_CACHE_EXPANSION_OUTPUT_DIR,
    selection_dir: Path = DEFAULT_SELECTION_DIR,
    dry_run: bool = False,
    max_targets: int = MAX_SELECTED_TARGETS,
) -> dict[str, Any]:
    inputs = load_expansion_inputs(
        false_positive_dir=false_positive_dir,
        source_manifest=source_manifest,
        cache_root=cache_root,
        selection_dir=selection_dir,
    )
    result = build_cache_expansion_plan(inputs, max_targets=max_targets)
    execute_cache_expansion(result, inputs, output_dir=output_dir, dry_run=dry_run)
    write_cache_expansion_outputs(result, output_dir)
    return result


def load_expansion_inputs(
    *,
    false_positive_dir: Path,
    source_manifest: Path,
    cache_root: Path,
    selection_dir: Path,
) -> dict[str, Any]:
    candidate_path = false_positive_dir / "batch_v2_cache_expansion_candidates.json"
    rejected_path = false_positive_dir / "rejected_ready_targets.json"
    if not candidate_path.exists() or not rejected_path.exists():
        msg = f"Missing false-positive audit inputs in {false_positive_dir}"
        raise FileNotFoundError(msg)
    sources = load_source_registry(source_manifest)
    source_by_name = {Path(source.logical_path).name: source for source in sources}
    source_by_name.update({Path(source.original_path).name: source for source in sources})
    selected_rows = _read_json(selection_dir / "candidate_scores.json")
    metadata_by_id = {str(item["target_id"]): dict(item) for item in _as_list(selected_rows)}
    rejected = _as_list(_read_json(rejected_path))
    rejected_by_id = {str(item["target_id"]): item for item in rejected}
    candidates = _as_list(_read_json(candidate_path))
    pages_before = load_cached_pages(cache_root)
    return {
        "false_positive_dir": str(false_positive_dir),
        "source_manifest": str(source_manifest),
        "cache_root": str(cache_root),
        "selection_dir": str(selection_dir),
        "source_by_name": source_by_name,
        "cache_pages_before": pages_before,
        "candidates": candidates,
        "rejected_by_id": rejected_by_id,
        "metadata_by_id": metadata_by_id,
    }


def build_cache_expansion_plan(inputs: dict[str, Any], *, max_targets: int) -> dict[str, Any]:
    ranked = [rank_candidate(candidate, inputs) for candidate in inputs["candidates"]]
    ranked = sorted(ranked, key=rank_sort_key)
    selected = [
        {**item, "expansion_rank": index}
        for index, item in enumerate(
            [item for item in ranked if item["eligible_for_expansion"]][:max_targets],
            start=1,
        )
    ]
    selected_ids = {str(item["target_id"]) for item in selected}
    excluded = [
        item
        for item in ranked
        if not item["eligible_for_expansion"] or item["target_id"] not in selected_ids
    ]
    ranges = [approved_range(item, inputs) for item in selected]
    merged = merge_parse_ranges(ranges, inputs)
    before = coverage_report(inputs["cache_pages_before"], ranges)
    result = {
        "ranked_expansion_candidates": ranked,
        "selected_expansion_targets": selected,
        "excluded_expansion_candidates": excluded,
        "approved_page_ranges": ranges,
        "merged_parse_plan": merged,
        "cache_coverage_before": before,
        "cache_coverage_after": {},
        "parse_execution_report": {},
        "target_readiness_before": [
            inputs["rejected_by_id"][str(item["target_id"])] for item in selected
        ],
        "target_readiness_after": [],
        "newly_confirmed_targets": [],
        "still_not_ready_targets": [],
        "cache_expansion_metrics": {},
        "cache_expansion_trace": [],
        "batch_v2_cache_expansion_review": [],
    }
    return result


def rank_candidate(candidate: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    target_id = str(candidate["target_id"])
    rejected = inputs["rejected_by_id"].get(target_id, {})
    metadata = inputs["metadata_by_id"].get(target_id, {})
    source = str(candidate.get("likely_source") or rejected.get("audit_source_file") or "")
    page_start, page_end = parse_page_range(str(candidate.get("recommended_page_range") or ""))
    source_known = source in inputs["source_by_name"]
    missing_pages = missing_pages_for_range(
        source_file=source,
        page_start=page_start,
        page_end=page_end,
        pages_by_source=inputs["cache_pages_before"],
    )
    classification = str(rejected.get("false_positive_classification") or "")
    reason = str(candidate.get("reason_additional_pages_may_help") or "")
    value_shape = str(rejected.get("value_shape") or metadata.get("value_shape") or "")
    eligible, exclusion = expansion_eligibility(
        candidate=candidate,
        rejected=rejected,
        source_known=source_known,
        page_start=page_start,
        page_end=page_end,
        missing_pages=missing_pages,
    )
    score = 0
    if source_known:
        score += 20
    if candidate.get("likely_hierarchy_section"):
        score += 15
    if classification in {"component_only", "insufficient_cached_evidence"}:
        score += 20
    if "adjacent or more specific pages" in reason.lower():
        score += 15
    if value_shape in {"date", "identifier_or_reference", "integer_count"}:
        score += 10
    if page_start and page_end:
        page_count = page_end - page_start + 1
        score += max(0, 10 - page_count)
    else:
        page_count = 0
    if classification == "wrong_event_or_attribute":
        score -= 15
    if candidate.get("priority") == "high":
        score += 8
    if candidate.get("remain_deferred_even_after_expansion"):
        score -= 50
    return {
        **candidate,
        "false_positive_classification": classification,
        "value_shape": value_shape,
        "source_known": source_known,
        "page_start": page_start,
        "page_end": page_end,
        "page_count": page_count,
        "missing_pages_before": missing_pages,
        "eligible_for_expansion": eligible,
        "exclusion_reason": exclusion,
        "expansion_score": score,
        "ranking_rationale": ranking_rationale(
            source_known=source_known,
            page_count=page_count,
            classification=classification,
            reason=reason,
            eligible=eligible,
            exclusion=exclusion,
        ),
    }


def expansion_eligibility(
    *,
    candidate: dict[str, Any],
    rejected: dict[str, Any],
    source_known: bool,
    page_start: int | None,
    page_end: int | None,
    missing_pages: list[int],
) -> tuple[bool, str]:
    classification = str(rejected.get("false_positive_classification") or "")
    if classification == "dictionary_ambiguous" or candidate.get(
        "remain_deferred_even_after_expansion"
    ):
        return False, "dictionary_ambiguous_or_deferred"
    if classification == "wrong_event_or_attribute":
        return False, "current_evidence_answers_wrong_attribute"
    if not source_known:
        return False, "likely_source_unavailable"
    if not candidate.get("likely_hierarchy_section"):
        return False, "missing_hierarchy_section"
    if page_start is None or page_end is None:
        return False, "unbounded_page_range"
    if page_end - page_start + 1 > DEFAULT_MAX_BATCH_PAGES:
        return False, "page_range_exceeds_worker_batch_limit"
    if not missing_pages:
        return False, "recommended_range_already_cached"
    if classification not in {"component_only", "insufficient_cached_evidence"}:
        return False, "low_expected_evidence_gain"
    return True, ""


def missing_pages_for_range(
    *,
    source_file: str,
    page_start: int | None,
    page_end: int | None,
    pages_by_source: dict[str, list[dict[str, Any]]],
) -> list[int]:
    if page_start is None or page_end is None:
        return []
    cached_pages = {int(page["page_number"]) for page in pages_by_source.get(source_file, [])}
    return [page for page in range(page_start, page_end + 1) if page not in cached_pages]


def approved_range(candidate: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    source = inputs["source_by_name"][str(candidate["likely_source"])]
    return {
        "target_id": candidate["target_id"],
        "field_name": candidate["field_name"],
        "source_file": candidate["likely_source"],
        "source_id": source.source_id,
        "source_path": source.original_path,
        "hierarchy_section": candidate.get("likely_hierarchy_section"),
        "page_start": candidate["page_start"],
        "page_end": candidate["page_end"],
        "page_count": candidate["page_count"],
        "range_key": f"{source.source_id}:{candidate['page_start']}-{candidate['page_end']}",
        "reason_range_is_relevant": candidate["reason_additional_pages_may_help"],
        "expected_evidence_form": expected_evidence_form(candidate),
        "expected_value_shape": candidate["value_shape"],
        "parsing_priority": candidate.get("priority") or "medium",
    }


def merge_parse_ranges(
    ranges: list[dict[str, Any]], inputs: dict[str, Any]
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in ranges:
        by_source[str(item["source_id"])].append(item)
    merged: list[dict[str, Any]] = []
    cached = inputs["cache_pages_before"]
    for source_id, source_ranges in sorted(by_source.items()):
        source_file = str(source_ranges[0]["source_file"])
        pages: set[int] = set()
        target_ids: set[str] = set()
        for item in source_ranges:
            target_ids.add(str(item["target_id"]))
            pages.update(range(int(item["page_start"]), int(item["page_end"]) + 1))
        cached_pages = {int(page["page_number"]) for page in cached.get(source_file, [])}
        missing = sorted(page for page in pages if page not in cached_pages)
        for start, end in split_contiguous(missing, DEFAULT_MAX_BATCH_PAGES):
            merged.append(
                {
                    "source_id": source_id,
                    "source_file": source_file,
                    "source_path": source_ranges[0]["source_path"],
                    "page_start": start,
                    "page_end": end,
                    "page_count": end - start + 1,
                    "target_ids": sorted(target_ids),
                    "worker_batch_page_limit": DEFAULT_MAX_BATCH_PAGES,
                    "parse_required": True,
                }
            )
    return merged


def execute_cache_expansion(
    result: dict[str, Any],
    inputs: dict[str, Any],
    *,
    output_dir: Path,
    dry_run: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_by_name = inputs["source_by_name"]
    service = CachedBatchParsingService(
        cache=CanonicalParsedPageCache(Path(inputs["cache_root"])),
        batch_config=BatchWorkerConfig(),
    )
    parse_results: list[CachedBatchParseResult] = []
    started = time.perf_counter()
    if not dry_run:
        for item in result["merged_parse_plan"]:
            source = source_by_name[str(item["source_file"])]
            parse_results.append(
                service.parse(
                    BatchRequest(
                        source=source,
                        source_path=source.original_path,
                        output_dir=str(
                            output_dir
                            / "parser_workers"
                            / f"{source.source_id}_{item['page_start']:04d}_{item['page_end']:04d}"
                        ),
                        page_start=int(item["page_start"]),
                        page_end=int(item["page_end"]),
                    )
                )
            )
    pages_after = load_cached_pages(Path(inputs["cache_root"]))
    after = coverage_report(pages_after, result["approved_page_ranges"])
    after_rows = re_audit_selected_targets(result["selected_expansion_targets"], pages_after)
    newly = [
        row
        for row in after_rows
        if row["false_positive_classification"] == "confirmed_execution_ready"
    ]
    still = [
        row
        for row in after_rows
        if row["false_positive_classification"] != "confirmed_execution_ready"
    ]
    result["cache_coverage_after"] = after
    result["parse_execution_report"] = parse_execution_report(parse_results, started, dry_run)
    result["target_readiness_after"] = after_rows
    result["newly_confirmed_targets"] = newly
    result["still_not_ready_targets"] = still
    result["cache_expansion_metrics"] = expansion_metrics(result)
    result["cache_expansion_trace"] = expansion_trace(result)
    result["batch_v2_cache_expansion_review"] = review_rows(result)


def re_audit_selected_targets(
    selected: list[dict[str, Any]], pages_by_source: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in selected:
        best = best_post_expansion_excerpt(item, pages_by_source)
        audited = audit_false_positive(
            {
                **item,
                "best_cached_excerpt": best["excerpt"],
                "best_cached_page": best["page_number"],
                "audit_source_file": item["likely_source"],
                "audit_classification": "execution_ready",
            }
        )
        rows.append(audited)
    return rows


def best_post_expansion_excerpt(
    target: dict[str, Any], pages_by_source: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    terms = [
        token
        for token in re.split(r"[_\W]+", str(target["field_name"]).lower())
        if len(token) > 2
    ]
    start, end = int(target["page_start"]), int(target["page_end"])
    best_score = -1
    best: dict[str, Any] = {"score": -1, "excerpt": "", "page_number": None}
    for page in pages_by_source.get(str(target["likely_source"]), []):
        page_number = int(page["page_number"])
        if page_number < start or page_number > end:
            continue
        text = str(page.get("extracted_text") or "")
        for excerpt in candidate_excerpts(text):
            lower = excerpt.lower()
            score = sum(1 for term in terms if term in lower)
            if score > best_score:
                best_score = score
                best = {
                    "score": score,
                    "excerpt": trim_excerpt(excerpt),
                    "page_number": page_number,
                }
    return best


def coverage_report(
    pages_by_source: dict[str, list[dict[str, Any]]], ranges: list[dict[str, Any]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_unique: set[tuple[str, int]] = set()
    cached_unique: set[tuple[str, int]] = set()
    for item in ranges:
        source_file = str(item["source_file"])
        wanted = set(range(int(item["page_start"]), int(item["page_end"]) + 1))
        cached = {int(page["page_number"]) for page in pages_by_source.get(source_file, [])}
        hits = sorted(wanted & cached)
        missing = sorted(wanted - cached)
        total_unique.update((source_file, page) for page in wanted)
        cached_unique.update((source_file, page) for page in hits)
        rows.append(
            {
                "target_id": item["target_id"],
                "source_file": source_file,
                "page_start": item["page_start"],
                "page_end": item["page_end"],
                "requested_pages": len(wanted),
                "cached_pages": hits,
                "missing_pages": missing,
                "cache_complete": not missing,
            }
        )
    return {
        "ranges": rows,
        "unique_pages_planned": len(total_unique),
        "unique_pages_cached": len(cached_unique),
        "unique_pages_missing": len(total_unique - cached_unique),
    }


def parse_execution_report(
    parse_results: list[CachedBatchParseResult], started: float, dry_run: bool
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for result in parse_results:
        for batch in result.batch_results:
            failures.extend(failure.model_dump(mode="json") for failure in batch.pages_failed)
    return {
        "dry_run": dry_run,
        "parser_batches_run": sum(result.worker_invocation_count for result in parse_results),
        "cache_service_calls": len(parse_results),
        "pages_loaded_from_cache": sum(
            len(result.pages_loaded_from_cache) for result in parse_results
        ),
        "new_pages_parsed": sum(len(result.pages_newly_parsed) for result in parse_results),
        "page_artifacts_written": sum(result.page_artifacts_written for result in parse_results),
        "parse_failures": failures,
        "timeout_or_restart_count": sum(result.restart_count for result in parse_results),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 2),
        "active_child_count_after_cleanup": max(
            [0, *(result.active_child_count_after_cleanup for result in parse_results)]
        ),
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


def expansion_metrics(result: dict[str, Any]) -> dict[str, Any]:
    before = result["cache_coverage_before"]
    after = result["cache_coverage_after"]
    parse_report = result["parse_execution_report"]
    newly = len(result["newly_confirmed_targets"])
    parsed = int(parse_report["new_pages_parsed"])
    return {
        "candidates_reviewed": len(result["ranked_expansion_candidates"]),
        "selected_target_count": len(result["selected_expansion_targets"]),
        "excluded_target_count": len(result["excluded_expansion_candidates"]),
        "unique_pages_planned": before["unique_pages_planned"],
        "pages_already_cached_before": before["unique_pages_cached"],
        "pages_missing_before": before["unique_pages_missing"],
        "pages_cached_after": after["unique_pages_cached"],
        "new_pages_parsed": parsed,
        "parser_batches_run": parse_report["parser_batches_run"],
        "parse_failure_count": len(parse_report["parse_failures"]),
        "newly_confirmed_count": newly,
        "still_not_ready_count": len(result["still_not_ready_targets"]),
        "confirmation_rate": round(newly / len(result["selected_expansion_targets"]), 4)
        if result["selected_expansion_targets"]
        else 0.0,
        "page_cost_per_newly_confirmed_target": round(parsed / newly, 2) if newly else None,
        "targeted_cache_expansion_materially_improved_readiness": newly > 0,
        "source_performance": dict(
            Counter(str(item["audit_source_file"]) for item in result["newly_confirmed_targets"])
        ),
        "still_not_ready_by_reason": dict(
            Counter(
                str(item["false_positive_classification"])
                for item in result["still_not_ready_targets"]
            )
        ),
    }


def expansion_trace(result: dict[str, Any]) -> list[dict[str, Any]]:
    after_by_id = {str(item["target_id"]): item for item in result["target_readiness_after"]}
    return [
        {
            "target_id": item["target_id"],
            "field_name": item["field_name"],
            "expansion_rank": item["expansion_rank"],
            "selected_reason": item["ranking_rationale"],
            "page_range": f"{item['page_start']}-{item['page_end']}",
            "after_classification": after_by_id[str(item["target_id"])][
                "false_positive_classification"
            ],
            "after_rationale": after_by_id[str(item["target_id"])][
                "false_positive_rationale"
            ],
        }
        for item in result["selected_expansion_targets"]
    ]


def review_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    after_by_id = {str(item["target_id"]): item for item in result["target_readiness_after"]}
    rows = []
    for item in result["ranked_expansion_candidates"]:
        after = after_by_id.get(str(item["target_id"]), {})
        rows.append(
            {
                "target ID": item["target_id"],
                "field name": item["field_name"],
                "eligible": item["eligible_for_expansion"],
                "selected": bool(after),
                "score": item["expansion_score"],
                "source": item.get("likely_source"),
                "page range": item.get("recommended_page_range"),
                "exclusion reason": item["exclusion_reason"],
                "ranking rationale": item["ranking_rationale"],
                "after classification": after.get("false_positive_classification", ""),
                "after rationale": after.get("false_positive_rationale", ""),
            }
        )
    return rows


def ranking_rationale(
    *,
    source_known: bool,
    page_count: int,
    classification: str,
    reason: str,
    eligible: bool,
    exclusion: str,
) -> str:
    if not eligible:
        return f"Excluded: {exclusion}."
    return (
        f"Eligible {classification} target with known source={source_known}, "
        f"{page_count} planned pages, and expansion reason: {reason}"
    )


def expected_evidence_form(candidate: dict[str, Any]) -> str:
    value_shape = str(candidate["value_shape"])
    if value_shape in {"integer_count", "decimal_measurement"}:
        return "local numeric/table row"
    if value_shape in {"identifier_or_reference", "date"}:
        return "labelled field or schedule row"
    return "component-specific narrative or schedule entry"


def parse_page_range(value: str) -> tuple[int | None, int | None]:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", value)
    if not match:
        return None, None
    start, end = int(match.group(1)), int(match.group(2))
    if start < 1 or end < start:
        return None, None
    return start, end


def split_contiguous(pages: list[int], max_pages: int) -> list[tuple[int, int]]:
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


def rank_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        not item["eligible_for_expansion"],
        -int(item["expansion_score"]),
        int(item["page_count"] or 999),
        str(item["target_id"]),
    )


def candidate_excerpts(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [
        " ".join(lines[max(0, index - 1) : index + 2])
        for index, _line in enumerate(lines)
    ] or [text[:600]]


def trim_excerpt(text: str, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def write_cache_expansion_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_outputs = {
        "ranked_expansion_candidates.json": result["ranked_expansion_candidates"],
        "selected_expansion_targets.json": result["selected_expansion_targets"],
        "excluded_expansion_candidates.json": result["excluded_expansion_candidates"],
        "approved_page_ranges.json": result["approved_page_ranges"],
        "merged_parse_plan.json": result["merged_parse_plan"],
        "cache_coverage_before.json": result["cache_coverage_before"],
        "cache_coverage_after.json": result["cache_coverage_after"],
        "parse_execution_report.json": result["parse_execution_report"],
        "target_readiness_before.json": result["target_readiness_before"],
        "target_readiness_after.json": result["target_readiness_after"],
        "newly_confirmed_targets.json": result["newly_confirmed_targets"],
        "still_not_ready_targets.json": result["still_not_ready_targets"],
        "cache_expansion_metrics.json": result["cache_expansion_metrics"],
        "cache_expansion_trace.json": result["cache_expansion_trace"],
    }
    for filename, payload in json_outputs.items():
        _atomic_write_json(output_dir / filename, payload)
    write_csv(
        output_dir / "ranked_expansion_candidates.csv",
        result["ranked_expansion_candidates"],
        ranked_fields(),
    )
    write_csv(
        output_dir / "selected_expansion_targets.csv",
        result["selected_expansion_targets"],
        selected_fields(),
    )
    write_csv(
        output_dir / "batch_v2_cache_expansion_review.csv",
        result["batch_v2_cache_expansion_review"],
        review_fields(),
    )
    (output_dir / "batch_v2_cache_expansion_summary.md").write_text(
        summary_markdown(result), encoding="utf-8"
    )
    (output_dir / "next_expansion_recommendations.md").write_text(
        recommendations_markdown(result), encoding="utf-8"
    )


def ranked_fields() -> list[str]:
    return [
        "target_id",
        "field_name",
        "false_positive_classification",
        "likely_source",
        "likely_hierarchy_section",
        "recommended_page_range",
        "value_shape",
        "priority",
        "eligible_for_expansion",
        "exclusion_reason",
        "expansion_score",
        "ranking_rationale",
    ]


def selected_fields() -> list[str]:
    return ["expansion_rank", *ranked_fields()]


def review_fields() -> list[str]:
    return [
        "target ID",
        "field name",
        "eligible",
        "selected",
        "score",
        "source",
        "page range",
        "exclusion reason",
        "ranking rationale",
        "after classification",
        "after rationale",
    ]


def summary_markdown(result: dict[str, Any]) -> str:
    metrics = result["cache_expansion_metrics"]
    lines = [
        "# Batch V2 Targeted Cache Expansion V1",
        "",
        f"- Candidates reviewed: {metrics['candidates_reviewed']}",
        f"- Selected targets: {metrics['selected_target_count']}",
        f"- Unique pages planned: {metrics['unique_pages_planned']}",
        f"- New pages parsed: {metrics['new_pages_parsed']}",
        f"- Newly confirmed targets: {metrics['newly_confirmed_count']}",
        f"- Still not ready: {metrics['still_not_ready_count']}",
        f"- Confirmation rate: {metrics['confirmation_rate']}",
    ]
    return "\n".join(lines) + "\n"


def recommendations_markdown(result: dict[str, Any]) -> str:
    metrics = result["cache_expansion_metrics"]
    if metrics["newly_confirmed_count"]:
        recommendation = "A follow-up tranche may be justified for similar source sections."
    else:
        recommendation = "Do not continue text cache expansion without stronger source guidance."
    return (
        "# Next Expansion Recommendations\n\n"
        f"- {recommendation}\n"
        "- Keep dictionary-ambiguous and visual-only targets out of cache expansion.\n"
    )


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


def _as_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Expected JSON list")
    return [dict(item) for item in value]
