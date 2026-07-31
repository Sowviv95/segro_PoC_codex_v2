from __future__ import annotations

import shutil
from pathlib import Path

from segro_evidence_extraction.batch_v2_cache_expansion import (
    DEFAULT_SOURCE_MANIFEST,
    build_cache_expansion_plan,
    merge_parse_ranges,
    parse_page_range,
    run_batch_v2_cache_expansion_v1,
    split_contiguous,
)
from segro_evidence_extraction.batch_v2_evidence_readiness import DEFAULT_PAGE_CACHE_ROOT
from segro_evidence_extraction.batch_v2_false_positive_audit import (
    DEFAULT_FALSE_POSITIVE_OUTPUT_DIR,
)


def test_excludes_ambiguous_visual_unavailable_and_wrong_attribute_candidates() -> None:
    inputs = synthetic_inputs(
        candidates=[
            candidate("ambiguous", classification="dictionary_ambiguous", deferred=True),
            candidate("wrong", classification="wrong_event_or_attribute"),
            candidate("missing_source", source="Missing.pdf", classification="component_only"),
            candidate("ready", classification="component_only"),
        ]
    )

    result = build_cache_expansion_plan(inputs, max_targets=30)
    by_id = {item["target_id"]: item for item in result["ranked_expansion_candidates"]}

    assert by_id["trg_ambiguous"]["eligible_for_expansion"] is False
    assert by_id["trg_wrong"]["eligible_for_expansion"] is False
    assert by_id["trg_missing_source"]["eligible_for_expansion"] is False
    assert result["selected_expansion_targets"][0]["target_id"] == "trg_ready"


def test_deterministic_ranking_and_maximum_target_count() -> None:
    inputs = synthetic_inputs(
        candidates=[
            candidate(f"target_{index}", classification="component_only")
            for index in range(40)
        ]
    )

    first = build_cache_expansion_plan(inputs, max_targets=30)
    second = build_cache_expansion_plan(inputs, max_targets=30)

    assert len(first["selected_expansion_targets"]) == 30
    assert first["selected_expansion_targets"] == second["selected_expansion_targets"]


def test_bounded_page_ranges_and_merging_avoid_duplicate_parsing() -> None:
    assert parse_page_range("3-7") == (3, 7)
    assert parse_page_range("7-3") == (None, None)
    assert split_contiguous(list(range(1, 13)), 10) == [(1, 10), (11, 12)]
    inputs = synthetic_inputs(candidates=[])
    ranges = [
        approved("trg_a", 1, 5),
        approved("trg_b", 4, 8),
        approved("trg_c", 20, 21),
    ]

    merged = merge_parse_ranges(ranges, inputs)

    assert [(item["page_start"], item["page_end"]) for item in merged] == [(1, 8), (20, 21)]
    assert all(item["page_count"] <= 10 for item in merged)


def test_cache_reuse_omits_already_cached_pages_from_parse_plan() -> None:
    inputs = synthetic_inputs(candidates=[])
    inputs["cache_pages_before"] = {
        "Building Manual - Part 1 General.pdf": [
            {"page_number": 1},
            {"page_number": 2},
            {"page_number": 5},
        ]
    }

    merged = merge_parse_ranges([approved("trg_a", 1, 5)], inputs)

    assert [(item["page_start"], item["page_end"]) for item in merged] == [(3, 4)]


def test_real_cache_expansion_dry_run_reconciles_without_parser_invocation() -> None:
    if not all(
        path.exists()
        for path in [
            DEFAULT_FALSE_POSITIVE_OUTPUT_DIR,
            DEFAULT_SOURCE_MANIFEST,
            DEFAULT_PAGE_CACHE_ROOT,
        ]
    ):
        return
    output_root = Path("output/test_batch_v2_cache_expansion")
    if output_root.exists():
        shutil.rmtree(output_root)
    try:
        result = run_batch_v2_cache_expansion_v1(output_dir=output_root, dry_run=True)
        metrics = result["cache_expansion_metrics"]

        assert metrics["selected_target_count"] <= 30
        assert metrics["new_pages_parsed"] == 0
        assert result["parse_execution_report"]["parser_batches_run"] == 0
        assert len(result["target_readiness_after"]) == metrics["selected_target_count"]
        assert (output_root / "merged_parse_plan.json").exists()
    finally:
        if output_root.exists():
            shutil.rmtree(output_root)


def synthetic_inputs(*, candidates: list[dict[str, object]]) -> dict[str, object]:
    source = {
        "source_id": "src_part1",
        "original_path": "data/input/Building Manual - Part 1 General.pdf",
        "logical_path": "Building Manual - Part 1 General.pdf",
        "file_hash": "abc",
        "size_bytes": 1,
        "file_type": "pdf",
    }
    from segro_evidence_extraction.models.source import SourceRegistryEntry

    source_entry = SourceRegistryEntry.model_validate(source)
    rejected = {
        str(item["target_id"]): {
            **item,
            "false_positive_classification": item["classification"],
            "audit_source_file": item["likely_source"],
            "value_shape": item["value_shape"],
            "best_cached_excerpt": "component appears",
            "audit_classification": "execution_ready",
        }
        for item in candidates
    }
    return {
        "false_positive_dir": "synthetic",
        "source_manifest": "synthetic",
        "cache_root": "synthetic",
        "selection_dir": "synthetic",
        "source_by_name": {"Building Manual - Part 1 General.pdf": source_entry},
        "cache_pages_before": {},
        "candidates": candidates,
        "rejected_by_id": rejected,
        "metadata_by_id": {},
    }


def candidate(
    name: str,
    *,
    classification: str,
    source: str = "Building Manual - Part 1 General.pdf",
    deferred: bool = False,
) -> dict[str, object]:
    return {
        "target_id": f"trg_{name}",
        "field_name": name,
        "likely_source": source,
        "likely_hierarchy_section": "Section",
        "recommended_page_range": "1-5",
        "reason_additional_pages_may_help": (
            "Adjacent or more specific pages may contain the missing requested attribute."
        ),
        "priority": "medium",
        "remain_deferred_even_after_expansion": deferred,
        "classification": classification,
        "value_shape": "short_text",
    }


def approved(target_id: str, start: int, end: int) -> dict[str, object]:
    return {
        "target_id": target_id,
        "field_name": target_id,
        "source_file": "Building Manual - Part 1 General.pdf",
        "source_id": "src_part1",
        "source_path": "data/input/Building Manual - Part 1 General.pdf",
        "page_start": start,
        "page_end": end,
        "page_count": end - start + 1,
    }
