from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

from segro_evidence_extraction.extraction_batch import (
    BATCH_REVIEW_SAMPLE_COUNT,
    BATCH_TARGET_COUNT,
    BatchSelectedTarget,
    EvidenceFirstBatchService,
    SourceRangePlan,
    batch_selected_target,
    build_preflight_report,
    preflight_errors,
    select_batch_targets,
    should_skip_extraction,
    split_range_for_pages,
)
from segro_evidence_extraction.models.common import ExpectedDataType, ProvenanceRef
from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry
from segro_evidence_extraction.models.target import DictionaryProvenance, TargetSpecification
from segro_evidence_extraction.parsing.models import (
    OcrRouting,
    ParsedPage,
    ParsingStatus,
    TextQualityMetrics,
)
from segro_evidence_extraction.parsing.page_cache import PageCacheLookupResult
from segro_evidence_extraction.vertical_slice import (
    EvidenceBundleRecord,
    ExtractionClient,
    ExtractionResult,
    ModelUsage,
    SourceRange,
    ValueShapeAssignment,
)


def test_selects_exactly_75_real_targets_deterministically() -> None:
    dictionary_path = Path("data/input/data_dictionary/SEGRO_Extraction_Template.xlsx")

    first = select_batch_targets(dictionary_path, _test_dir("selection_first"))
    second = select_batch_targets(dictionary_path, _test_dir("selection_second"))

    assert len(first) == BATCH_TARGET_COUNT
    assert [item.target.target_row_id for item in first] == [
        item.target.target_row_id for item in second
    ]
    assert all(item.target.source_dictionary_provenance for item in first)


def test_selection_balances_domains_shapes_and_unsupported_targets() -> None:
    selected = select_batch_targets(
        Path("data/input/data_dictionary/SEGRO_Extraction_Template.xlsx"),
        _test_dir("selection_balance"),
    )

    domains = {item.target.sub_domain for item in selected}
    shapes = {item.value_shape.value_shape_family for item in selected}
    unsupported = [
        item for item in selected if item.expected_support_status == "expected_unsupported"
    ]

    assert {"Component", "Statutory Compliance", "Technical Specification"} <= domains
    assert len(domains) >= 5
    assert {
        "descriptive_text",
        "categorical",
        "integer_count",
        "decimal_measurement",
        "date",
        "identifier_or_reference",
    } <= shapes
    assert len(unsupported) == 5
    assert all(item.target.sub_domain in {"Property", "Location", "Legal"} for item in unsupported)


def test_source_requests_are_explicitly_bounded_and_whole_manual_is_rejected() -> None:
    source = _source("src-manual", "manual.pdf", page_count=60)
    plan = SourceRangePlan(
        ranges=[
            SourceRange(
                source_id=source.source_id,
                logical_path=source.logical_path,
                page_start=1,
                page_end=60,
                reason="bad",
            )
        ],
        unique_pages_requested=60,
    )

    errors = preflight_errors(
        selected_targets=_selected_targets(75),
        source_plan=plan,
        expected_calls=75,
        cost=0.01,
        ceiling=0.10,
        source_registry={source.source_id: source},
    )

    assert "whole-manual parsing requested: manual.pdf" in errors


def test_source_plan_rejects_more_than_200_unique_pages() -> None:
    source = _source("src-manual", "manual.pdf", page_count=300)
    plan = SourceRangePlan(
        ranges=[
            SourceRange(
                source_id=source.source_id,
                logical_path=source.logical_path,
                page_start=1,
                page_end=201,
                reason="too broad",
            )
        ],
        unique_pages_requested=201,
    )

    errors = preflight_errors(
        selected_targets=_selected_targets(75),
        source_plan=plan,
        expected_calls=75,
        cost=0.01,
        ceiling=0.10,
        source_registry={source.source_id: source},
    )

    assert any("bounded batch limit is 200" in error for error in errors)


def test_cache_hits_misses_and_worker_invocations_are_reported(
    monkeypatch: Any,
) -> None:
    output_dir = _test_dir("preflight_cache")
    source = _source("src-manual", "manual.pdf", page_count=20)
    plan = SourceRangePlan(
        ranges=[
            SourceRange(
                source_id=source.source_id,
                logical_path=source.logical_path,
                page_start=1,
                page_end=15,
                reason="bounded",
            )
        ],
        unique_pages_requested=15,
    )

    class FakeCache:
        def read_range(self, **_: Any) -> PageCacheLookupResult:
            return PageCacheLookupResult(hits=11, misses=4, missing_pages=[1, 2, 5, 15])

    class FakeCachedService:
        parser_name = "fake"
        parser_version = "1"
        parser_config_fingerprint = "cfg"
        cache = FakeCache()

        def __init__(self, **_: Any) -> None:
            pass

    monkeypatch.setattr(
        "segro_evidence_extraction.extraction_batch.CachedBatchParsingService",
        FakeCachedService,
    )

    report = build_preflight_report(
        selected_targets=_selected_targets(75),
        source_plan=plan,
        source_registry={source.source_id: source},
        output_dir=output_dir,
        cache_root=output_dir / "cache",
        model_name="gpt-4o-mini",
        cost_ceiling_usd=0.10,
    )

    assert report.cache_hits_expected == 11
    assert report.cache_misses_expected == 4
    assert report.expected_parser_worker_invocations == 3
    assert split_range_for_pages([1, 2, 5, 15]) == [(1, 2), (5, 5), (15, 15)]


def test_retrieval_threshold_skips_extraction_without_forcing_bundle() -> None:
    target = _target("unsupported", ExpectedDataType.STRING, "Unavailable field")
    selected = batch_selected_target(target, "short_text")
    bundle = EvidenceBundleRecord(
        target_row_id=target.target_row_id,
        evidence_items=[],
        retrieval_status="no_relevant_evidence",
        combined_text="",
        character_count=0,
        token_estimate=0,
    )

    assert should_skip_extraction(selected, bundle)


def test_cost_ceiling_is_enforced() -> None:
    source = _source("src-manual", "manual.pdf", page_count=300)
    plan = SourceRangePlan(
        ranges=[
            SourceRange(
                source_id=source.source_id,
                logical_path=source.logical_path,
                page_start=1,
                page_end=10,
                reason="bounded",
            )
        ],
        unique_pages_requested=10,
    )

    errors = preflight_errors(
        selected_targets=_selected_targets(75),
        source_plan=plan,
        expected_calls=75,
        cost=0.11,
        ceiling=0.10,
        source_registry={source.source_id: source},
    )

    assert any("estimated cost" in error for error in errors)


def test_batch_run_uses_at_most_one_model_call_per_eligible_target_and_no_children(
    monkeypatch: Any,
) -> None:
    output_dir = _test_dir("service_run")
    source = _source("src-test", "manual.pdf", page_count=5)
    selected = _selected_targets(75)

    monkeypatch.setattr(
        "segro_evidence_extraction.extraction_batch.load_source_registry",
        lambda _: [source],
    )
    monkeypatch.setattr(
        "segro_evidence_extraction.extraction_batch.select_batch_targets",
        lambda *_: selected,
    )
    monkeypatch.setattr(
        "segro_evidence_extraction.extraction_batch.plan_batch_source_ranges",
        lambda _: SourceRangePlan(
            ranges=[
                SourceRange(
                    source_id=source.source_id,
                    logical_path=source.logical_path,
                    page_start=1,
                    page_end=1,
                    reason="fixture",
                )
            ],
            unique_pages_requested=1,
        ),
    )
    monkeypatch.setattr(
        EvidenceFirstBatchService,
        "_load_pages",
        lambda self, *_: (
            {source.source_id: [_page(source, 1, "Roof construction is steel.")]},
            [],
        ),
    )

    client = CountingExtractionClient()
    service = EvidenceFirstBatchService(
        source_manifest=output_dir / "manifest.json",
        dictionary_path=output_dir / "dictionary.xlsx",
        output_dir=output_dir / "out",
        cache_root=output_dir / "cache",
        extraction_client=client,
    )

    result = service.run()

    assert client.calls <= 75
    assert len(result.extraction_results) == 75
    assert (output_dir / "out" / "selected_targets.json").exists()
    assert len((output_dir / "out" / "manual_review_sample.csv").read_text().splitlines()) == (
        BATCH_REVIEW_SAMPLE_COUNT + 1
    )
    assert not multiprocessing.active_children()


def test_no_ocr_vlm_embeddings_or_vector_db_are_introduced() -> None:
    module_text = Path("src/segro_evidence_extraction/extraction_batch.py").read_text(
        encoding="utf-8"
    )

    forbidden = ["vector", "embedding", "vlm", "rerank"]
    assert not any(term in module_text.lower() for term in forbidden)


class CountingExtractionClient(ExtractionClient):
    provider = "mock"
    model_name = "mock-model"

    def __init__(self) -> None:
        self.calls = 0

    def extract(
        self,
        *,
        target: TargetSpecification,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> ExtractionResult:
        self.calls += 1
        span_ids = [bundle.evidence_spans[0].span_id] if bundle.evidence_spans else []
        return ExtractionResult(
            target_row_id=target.target_row_id,
            requirement_id=target.requirement_id,
            raw_model_value="Roof construction is steel",
            evidence_value="Roof construction is steel",
            normalized_value="Roof construction is steel",
            display_value="Roof construction is steel",
            proposed_value_shape="descriptive_text",
            value_bearing_quote="Roof construction is steel",
            unit=target.unit,
            status="extracted",
            confidence=0.8,
            supporting_span_ids=span_ids,
            model_provider=self.provider,
            model_name=self.model_name,
            model_usage=ModelUsage(input_tokens=10, output_tokens=5, estimated_cost_usd=0.0),
        )


def _selected_targets(count: int) -> list[BatchSelectedTarget]:
    return [
        BatchSelectedTarget(
            target=_target(
                f"roof_construction_description_{index}",
                ExpectedDataType.STRING,
                "Roof",
            ),
                value_shape=ValueShapeAssignment(
                    target_row_id=f"target-roof_construction_description_{index}",
                    field_name=f"roof_construction_description_{index}",
                    dictionary_declared_datatype=ExpectedDataType.STRING,
                    dictionary_declared_unit=None,
                value_shape_family="descriptive_text",
                inference_basis=["fixture"],
                confidence=0.8,
            ),
            selection_reason="fixture",
            expected_evidence_source_category="building manuals",
            expected_support_status="likely_supported",
        )
        for index in range(count)
    ]


def _test_dir(name: str) -> Path:
    path = Path("output/test_extraction_batch") / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _target(
    expected_field: str,
    datatype: ExpectedDataType,
    requirement_text: str,
    unit: str | None = None,
) -> TargetSpecification:
    return TargetSpecification(
        target_row_id=f"target-{expected_field}",
        requirement_id="REQ-1",
        sub_domain="Technical Specification",
        requirement_text=requirement_text,
        expected_field=expected_field,
        expected_data_type=datatype,
        unit=unit,
        source_dictionary_provenance=DictionaryProvenance(
            dictionary_id="dictionary",
            sheet_name="Extraction Template",
            row_number=1,
        ),
    )


def _source(source_id: str, logical_path: str, *, page_count: int | None) -> SourceRegistryEntry:
    return SourceRegistryEntry(
        source_id=source_id,
        original_path=logical_path,
        logical_path=logical_path,
        file_type=FileType.PDF,
        extension=".pdf",
        size_bytes=100,
        file_hash="hash",
        page_count=page_count,
    )


def _page(source: SourceRegistryEntry, page_number: int, text: str) -> ParsedPage:
    return ParsedPage(
        page_id=f"{source.source_id}:p{page_number}",
        source_id=source.source_id,
        content_hash="hash",
        page_number=page_number,
        physical_page_index=page_number - 1,
        parser_name="fake",
        parser_version="fake-1",
        text=text,
        character_count=len(text),
        word_count=len(text.split()),
        quality=TextQualityMetrics(
            character_count=len(text),
            word_count=len(text.split()),
            line_count=max(1, text.count("\n") + 1),
            average_line_length=10,
            alphabetic_ratio=0.5,
            numeric_ratio=0.1,
            repeated_character_ratio=0,
            text_density=1,
            scan_likelihood=0,
        ),
        scan_likelihood=0,
        ocr_routing=OcrRouting.NOT_REQUIRED,
        ocr_rationale="fixture",
        status=ParsingStatus.PARSED,
        parsing_duration_ms=1,
        provenance=[
            ProvenanceRef(source_id=source.source_id, page_or_sheet=f"page-{page_number}")
        ],
    )
