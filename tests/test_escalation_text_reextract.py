from __future__ import annotations

import multiprocessing
from pathlib import Path

from segro_evidence_extraction.escalation_text_reextract import (
    DEFAULT_BATCH_V1_OUTPUT_DIR,
    DEFAULT_ESCALATION_OUTPUT_DIR,
    DEFAULT_REEXTRACT_V1_OUTPUT_DIR,
    DEFAULT_REEXTRACT_V2_OUTPUT_DIR,
    EXPECTED_TEXT_REEXTRACT_IDS,
    ApprovedEvidenceRecord,
    TextReextractArtifacts,
    dock_count_support,
    floor_capacity_support,
    load_ready_text_target_ids,
    run_escalation_text_reextract_v1,
    validate_text_canonical_evidence,
)
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.reextract_pass import (
    ParsedResponseDiagnostic,
    StructuredExtractionEnvelope,
    parse_structured_attribute_response,
)
from segro_evidence_extraction.target_semantics import TargetIntent
from segro_evidence_extraction.vertical_slice import EvidenceBundleRecord, ExtractionResult


class RecordingTextClient:
    provider = "test"
    model_name = "gpt-4o-mini"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract_structured(
        self,
        *,
        target: TargetSpecification,
        intent: TargetIntent,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> tuple[ExtractionResult, ParsedResponseDiagnostic, StructuredExtractionEnvelope | None]:
        _ = intent, max_output_tokens
        self.calls.append(target.target_row_id)
        span = bundle.evidence_spans[0]
        if target.expected_field == "floor_construction_capacity":
            raw = "50 kN/m2"
            quote = "Floor slab loading capacity is 50 kN/m2"
            unit = "kN/m2"
        else:
            raw = "5"
            quote = "5No dock levellers"
            unit = None
        response = {
            "target_id": target.target_row_id,
            "status": "extracted",
            "requested_attribute": intent.requested_attribute,
            "raw_value": raw,
            "normalized_value": raw,
            "unit": unit,
            "supporting_span_ids": [span.span_id],
            "value_bearing_text": quote,
            "confidence": 0.82,
            "ambiguity": None,
            "rejection_reason": None,
        }
        import json

        return parse_structured_attribute_response(
            target=target,
            bundle=bundle,
            provider=self.provider,
            model_name=self.model_name,
            response_text=json.dumps(response),
            finish_reason="stop",
        )


def test_exactly_two_escalation_ready_targets_loaded() -> None:
    artifacts = _artifacts()

    ready = load_ready_text_target_ids(artifacts)

    assert set(ready) == EXPECTED_TEXT_REEXTRACT_IDS
    assert len(ready) == 2


def test_manual_review_targets_cannot_enter_scope() -> None:
    artifacts = _artifacts()
    ready = set(load_ready_text_target_ids(artifacts))
    manual = {
        target_id
        for target_id, item in artifacts.readiness_by_id.items()
        if item.readiness == "manual_review_only"
    }

    assert ready.isdisjoint(manual)


def test_no_new_source_range_is_requested_in_real_preflight() -> None:
    client = RecordingTextClient()
    result = run_escalation_text_reextract_v1(
        output_dir=Path("output/test_escalation_text_reextract_real_preflight"),
        extraction_client=client,
    )
    escalation_pages = _escalation_page_set()
    requested_pages = {
        page
        for pages in result.preflight_report.approved_source_pages.values()
        for page in pages
    }

    assert requested_pages <= escalation_pages
    assert result.telemetry.parser_worker_invocations == 0
    assert client.calls == []


def test_floor_capacity_requires_value_unit_and_floor_association() -> None:
    supported, candidates, issues = floor_capacity_support(
        "Warehouse floor slab loading capacity is 50 kN/m2."
    )
    unrelated_strength, _, strength_issues = floor_capacity_support(
        "Concrete slab compressive strength is 40 N/mm2."
    )

    assert supported == "supported"
    assert candidates == ["50 kN/m2"]
    assert not issues
    assert unrelated_strength != "supported"
    assert "material strength evidence is not floor loading capacity" in strength_issues


def test_multiple_floor_load_cases_remain_multiple_candidates() -> None:
    status, candidates, _ = floor_capacity_support(
        "Warehouse floor slab loading capacity is 50 kN/m2 and office floor load is 4 kN/m2."
    )

    assert status == "supported"
    assert candidates == ["50 kN/m2", "4 kN/m2"]


def test_dock_count_requires_integer_associated_with_dock_component() -> None:
    supported, candidates, _ = dock_count_support("There are 5No dock levellers.")
    fire_doors, _, fire_issues = dock_count_support("There are 4No fire exit doors.")
    section_number, section_candidates, _ = dock_count_support("2.6.4 | DOCK LEVELLERS | 2.7")

    assert supported == "supported"
    assert candidates == ["5No dock levellers"]
    assert fire_doors != "supported"
    assert "fire-door numbers are not dock count evidence" in fire_issues
    assert section_number == "component_only"
    assert section_candidates == []


def test_ambiguous_dock_door_and_leveller_counts_are_not_collapsed() -> None:
    status, candidates, _ = dock_count_support("5No dock levellers and 6No loading doors.")

    assert status == "supported"
    assert candidates == ["5No dock levellers", "6No loading doors"]


def test_canonical_span_validation_remains_strict() -> None:
    artifacts = _artifacts()
    target = artifacts.target_by_id["trg_af5d4799dad79a68"]
    intent = artifacts.intent_by_id[target.target_row_id]
    span_text = "There are 5No dock levellers."
    evidence = ApprovedEvidenceRecord(
        target_row_id=target.target_row_id,
        field_name=target.expected_field,
        source_id="src_test",
        source_file="test.pdf",
        page_number=1,
        span_id="span-1",
        hierarchy_node_id="page:src_test:000001",
        text=span_text,
        value_candidates=["5No dock levellers"],
        support_status="supported",
    )
    from segro_evidence_extraction.escalation_text_reextract import build_text_reextract_bundle

    bundle = build_text_reextract_bundle(target, [evidence])
    extraction = ExtractionResult(
        target_row_id=target.target_row_id,
        requirement_id=target.requirement_id,
        raw_model_value=5,
        extracted_value=5,
        normalized_value=5,
        value_bearing_quote="6No dock levellers",
        status="extracted",
        confidence=0.8,
        supporting_span_ids=["span-1"],
        model_provider="test",
        model_name="gpt-4o-mini",
    )

    validation = validate_text_canonical_evidence(extraction, intent, bundle)

    assert validation.status == "invalid"
    assert "Value-bearing phrase is not present in selected canonical spans." in validation.issues


def test_one_model_call_per_eligible_target_is_enforced(
    monkeypatch,
) -> None:
    def supported_evidence(
        artifacts: TextReextractArtifacts,
        targets: list[TargetSpecification],
        *_args,
    ) -> list[ApprovedEvidenceRecord]:
        _ = artifacts
        records: list[ApprovedEvidenceRecord] = []
        for target in targets:
            text = (
                "Floor slab loading capacity is 50 kN/m2."
                if target.expected_field == "floor_construction_capacity"
                else "There are 5No dock levellers."
            )
            records.append(
                ApprovedEvidenceRecord(
                    target_row_id=target.target_row_id,
                    field_name=target.expected_field,
                    source_id="src_test",
                    source_file="test.pdf",
                    page_number=1,
                    span_id=f"span-{target.expected_field}",
                    hierarchy_node_id="page:src_test:000001",
                    text=text,
                    value_candidates=[
                        "50 kN/m2"
                        if "floor" in target.expected_field
                        else "5No dock levellers"
                    ],
                    support_status="supported",
                )
            )
        return records

    monkeypatch.setattr(
        "segro_evidence_extraction.escalation_text_reextract.build_approved_evidence",
        supported_evidence,
    )
    client = RecordingTextClient()

    result = run_escalation_text_reextract_v1(
        output_dir=Path("output/test_escalation_text_reextract_positive"),
        extraction_client=client,
    )

    assert result.preflight_report.expected_calls == 2
    assert sorted(client.calls) == sorted(EXPECTED_TEXT_REEXTRACT_IDS)
    assert result.telemetry.model_call_count == 0
    assert len(client.calls) == 2


def test_prior_artifacts_are_not_overwritten() -> None:
    prior = DEFAULT_ESCALATION_OUTPUT_DIR / "escalation_decisions.json"
    before = prior.stat().st_mtime_ns

    run_escalation_text_reextract_v1(
        output_dir=Path("output/test_escalation_text_reextract_no_overwrite"),
        extraction_client=RecordingTextClient(),
    )

    assert prior.stat().st_mtime_ns == before


def test_no_active_parser_child_remains_after_run() -> None:
    result = run_escalation_text_reextract_v1(
        output_dir=Path("output/test_escalation_text_reextract_no_child"),
        extraction_client=RecordingTextClient(),
    )

    assert result.telemetry.parser_worker_invocations == 0
    assert result.telemetry.active_child_count_after_cleanup == 0
    assert multiprocessing.active_children() == []


def _artifacts() -> TextReextractArtifacts:
    return TextReextractArtifacts(
        escalation_dir=DEFAULT_ESCALATION_OUTPUT_DIR,
        reextract_v2_dir=DEFAULT_REEXTRACT_V2_OUTPUT_DIR,
        batch_v1_dir=DEFAULT_BATCH_V1_OUTPUT_DIR,
        reextract_v1_dir=DEFAULT_REEXTRACT_V1_OUTPUT_DIR,
    )


def _escalation_page_set() -> set[str]:
    import json

    payload = json.loads(
        (DEFAULT_ESCALATION_OUTPUT_DIR / "adjacent_text_search.json").read_text(
            encoding="utf-8"
        )
    )
    ready = EXPECTED_TEXT_REEXTRACT_IDS
    return {
        str(page)
        for item in payload
        if item["target_row_id"] in ready
        for page in item["searched_pages"]
    }
