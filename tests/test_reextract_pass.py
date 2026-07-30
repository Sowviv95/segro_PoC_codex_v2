from __future__ import annotations

import multiprocessing
import shutil
from pathlib import Path

from segro_evidence_extraction.models.common import ExpectedDataType
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.reextract_pass import (
    EXPECTED_REEXTRACT_TARGET_COUNT,
    AttributeEvidenceValidationResult,
    ParsedResponseDiagnostic,
    ReextractArtifacts,
    StructuredExtractionEnvelope,
    build_preflight,
    build_reextract_validation,
    gate_candidate,
    parse_structured_attribute_response,
    run_reextract_pass_v1,
    run_reextract_pass_v2,
    structured_response_schema,
    validate_attribute_evidence,
    value_shape_supported,
)
from segro_evidence_extraction.retrieval_diagnostics import score_support
from segro_evidence_extraction.target_semantics import TargetIntent, derive_target_intent
from segro_evidence_extraction.vertical_slice import (
    EvidenceBundleRecord,
    EvidenceSpan,
    ExtractionResult,
    ExtractionStatus,
    ModelUsage,
    SchemaCompatibilityResult,
    ShapeValidationResult,
)

BATCH_V1_DIR = Path("output/enfield_unit1_evidence_first_batch_v1")
DIAGNOSTIC_DIR = Path("output/enfield_unit1_evidence_first_batch_v1_retrieval_diagnostic")


class RecordingClient:
    provider = "test"
    model_name = "gpt-4o-mini"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(
        self,
        *,
        target: TargetSpecification,
        intent: TargetIntent,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> ExtractionResult:
        _ = intent, max_output_tokens
        self.calls.append(target.target_row_id)
        span = bundle.evidence_spans[0]
        quote = span.text[:80]
        return ExtractionResult(
            target_row_id=target.target_row_id,
            requirement_id=target.requirement_id,
            raw_model_value=quote,
            extracted_value=quote,
            proposed_value_shape="short_text",
            value_bearing_quote=quote,
            status="extracted",
            confidence=0.8,
            supporting_span_ids=[span.span_id],
            model_provider=self.provider,
            model_name=self.model_name,
            model_usage=ModelUsage(input_tokens=1, output_tokens=1),
        )


class RecordingStructuredClient:
    provider = "test"
    model_name = "gpt-4o-mini"

    def __init__(self, response_text: str | None = None) -> None:
        self.calls: list[str] = []
        self.response_text = response_text or (
            '{"target_id":"unused","status":"insufficient_evidence",'
            '"requested_attribute":"description","raw_value":null,'
            '"normalized_value":null,"unit":null,"supporting_span_ids":[],'
            '"value_bearing_text":null,"confidence":0.2,"ambiguity":null,'
            '"rejection_reason":"no supported value"}'
        )

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
        response = self.response_text.replace("unused", target.target_row_id)
        return parse_structured_attribute_response(
            target=target,
            bundle=bundle,
            provider=self.provider,
            model_name=self.model_name,
            response_text=response,
            finish_reason="stop",
        )


def test_only_diagnostic_approved_targets_are_selected() -> None:
    artifacts = ReextractArtifacts(BATCH_V1_DIR, DIAGNOSTIC_DIR)

    assert len(artifacts.reextract_candidate_ids) == EXPECTED_REEXTRACT_TARGET_COUNT
    assert len(set(artifacts.reextract_candidate_ids)) == EXPECTED_REEXTRACT_TARGET_COUNT


def test_blocked_diagnostic_statuses_cannot_reach_model_gate() -> None:
    artifacts = ReextractArtifacts(BATCH_V1_DIR, DIAGNOSTIC_DIR)
    corrected_pages = set()
    for ranges in artifacts.corrected_source_plan.get("source_ranges", {}).values():
        for item in ranges:
            source_id = str(item["source_id"])
            for page in range(int(item["page_start"]), int(item["page_end"]) + 1):
                corrected_pages.add((source_id, page))

    for blocked in [
        "unsupported_in_available_sources",
        "dictionary_target_ambiguous",
        "component_present_attribute_absent",
    ]:
        target_id = next(
            target_id
            for target_id, item in artifacts.eligibility_by_id.items()
            if item["status"] == blocked
        )
        gate = gate_candidate(
            target=artifacts.target_by_id[target_id],
            artifacts=artifacts,
            corrected_pages=corrected_pages,
        )

        assert not gate.accepted
        assert blocked in gate.reason


def test_attribute_family_evidence_requirements_are_specific() -> None:
    manufacturer = derive_target_intent(
        _target("dock_manufacturer", "Dock manufacturer", ExpectedDataType.STRING)
    )
    install_date = derive_target_intent(
        _target("dock_installation_date", "Dock installation date", ExpectedDataType.DATE)
    )
    count = derive_target_intent(
        _target("dock_count", "Dock count", ExpectedDataType.INTEGER)
    )
    capacity = derive_target_intent(
        _target("floor_construction_capacity", "Floor capacity", ExpectedDataType.DECIMAL)
    )

    assert score_support(manufacturer, "Dock leveller description only.").attribute_score == 0
    assert score_support(
        manufacturer, "Dock leveller manufacturer is Example Doors Ltd."
    ).attribute_score > 0
    assert score_support(install_date, "Document issue date 23/04/2020.").component_score == 0
    assert score_support(count, "There are 5No dock levellers.").cooccurrence_score > 0
    unrelated_count = score_support(count, "Dock levellers are inspected. PV panels 24No.")
    capacity_score = score_support(capacity, "Floor slab loading capacity is 50 kN/m2.")
    capacity_text = "Floor slab loading capacity is 50 kN/m2."
    assert not value_shape_supported(
        count, unrelated_count, "Dock levellers are inspected. PV panels 24No."
    )
    assert value_shape_supported(capacity, capacity_score, capacity_text)


def test_unknown_span_ids_and_missing_value_phrase_fail_evidence_validation() -> None:
    target = _target("dock_count", "Dock count", ExpectedDataType.INTEGER)
    intent = derive_target_intent(target)
    bundle = _bundle(target.target_row_id, "The warehouse has 5No dock levellers.")
    unknown = _extraction(target, ["missing"], "5No dock levellers")
    missing_phrase = _extraction(target, ["span-1"], "six dock levellers")

    unknown_result = validate_attribute_evidence(unknown, intent, bundle)
    missing_phrase_result = validate_attribute_evidence(missing_phrase, intent, bundle)

    assert unknown_result.status == "invalid"
    assert "Unknown selected span ID." in unknown_result.issues
    assert missing_phrase_result.status == "invalid"
    assert "Value-bearing phrase is not present in selected canonical spans." in (
        missing_phrase_result.issues
    )


def test_multiple_candidates_are_review_required_not_collapsed() -> None:
    target_id = "target-dock_count"
    validation = build_reextract_validation(
        [_extraction(_target("dock_count", "Dock count"), ["span-1"], "5", "multiple_candidates")],
        [
            AttributeEvidenceValidationResult(
                target_row_id=target_id,
                status="valid",
                selected_span_ids=["span-1"],
            )
        ],
        [
            ShapeValidationResult(
                target_row_id=target_id,
                value_shape_family="integer_count",
                status="review_required",
            )
        ],
        [
            SchemaCompatibilityResult(
                target_row_id=target_id,
                dictionary_datatype="integer",
                compatibility="not_evaluated",
                evidence_validity="valid",
                value_format_validity="not_evaluated",
                overall_review_status="review_required",
            )
        ],
    )

    assert validation[0].status == "review_required"


def test_preflight_cost_ceiling_is_enforced() -> None:
    artifacts = ReextractArtifacts(BATCH_V1_DIR, DIAGNOSTIC_DIR)
    targets = [
        artifacts.target_by_id[target_id] for target_id in artifacts.reextract_candidate_ids
    ]

    preflight = build_preflight(
        artifacts=artifacts,
        targets=targets,
        model_name="gpt-4o-mini",
        cost_ceiling_usd=0.0,
    )

    assert not preflight.preflight_report.safety_gate_passed
    assert any("exceeds ceiling" in item for item in preflight.preflight_report.safety_gate_errors)


def test_fake_run_calls_model_at_most_once_per_eligible_target_and_preserves_batch_v1() -> None:
    output_dir = Path("output/test_reextract_pass")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    before = (BATCH_V1_DIR / "telemetry.json").stat().st_mtime_ns
    client = RecordingClient()

    result = run_reextract_pass_v1(
        batch_v1_dir=BATCH_V1_DIR,
        diagnostic_dir=DIAGNOSTIC_DIR,
        output_dir=output_dir,
        extraction_client=client,
    )

    after = (BATCH_V1_DIR / "telemetry.json").stat().st_mtime_ns
    assert before == after
    assert len(client.calls) == len(set(client.calls))
    assert len(client.calls) == result.preflight_report.preflight_eligible_count
    assert result.telemetry.parser_worker_invocations == 0
    assert not multiprocessing.active_children()
    shutil.rmtree(output_dir)


def test_no_forbidden_runtime_capabilities_are_invoked_by_reextract_module() -> None:
    text = Path("src/segro_evidence_extraction/reextract_pass.py").read_text(encoding="utf-8")

    assert "vector database" not in text.lower()
    assert "embedding" not in text.lower()
    assert "ocr" not in text.lower()
    assert "vlm" not in text.lower()


def test_structured_parser_accepts_strict_json_and_markdown_fence() -> None:
    target = _target("dock_count", "Dock count", ExpectedDataType.INTEGER)
    bundle = _bundle(target.target_row_id, "There are 5No dock levellers.")
    payload = (
        '{"target_id":"target-dock_count","status":"extracted",'
        '"requested_attribute":"count","raw_value":5,"normalized_value":5,'
        '"unit":null,"supporting_span_ids":["span-1"],'
        '"value_bearing_text":"5No dock levellers","confidence":0.8,'
        '"ambiguity":null,"rejection_reason":null}'
    )

    extraction, diagnostic, parsed = parse_structured_attribute_response(
        target=target,
        bundle=bundle,
        provider="test",
        model_name="model",
        response_text=payload,
    )
    fenced_extraction, fenced_diagnostic, _ = parse_structured_attribute_response(
        target=target,
        bundle=bundle,
        provider="test",
        model_name="model",
        response_text=f"```json\n{payload}\n```",
    )

    assert extraction.status == "extracted"
    assert parsed is not None
    assert diagnostic.schema_valid
    assert fenced_extraction.status == "extracted"
    assert fenced_diagnostic.safe_fence_removed


def test_structured_parser_rejects_prose_unknown_status_and_bad_span() -> None:
    target = _target("dock_count", "Dock count", ExpectedDataType.INTEGER)
    bundle = _bundle(target.target_row_id, "There are 5No dock levellers.")
    prose, prose_diag, _ = parse_structured_attribute_response(
        target=target,
        bundle=bundle,
        provider="test",
        model_name="model",
        response_text='Here is JSON {"status":"extracted"}',
    )
    bad_status_payload = (
        '{"target_id":"target-dock_count","status":"done",'
        '"requested_attribute":"count","raw_value":5,"normalized_value":5,'
        '"unit":null,"supporting_span_ids":["span-1"],'
        '"value_bearing_text":"5No dock levellers","confidence":0.8,'
        '"ambiguity":null,"rejection_reason":null}'
    )
    bad_status, status_diag, _ = parse_structured_attribute_response(
        target=target,
        bundle=bundle,
        provider="test",
        model_name="model",
        response_text=bad_status_payload,
    )
    bad_span_payload = bad_status_payload.replace('"done"', '"extracted"').replace(
        '"span-1"', '"missing"'
    )
    bad_span, span_diag, _ = parse_structured_attribute_response(
        target=target,
        bundle=bundle,
        provider="test",
        model_name="model",
        response_text=bad_span_payload,
    )

    assert prose.status == "invalid_format"
    assert prose_diag.invalid_format_reason == "response_is_not_single_json_object"
    assert bad_status.status == "invalid_format"
    assert status_diag.enum_mismatches
    assert bad_span.status == "invalid_format"
    assert span_diag.invalid_span_ids == ["missing"]


def test_structured_parser_allows_null_abstention_and_non_measurement_unit_absence() -> None:
    target = _target("dock_component_description", "Dock description")
    bundle = _bundle(target.target_row_id, "Dock levellers are present.")
    response = (
        '{"target_id":"target-dock_component_description",'
        '"status":"insufficient_evidence","requested_attribute":"description",'
        '"raw_value":null,"normalized_value":null,"unit":null,'
        '"supporting_span_ids":[],"value_bearing_text":null,"confidence":0.3,'
        '"ambiguity":null,"rejection_reason":"component only"}'
    )

    extraction, diagnostic, parsed = parse_structured_attribute_response(
        target=target,
        bundle=bundle,
        provider="test",
        model_name="model",
        response_text=response,
    )

    assert extraction.status == "insufficient_evidence"
    assert diagnostic.schema_valid
    assert parsed is not None
    assert parsed.unit is None


def test_structured_schema_carries_measurements_counts_descriptions_and_lists() -> None:
    schema = structured_response_schema()
    raw_value_schema = schema["properties"]["raw_value"]

    assert raw_value_schema["anyOf"][0]["type"] == "string"
    assert raw_value_schema["anyOf"][1]["type"] == "integer"
    assert raw_value_schema["anyOf"][2]["type"] == "number"
    assert raw_value_schema["anyOf"][3]["type"] == "array"


def test_extracted_response_requires_value_and_value_bearing_text() -> None:
    target = _target("dock_count", "Dock count", ExpectedDataType.INTEGER)
    bundle = _bundle(target.target_row_id, "There are 5No dock levellers.")
    response = (
        '{"target_id":"target-dock_count","status":"extracted",'
        '"requested_attribute":"count","raw_value":null,"normalized_value":null,'
        '"unit":null,"supporting_span_ids":["span-1"],'
        '"value_bearing_text":null,"confidence":0.8,'
        '"ambiguity":null,"rejection_reason":null}'
    )

    extraction, diagnostic, _ = parse_structured_attribute_response(
        target=target,
        bundle=bundle,
        provider="test",
        model_name="model",
        response_text=response,
    )

    assert extraction.status == "invalid_format"
    assert "raw_value" in diagnostic.missing_required_fields
    assert "value_bearing_text" in diagnostic.missing_required_fields


def test_reextract_v2_uses_only_frozen_seven_and_persists_diagnostics() -> None:
    output_dir = Path("output/test_reextract_pass_v2")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    v1_telemetry = (
        Path("output/enfield_unit1_evidence_first_batch_v1_reextract_v1")
        / "telemetry.json"
    )
    before = v1_telemetry.stat().st_mtime_ns
    client = RecordingStructuredClient()

    result = run_reextract_pass_v2(
        batch_v1_dir=BATCH_V1_DIR,
        diagnostic_dir=DIAGNOSTIC_DIR,
        v1_reextract_dir=Path("output/enfield_unit1_evidence_first_batch_v1_reextract_v1"),
        output_dir=output_dir,
        extraction_client=client,
    )

    after = v1_telemetry.stat().st_mtime_ns
    assert before == after
    assert len(result.frozen_targets) == 7
    assert len(client.calls) == 7
    assert len(client.calls) == len(set(client.calls))
    assert result.telemetry.parser_worker_invocations == 0
    assert result.telemetry.llm_calls_skipped == 11
    assert (output_dir / "raw_response_diagnostics.json").exists()
    assert (output_dir / "v1_invalid_format_diagnostic.json").exists()
    assert not multiprocessing.active_children()
    shutil.rmtree(output_dir)


def _target(
    field_name: str,
    definition: str,
    datatype: ExpectedDataType = ExpectedDataType.STRING,
) -> TargetSpecification:
    return TargetSpecification(
        target_row_id=f"target-{field_name}",
        requirement_id="REQ-1",
        sub_domain="Component",
        requirement_text=definition,
        expected_field=field_name,
        expected_data_type=datatype,
    )


def _bundle(target_id: str, text: str) -> EvidenceBundleRecord:
    span = EvidenceSpan(
        span_id="span-1",
        source_id="src",
        source_file="manual.pdf",
        page_number=1,
        hierarchy_node_id="node-1",
        text=text,
        start_char=0,
        end_char=len(text),
        retrieval_rank=1,
        score=10,
    )
    return EvidenceBundleRecord(
        target_row_id=target_id,
        evidence_items=[],
        evidence_spans=[span],
        combined_text=text,
        character_count=len(text),
        token_estimate=10,
    )


def _extraction(
    target: TargetSpecification,
    span_ids: list[str],
    quote: str,
    status: ExtractionStatus = "extracted",
) -> ExtractionResult:
    return ExtractionResult(
        target_row_id=target.target_row_id,
        requirement_id=target.requirement_id,
        raw_model_value=quote,
        extracted_value=quote,
        value_bearing_quote=quote,
        status=status,
        confidence=0.8,
        supporting_span_ids=span_ids,
        model_provider="test",
        model_name="test-model",
    )
