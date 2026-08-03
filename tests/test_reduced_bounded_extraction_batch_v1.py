from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from segro_evidence_extraction.reduced_bounded_extraction_batch_v1 import (
    DEFAULT_CHECKPOINT_AUDIT_DIR,
    DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR,
    DEFAULT_PAGE_CACHE_ROOT,
    INCLUDED_TARGET_IDS,
    REJECTED_TARGET_REASONS,
    AbstainingReducedExtractionClient,
    build_reduced_batch_package,
    dictionary_valid,
    event_valid,
    execute_reduced_batch,
    load_reduced_inputs,
    run_reduced_bounded_extraction_batch_v1,
    typed_value_valid,
    validate_reduced_batch_contract,
)
from segro_evidence_extraction.vertical_slice import ExtractionResult, parse_extraction_response


def test_real_artifacts_reduce_to_exact_approved_and_rejected_ids() -> None:
    if (
        not DEFAULT_CHECKPOINT_AUDIT_DIR.exists()
        or not DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR.exists()
    ):
        return
    inputs = load_reduced_inputs(
        checkpoint_dir=DEFAULT_CHECKPOINT_AUDIT_DIR,
        constructed_batch_dir=DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR,
        cache_root=DEFAULT_PAGE_CACHE_ROOT,
    )
    package = build_reduced_batch_package(inputs)

    assert [item["target_id"] for item in package["reduced_batch"]] == INCLUDED_TARGET_IDS
    assert [item["target_id"] for item in package["rejected_targets"]] == list(
        REJECTED_TARGET_REASONS
    )
    assert package["batch_summary"]["checkpoint_status_counts"] == {
        "approved_as_is": 7,
        "approved_with_caveat": 3,
    }


def test_rejection_manifest_retains_reasons_and_record_reference() -> None:
    inputs = load_reduced_inputs(
        checkpoint_dir=DEFAULT_CHECKPOINT_AUDIT_DIR,
        constructed_batch_dir=DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR,
        cache_root=DEFAULT_PAGE_CACHE_ROOT,
    )
    package = build_reduced_batch_package(inputs)

    by_id = {item["target_id"]: item for item in package["rejected_targets"]}
    assert by_id["trg_7aefe5e9082729d6"]["rejection_reason"] == "wrong event"
    assert by_id["trg_8e2e33558f11916b"]["rejection_reason"] == "component only"
    assert by_id["trg_ea2bde50a7821e45"]["rejection_reason"] == "wrong system"
    assert by_id["trg_8e2e33558f11916b"]["original_constructed_batch_record_reference"][
        "request_id"
    ] == "extract_trg_8e2e33558f11916b"


def test_failure_when_included_target_has_no_evidence_bundle() -> None:
    inputs = load_reduced_inputs(
        checkpoint_dir=DEFAULT_CHECKPOINT_AUDIT_DIR,
        constructed_batch_dir=DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR,
        cache_root=DEFAULT_PAGE_CACHE_ROOT,
    )
    inputs["payloads"] = [
        item for item in inputs["payloads"] if item["target_id"] != INCLUDED_TARGET_IDS[0]
    ]

    try:
        build_reduced_batch_package(inputs)
    except ValueError as exc:
        assert "missing evidence payload" in str(exc)
    else:
        raise AssertionError("missing evidence payload should fail")


def test_failure_when_evidence_page_absent_from_cache() -> None:
    inputs = load_reduced_inputs(
        checkpoint_dir=DEFAULT_CHECKPOINT_AUDIT_DIR,
        constructed_batch_dir=DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR,
        cache_root=DEFAULT_PAGE_CACHE_ROOT,
    )
    package = build_reduced_batch_package(inputs)
    validation = validate_reduced_batch_contract(package, cached_page_keys=set())

    assert validation["dry_run_validation"]["overall_status"] == "failed"
    assert set(validation["failed_target_ids"]) == set(INCLUDED_TARGET_IDS)


def test_dry_run_schema_validation_passes_for_real_reduced_package() -> None:
    inputs = load_reduced_inputs(
        checkpoint_dir=DEFAULT_CHECKPOINT_AUDIT_DIR,
        constructed_batch_dir=DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR,
        cache_root=DEFAULT_PAGE_CACHE_ROOT,
    )
    package = build_reduced_batch_package(inputs)
    validation = validate_reduced_batch_contract(package, inputs["cached_page_keys"])

    assert validation["dry_run_validation"]["overall_status"] == "passed"
    assert validation["dry_run_validation"]["preflight_rejections"] == []


def test_rejected_target_ids_cannot_enter_execution_batch() -> None:
    inputs = load_reduced_inputs(
        checkpoint_dir=DEFAULT_CHECKPOINT_AUDIT_DIR,
        constructed_batch_dir=DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR,
        cache_root=DEFAULT_PAGE_CACHE_ROOT,
    )
    inputs["approved"].append(inputs["rejected"][0])

    package = build_reduced_batch_package(inputs)

    assert set(REJECTED_TARGET_REASONS).isdisjoint(
        {item["target_id"] for item in package["reduced_batch"]}
    )


def test_evidence_containment_wrong_event_system_and_component_validation() -> None:
    inputs = load_reduced_inputs(
        checkpoint_dir=DEFAULT_CHECKPOINT_AUDIT_DIR,
        constructed_batch_dir=DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR,
        cache_root=DEFAULT_PAGE_CACHE_ROOT,
    )
    package = build_reduced_batch_package(inputs)
    item = package["reduced_batch"][0]
    target = item["constructed_request"]["target"]
    extraction = ExtractionResult(
        target_row_id=item["target_id"],
        requirement_id=target["requirement_id"],
        raw_model_value=43830,
        normalized_value=43830,
        status="extracted",
        confidence=0.9,
        source_id=item["canonical_evidence_payload"]["span"]["source_id"],
        page_number=item["canonical_evidence_payload"]["span"]["page_number"],
        value_bearing_quote="Certificate Number: 43830",
        model_provider="test",
        model_name="mock",
    )
    assignment = item_assignment(item)

    assert typed_value_valid(extraction, assignment)
    assert event_valid(extraction, item)
    assert dictionary_valid(extraction, item_target(item))

    wrong_event = {**item, "target": {**item["target"], "expected_field": "x_installation_date"}}
    assert not event_valid(extraction, wrong_event)


def test_mocked_execution_persists_raw_responses_and_is_deterministic() -> None:
    inputs = load_reduced_inputs(
        checkpoint_dir=DEFAULT_CHECKPOINT_AUDIT_DIR,
        constructed_batch_dir=DEFAULT_EXTRACTION_BATCH_CONSTRUCTION_OUTPUT_DIR,
        cache_root=DEFAULT_PAGE_CACHE_ROOT,
    )
    package = build_reduced_batch_package(inputs)
    validation = validate_reduced_batch_contract(package, inputs["cached_page_keys"])
    first = execute_reduced_batch(
        package,
        dry_run=validation,
        extraction_client=FixtureReducedClient(),
        dry_run_only=False,
    )
    second = execute_reduced_batch(
        package,
        dry_run=validation,
        extraction_client=FixtureReducedClient(),
        dry_run_only=False,
    )

    assert first["model_requests"] == second["model_requests"]
    assert first["extraction_results_validated"] == second["extraction_results_validated"]
    assert len(first["raw_model_responses"]) == 10
    assert all(row["provider_response"] for row in first["raw_model_responses"])

    out = Path("output/test_reduced_bounded_mocked_execution")
    if out.exists():
        shutil.rmtree(out)
    try:
        result = run_reduced_bounded_extraction_batch_v1(
            output_dir=out,
            extraction_client=FixtureReducedClient(),
        )
        assert (out / "model_requests.jsonl").exists()
        assert (out / "raw_model_responses.jsonl").exists()
        assert len(result["final_adjudication"]) == 10
    finally:
        if out.exists():
            shutil.rmtree(out)


def test_abstaining_client_records_no_parser_or_retrieval_invocation() -> None:
    out = Path("output/test_reduced_bounded_abstaining_execution")
    if out.exists():
        shutil.rmtree(out)
    try:
        result = run_reduced_bounded_extraction_batch_v1(
            output_dir=out,
            extraction_client=AbstainingReducedExtractionClient(),
        )
    finally:
        if out.exists():
            shutil.rmtree(out)

    assert result["execution_summary"]["model_calls"] == 10
    assert result["execution_summary"]["retrieval_invocations"] == 0
    assert result["execution_summary"]["parser_invocations"] == 0
    assert result["execution_summary"]["cache_expansion_invocations"] == 0


def item_target(item: dict[str, Any]):
    from segro_evidence_extraction.models.target import TargetSpecification

    return TargetSpecification.model_validate(item["target"])


def item_assignment(item: dict[str, Any]):
    from segro_evidence_extraction.vertical_slice import infer_value_shape_assignment

    return infer_value_shape_assignment(item_target(item))


class FixtureReducedClient:
    provider = "fixture"
    model_name = "fixture-model"

    def extract(self, *, target, bundle, max_output_tokens):
        _ = max_output_tokens
        span = bundle.evidence_spans[0]
        value = fixture_value(target.target_row_id, span.text)
        content = {
            "raw_value": value,
            "extracted_value": value,
            "normalized_value": value,
            "proposed_value_shape": "short_text",
            "unit": None,
            "status": "extracted",
            "confidence": 0.82,
            "selected_supporting_span_ids": [span.span_id],
            "value_bearing_quote": str(value),
            "reasoning_summary": "fixture response",
            "ambiguity_or_caveat": None,
        }
        request = {"model": self.model_name, "messages": [], "target_id": target.target_row_id}
        response = {
            "model": self.model_name,
            "choices": [{"message": {"content": json.dumps(content, sort_keys=True)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = parse_extraction_response(
            target=target,
            bundle=bundle,
            provider=self.provider,
            model_name=self.model_name,
            response_text=json.dumps(content, sort_keys=True),
        )
        result.model_usage.input_tokens = 10
        result.model_usage.output_tokens = 5
        return result, request, response


def fixture_value(target_id: str, evidence: str) -> str | int:
    _ = target_id
    if "5No Dock Levellers" in evidence:
        return 5
    if "Certificate Number: 43830" in evidence:
        return "43830"
    if "CP Electronics" in evidence:
        return "CP Electronics"
    if "Profiled Wall Cladding" in evidence:
        return "Horizontally / vertically laid trapezoidal profiled steel cladding"
    return "supported value"
