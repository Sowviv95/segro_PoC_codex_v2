from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from segro_evidence_extraction.expanded_bounded_extraction_batch_v1 import (
    CHECKPOINT_REJECTED_TARGET_IDS,
    COMPLETED_TARGET_IDS,
    EXPECTED_SELECTED_TARGET_IDS,
    evidence_bundle_record_from_selected,
    run_expanded_bounded_extraction_batch_v1,
    validate_expanded_preflight,
)
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.reduced_bounded_extraction_batch_v1 import (
    build_model_request_payload,
    model_error_result,
)
from segro_evidence_extraction.vertical_slice import (
    EvidenceBundleRecord,
    ExtractionResult,
    parse_extraction_response,
)


class MockExpandedClient:
    provider = "mock"
    model_name = "mock-expanded-model"

    def __init__(self, *, repair_target_id: str | None = None) -> None:
        self.calls: list[str] = []
        self.repair_target_id = repair_target_id

    def extract(
        self,
        *,
        target: TargetSpecification,
        bundle: EvidenceBundleRecord,
        max_output_tokens: int,
    ) -> tuple[ExtractionResult, dict[str, Any], dict[str, Any]]:
        self.calls.append(target.target_row_id)
        request = build_model_request_payload(target, bundle, self.model_name, max_output_tokens)
        payload = response_payload_for_target(target, bundle)
        response = provider_response(self.model_name, payload)
        if target.target_row_id == self.repair_target_id:
            payload = {**payload, "proposed_value_shape": "unordered_or_unordered_list"}
            response = provider_response(self.model_name, payload)
            return (
                model_error_result(
                    target,
                    self.provider,
                    self.model_name,
                    "invalid proposed_value_shape literal",
                ),
                request,
                response,
            )
        parsed = parse_extraction_response(
            target=target,
            bundle=bundle,
            provider=self.provider,
            model_name=self.model_name,
            response_text=json.dumps(payload, sort_keys=True),
        )
        parsed.model_usage.input_tokens = 101
        parsed.model_usage.output_tokens = 17
        return parsed, request, response


def test_expanded_batch_executes_exact_three_targets_and_exports_contracts() -> None:
    output_dirs = cleanup_dirs()
    client = MockExpandedClient()
    try:
        result = run_expanded_bounded_extraction_batch_v1(
            batch_output_dir=output_dirs[0],
            adjudication_output_dir=output_dirs[1],
            handoff_output_dir=output_dirs[2],
            extraction_client=client,
        )

        assert client.calls == EXPECTED_SELECTED_TARGET_IDS
        assert result["batch"]["dry_run_validation"]["overall_status"] == "passed"
        assert result["batch"]["execution_summary"]["model_calls"] == 3
        assert result["batch"]["execution_summary"]["retrieval_invocations"] == 0
        assert result["batch"]["execution_summary"]["parser_invocations"] == 0
        assert result["batch"]["execution_summary"]["ocr_invocations"] == 0
        assert result["batch"]["execution_summary"]["vlm_invocations"] == 0
        assert result["batch"]["execution_summary"]["cache_expansion_invocations"] == 0

        decisions = {
            row["target_id"]: row["final_decision"]
            for row in result["adjudication"]["final_adjudication"]
        }
        assert decisions["trg_b16ea18d75c226c0"] == "accepted"
        assert decisions["trg_2ab51d5b0cc8b48d"] == "abstained"
        assert decisions["trg_199c5560cea7955c"] == "accepted"
        assert result["handoff"]["export_validation"]["contract_validation"]["status"] == "passed"
    finally:
        remove_dirs(output_dirs)


def test_preflight_excludes_completed_rejected_and_missing_cache_candidates() -> None:
    selected = load_selected_batch()
    source_paths = {"src_48403a16aee1d4b3": "source.pdf"}
    selected[0]["target_id"] = next(iter(COMPLETED_TARGET_IDS))
    selected[1]["target_id"] = next(iter(CHECKPOINT_REJECTED_TARGET_IDS))
    selected[2]["cache_path"] = "output/not-present/page.json"

    validation = validate_expanded_preflight(selected, source_paths)

    assert validation["overall_status"] == "failed"
    by_id = {row["target_id"]: row for row in validation["target_validations"]}
    assert "not_previously_completed" in by_id[selected[0]["target_id"]]["failure_reasons"]
    assert "not_checkpoint_rejected" in by_id[selected[1]["target_id"]]["failure_reasons"]
    assert "selected_page_exists_in_cache" in by_id[selected[2]["target_id"]]["failure_reasons"]
    assert validation["retrieval_invocations"] == 0
    assert validation["parser_invocations"] == 0
    assert validation["ocr_invocations"] == 0
    assert validation["vlm_invocations"] == 0
    assert validation["cache_expansion_invocations"] == 0


def test_metadata_propagation_and_raw_response_persistence() -> None:
    output_dirs = cleanup_dirs("metadata")
    try:
        result = run_expanded_bounded_extraction_batch_v1(
            batch_output_dir=output_dirs[0],
            adjudication_output_dir=output_dirs[1],
            handoff_output_dir=output_dirs[2],
            extraction_client=MockExpandedClient(),
        )
        internal = result["handoff"]["internal_handoff"]

        assert all(record["model"] == "mock-expanded-model" for record in internal)
        assert all(record["source_path"] for record in internal)
        assert all(record["extraction_run_id"] for record in internal)
        assert all(record["adjudication_run_id"] for record in internal)
        assert len(read_jsonl(output_dirs[0] / "raw_model_responses.jsonl")) == 3
        assert len(read_jsonl(output_dirs[0] / "model_requests.jsonl")) == 3
    finally:
        remove_dirs(output_dirs)


def test_deterministic_repair_from_persisted_response() -> None:
    output_dirs = cleanup_dirs("repair")
    try:
        result = run_expanded_bounded_extraction_batch_v1(
            batch_output_dir=output_dirs[0],
            adjudication_output_dir=output_dirs[1],
            handoff_output_dir=output_dirs[2],
            extraction_client=MockExpandedClient(
                repair_target_id="trg_199c5560cea7955c",
            ),
        )
        row = next(
            item
            for item in result["adjudication"]["final_adjudication"]
            if item["target_id"] == "trg_199c5560cea7955c"
        )

        assert row["deterministic_repair_applied"] is True
        assert row["model_retry_occurred"] is False
        assert row["final_decision"] == "accepted"
    finally:
        remove_dirs(output_dirs)


def test_json_jsonl_and_csv_outputs_are_consistent() -> None:
    output_dirs = cleanup_dirs("formats")
    try:
        run_expanded_bounded_extraction_batch_v1(
            batch_output_dir=output_dirs[0],
            adjudication_output_dir=output_dirs[1],
            handoff_output_dir=output_dirs[2],
            extraction_client=MockExpandedClient(),
        )

        internal_json = json.loads(
            (output_dirs[2] / "internal_handoff.json").read_text(encoding="utf-8")
        )
        assert len(internal_json) == len(read_jsonl(output_dirs[2] / "internal_handoff.jsonl"))
        with (output_dirs[2] / "internal_handoff.csv").open(encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == len(internal_json)
    finally:
        remove_dirs(output_dirs)


def test_evidence_bundle_uses_selected_artifact_only() -> None:
    selected = load_selected_batch()
    bundle = evidence_bundle_record_from_selected(selected[0])

    assert bundle.target_row_id == selected[0]["target_id"]
    assert bundle.combined_text == selected[0]["canonical_evidence_payload"]["bounded_text"]
    assert bundle.evidence_spans[0].source_id == selected[0]["source_id"]


def response_payload_for_target(
    target: TargetSpecification,
    bundle: EvidenceBundleRecord,
) -> dict[str, Any]:
    quote = quote_for_target(target, bundle)
    value: str | None
    if target.expected_field.endswith("certificate_available"):
        value = "Y"
    elif target.expected_field.endswith("certificate_type"):
        value = "Certificate of Installation & Testing of Horizontal Lifeline System"
    else:
        value = "Horizontal Lifeline System"
    return {
        "raw_value": value,
        "extracted_value": value,
        "normalized_value": value,
        "status": "extracted",
        "confidence": 0.91,
        "selected_supporting_span_ids": [bundle.evidence_spans[0].span_id],
        "value_bearing_quote": quote,
        "reasoning_summary": "Mocked bounded extraction response.",
        "ambiguity_or_caveat": None,
    }


def quote_for_target(target: TargetSpecification, bundle: EvidenceBundleRecord) -> str:
    if target.expected_field.endswith("certificate_available"):
        return "Certificate of Installation & Testing of Horizontal Lifeline System"
    if target.expected_field.endswith("certificate_type"):
        return "Certificate of Installation & Testing of Horizontal Lifeline System"
    return "Horizontal Lifeline System"


def provider_response(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": "mock",
        "model": model,
        "choices": [{"message": {"content": json.dumps(payload, sort_keys=True)}}],
        "usage": {"prompt_tokens": 101, "completion_tokens": 17},
    }


def load_selected_batch() -> list[dict[str, Any]]:
    path = Path(
        "output/enfield_unit1_bounded_extraction_expansion_preparation_v1/selected_batch.json"
    )
    return [dict(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def cleanup_dirs(suffix: str = "run") -> tuple[Path, Path, Path]:
    dirs = (
        Path(f"output/test_expanded_batch_{suffix}"),
        Path(f"output/test_expanded_adjudication_{suffix}"),
        Path(f"output/test_expanded_handoff_{suffix}"),
    )
    remove_dirs(dirs)
    return dirs


def remove_dirs(paths: tuple[Path, Path, Path]) -> None:
    for path in paths:
        if path.exists():
            shutil.rmtree(path)
