from __future__ import annotations

import multiprocessing
import shutil
from pathlib import Path

from segro_evidence_extraction.evidence_escalation import (
    MAX_ADJACENT_NEW_PAGES_PER_TARGET,
    MAX_TOTAL_ADJACENT_NEW_PAGES,
    AdjacentPagePlanItem,
    AdjacentTextSearchResult,
    EscalationArtifacts,
    TableEvidenceDiagnostic,
    VisualPageTriage,
    assess_source_availability,
    decide_ocr_candidate,
    decide_vlm_candidate,
    dedupe_adjacent_plans,
    run_evidence_escalation_v1,
    table_like_line,
)
from segro_evidence_extraction.models.common import ExpectedDataType
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.reextract_pass import DEFAULT_REEXTRACT_V2_OUTPUT_DIR
from segro_evidence_extraction.retrieval_diagnostics import DEFAULT_DIAGNOSTIC_OUTPUT_DIR
from segro_evidence_extraction.target_semantics import derive_target_intent


def test_only_frozen_seven_targets_are_loaded() -> None:
    artifacts = EscalationArtifacts(
        reextract_v2_dir=DEFAULT_REEXTRACT_V2_OUTPUT_DIR,
        diagnostic_dir=DEFAULT_DIAGNOSTIC_OUTPUT_DIR,
    )

    assert len(artifacts.frozen_targets) == 7
    assert len({target.target_row_id for target in artifacts.frozen_targets}) == 7


def test_adjacent_page_ranges_are_explicit_bounded_and_deduped() -> None:
    duplicate = _plan("target", "src", 5, 1, 10, [11, 12])
    deduped = dedupe_adjacent_plans([duplicate, duplicate])

    assert len(deduped) == 1
    assert deduped[0].page_start >= 1
    assert deduped[0].page_end >= deduped[0].page_start
    assert len(deduped[0].new_pages) <= MAX_ADJACENT_NEW_PAGES_PER_TARGET


def test_new_page_ceiling_constant_is_enforced_by_planning_contract() -> None:
    assert MAX_TOTAL_ADJACENT_NEW_PAGES == 50


def test_table_like_evidence_is_distinguished_from_generic_text() -> None:
    assert table_like_line("Component | Manufacturer | Model | Date")
    assert table_like_line("Manufacturer: Example Ltd  Model: ABC")
    assert not table_like_line("This is generic narrative text about a roof.")


def test_weak_parsed_text_can_trigger_visual_triage_and_ocr_candidate() -> None:
    target = _target("certificate_component_description")
    visual = VisualPageTriage(
        target_row_id=target.target_row_id,
        source_page="src:1",
        text_character_count=50,
        page_type="weak_or_sparse_text",
        visual_need="scanned_certificate_candidate",
        explanation="weak",
    )
    table = TableEvidenceDiagnostic(
        target_row_id=target.target_row_id,
        status="no_table_evidence",
        explanation="none",
    )

    ocr = decide_ocr_candidate(target, [visual], [table])

    assert ocr.ocr_status == "candidate"


def test_ocr_not_selected_for_spatial_drawing_interpretation() -> None:
    target = _target("dock_count", ExpectedDataType.INTEGER)
    visual = VisualPageTriage(
        target_row_id=target.target_row_id,
        source_page="src:8",
        text_character_count=500,
        page_type="drawing_or_schedule",
        visual_need="VLM_candidate",
        explanation="drawing",
    )
    table = TableEvidenceDiagnostic(
        target_row_id=target.target_row_id,
        status="no_table_evidence",
        explanation="none",
    )

    ocr = decide_ocr_candidate(target, [visual], [table])

    assert ocr.ocr_status == "not_selected"


def test_vlm_selected_only_for_layout_or_spatial_evidence() -> None:
    target = _target("dock_count", ExpectedDataType.INTEGER)
    visual = VisualPageTriage(
        target_row_id=target.target_row_id,
        source_page="src:8",
        text_character_count=500,
        page_type="drawing_or_schedule",
        visual_need="VLM_candidate",
        explanation="drawing",
    )
    table = TableEvidenceDiagnostic(
        target_row_id=target.target_row_id,
        status="no_table_evidence",
        explanation="none",
    )
    intent = derive_target_intent(target)
    artifacts = type("A", (), {"intent_by_id": {target.target_row_id: intent}})()

    vlm = decide_vlm_candidate(target, artifacts, [visual], [table])
    no_vlm = decide_vlm_candidate(
        target,
        artifacts,
        [
            visual.model_copy(
                update={"visual_need": "no_visual_needed", "page_type": "text_page"}
            )
        ],
        [table],
    )

    assert vlm.vlm_status == "candidate"
    assert no_vlm.vlm_status == "not_selected"


def test_source_unavailable_requires_all_avenues_exhausted() -> None:
    target = _target("roof_component_description")
    text = AdjacentTextSearchResult(
        target_row_id=target.target_row_id,
        searched_pages=["src:1"],
        status="no_text_evidence",
        explanation="none",
    )
    table = TableEvidenceDiagnostic(
        target_row_id=target.target_row_id,
        status="no_table_evidence",
        explanation="none",
    )
    visual = VisualPageTriage(
        target_row_id=target.target_row_id,
        page_type="text_page",
        visual_need="no_visual_needed",
        explanation="none",
    )
    vlm = type(
        "V",
        (),
        {"target_row_id": target.target_row_id, "vlm_status": "not_selected"},
    )()

    result = assess_source_availability(target, [text], [table], [visual], [vlm])

    assert result.source_unavailable


def test_real_escalation_run_writes_all_targets_without_overwriting_v2() -> None:
    output_dir = Path("output/test_evidence_escalation")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    v2_telemetry = DEFAULT_REEXTRACT_V2_OUTPUT_DIR / "telemetry.json"
    before = v2_telemetry.stat().st_mtime_ns

    result = run_evidence_escalation_v1(output_dir=output_dir)

    after = v2_telemetry.stat().st_mtime_ns
    assert before == after
    assert len(result.escalation_decisions) == 7
    assert len({item.target_row_id for item in result.escalation_decisions}) == 7
    assert result.telemetry.extraction_model_calls == 0
    assert result.telemetry.parser_worker_invocations >= 0
    assert not multiprocessing.active_children()
    assert (output_dir / "escalation_review.csv").exists()
    shutil.rmtree(output_dir)


def _plan(
    target_id: str,
    source_id: str,
    anchor: int,
    start: int,
    end: int,
    new_pages: list[int],
) -> AdjacentPagePlanItem:
    return AdjacentPagePlanItem(
        target_row_id=target_id,
        source_id=source_id,
        logical_path="manual.pdf",
        anchor_page=anchor,
        page_start=start,
        page_end=end,
        adjacent_pages=list(range(start, end + 1)),
        new_pages=new_pages,
        reason="test",
    )


def _target(
    field_name: str,
    datatype: ExpectedDataType = ExpectedDataType.STRING,
) -> TargetSpecification:
    return TargetSpecification(
        target_row_id=f"target-{field_name}",
        requirement_id="REQ-1",
        sub_domain="Component",
        requirement_text=field_name,
        expected_field=field_name,
        expected_data_type=datatype,
    )
