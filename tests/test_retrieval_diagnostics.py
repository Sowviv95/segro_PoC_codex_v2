from __future__ import annotations

import multiprocessing

from segro_evidence_extraction.models.common import ExpectedDataType
from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.retrieval_diagnostics import (
    CurrentCorpusDiagnosticItem,
    ExtractedResultAuditItem,
    build_page_probe_plan,
    classify_eligibility,
    classify_support,
    retrieve_attribute_aware,
    score_support,
)
from segro_evidence_extraction.target_semantics import TargetIntent, derive_target_intent
from segro_evidence_extraction.vertical_slice import (
    HierarchyNode,
    LightweightHierarchy,
    RetrievalResult,
    SourceRange,
)


def test_component_and_attribute_are_derived_separately() -> None:
    intent = derive_target_intent(
        _target("floor_construction_manufacturer", "Floor manufacturer")
    )

    assert "floor" in intent.primary_component
    assert intent.requested_attribute == "manufacturer"
    assert "manufacturer" in intent.attribute_terms


def test_manufacturer_and_model_fields_require_associated_attribute() -> None:
    manufacturer = derive_target_intent(
        _target("floor_construction_manufacturer", "Floor manufacturer")
    )
    model = derive_target_intent(_target("floor_construction_model_name", "Floor model"))

    component_only = score_support(manufacturer, "Concrete floor slab construction.")
    manufacturer_score = score_support(
        manufacturer,
        "Concrete floor slab manufacturer is Example Concrete Ltd.",
    )
    model_score = score_support(model, "Floor slab model reference is ABC-123.")

    assert classify_support(component_only) == "component_only"
    assert classify_support(manufacturer_score) == "supports_requested_attribute"
    assert classify_support(model_score) == "supports_requested_attribute"


def test_count_fields_require_number_associated_with_component() -> None:
    intent = derive_target_intent(
        _target("dock_leveller_count", "Dock Levellers - Count", ExpectedDataType.INTEGER)
    )

    unrelated_number = score_support(intent, "PV panel access hatch 5 is maintained.")
    associated = score_support(intent, "The warehouse has 5No dock levellers.")

    assert classify_support(unrelated_number) != "supports_requested_attribute"
    assert classify_support(associated) == "supports_requested_attribute"


def test_attribute_aware_retrieval_ranks_complete_evidence_first() -> None:
    target = _target("floor_construction_manufacturer", "Floor manufacturer")
    intent = derive_target_intent(target)
    hierarchy = LightweightHierarchy(
        hierarchy_id="h",
        nodes=[
            _node("n1", 1, "Floor", "Concrete floor slab construction."),
            _node("n2", 2, "Floor manufacturer", "Floor slab manufacturer is Example Ltd."),
        ],
    )
    registry = {"src": _source()}

    result = retrieve_attribute_aware(
        target=target,
        intent=intent,
        hierarchy=hierarchy,
        source_registry=registry,
    )

    assert result.results[0].page_start == 2


def test_current_corpus_can_find_relevant_evidence_below_previous_top_three() -> None:
    target = _target("floor_construction_manufacturer", "Floor manufacturer")
    intent = derive_target_intent(target)
    hierarchy = LightweightHierarchy(
        hierarchy_id="h",
        nodes=[
            _node("n1", 1, "Unrelated", "General maintenance."),
            _node("n2", 2, "Floor", "Concrete floor slab."),
            _node("n3", 3, "Floor", "Floor cleaning schedule."),
            _node("n4", 4, "Manufacturer", "Floor slab manufacturer is Example Ltd."),
        ],
    )
    result = retrieve_attribute_aware(
        target=target,
        intent=intent,
        hierarchy=hierarchy,
        source_registry={"src": _source()},
    )

    assert any(item.page_start == 4 for item in result.results[:3])


def test_no_relevant_current_corpus_triggers_bounded_deduped_probe_plan() -> None:
    target_id = "target-floor"
    intent = TargetIntent(
        target_row_id=target_id,
        field_name="floor_construction_manufacturer",
        field_definition="Floor manufacturer",
        domain="Component",
        sub_domain="Component",
        value_shape_family="short_text",
        primary_component="floor",
        component_terms=["floor"],
        requested_attribute="manufacturer",
        attribute_terms=["manufacturer"],
        likely_source_types=["building fabric/general"],
        likely_evidence_forms=["schedule"],
    )
    diagnostic = CurrentCorpusDiagnosticItem(
        target_row_id=target_id,
        field_name="floor_construction_manufacturer",
        prior_review_status="insufficient_evidence",
        prior_retrieval_status="weak_evidence",
        best_component_nodes=[],
        best_attribute_nodes=[],
        best_component_plus_attribute_nodes=[],
        current_corpus_evidence_found=False,
        failure_classification="no_relevant_evidence_in_current_corpus",
        explanation="fixture",
    )
    registry = {"src": _source("Building Manual - Part 2 Building Fabric.pdf")}

    plan = build_page_probe_plan(
        [diagnostic, diagnostic.model_copy()],
        {target_id: intent},
        registry,
        [
            SourceRange(
                source_id="src",
                logical_path="manual.pdf",
                page_start=1,
                page_end=10,
                reason="old",
            )
        ],
    )

    assert plan.ranges
    assert all(item.page_end - item.page_start + 1 <= 10 for item in plan.ranges)
    assert all(len([r for r in plan.ranges if target_id in r.target_row_ids]) <= 3 for _ in [0])
    assert plan.unique_new_pages_requested <= 150


def test_extraction_eligibility_requires_requested_attribute_support() -> None:
    target = _target("floor_construction_manufacturer", "Floor manufacturer")
    intent = derive_target_intent(target)
    retrieval = RetrievalResult(
        target_row_id=target.target_row_id,
        query="floor manufacturer",
        retrieval_status="weak_evidence",
        results=[],
        retrieval_time_ms=1,
        top_score=0,
    )

    eligibility = classify_eligibility(target=target, intent=intent, retrieval=retrieval)

    assert eligibility.status in {"unsupported_in_available_sources", "dictionary_target_ambiguous"}


def test_existing_success_can_be_flagged_component_only_overgeneralization() -> None:
    target = _target("floor_construction_manufacturer", "Floor manufacturer")
    intent = derive_target_intent(target)
    score = score_support(intent, "Concrete floor slab construction.")

    audit = ExtractedResultAuditItem(
        target_row_id=target.target_row_id,
        field_name=target.expected_field,
        audit_classification="component_only_overgeneralization",
        support_classification=classify_support(score),
        explanation="fixture",
    )

    assert audit.support_classification == "component_only"


def test_passing_control_evidence_remains_supported_and_no_children() -> None:
    intent = derive_target_intent(
        _target("dock_leveller_count", "Dock count", ExpectedDataType.INTEGER)
    )
    score = score_support(intent, "Dock leveller count: 5No dock levellers.")

    assert classify_support(score) == "supports_requested_attribute"
    assert not multiprocessing.active_children()


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


def _source(logical_path: str = "manual.pdf") -> SourceRegistryEntry:
    return SourceRegistryEntry(
        source_id="src",
        original_path=logical_path,
        logical_path=logical_path,
        file_type=FileType.PDF,
        extension=".pdf",
        size_bytes=100,
        file_hash="hash",
        page_count=100,
    )


def _node(node_id: str, page: int, title: str, text: str) -> HierarchyNode:
    return HierarchyNode(
        node_id=node_id,
        source_id="src",
        node_type="text_block",
        title=title,
        parent_node_id=None,
        page_start=page,
        page_end=page,
        text_summary=text,
    )
