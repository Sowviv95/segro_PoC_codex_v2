from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from segro_evidence_extraction.interfaces.capabilities import ExtractionCapability
from segro_evidence_extraction.models import (
    CandidateExtractionResult,
    Cardinality,
    EvidenceBundle,
    EvidenceIndex,
    EvidenceIndexNode,
    EvidenceType,
    ExpectedDataType,
    FileType,
    PageSheetRange,
    ProvenanceRef,
    RetrievalScore,
    SourceRegistry,
    SourceRegistryEntry,
    TargetSpecification,
    TargetSpecificationSet,
    TextEvidenceUnit,
)
from segro_evidence_extraction.models.evidence_index import EvidenceNodeType
from segro_evidence_extraction.telemetry.models import ModelCallTelemetry, RunTelemetry
from segro_evidence_extraction.validation.results import (
    IndependentValidationResult,
    ValidationFinding,
    ValidationStatus,
)


def test_target_specification_validation_and_serialization() -> None:
    spec = _target_spec()
    payload = spec.model_dump(mode="json")

    assert payload["target_row_id"] == "row-1"
    assert payload["expected_data_type"] == "string"

    with pytest.raises(ValidationError, match="accepted_values"):
        TargetSpecification(
            target_row_id="row-2",
            requirement_id="REQ-2",
            sub_domain="Energy",
            requirement_text="Tariff",
            expected_field="tariff",
            accepted_values=["A", "a"],
        )


def test_target_specification_set_rejects_duplicate_rows() -> None:
    with pytest.raises(ValidationError, match="target_row_id"):
        TargetSpecificationSet(specifications=[_target_spec(), _target_spec()])


def test_source_registry_serialization() -> None:
    entry = SourceRegistryEntry(
        source_id="src-1",
        original_path="C:/input/manual.pdf",
        logical_path="manual.pdf",
        logical_role="manual",
        file_type=FileType.PDF,
        file_hash="abc",
        size_bytes=3,
        page_count=10,
        classification="supplier_manual",
    )
    registry = SourceRegistry(
        registry_id="reg-1",
        entries=[entry],
        created_at=datetime(2026, 7, 29, tzinfo=UTC),
    )

    dumped = registry.model_dump(mode="json")
    assert dumped["entries"][0]["file_type"] == "pdf"
    assert dumped["entries"][0]["logical_path"] == "manual.pdf"


def test_evidence_hierarchy_relationships() -> None:
    index = EvidenceIndex(
        index_id="idx-1",
        source_pack_id="pack-1",
        nodes=[
            EvidenceIndexNode(node_id="pack", node_type=EvidenceNodeType.SOURCE_PACK),
            EvidenceIndexNode(
                node_id="doc",
                parent_id="pack",
                source_id="src-1",
                node_type=EvidenceNodeType.DOCUMENT,
                title="Manual",
            ),
            EvidenceIndexNode(
                node_id="page-1",
                parent_id="doc",
                source_id="src-1",
                node_type=EvidenceNodeType.PAGE,
                page_or_sheet_range=PageSheetRange(start=1, end=1),
                table_indicators=["schedule"],
            ),
        ],
    )

    assert index.nodes[2].parent_id == "doc"

    with pytest.raises(ValidationError, match="parent_id"):
        EvidenceIndex(
            index_id="bad",
            source_pack_id="pack-1",
            nodes=[EvidenceIndexNode(node_id="orphan", parent_id="missing", node_type="page")],
        )


def test_evidence_bundle_serialization() -> None:
    bundle = EvidenceBundle(
        bundle_id="bundle-1",
        target_specification=_target_spec(),
        candidate_document_ids=["doc"],
        candidate_section_ids=["section"],
        selected_pages_or_sheets=["p1"],
        text_evidence_units=[
            TextEvidenceUnit(unit_id="txt-1", node_id="page-1", excerpt="AHU manufacturer")
        ],
        retrieval_scores=[RetrievalScore(item_id="page-1", score=0.8, strategy="keyword")],
        retrieval_strategy="keyword",
        selection_reasons=["matched field semantics"],
        provenance=[ProvenanceRef(source_id="src-1", node_id="page-1")],
    )

    assert bundle.model_dump(mode="json")["target_specification"]["requirement_id"] == "REQ-1"


def test_fake_capability_protocol_and_candidate_validation() -> None:
    capability = FakeCapability()
    assert isinstance(capability, ExtractionCapability)

    result = capability.extract(
        EvidenceBundle(
            bundle_id="bundle-1",
            target_specification=_target_spec(),
            retrieval_strategy="test",
        )
    )[0]

    assert result.capability_id == "fake"
    assert result.confidence == 0.1

    with pytest.raises(ValidationError, match="confidence"):
        CandidateExtractionResult(
            target_row_id="row-1",
            requirement_id="REQ-1",
            candidate_value="x",
            capability_id="fake",
            confidence=2,
        )


def test_independent_validation_statuses() -> None:
    finding = ValidationFinding(
        check="evidence_grounding",
        status=ValidationStatus.REVIEW_REQUIRED,
        message="Evidence is ambiguous",
    )
    result = IndependentValidationResult(
        target_row_id="row-1",
        status=ValidationStatus.REVIEW_REQUIRED,
        evidence_grounding_result=finding,
        findings=[finding],
    )

    assert result.status == "review_required"
    assert result.findings[0].check == "evidence_grounding"


def test_telemetry_aggregation() -> None:
    telemetry = RunTelemetry(
        model_calls=[
            ModelCallTelemetry(
                input_tokens=10,
                output_tokens=4,
                cached_tokens=3,
                estimated_cost=Decimal("0.02"),
            ),
            ModelCallTelemetry(input_tokens=5, output_tokens=1, estimated_cost=Decimal("0.01")),
        ],
        accepted_values=2,
        rejected_values=1,
    )

    assert telemetry.total_input_tokens == 15
    assert telemetry.total_output_tokens == 5
    assert telemetry.total_cached_tokens == 3
    assert telemetry.estimated_cost == Decimal("0.03")
    assert telemetry.cost_per_accepted_value == Decimal("0.015")


class FakeCapability:
    capability_id = "fake"
    supported_evidence_types = {EvidenceType.TEXT}
    supported_field_semantics = {"manufacturer"}
    input_requirements = {"text_evidence_units"}
    output_schema = "CandidateExtractionResult"
    confidence_approach = "fixed test confidence"
    rejection_rules = ["no evidence"]
    validation_requirements = ["evidence_grounding"]

    def extract(self, bundle: EvidenceBundle) -> list[CandidateExtractionResult]:
        return [
            CandidateExtractionResult(
                target_row_id=bundle.target_specification.target_row_id,
                requirement_id=bundle.target_specification.requirement_id,
                candidate_value="candidate",
                capability_id=self.capability_id,
                confidence=0.1,
                reasoning_summary="fake capability for protocol test",
            )
        ]


def _target_spec() -> TargetSpecification:
    return TargetSpecification(
        target_row_id="row-1",
        requirement_id="REQ-1",
        sub_domain="Component",
        requirement_text="AHU manufacturer",
        expected_field="manufacturer",
        expected_data_type=ExpectedDataType.STRING,
        cardinality=Cardinality.SINGLE,
        component_type="AHU",
        source_guidance="supplier manuals",
        likely_evidence_types=[EvidenceType.TEXT, EvidenceType.TABLE],
    )
