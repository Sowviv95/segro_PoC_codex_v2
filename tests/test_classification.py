from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from segro_evidence_extraction.models import (
    AlternativeLabel,
    ClassificationMethodType,
    ClassificationResult,
    ClassificationStage,
    ClassificationSubjectType,
    HumanOverride,
    ProvenanceRef,
)


def test_valid_classification_result_serializes() -> None:
    result = _classification(ClassificationStage.SOURCE_DOCUMENT)

    payload = result.model_dump(mode="json")

    assert payload["classification_stage"] == "source_document"
    assert payload["method_type"] == "hybrid"
    assert payload["alternative_labels"][0]["label"] == "certificate"
    assert payload["evidence_references"][0]["source_id"] == "src-1"


def test_confidence_validation() -> None:
    with pytest.raises(ValidationError, match="confidence"):
        _classification(ClassificationStage.PAGE_EVIDENCE_TYPE, confidence=1.5)


def test_alternative_labels_are_serializable() -> None:
    result = _classification(
        ClassificationStage.PAGE_EVIDENCE_TYPE,
        alternatives=[
            AlternativeLabel(label="table", confidence=0.7, rationale="grid lines detected"),
            AlternativeLabel(label="drawing", confidence=0.2),
        ],
    )

    dumped = result.model_dump(mode="json")

    assert [item["label"] for item in dumped["alternative_labels"]] == ["table", "drawing"]


def test_human_override_sets_effective_label_without_replacing_original() -> None:
    result = _classification(
        ClassificationStage.DOMAIN_COMPONENT,
        override=HumanOverride(
            override_label="lighting",
            override_reason="Reviewer found fixture schedule context",
            reviewer_id="reviewer-1",
            timestamp=datetime(2026, 7, 29, tzinfo=UTC),
        ),
    )

    assert result.primary_label == "electrical"
    assert result.effective_label == "lighting"
    assert result.model_dump(mode="json")["human_override"]["override_label"] == "lighting"


def test_all_four_classification_stages_are_supported() -> None:
    stages = {
        ClassificationStage.SOURCE_DOCUMENT,
        ClassificationStage.PAGE_EVIDENCE_TYPE,
        ClassificationStage.DOMAIN_COMPONENT,
        ClassificationStage.EXTRACTION_ROUTE,
    }

    results = [_classification(stage) for stage in stages]

    assert {result.classification_stage for result in results} == stages


def _classification(
    stage: ClassificationStage,
    *,
    confidence: float = 0.82,
    alternatives: list[AlternativeLabel] | None = None,
    override: HumanOverride | None = None,
) -> ClassificationResult:
    return ClassificationResult(
        classification_id=f"cls-{stage.value}",
        classifier_name="foundation-test-classifier",
        classifier_version="0.1",
        classification_stage=stage,
        subject_type=_subject_type(stage),
        subject_id="subject-1",
        primary_label=_primary_label(stage),
        secondary_labels=["manual", "supplier"],
        confidence=confidence,
        alternative_labels=alternatives
        if alternatives is not None
        else [AlternativeLabel(label="certificate", confidence=0.2)],
        evidence_references=[ProvenanceRef(source_id="src-1", node_id="node-1")],
        rationale="Short classifier rationale suitable for evidence pack inclusion.",
        method_type=ClassificationMethodType.HYBRID,
        model_name="mock-model",
        prompt_version="prompt-v1",
        warnings=["low heading quality"],
        human_override=override,
        timestamp=datetime(2026, 7, 29, tzinfo=UTC),
        metadata={"fixture": True},
    )


def _subject_type(stage: ClassificationStage) -> ClassificationSubjectType:
    if stage == ClassificationStage.SOURCE_DOCUMENT:
        return ClassificationSubjectType.DOCUMENT
    if stage == ClassificationStage.PAGE_EVIDENCE_TYPE:
        return ClassificationSubjectType.PAGE
    if stage == ClassificationStage.DOMAIN_COMPONENT:
        return ClassificationSubjectType.SECTION
    return ClassificationSubjectType.EVIDENCE_BUNDLE


def _primary_label(stage: ClassificationStage) -> str:
    if stage == ClassificationStage.EXTRACTION_ROUTE:
        return "table_extraction"
    if stage == ClassificationStage.PAGE_EVIDENCE_TYPE:
        return "text"
    if stage == ClassificationStage.DOMAIN_COMPONENT:
        return "electrical"
    return "supplier_manual"
