"""Rule-based page and sheet evidence classification."""

from datetime import UTC, datetime
from hashlib import sha1

from segro_evidence_extraction.models.classification import (
    AlternativeLabel,
    ClassificationMethodType,
    ClassificationResult,
    ClassificationStage,
    ClassificationSubjectType,
)
from segro_evidence_extraction.models.common import ProvenanceRef
from segro_evidence_extraction.parsing.models import TextQualityMetrics

CLASSIFIER_NAME = "segro-rule-page-evidence-classifier"
CLASSIFIER_VERSION = "page-evidence-classifier-v1"


def classify_page_evidence(
    *,
    source_id: str,
    page_or_sheet: str,
    subject_type: ClassificationSubjectType,
    text: str,
    quality: TextQualityMetrics,
    file_type: str,
    table_indicator: bool,
    drawing_indicator: bool,
) -> ClassificationResult:
    lowered = text.lower()
    labels: list[tuple[str, float, str]] = []
    if quality.character_count == 0:
        labels.append(("blank page", 0.9, "No extractable text was present."))
    if quality.scan_likelihood >= 0.8 and file_type in {"pdf", "image"}:
        labels.append(("scanned page", 0.82, "Text quality signals indicate likely scan."))
    if table_indicator:
        labels.append(("table", 0.76, "Tabular delimiters or aligned numeric rows were detected."))
    if any(term in lowered for term in ["certificate", "certification", "test certificate"]):
        labels.append(("certificate", 0.78, "Certificate terminology appears in extracted text."))
    if any(term in lowered for term in ["commissioning", "test report", "inspection report"]):
        labels.append(("test report", 0.7, "Testing or commissioning terminology appears."))
    if any(term in lowered for term in ["schedule", "asset register", "equipment list"]):
        labels.append(("equipment schedule", 0.72, "Schedule-like terminology appears."))
    if drawing_indicator:
        labels.append(("drawing", 0.74, "Drawing/title-block signals were detected."))
    if any(term in lowered for term in ["specification", "technical data", "performance data"]):
        labels.append(("technical specification", 0.68, "Technical specification terms appear."))
    if any(term in lowered for term in ["contents", "index", "table of contents"]):
        labels.append(("cover/index page", 0.66, "Index or contents terminology appears."))
    if not labels and quality.character_count > 120:
        labels.append(("narrative text", 0.62, "The page has text but no stronger evidence label."))
    if not labels:
        labels.append(("unknown", 0.35, "Deterministic signals were insufficient."))

    labels.sort(key=lambda item: item[1], reverse=True)
    primary, confidence, rationale = labels[0]
    secondary = [label for label, score, _ in labels[1:4] if score >= 0.6]
    alternatives = [
        AlternativeLabel(label=label, confidence=score, rationale=reason)
        for label, score, reason in labels[1:4]
    ]
    subject_id = f"{source_id}:{page_or_sheet}"
    digest = sha1(f"{subject_id}:{primary}:{CLASSIFIER_VERSION}".encode()).hexdigest()[:12]
    return ClassificationResult(
        classification_id=f"cls_page_{digest}",
        classifier_name=CLASSIFIER_NAME,
        classifier_version=CLASSIFIER_VERSION,
        classification_stage=ClassificationStage.PAGE_EVIDENCE_TYPE,
        subject_type=subject_type,
        subject_id=subject_id,
        primary_label=primary,
        secondary_labels=secondary,
        confidence=confidence,
        alternative_labels=alternatives,
        evidence_references=[
            ProvenanceRef(source_id=source_id, page_or_sheet=page_or_sheet, notes=rationale)
        ],
        rationale=rationale,
        method_type=ClassificationMethodType.RULE,
        warnings=[],
        timestamp=datetime.now(UTC),
        metadata={"file_type": file_type, "character_count": quality.character_count},
    )
