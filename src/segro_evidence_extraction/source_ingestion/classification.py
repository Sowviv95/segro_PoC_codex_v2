"""Deterministic source/document classification."""

from datetime import UTC, datetime
from hashlib import sha256

from segro_evidence_extraction.models.classification import (
    AlternativeLabel,
    ClassificationMethodType,
    ClassificationResult,
    ClassificationStage,
    ClassificationSubjectType,
)
from segro_evidence_extraction.models.common import ProvenanceRef
from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry

CLASSIFIER_NAME = "source-document-rule-classifier"
CLASSIFIER_VERSION = "0.1.0"


def classify_source(source: SourceRegistryEntry) -> ClassificationResult:
    label, confidence, secondary, alternatives, rationale = _label_source(source)
    classification_id = _classification_id(source.source_id, label)
    return ClassificationResult(
        classification_id=classification_id,
        classifier_name=CLASSIFIER_NAME,
        classifier_version=CLASSIFIER_VERSION,
        classification_stage=ClassificationStage.SOURCE_DOCUMENT,
        subject_type=ClassificationSubjectType.SOURCE,
        subject_id=source.source_id,
        primary_label=label,
        secondary_labels=secondary,
        confidence=confidence,
        alternative_labels=alternatives,
        evidence_references=[
            ProvenanceRef(
                source_id=source.source_id,
                text_ref=source.logical_path,
                notes="filename, path, extension and lightweight metadata",
            )
        ],
        rationale=rationale,
        method_type=ClassificationMethodType.RULE,
        warnings=[] if label != "unknown" else ["Insufficient deterministic evidence."],
        timestamp=datetime.now(UTC),
        metadata={"file_type": source.file_type, "logical_path": source.logical_path},
    )


def _label_source(
    source: SourceRegistryEntry,
) -> tuple[str, float, list[str], list[AlternativeLabel], str]:
    text = f"{source.logical_path} {source.metadata.get('sheet_names', '')}".casefold()
    file_type = FileType(source.file_type)
    if file_type == FileType.ZIP:
        return "archive", 0.95, [], [], "ZIP container detected."
    if file_type == FileType.IMAGE:
        return "photograph/image", 0.85, [], [], "Image file signature detected."
    if file_type in {FileType.XLSX, FileType.XLS, FileType.CSV}:
        if "maintenance" in text:
            label = "maintenance schedule"
        elif "schedule" in text or "matrix" in text:
            label = "equipment schedule"
        else:
            label = "spreadsheet"
        return label, 0.85, ["spreadsheet"], [], "Spreadsheet/CSV source metadata detected."
    if "certificate" in text or "certificat" in text:
        return (
            "individual certificate",
            0.8,
            ["certificate"],
            [AlternativeLabel(label="certificate pack", confidence=0.35)],
            "Certificate keyword found in source path or metadata.",
        )
    if "commissioning" in text or "pre commissioning" in text:
        return "commissioning report", 0.82, [], [], "Commissioning keyword found."
    if "drawing" in text or "layout" in text or "plan" in text:
        return "drawing set", 0.78, [], [], "Drawing/layout keyword found."
    if "technical specification" in text or "specification" in text:
        return "technical specification", 0.78, [], [], "Specification keyword found."
    if "supplier" in text or "manufacturer" in text:
        return "supplier manual", 0.72, [], [], "Supplier/manufacturer keyword found."
    if "o&m" in text or "operation" in text or "maintenance" in text:
        return "O&M manual", 0.72, [], [], "O&M or maintenance keyword found."
    if "manual" in text:
        return (
            "building manual",
            0.7,
            ["manual"],
            [AlternativeLabel(label="O&M manual", confidence=0.45)],
            "Manual keyword found without relying on fixed part routing.",
        )
    if file_type == FileType.PDF:
        return (
            "unknown",
            0.25,
            ["pdf"],
            [AlternativeLabel(label="mixed document", confidence=0.2)],
            "PDF detected but deterministic label evidence is weak.",
        )
    return "unknown", 0.1, [], [], "No deterministic classification rule matched."


def _classification_id(source_id: str, label: str) -> str:
    return f"cls_{sha256(f'{source_id}|{label}|{CLASSIFIER_VERSION}'.encode()).hexdigest()[:16]}"
