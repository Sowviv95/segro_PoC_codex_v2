"""Public model exports for foundation contracts."""

from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.candidate import CandidateExtractionResult
from segro_evidence_extraction.models.classification import (
    AlternativeLabel,
    ClassificationMethodType,
    ClassificationResult,
    ClassificationStage,
    ClassificationSubjectType,
    HumanOverride,
)
from segro_evidence_extraction.models.common import (
    BoundingBox,
    Cardinality,
    EvidenceType,
    ExpectedDataType,
    PageSheetRange,
    ProvenanceRef,
)
from segro_evidence_extraction.models.evidence_bundle import (
    EvidenceBundle,
    RetrievalScore,
    TableEvidenceUnit,
    TextEvidenceUnit,
    VisualEvidenceRef,
)
from segro_evidence_extraction.models.evidence_index import EvidenceIndex, EvidenceIndexNode
from segro_evidence_extraction.models.source import (
    ExtractionStatus,
    FileType,
    SourceRegistry,
    SourceRegistryEntry,
)
from segro_evidence_extraction.models.target import (
    DictionaryProvenance,
    TargetSpecification,
    TargetSpecificationSet,
)

__all__ = [
    "BoundingBox",
    "CandidateExtractionResult",
    "Cardinality",
    "AlternativeLabel",
    "ClassificationMethodType",
    "ClassificationResult",
    "ClassificationStage",
    "ClassificationSubjectType",
    "DictionaryProvenance",
    "EvidenceBundle",
    "EvidenceIndex",
    "EvidenceIndexNode",
    "EvidenceType",
    "ExpectedDataType",
    "ExtractionStatus",
    "FileType",
    "HumanOverride",
    "PageSheetRange",
    "ProvenanceRef",
    "RetrievalScore",
    "SourceRegistry",
    "SourceRegistryEntry",
    "StrictBaseModel",
    "TableEvidenceUnit",
    "TargetSpecification",
    "TargetSpecificationSet",
    "TextEvidenceUnit",
    "VisualEvidenceRef",
]
