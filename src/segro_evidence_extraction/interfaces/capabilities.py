"""Protocols for future extraction capabilities."""

from typing import Protocol, runtime_checkable

from segro_evidence_extraction.models.candidate import CandidateExtractionResult
from segro_evidence_extraction.models.common import EvidenceType
from segro_evidence_extraction.models.evidence_bundle import EvidenceBundle


@runtime_checkable
class ExtractionCapability(Protocol):
    capability_id: str
    supported_evidence_types: set[EvidenceType]
    supported_field_semantics: set[str]
    input_requirements: set[str]
    output_schema: str
    confidence_approach: str
    rejection_rules: list[str]
    validation_requirements: list[str]

    def extract(self, bundle: EvidenceBundle) -> list[CandidateExtractionResult]:
        """Return candidate values from a bounded evidence bundle."""
        ...
