"""Capability-neutral bounded evidence bundle contracts."""

from pydantic import Field

from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.common import BoundingBox, ProvenanceRef
from segro_evidence_extraction.models.target import TargetSpecification


class TextEvidenceUnit(StrictBaseModel):
    unit_id: str
    node_id: str
    text_ref: str | None = None
    excerpt: str | None = None
    page_or_sheet: str | None = None
    relevance_reason: str | None = None


class TableEvidenceUnit(StrictBaseModel):
    unit_id: str
    node_id: str
    table_ref: str
    page_or_sheet: str | None = None
    header_hint: str | None = None
    relevance_reason: str | None = None


class VisualEvidenceRef(StrictBaseModel):
    ref_id: str
    node_id: str
    image_ref: str | None = None
    crop_ref: str | None = None
    region_ref: str | None = None
    bounding_box: BoundingBox | None = None
    page_or_sheet: str | None = None


class RetrievalScore(StrictBaseModel):
    item_id: str
    score: float
    strategy: str
    reason: str | None = None


class EvidenceBundle(StrictBaseModel):
    bundle_id: str
    target_specification: TargetSpecification
    candidate_document_ids: list[str] = Field(default_factory=list)
    candidate_section_ids: list[str] = Field(default_factory=list)
    selected_pages_or_sheets: list[str] = Field(default_factory=list)
    text_evidence_units: list[TextEvidenceUnit] = Field(default_factory=list)
    table_evidence_units: list[TableEvidenceUnit] = Field(default_factory=list)
    visual_evidence_refs: list[VisualEvidenceRef] = Field(default_factory=list)
    component_context: dict[str, str] = Field(default_factory=dict)
    retrieval_scores: list[RetrievalScore] = Field(default_factory=list)
    retrieval_strategy: str
    selection_reasons: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
