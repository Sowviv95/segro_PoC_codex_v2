"""Hierarchical evidence index contracts."""

from enum import StrEnum

from pydantic import Field, field_validator

from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.common import BoundingBox, PageSheetRange, ProvenanceRef


class EvidenceNodeType(StrEnum):
    SOURCE_PACK = "source_pack"
    DOCUMENT = "document"
    SECTION = "section"
    SUBSECTION = "subsection"
    PAGE = "page"
    SHEET = "sheet"
    TEXT_BLOCK = "text_block"
    TABLE = "table"
    DRAWING = "drawing"
    IMAGE_REGION = "image_region"


class EvidenceIndexNode(StrictBaseModel):
    node_id: str
    parent_id: str | None = None
    source_id: str | None = None
    node_type: EvidenceNodeType
    title: str | None = None
    page_or_sheet_range: PageSheetRange | None = None
    summary: str | None = None
    likely_sub_domain: str | None = None
    component_types: list[str] = Field(default_factory=list)
    manufacturers_or_models: list[str] = Field(default_factory=list)
    certificate_indicators: list[str] = Field(default_factory=list)
    test_indicators: list[str] = Field(default_factory=list)
    table_indicators: list[str] = Field(default_factory=list)
    drawing_indicators: list[str] = Field(default_factory=list)
    ocr_required: bool = False
    source_role: str | None = None
    searchable_text_ref: str | None = None
    semantic_representation_ref: str | None = None
    bounding_box: BoundingBox | None = None
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRef] = Field(default_factory=list)

    @field_validator("node_id")
    @classmethod
    def node_id_must_not_be_blank(cls, value: str) -> str:
        if not value:
            msg = "node_id must not be blank"
            raise ValueError(msg)
        return value


class EvidenceIndex(StrictBaseModel):
    index_id: str
    source_pack_id: str
    nodes: list[EvidenceIndexNode]
    warnings: list[str] = Field(default_factory=list)

    @field_validator("nodes")
    @classmethod
    def relationships_must_reference_known_nodes(
        cls, nodes: list[EvidenceIndexNode]
    ) -> list[EvidenceIndexNode]:
        ids = [node.node_id for node in nodes]
        if len(ids) != len(set(ids)):
            msg = "node_id values must be unique"
            raise ValueError(msg)
        known = set(ids)
        missing = [
            node.parent_id
            for node in nodes
            if node.parent_id and node.parent_id not in known
        ]
        if missing:
            msg = f"parent_id values must reference known nodes: {missing}"
            raise ValueError(msg)
        return nodes
