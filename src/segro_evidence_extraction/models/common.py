"""Common reusable evidence model primitives."""

from enum import StrEnum

from pydantic import Field, field_validator

from segro_evidence_extraction.models.base import StrictBaseModel


class Cardinality(StrEnum):
    SINGLE = "single"
    MULTIPLE = "multiple"
    OPTIONAL = "optional"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


class ExpectedDataType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATE = "date"
    ENUM = "enum"
    LIST = "list"
    OBJECT = "object"
    UNKNOWN = "unknown"


class EvidenceType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    DRAWING = "drawing"
    IMAGE = "image"
    OCR_TEXT = "ocr_text"
    SPREADSHEET = "spreadsheet"
    CERTIFICATE = "certificate"
    HUMAN_REVIEW = "human_review"
    UNKNOWN = "unknown"


class PageSheetRange(StrictBaseModel):
    start: int = Field(ge=1)
    end: int = Field(ge=1)
    label: str | None = None

    @field_validator("end")
    @classmethod
    def end_must_not_precede_start(cls, value: int, info: object) -> int:
        data = getattr(info, "data", {})
        start = data.get("start")
        if isinstance(start, int) and value < start:
            msg = "end must be greater than or equal to start"
            raise ValueError(msg)
        return value


class BoundingBox(StrictBaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    coordinate_system: str = "relative"


class ProvenanceRef(StrictBaseModel):
    source_id: str | None = None
    node_id: str | None = None
    page_or_sheet: str | None = None
    text_ref: str | None = None
    table_ref: str | None = None
    region_ref: str | None = None
    notes: str | None = None


class WarningItem(StrictBaseModel):
    code: str
    message: str
    severity: str = "warning"
