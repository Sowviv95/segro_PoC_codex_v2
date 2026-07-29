"""Provider-neutral telemetry models."""

from decimal import Decimal

from pydantic import Field

from segro_evidence_extraction.models.base import StrictBaseModel


class StageTiming(StrictBaseModel):
    stage: str
    duration_ms: float = Field(ge=0)


class ModelCallTelemetry(StrictBaseModel):
    provider: str | None = None
    model_name: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    estimated_cost: Decimal = Decimal("0")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class RetrievalTelemetry(StrictBaseModel):
    strategy: str
    target_row_id: str
    candidate_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)


class RunTelemetry(StrictBaseModel):
    stage_timings: list[StageTiming] = Field(default_factory=list)
    document_count: int = Field(default=0, ge=0)
    page_or_sheet_count: int = Field(default=0, ge=0)
    retrieval_operations: list[RetrievalTelemetry] = Field(default_factory=list)
    candidate_count: int = Field(default=0, ge=0)
    model_calls: list[ModelCallTelemetry] = Field(default_factory=list)
    accepted_values: int = Field(default=0, ge=0)
    rejected_values: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(call.input_tokens for call in self.model_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.model_calls)

    @property
    def total_cached_tokens(self) -> int:
        return sum(call.cached_tokens for call in self.model_calls)

    @property
    def estimated_cost(self) -> Decimal:
        return sum((call.estimated_cost for call in self.model_calls), Decimal("0"))

    @property
    def cost_per_accepted_value(self) -> Decimal | None:
        if self.accepted_values == 0:
            return None
        return self.estimated_cost / Decimal(self.accepted_values)
