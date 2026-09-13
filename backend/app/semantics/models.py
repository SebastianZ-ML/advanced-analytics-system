"""
Semantic Layer Models and Executable Contracts.
Decouples analytical operations from domain-specific column and table names.
"""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from app.contracts import RelationshipSpec


class MetricDefinition(BaseModel):
    name: str  # Column or alias name
    label: str  # Human-readable title
    formula: str  # SQL or descriptive expression (e.g. "SUM(charge_amount)")
    aggregation: Literal["sum", "avg", "count", "min", "max", "ratio"] = "sum"
    unit: str = "USD / Monetary units"
    currency: Optional[str] = "USD"
    null_policy: Literal["exclude", "zero_if_missing", "fail_if_null"] = "exclude"
    is_primary: bool = False


class EntityGranularity(BaseModel):
    entity_name: str  # e.g. "orders", "admissions", "subscriptions"
    primary_key: str  # e.g. "order_id", "encounter_id", "subscription_id"
    row_semantic: str = "One record per entity occurrence"


class DimensionSpec(BaseModel):
    dimension_name: str
    source_column: str
    table_name: str
    data_type: Literal["string", "integer", "date", "boolean"] = "string"
    is_hierarchical: bool = False
    parent_dimension: Optional[str] = None


class CalendarSpec(BaseModel):
    date_column: str
    frequency: Literal["D", "W", "M", "Y"] = "M"
    timezone: str = "UTC"
    coverage_start: Optional[str] = None
    coverage_end: Optional[str] = None
    closed_periods: List[str] = Field(default_factory=list)
    incomplete_period_cutoff: Optional[str] = None


class BusinessStateMapping(BaseModel):
    status_column: Optional[str] = None
    completed_values: List[str] = Field(default_factory=lambda: ["COMPLETED", "PAID", "DELIVERED", "ACTIVE", "DISCHARGED"])
    cancelled_values: List[str] = Field(default_factory=lambda: ["CANCELLED", "REFUNDED", "FAILED", "VOID"])
    pending_values: List[str] = Field(default_factory=lambda: ["PENDING", "IN_PROGRESS", "PROCESSING"])


class SemanticAmbiguity(BaseModel):
    id: str
    concept: str  # e.g. "primary_date", "primary_metric", "status_filter"
    question: str
    context: str
    candidates: List[str]
    selected_choice: Optional[str] = None
    is_resolved: bool = False
    impact_level: Literal["critical", "moderate", "minor"] = "critical"


class SemanticModel(BaseModel):
    schema_version: str = "1.0"
    project_id: str
    fact_table: str
    granularity: EntityGranularity
    calendar: CalendarSpec
    metrics: List[MetricDefinition] = Field(default_factory=list)
    dimensions: List[DimensionSpec] = Field(default_factory=list)
    business_states: BusinessStateMapping = Field(default_factory=BusinessStateMapping)
    dimension_relationships: List[RelationshipSpec] = Field(default_factory=list)
    ambiguities: List[SemanticAmbiguity] = Field(default_factory=list)

    def get_primary_metric(self) -> MetricDefinition:
        for m in self.metrics:
            if m.is_primary:
                return m
        return self.metrics[0] if self.metrics else MetricDefinition(name="value", label="Value", formula="SUM(value)", is_primary=True)
