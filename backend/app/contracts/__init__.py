"""
Typed Pydantic contracts for the Adaptive Multi-Agent Analytics System.
Every agent produces and consumes typed specifications according to these contracts.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# A. Objective Contracts
# ---------------------------------------------------------------------------
class AmbiguityItem(BaseModel):
    id: str
    question: str
    context: str
    proposed_default: str
    user_decision: Optional[str] = None
    impact_level: Literal["low", "medium", "high"] = "medium"
    status: Literal["pending", "resolved"] = "pending"


class ObjectiveSpec(BaseModel):
    schema_version: str = "1.0"
    project_id: str
    original_question: str
    operational_objective: str
    decision_to_inform: str
    primary_metric: str
    auxiliary_metrics: List[str] = Field(default_factory=list)
    time_period: str
    comparison_period: Optional[str] = None
    relevant_dimensions: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    pending_ambiguities: List[AmbiguityItem] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    excluded_scope: List[str] = Field(default_factory=list)
    status: Literal["draft", "confirmed", "requires_revision"] = "draft"
    is_demo_mode: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# B. Data Catalog, Profiling & Quality Contracts
# ---------------------------------------------------------------------------
class ColumnProfile(BaseModel):
    name: str
    inferred_type: Literal["string", "integer", "float", "date", "datetime", "boolean", "unknown"]
    confirmed_type: Optional[str] = None
    total_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    has_leading_zeros: bool = False
    has_mixed_currency: bool = False
    sample_values: List[Any] = Field(default_factory=list)
    min_value: Optional[Union[float, int, str]] = None
    max_value: Optional[Union[float, int, str]] = None
    mean_value: Optional[float] = None
    std_dev: Optional[float] = None


class TableProfile(BaseModel):
    table_id: str
    source_filename: str
    sheet_name: Optional[str] = None
    display_name: str
    row_count: int
    column_count: int
    columns: Dict[str, ColumnProfile]
    file_hash_sha256: str
    row_semantic_meaning: str
    date_columns: List[str] = Field(default_factory=list)
    id_columns: List[str] = Field(default_factory=list)
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)


class DataCatalog(BaseModel):
    schema_version: str = "1.0"
    project_id: str
    tables: Dict[str, TableProfile] = Field(default_factory=dict)
    total_tables: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QualityIssue(BaseModel):
    id: str
    issue_type: Literal[
        "invalid_date",
        "leading_zeros_preserved",
        "numeric_as_text",
        "duplicate_keys",
        "many_to_many",
        "orphan_records",
        "incomplete_period",
        "returns_cancellations",
        "granularity_mismatch",
        "missing_values",
        "empty_file",
        "malformed_file"
    ]
    severity: Literal["info", "warning", "blocking"]
    table_id: str
    column: Optional[str] = None
    affected_rows: int
    total_rows: int
    affected_percentage: float
    evidence_samples: List[Any] = Field(default_factory=list)
    description: str
    recommended_action: str
    user_resolution: Optional[str] = None


class DataQualityReport(BaseModel):
    schema_version: str = "1.0"
    project_id: str
    issues: List[QualityIssue] = Field(default_factory=list)
    has_blocking_issues: bool = False
    coverage_start: Optional[str] = None
    coverage_end: Optional[str] = None
    is_last_period_incomplete: bool = False
    incomplete_period_details: Optional[str] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RelationshipSpec(BaseModel):
    id: str
    left_table: str
    left_key: str
    right_table: str
    right_key: str
    cardinality: Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]
    match_rate_left: float
    orphan_count_left: int
    match_rate_right: float
    orphan_count_right: int
    risk_level: Literal["low", "medium", "high"]
    is_confirmed: bool = False
    rejection_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# C. Methodologist & Analysis Plan Contracts
# ---------------------------------------------------------------------------
class OperationSpec(BaseModel):
    step_id: str
    operation_name: str
    category: Literal[
        "metric_summary",
        "time_aggregation",
        "period_comparison",
        "dimension_breakdown",
        "customer_dynamics",
        "concentration_pareto",
        "baseline_forecast"
    ]
    description: str
    required_inputs: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)
    pre_execution_validations: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    is_implemented: bool = True
    method_documentation_ref: Optional[str] = None


class ExternalResearchCitation(BaseModel):
    url: str
    title: str
    consulted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    supported_claim: str
    is_primary_source: bool = True


class AnalysisPlan(BaseModel):
    schema_version: str = "1.0"
    project_id: str
    run_id: str
    title: str
    rationale: str
    operations: List[OperationSpec] = Field(default_factory=list)
    excluded_methods: List[str] = Field(default_factory=list)
    citations: List[ExternalResearchCitation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# D. Data Preparation & Transformation Contracts
# ---------------------------------------------------------------------------
class JoinCheckResult(BaseModel):
    left_table: str
    right_table: str
    join_type: str
    join_keys: List[str]
    left_rows: int
    right_rows: int
    result_rows: int
    multiplication_factor: float
    orphans_left: int
    orphans_right: int
    metric_reconciled: bool
    is_safe: bool
    warning: Optional[str] = None


class TransformationRecord(BaseModel):
    transformation_id: str
    step_index: int
    input_table: str
    output_table: str
    operation: str
    columns_affected: List[str]
    rationale: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    rows_before: int
    rows_after: int
    rows_excluded: int
    exclusion_reason: Optional[str] = None
    exclusion_samples: List[Dict[str, Any]] = Field(default_factory=list)
    join_check: Optional[JoinCheckResult] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# E. Analytical Results & Execution Contracts
# ---------------------------------------------------------------------------
class DimensionContribution(BaseModel):
    dimension_value: str
    baseline_value: float
    current_value: float
    absolute_change: float
    percentage_change: float
    contribution_to_total_change: float  # Percentage of total decline explained (sums to 100%)


class AnalysisResult(BaseModel):
    result_id: str
    step_id: str
    operation_name: str
    method: str
    parameters: Dict[str, Any]
    data_source_version: str
    unit_of_measure: str
    time_period_covered: str
    calculated_values: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)
    version: int = 1
    validation_status: Literal["pending", "approved", "approved_with_warnings", "rejected"] = "pending"
    rejection_details: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# F. Analytical Validation Contracts
# ---------------------------------------------------------------------------
class ValidationCheck(BaseModel):
    check_name: str
    description: str
    passed: bool
    severity: Literal["info", "warning", "blocking"]
    evidence: str
    remedy_action: Optional[str] = None


class ValidationReport(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    overall_status: Literal["approved", "approved_with_warnings", "rejected"]
    checks_executed: List[ValidationCheck] = Field(default_factory=list)
    failed_results: List[str] = Field(default_factory=list)
    repair_attempt: int = 0
    repair_instruction: Optional[str] = None
    can_retry: bool = False
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# G. Insights & Interpretation Contracts
# ---------------------------------------------------------------------------
class InsightFinding(BaseModel):
    id: str
    claim: str
    result_id: str
    metric_name: str
    observed_value: Any
    is_empirically_proven: bool = True
    evidence_text: str


class InsightInterpretation(BaseModel):
    id: str
    interpretation_text: str
    grounded_in_finding_ids: List[str]
    confidence_rationale: str
    distinction_from_causality: str


class ActionRecommendation(BaseModel):
    id: str
    title: str
    description: str
    hypothesis_to_investigate: str
    expected_impact: str
    data_needed_to_confirm: str
    is_action_proposal_only: bool = True  # Strict flag: not proven causality


class InsightReport(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    executive_summary: str
    observed_findings: List[InsightFinding] = Field(default_factory=list)
    interpretations: List[InsightInterpretation] = Field(default_factory=list)
    unproven_hypotheses: List[str] = Field(default_factory=list)
    data_limitations: List[str] = Field(default_factory=list)
    recommended_actions: List[ActionRecommendation] = Field(default_factory=list)
    suggested_further_analyses: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# H. Dashboard Specifications
# ---------------------------------------------------------------------------
class MetricCardSpec(BaseModel):
    id: str
    title: str
    value: str
    comparison_text: Optional[str] = None
    trend_direction: Optional[Literal["up", "down", "flat"]] = None
    result_id: str
    period: str
    validation_status: str


class ChartComponentSpec(BaseModel):
    id: str
    question_answered: str
    chart_type: Literal["time_series", "waterfall_bars", "horizontal_bars", "table_breakdown", "line_comparison"]
    title: str
    result_id: str
    metric: str
    unit: str
    dimensions: List[str] = Field(default_factory=list)
    data: List[Dict[str, Any]]
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    period: str
    validation_status: str
    is_primary_objective: bool = False


class DashboardSpec(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    title: str
    subtitle: str
    objective_question: str
    metric_cards: List[MetricCardSpec] = Field(default_factory=list)
    charts: List[ChartComponentSpec] = Field(default_factory=list)
    quality_alerts: List[str] = Field(default_factory=list)
    methodology_notes: List[str] = Field(default_factory=list)
    provenance_chain_summary: List[str] = Field(default_factory=list)
    is_dashboard_validated: bool = False
    validation_notes: List[str] = Field(default_factory=list)
    is_demo_mode: bool = False
    operating_mode: str = "DEMO_WITHOUT_LLM"
    llm_model: Optional[str] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# J. Chat & Provenance Contracts
# ---------------------------------------------------------------------------
class ProvenanceCitation(BaseModel):
    source_type: Literal["raw_file", "transformed_table", "analytic_step", "validation_check"]
    source_name: str
    filter_or_condition: str
    result_id: Optional[str] = None
    details: str


class ChatRequest(BaseModel):
    project_id: str
    run_id: str
    question: str
    active_filters: Dict[str, Any] = Field(default_factory=dict)


class ChatAnswer(BaseModel):
    schema_version: str = "1.0"
    query_type: Literal["explain_existing_result", "recalculate_with_filters", "request_new_analysis", "unanswerable_by_data"]
    answer_text: str
    citations: List[ProvenanceCitation] = Field(default_factory=list)
    calculation_summary: Optional[str] = None
    data_limitation_notice: Optional[str] = None
    requires_full_pipeline: bool = False
    is_demo_mode: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
