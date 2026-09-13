/**
 * TypeScript contracts mirroring Pydantic backend models.
 */

export interface AmbiguityItem {
  id: string;
  question: string;
  context: string;
  proposed_default: string;
  user_decision?: string;
  impact_level: 'low' | 'medium' | 'high';
  status: 'pending' | 'resolved';
}

export interface ObjectiveSpec {
  schema_version: string;
  project_id: string;
  original_question: string;
  operational_objective: string;
  decision_to_inform: string;
  primary_metric: string;
  auxiliary_metrics: string[];
  time_period: string;
  comparison_period?: string;
  relevant_dimensions: string[];
  assumptions: string[];
  pending_ambiguities: AmbiguityItem[];
  acceptance_criteria: string[];
  excluded_scope: string[];
  status: 'draft' | 'confirmed' | 'requires_revision';
  created_at: string;
}

export interface ColumnProfile {
  name: string;
  inferred_type: 'string' | 'integer' | 'float' | 'date' | 'datetime' | 'boolean' | 'unknown';
  confirmed_type?: string;
  total_count: number;
  null_count: number;
  null_percentage: number;
  unique_count: number;
  has_leading_zeros: boolean;
  has_mixed_currency: boolean;
  sample_values: any[];
  min_value?: number | string;
  max_value?: number | string;
  mean_value?: number;
  std_dev?: number;
}

export interface TableProfile {
  table_id: string;
  source_filename: string;
  sheet_name?: string;
  display_name: string;
  row_count: number;
  column_count: number;
  columns: Record<string, ColumnProfile>;
  file_hash_sha256: string;
  row_semantic_meaning: string;
  date_columns: string[];
  id_columns: string[];
  numeric_columns: string[];
  categorical_columns: string[];
}

export interface DataCatalog {
  schema_version: string;
  project_id: string;
  tables: Record<string, TableProfile>;
  total_tables: number;
  generated_at: string;
}

export interface QualityIssue {
  id: string;
  issue_type: string;
  severity: 'info' | 'warning' | 'blocking';
  table_id: string;
  column?: string;
  affected_rows: number;
  total_rows: number;
  affected_percentage: number;
  evidence_samples: any[];
  description: string;
  recommended_action: string;
  user_resolution?: string;
}

export interface DataQualityReport {
  schema_version: string;
  project_id: string;
  issues: QualityIssue[];
  has_blocking_issues: boolean;
  coverage_start?: string;
  coverage_end?: string;
  is_last_period_incomplete: boolean;
  incomplete_period_details?: string;
  generated_at: string;
}

export interface RelationshipSpec {
  id: string;
  left_table: string;
  left_key: string;
  right_table: string;
  right_key: string;
  cardinality: 'one_to_one' | 'one_to_many' | 'many_to_one' | 'many_to_many';
  match_rate_left: number;
  orphan_count_left: number;
  match_rate_right: number;
  orphan_count_right: number;
  risk_level: 'low' | 'medium' | 'high';
  is_confirmed: boolean;
  rejection_reason?: string;
}

export interface OperationSpec {
  step_id: string;
  operation_name: string;
  category: string;
  description: string;
  required_inputs: string[];
  parameters: Record<string, any>;
  assumptions: string[];
  pre_execution_validations: string[];
  depends_on: string[];
  is_implemented: boolean;
  method_documentation_ref?: string;
}

export interface ExternalResearchCitation {
  url: string;
  title: string;
  consulted_at: string;
  supported_claim: string;
  is_primary_source: boolean;
}

export interface AnalysisPlan {
  schema_version: string;
  project_id: string;
  run_id: string;
  title: string;
  rationale: string;
  operations: OperationSpec[];
  excluded_methods: string[];
  citations: ExternalResearchCitation[];
  created_at: string;
}

export interface JoinCheckResult {
  left_table: string;
  right_table: string;
  join_type: string;
  join_keys: string[];
  left_rows: number;
  right_rows: number;
  result_rows: number;
  multiplication_factor: number;
  orphans_left: number;
  orphans_right: number;
  metric_reconciled: boolean;
  is_safe: boolean;
  warning?: string;
}

export interface TransformationRecord {
  transformation_id: string;
  step_index: number;
  input_table: string;
  output_table: string;
  operation: string;
  columns_affected: string[];
  rationale: string;
  parameters: Record<string, any>;
  rows_before: number;
  rows_after: number;
  rows_excluded: number;
  exclusion_reason?: string;
  exclusion_samples: any[];
  join_check?: JoinCheckResult;
  created_at: string;
}

export interface DimensionContribution {
  dimension_value: string;
  baseline_value: number;
  current_value: number;
  absolute_change: number;
  percentage_change: number;
  contribution_to_total_change: number;
}

export interface AnalysisResult {
  result_id: string;
  step_id: string;
  operation_name: string;
  method: string;
  parameters: Record<string, any>;
  data_source_version: string;
  unit_of_measure: string;
  time_period_covered: string;
  calculated_values: Record<string, any>;
  warnings: string[];
  version: number;
  validation_status: 'pending' | 'approved' | 'approved_with_warnings' | 'rejected';
  rejection_details?: string;
  created_at: string;
}

export interface ValidationCheck {
  check_name: string;
  description: string;
  passed: boolean;
  severity: 'info' | 'warning' | 'blocking';
  evidence: string;
  remedy_action?: string;
}

export interface ValidationReport {
  schema_version: string;
  run_id: string;
  overall_status: 'approved' | 'approved_with_warnings' | 'rejected';
  checks_executed: ValidationCheck[];
  failed_results: string[];
  repair_attempt: number;
  repair_instruction?: string;
  can_retry: boolean;
  generated_at: string;
}

export interface InsightFinding {
  id: string;
  claim: string;
  result_id: string;
  metric_name: string;
  observed_value: any;
  is_empirically_proven: boolean;
  evidence_text: string;
}

export interface InsightInterpretation {
  id: string;
  interpretation_text: string;
  grounded_in_finding_ids: string[];
  confidence_rationale: string;
  distinction_from_causality: string;
}

export interface ActionRecommendation {
  id: string;
  title: string;
  description: string;
  hypothesis_to_investigate: string;
  expected_impact: string;
  data_needed_to_confirm: string;
  is_action_proposal_only: boolean;
}

export interface InsightReport {
  schema_version: string;
  run_id: string;
  executive_summary: string;
  observed_findings: InsightFinding[];
  interpretations: InsightInterpretation[];
  unproven_hypotheses: string[];
  data_limitations: string[];
  recommended_actions: ActionRecommendation[];
  suggested_further_analyses: string[];
  generated_at: string;
}

export interface MetricCardSpec {
  id: string;
  title: string;
  value: string;
  comparison_text?: string;
  trend_direction?: 'up' | 'down' | 'flat';
  result_id: string;
  period: string;
  validation_status: string;
}

export interface ChartComponentSpec {
  id: string;
  question_answered: string;
  chart_type: 'time_series' | 'waterfall_bars' | 'horizontal_bars' | 'table_breakdown' | 'line_comparison';
  title: string;
  result_id: string;
  metric: string;
  unit: string;
  dimensions: string[];
  data: any[];
  filters_applied: Record<string, any>;
  period: string;
  validation_status: string;
  is_primary_objective: boolean;
}

export interface DashboardSpec {
  schema_version: string;
  run_id: string;
  title: string;
  subtitle: string;
  objective_question: string;
  metric_cards: MetricCardSpec[];
  charts: ChartComponentSpec[];
  quality_alerts: string[];
  methodology_notes: string[];
  provenance_chain_summary: string[];
  is_dashboard_validated: boolean;
  validation_notes: string[];
  generated_at: string;
}

export interface ProvenanceCitation {
  source_type: 'raw_file' | 'transformed_table' | 'analytic_step' | 'validation_check';
  source_name: string;
  filter_or_condition: string;
  result_id?: string;
  details: string;
}

export interface ChatAnswer {
  schema_version: string;
  query_type: 'explain_existing_result' | 'recalculate_with_filters' | 'request_new_analysis' | 'unanswerable_by_data';
  answer_text: string;
  citations: ProvenanceCitation[];
  calculation_summary?: string;
  data_limitation_notice?: string;
  requires_full_pipeline: boolean;
  is_demo_mode: boolean;
  timestamp: string;
}

export interface Project {
  project_id: string;
  name: string;
  description: string;
  created_at: string;
  files?: ProjectFile[];
}

export interface ProjectFile {
  file_id: string;
  project_id: string;
  filename: string;
  sheet_name?: string;
  table_name: string;
  file_path: string;
  file_hash: string;
  row_count: number;
  column_count: number;
  uploaded_at: string;
}

export interface RunEvent {
  event_id: number;
  run_id: string;
  stage: string;
  level: string;
  message: string;
  metadata_json?: string;
  created_at: string;
}
