"""
Agent C: Methodologist & Researcher.
Selects mathematically and empirically appropriate methods from a registered internal catalog,
documents assumptions, and produces a structured AnalysisPlan.
Supports Gemini LLM proposal with deterministic validation and fallback.
"""
from typing import Any, Dict, List, Optional
from app.agents.base import BaseAgent
from app.contracts import (
    AnalysisPlan,
    DataCatalog,
    DataQualityReport,
    ExternalResearchCitation,
    ObjectiveSpec,
    OperationSpec,
    RelationshipSpec,
)
from app.providers import ContextBuilder, LLMProvider, get_llm_provider

REGISTERED_METHODS_CATALOG: List[Dict[str, Any]] = [
    {
        "step_id": "OP_01_METRIC_SUMMARY",
        "operation_name": "Descriptive summary of global metrics",
        "category": "metric_summary",
        "description": "Calculation of total gross sales, net sales, units, and order volume over the closed period.",
        "required_inputs": ["orders_analytical"],
        "parameters": {"metric_columns": ["net_sales", "gross_sales", "units", "discount_amount"]},
        "assumptions": ["Amounts correspond to closed and completed transactions."],
        "pre_execution_validations": ["Verify that numeric columns do not contain NaN values."],
        "depends_on": []
    },
    {
        "step_id": "OP_02_TIME_AGGREGATION",
        "operation_name": "Monthly temporal aggregation",
        "category": "time_aggregation",
        "description": "Monthly evolution of accrued net sales and order count between January and May 2026.",
        "required_inputs": ["orders_analytical"],
        "parameters": {"date_column": "order_date_clean", "freq": "M"},
        "assumptions": ["Incomplete month of June is excluded to ensure comparability."],
        "pre_execution_validations": ["Verify continuous temporal coverage."],
        "depends_on": ["OP_01_METRIC_SUMMARY"]
    },
    {
        "step_id": "OP_03_PERIOD_COMPARISON",
        "operation_name": "Period-over-period peak to valley comparison",
        "category": "period_comparison",
        "description": "Measurement of absolute and percentage variation between March 2026 (pre-drop peak month) and May 2026.",
        "required_inputs": ["orders_analytical"],
        "parameters": {"baseline_period": "2026-03", "current_period": "2026-05", "metric": "net_sales"},
        "assumptions": ["March represents the standard operational baseline prior to the decline."],
        "pre_execution_validations": ["Verify that both months have complete data."],
        "depends_on": ["OP_02_TIME_AGGREGATION"]
    },
    {
        "step_id": "OP_04_DIMENSION_BREAKDOWN_CHANNEL",
        "operation_name": "Additive commercial channel breakdown (Waterfall)",
        "category": "dimension_breakdown",
        "description": "Decomposition of total variance into mutually exclusive contributions by channel, reconciled to 100%.",
        "required_inputs": ["orders_analytical"],
        "parameters": {"dimension": "channel", "baseline_period": "2026-03", "current_period": "2026-05", "metric": "net_sales"},
        "assumptions": ["Commercial channels are exhaustive and mutually exclusive."],
        "pre_execution_validations": ["Verify that the sum of channel variations exactly equals the total variation."],
        "depends_on": ["OP_03_PERIOD_COMPARISON"]
    },
    {
        "step_id": "OP_05_DIMENSION_BREAKDOWN_SEGMENT",
        "operation_name": "Additive customer segment breakdown",
        "category": "dimension_breakdown",
        "description": "Decomposition of total variance across customer segments to evaluate corporate vs retail decline.",
        "required_inputs": ["orders_analytical"],
        "parameters": {"dimension": "segment", "baseline_period": "2026-03", "current_period": "2026-05", "metric": "net_sales"},
        "assumptions": ["Each customer belongs to a unique segment in the dimension."],
        "pre_execution_validations": ["Verify total reconciliation."],
        "depends_on": ["OP_03_PERIOD_COMPARISON"]
    },
    {
        "step_id": "OP_06_CUSTOMER_DYNAMICS",
        "operation_name": "New vs recurring customer dynamics",
        "category": "customer_dynamics",
        "description": "Evaluation of whether sales loss stems from lower customer acquisition or repurchase churn among recurring clients.",
        "required_inputs": ["orders_analytical"],
        "parameters": {"customer_id_col": "customer_id", "date_col": "order_date_clean"},
        "assumptions": ["New customer is formally defined as one whose first order occurs in the analyzed month."],
        "pre_execution_validations": ["Verify customer identifier consistency."],
        "depends_on": ["OP_02_TIME_AGGREGATION"]
    },
    {
        "step_id": "OP_07_BASELINE_FORECAST_EVAL",
        "operation_name": "Statistical forecast feasibility evaluation",
        "category": "baseline_forecast",
        "description": "Verification of minimum statistical conditions (history >= 12 periods) for predictive modeling.",
        "required_inputs": ["orders_analytical"],
        "parameters": {"min_periods": 12},
        "assumptions": ["Forecasting without sufficient historical depth introduces overfitting and misleading certainty."],
        "pre_execution_validations": ["Check monthly series length."],
        "depends_on": ["OP_02_TIME_AGGREGATION"]
    }
]


class MethodologistAgent(BaseAgent):
    def __init__(self, override_provider: Optional[LLMProvider] = None):
        super().__init__(name="Methodologist Agent", role="Selection of registered analytical methods, definition of assumptions, and plan design")
        self._override_provider = override_provider

    def build_plan(
        self,
        project_id: str,
        run_id: str,
        objective: ObjectiveSpec,
        catalog: DataCatalog,
        dq_report: DataQualityReport,
        confirmed_relationships: Optional[List[RelationshipSpec]] = None
    ) -> AnalysisPlan:
        """
        Creates an AnalysisPlan tailored to decomposing sales variation.
        Uses Gemini to select operations if available, strictly validating against
        the registered catalog of implemented methods.
        """
        provider = get_llm_provider(self._override_provider)
        catalog_summary = ContextBuilder.build_catalog_summary(catalog)
        dq_summary = ContextBuilder.build_quality_summary(dq_report)
        relationships = confirmed_relationships or []

        if not provider.is_deterministic_fallback:
            try:
                proposed_plan = provider.propose_analysis_plan(
                    project_id=project_id,
                    run_id=run_id,
                    objective=objective,
                    catalog_summary=catalog_summary,
                    dq_summary=dq_summary,
                    confirmed_relationships=relationships,
                    registered_methods=REGISTERED_METHODS_CATALOG
                )

                # Deterministic Gatekeeper: Validate that all proposed operations are in the registered catalog
                # and do not reference non-existent columns
                registered_step_ids = {m["step_id"] for m in REGISTERED_METHODS_CATALOG}
                all_catalog_cols = set()
                if catalog:
                    for t in catalog.tables.values():
                        if isinstance(t.columns, dict):
                            for col_name in t.columns.keys():
                                all_catalog_cols.add(str(col_name).lower())
                        elif isinstance(t.columns, list):
                            for c in t.columns:
                                all_catalog_cols.add(getattr(c, "name", str(c)).lower())
                # Add known analytical derived columns
                all_catalog_cols.update([
                    "order_date_clean", "net_sales", "gross_sales", "units", 
                    "discount_amount", "channel", "region", "customer_name", "category_name"
                ])

                valid_ops = []
                for op in proposed_plan.operations:
                    if op.step_id not in registered_step_ids:
                        print(f"[MethodologistAgent] Unregistered operation rejected: {op.step_id}")
                        continue

                    # Check for non-existent columns in parameters
                    params = op.parameters or {}
                    cols_to_check = []
                    if "metric_columns" in params and isinstance(params["metric_columns"], list):
                        cols_to_check.extend([str(c).lower() for c in params["metric_columns"]])
                    if "dimension" in params and isinstance(params["dimension"], str):
                        cols_to_check.append(str(params["dimension"]).lower())
                    if "metric" in params and isinstance(params["metric"], str):
                        cols_to_check.append(str(params["metric"]).lower())
                    if "date_column" in params and isinstance(params["date_column"], str):
                        cols_to_check.append(str(params["date_column"]).lower())

                    invalid_cols = [c for c in cols_to_check if c not in all_catalog_cols]
                    if invalid_cols:
                        print(f"[MethodologistAgent] Operation rejected for referencing non-existent columns: {invalid_cols}")
                        continue

                    valid_ops.append(op)

                if len(valid_ops) >= 3:
                    proposed_plan.operations = valid_ops
                    return proposed_plan
                else:
                    print("[MethodologistAgent] Proposed Gemini plan had insufficient or unregistered operations. Applying deterministic plan.")

            except Exception as e:
                print(f"[MethodologistAgent] Fallback activated after error in plan proposal: {e}")

        # Deterministic fallback plan
        return self._build_deterministic_plan(project_id, run_id, objective)

    def _build_deterministic_plan(self, project_id: str, run_id: str, objective: Optional[ObjectiveSpec] = None) -> AnalysisPlan:
        ops = []
        target_metric = "net_sales"
        if objective and objective.primary_metric:
            target_metric = objective.primary_metric.split(" ")[0].split("(")[0].strip()

        for m in REGISTERED_METHODS_CATALOG:
            params = dict(m["parameters"])
            if "metric" in params:
                params["metric"] = target_metric

            ops.append(OperationSpec(
                step_id=m["step_id"],
                operation_name=m["operation_name"],
                category=m["category"],
                description=m["description"],
                required_inputs=m["required_inputs"],
                parameters=params,
                assumptions=m["assumptions"],
                pre_execution_validations=m["pre_execution_validations"],
                depends_on=m["depends_on"]
            ))

        citations = [
            ExternalResearchCitation(
                url="https://otexts.com/fpp3/decomposition.html",
                title="Forecasting: Principles and Practice - Time Series Decomposition",
                supported_claim="Additive decomposition requires mutually exclusive categories to reconcile component sums with the aggregate series.",
                is_primary_source=True
            ),
            ExternalResearchCitation(
                url="https://pandas.pydata.org/docs/user_guide/merging.html",
                title="Pandas Documentation: Merge and Join Integrity",
                supported_claim="Fact-to-dimension joins must enforce key uniqueness on the right table to prevent unintended row inflation.",
                is_primary_source=True
            )
        ]

        return AnalysisPlan(
            schema_version="1.0",
            project_id=project_id,
            run_id=run_id,
            title="Sales Decline Diagnosis and Variance Decomposition Plan",
            rationale="Analytical sequence designed to isolate segment and channel drivers explaining observed contraction, validating mathematical integrity at every step.",
            operations=ops,
            excluded_methods=[
                "Causal multivariate regression (dataset lacks exogenous confounding variables such as inflation or competitor stockouts).",
                "Complex ARIMA / Prophet (5-month historical series does not meet minimum statistical threshold for seasonality)."
            ],
            citations=citations
        )
