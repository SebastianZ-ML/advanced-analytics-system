"""
Agent E: Analytical Executor.
Executes operations governed strictly by an ExecutionDAG topological ordering.
Produces typed AnalysisResult artifacts with parameters, coverage, and warnings.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
from app.agents.base import BaseAgent
from app.contracts import AnalysisPlan, AnalysisResult, OperationSpec
from app.engine.duckdb_engine import DuckDBAnalyticsEngine
from app.orchestrator.dag import ExecutionDAG


class AnalyticalExecutorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Analytical Executor Agent", role="Mathematical execution of registered operations")

    def execute_plan(
        self,
        analytical_df: pd.DataFrame,
        plan: AnalysisPlan,
        data_source_version: str = "v1.0"
    ) -> List[AnalysisResult]:
        """
        Executes operations governed strictly by the ExecutionDAG topological order.
        Guarantees that dependencies execute before dependents.
        """
        if not plan.operations:
            return []

        # 1. Build and validate Execution DAG
        dag = ExecutionDAG(plan.operations)
        ordered_step_ids = dag.topological_sort()
        op_map = {op.step_id: op for op in plan.operations}

        results: List[AnalysisResult] = []
        results_by_id: Dict[str, AnalysisResult] = {}

        # Resolve primary date and numeric columns dynamically
        date_candidates = [c for c in analytical_df.columns if "date" in c.lower() or "fecha" in c.lower() or "time" in c.lower()]
        primary_date_col = date_candidates[0] if date_candidates else analytical_df.columns[0]

        numeric_candidates = [c for c in analytical_df.columns if pd.api.types.is_numeric_dtype(analytical_df[c]) and "id" not in c.lower()]
        primary_metric_col = "net_sales" if "net_sales" in analytical_df.columns else (numeric_candidates[0] if numeric_candidates else analytical_df.columns[0])

        # Monthly aggregation calculated with resolved columns
        monthly_trend = DuckDBAnalyticsEngine.calculate_monthly_trend(
            analytical_df,
            date_col=primary_date_col,
            metric_col=primary_metric_col
        )

        for step_id in ordered_step_ids:
            op = op_map[step_id]

            # Validate upstream dependencies
            missing_deps = [dep for dep in op.depends_on if dep not in results_by_id]
            if missing_deps:
                raise RuntimeError(f"Cannot execute '{step_id}': missing upstream dependencies: {missing_deps}")

            if op.category == "metric_summary":
                calc_vals: Dict[str, Any] = {}
                for num_col in numeric_candidates:
                    calc_vals[num_col] = round(float(analytical_df[num_col].sum()), 2)
                
                calc_vals["total_rows"] = int(len(analytical_df))
                calc_vals["total_orders"] = int(len(analytical_df))
                # Ensure primary metric is explicitly present
                if primary_metric_col in calc_vals:
                    calc_vals["net_sales"] = calc_vals[primary_metric_col]

                res = AnalysisResult(
                    result_id=f"RES_{step_id}",
                    step_id=step_id,
                    operation_name=op.operation_name,
                    method="Descriptive Aggregation",
                    parameters=op.parameters,
                    data_source_version=data_source_version,
                    unit_of_measure="USD / Monetary units",
                    time_period_covered=f"Full dataset ({len(monthly_trend)} periods)",
                    calculated_values=calc_vals,
                    warnings=[]
                )

            elif op.category == "time_aggregation":
                res = AnalysisResult(
                    result_id=f"RES_{step_id}",
                    step_id=step_id,
                    operation_name=op.operation_name,
                    method="Monthly Time Series Aggregation",
                    parameters=op.parameters,
                    data_source_version=data_source_version,
                    unit_of_measure="USD / Monetary units",
                    time_period_covered=f"{monthly_trend[0]['period']} to {monthly_trend[-1]['period']}" if monthly_trend else "N/A",
                    calculated_values={"monthly_series": monthly_trend},
                    warnings=[]
                )

            elif op.category == "period_comparison":
                # Use specified or infer peak to valley
                if monthly_trend:
                    default_base = monthly_trend[0]["period"]
                    default_curr = monthly_trend[-1]["period"]
                else:
                    default_base, default_curr = "P1", "P2"

                base_p = op.parameters.get("baseline_period", default_base)
                curr_p = op.parameters.get("current_period", default_curr)

                m_base = next((m for m in monthly_trend if m["period"] == base_p), None)
                m_curr = next((m for m in monthly_trend if m["period"] == curr_p), None)

                base_val = m_base.get(primary_metric_col, m_base.get("net_sales", 0.0)) if m_base else 0.0
                curr_val = m_curr.get(primary_metric_col, m_curr.get("net_sales", 0.0)) if m_curr else 0.0
                delta = round(curr_val - base_val, 2)
                pct_chg = round((delta / base_val) * 100, 2) if base_val != 0 else 0.0

                res = AnalysisResult(
                    result_id=f"RES_{step_id}",
                    step_id=step_id,
                    operation_name=op.operation_name,
                    method="Period-over-Period Delta",
                    parameters=op.parameters,
                    data_source_version=data_source_version,
                    unit_of_measure="USD / Monetary units",
                    time_period_covered=f"{base_p} vs {curr_p}",
                    calculated_values={
                        "baseline_period": base_p,
                        "current_period": curr_p,
                        "baseline_net_sales": base_val,
                        "current_net_sales": curr_val,
                        primary_metric_col: curr_val,
                        "absolute_change": delta,
                        "percentage_change": pct_chg
                    },
                    warnings=[]
                )

            elif op.category == "dimension_breakdown":
                dim = op.parameters.get("dimension")
                if not dim or dim not in analytical_df.columns:
                    # Pick first categorical column
                    cat_cols = [c for c in analytical_df.columns if not pd.api.types.is_numeric_dtype(analytical_df[c]) and "id" not in c.lower()]
                    dim = cat_cols[0] if cat_cols else "channel"

                base_p = op.parameters.get("baseline_period", monthly_trend[0]["period"] if monthly_trend else "2026-03")
                curr_p = op.parameters.get("current_period", monthly_trend[-1]["period"] if monthly_trend else "2026-05")
                metric = op.parameters.get("metric", primary_metric_col)

                breakdown = DuckDBAnalyticsEngine.calculate_period_breakdown(
                    df=analytical_df,
                    dimension=dim,
                    baseline_period=base_p,
                    current_period=curr_p,
                    metric=metric
                )

                res = AnalysisResult(
                    result_id=f"RES_{step_id}",
                    step_id=step_id,
                    operation_name=op.operation_name,
                    method="Additive Dimensional Waterfall",
                    parameters=op.parameters,
                    data_source_version=data_source_version,
                    unit_of_measure="USD / Monetary units",
                    time_period_covered=f"{base_p} vs {curr_p}",
                    calculated_values=breakdown,
                    warnings=[]
                )

            elif op.category == "customer_dynamics":
                cust_col = next((c for c in analytical_df.columns if "cust" in c.lower() or "client" in c.lower() or "user" in c.lower()), None)
                if cust_col and primary_date_col in analytical_df.columns:
                    dyn = DuckDBAnalyticsEngine.calculate_customer_dynamics(
                        df=analytical_df,
                        customer_id_col=cust_col,
                        date_col=primary_date_col
                    )
                else:
                    dyn = {"monthly_customer_dynamics": []}

                res = AnalysisResult(
                    result_id=f"RES_{step_id}",
                    step_id=step_id,
                    operation_name=op.operation_name,
                    method="Cohort and Recurrence Analysis",
                    parameters=op.parameters,
                    data_source_version=data_source_version,
                    unit_of_measure="Customer count",
                    time_period_covered="Historical cohort coverage",
                    calculated_values=dyn,
                    warnings=[]
                )

            elif op.category == "baseline_forecast":
                eligibility = DuckDBAnalyticsEngine.evaluate_forecast_eligibility(monthly_trend)
                res = AnalysisResult(
                    result_id=f"RES_{step_id}",
                    step_id=step_id,
                    operation_name=op.operation_name,
                    method="Evaluation of Forecast Eligibility",
                    parameters=op.parameters,
                    data_source_version=data_source_version,
                    unit_of_measure="Eligibility Specification",
                    time_period_covered="Next period",
                    calculated_values=eligibility,
                    warnings=[eligibility.get("reason", "")] if not eligibility.get("eligible") else []
                )

            else:
                res = AnalysisResult(
                    result_id=f"RES_{step_id}",
                    step_id=step_id,
                    operation_name=op.operation_name,
                    method="Generic Execution",
                    parameters=op.parameters,
                    data_source_version=data_source_version,
                    unit_of_measure="Units",
                    time_period_covered="All",
                    calculated_values={"status": "completed"},
                    warnings=[]
                )

            results.append(res)
            results_by_id[op.step_id] = res

        return results
