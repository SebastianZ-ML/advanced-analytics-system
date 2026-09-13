"""
Agent E: Analytical Executor.
Executes strictly registered operations from the approved AnalysisPlan.
Produces typed AnalysisResult artifacts with parameters, coverage, and warnings.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
from app.agents.base import BaseAgent
from app.contracts import AnalysisPlan, AnalysisResult
from app.engine.duckdb_engine import DuckDBAnalyticsEngine


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
        Executes operations from the plan sequentially.
        """
        results: List[AnalysisResult] = []

        # Monthly aggregation needed for several steps
        monthly_trend = DuckDBAnalyticsEngine.calculate_monthly_trend(analytical_df)

        for op in plan.operations:
            step_id = op.step_id

            if op.category == "metric_summary":
                gross = float(analytical_df["gross_sales"].sum()) if "gross_sales" in analytical_df.columns else 0.0
                net = float(analytical_df["net_sales"].sum()) if "net_sales" in analytical_df.columns else 0.0
                units = int(analytical_df["units"].sum()) if "units" in analytical_df.columns else 0
                orders = int(len(analytical_df))
                discounts = float(analytical_df["discount_amount"].sum()) if "discount_amount" in analytical_df.columns else 0.0

                calc_vals = {
                    "net_sales": round(net, 2),
                    "gross_sales": round(gross, 2),
                    "total_units": units,
                    "total_orders": orders,
                    "total_discounts": round(discounts, 2),
                    "avg_order_value": round(net / max(orders, 1), 2)
                }

                results.append(AnalysisResult(
                    result_id=f"RES_{step_id}",
                    step_id=step_id,
                    operation_name=op.operation_name,
                    method="Descriptive Aggregation",
                    parameters=op.parameters,
                    data_source_version=data_source_version,
                    unit_of_measure="USD / Monetary units",
                    time_period_covered="2026-01 to 2026-05",
                    calculated_values=calc_vals,
                    warnings=[]
                ))

            elif op.category == "time_aggregation":
                results.append(AnalysisResult(
                    result_id=f"RES_{step_id}",
                    step_id=step_id,
                    operation_name=op.operation_name,
                    method="Monthly Time Series Aggregation",
                    parameters=op.parameters,
                    data_source_version=data_source_version,
                    unit_of_measure="USD / Monetary units",
                    time_period_covered="2026-01 to 2026-05",
                    calculated_values={"monthly_series": monthly_trend},
                    warnings=[]
                ))

            elif op.category == "period_comparison":
                base_p = op.parameters.get("baseline_period", "2026-03")
                curr_p = op.parameters.get("current_period", "2026-05")
                m_base = next((m for m in monthly_trend if m["period"] == base_p), None)
                m_curr = next((m for m in monthly_trend if m["period"] == curr_p), None)

                base_val = m_base["net_sales"] if m_base else 0.0
                curr_val = m_curr["net_sales"] if m_curr else 0.0
                delta = round(curr_val - base_val, 2)
                pct_chg = round((delta / base_val) * 100, 2) if base_val != 0 else 0.0

                results.append(AnalysisResult(
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
                        "absolute_change": delta,
                        "percentage_change": pct_chg
                    },
                    warnings=[]
                ))

            elif op.category == "dimension_breakdown":
                dim = op.parameters.get("dimension", "channel")
                base_p = op.parameters.get("baseline_period", "2026-03")
                curr_p = op.parameters.get("current_period", "2026-05")
                metric = op.parameters.get("metric", "net_sales")

                breakdown = DuckDBAnalyticsEngine.calculate_period_breakdown(
                    df=analytical_df,
                    dimension=dim,
                    baseline_period=base_p,
                    current_period=curr_p,
                    metric=metric
                )

                results.append(AnalysisResult(
                    result_id=f"RES_{step_id}",
                    step_id=step_id,
                    operation_name=op.operation_name,
                    method="Additive Waterfall Dimension Decomposition",
                    parameters=op.parameters,
                    data_source_version=data_source_version,
                    unit_of_measure="USD / Monetary units",
                    time_period_covered=f"{base_p} vs {curr_p}",
                    calculated_values=breakdown,
                    warnings=[] if breakdown["is_perfectly_reconciled"] else ["Reconciliation discrepancy exceeds tolerance threshold."]
                ))

            elif op.category == "customer_dynamics":
                dynamics = DuckDBAnalyticsEngine.calculate_customer_dynamics(analytical_df)
                results.append(AnalysisResult(
                    result_id=f"RES_{step_id}",
                    step_id=step_id,
                    operation_name=op.operation_name,
                    method="First Purchase Cohort Comparison",
                    parameters=op.parameters,
                    data_source_version=data_source_version,
                    unit_of_measure="USD / Monetary units",
                    time_period_covered="2026-01 to 2026-05",
                    calculated_values=dynamics,
                    warnings=[]
                ))

            elif op.category == "baseline_forecast":
                eval_res = DuckDBAnalyticsEngine.evaluate_forecast_eligibility(monthly_trend)
                results.append(AnalysisResult(
                    result_id=f"RES_{step_id}",
                    step_id=step_id,
                    operation_name=op.operation_name,
                    method="Statistical Forecast Eligibility Audit",
                    parameters=op.parameters,
                    data_source_version=data_source_version,
                    unit_of_measure="Methodological eligibility",
                    time_period_covered="Closed historical period",
                    calculated_values=eval_res,
                    warnings=[eval_res["reason"]] if not eval_res["eligible"] else []
                ))

        return results
