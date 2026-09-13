"""
Agent F: Analytical Validator.
Performs independent, deterministic verification of analytical results.
Does NOT rely on LLM intuition. Enforces mathematical reconciliation,
finite value checks, cardinality checks, and automated repair cycles.
"""
import math
from typing import Any, Dict, List, Tuple
from app.agents.base import BaseAgent
from app.contracts import AnalysisResult, TransformationRecord, ValidationCheck, ValidationReport


class AnalyticalValidatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Analytical Validator Agent", role="Independent mathematical audit and quality control of results")

    @staticmethod
    def _is_finite_number(val: Any) -> bool:
        if val is None:
            return False
        if isinstance(val, (int, float)):
            return not (math.isnan(val) or math.isinf(val))
        return True

    @staticmethod
    def _check_nested_finite(obj: Any) -> bool:
        if isinstance(obj, dict):
            return all(AnalyticalValidatorAgent._check_nested_finite(v) for v in obj.values())
        elif isinstance(obj, list):
            return all(AnalyticalValidatorAgent._check_nested_finite(v) for v in obj)
        elif isinstance(obj, float):
            return not (math.isnan(obj) or math.isinf(obj))
        return True

    def validate_results(
        self,
        run_id: str,
        results: List[AnalysisResult],
        transformations: List[TransformationRecord],
        repair_attempt: int = 0
    ) -> ValidationReport:
        """
        Executes strict deterministic controls across all results and transformations.
        """
        checks: List[ValidationCheck] = []
        failed_result_ids: List[str] = []
        has_blocking = False
        has_warnings = False
        repair_instruction = None

        # 1. Check Join Cardinality in Transformations
        for trf in transformations:
            if trf.join_check:
                jc = trf.join_check
                if not jc.is_safe:
                    has_blocking = True
                    checks.append(ValidationCheck(
                        check_name=f"Join_Safety_{jc.right_table}",
                        description=f"Cardinality and non-inflation verification for join with {jc.right_table}",
                        passed=False,
                        severity="blocking",
                        evidence=f"Multiplication factor: {jc.multiplication_factor}. Unreconciled variation.",
                        remedy_action=f"Deduplicate dimension {jc.right_table} before join."
                    ))
                else:
                    checks.append(ValidationCheck(
                        check_name=f"Join_Safety_{jc.right_table}",
                        description=f"Cardinality verification for join with {jc.right_table}",
                        passed=True,
                        severity="info",
                        evidence=f"Exact factor: {jc.multiplication_factor}x. Metric perfectly reconciled."
                    ))

        # Find period comparison result to reconcile against
        period_comp_res = next((r for r in results if r.step_id == "OP_03_PERIOD_COMPARISON"), None)
        expected_total_change = None
        if period_comp_res:
            expected_total_change = period_comp_res.calculated_values.get("absolute_change")

        # 2. Check Each Analysis Result
        for res in results:
            calc = res.calculated_values

            # Check 2A: Finite Values (No NaN / Inf)
            finite_ok = self._check_nested_finite(calc)
            if not finite_ok:
                has_blocking = True
                failed_result_ids.append(res.result_id)
                res.validation_status = "rejected"
                res.rejection_details = "Contains NaN or infinite values in calculated metrics."
                checks.append(ValidationCheck(
                    check_name=f"Finite_Values_{res.result_id}",
                    description="Verification of absence of NaN and infinite values",
                    passed=False,
                    severity="blocking",
                    evidence=f"Result {res.result_id} contains non-finite values.",
                    remedy_action="Filter null values or prevent zero division in calculation."
                ))
            else:
                checks.append(ValidationCheck(
                    check_name=f"Finite_Values_{res.result_id}",
                    description="Verification of absence of NaN and infinite values",
                    passed=True,
                    severity="info",
                    evidence=f"All calculated values in {res.result_id} are finite and defined."
                ))

            # Check 2B: Mathematical Reconciliation of Dimension Decompositions
            if "contributions" in calc:
                contribs = calc["contributions"]
                target_total = expected_total_change if expected_total_change is not None else calc.get("total_change")
                is_flagged_unreconciled = calc.get("is_perfectly_reconciled") is False or calc.get("reconciliation_diff", 0.0) > 0.05

                if target_total is not None or is_flagged_unreconciled:
                    sum_contrib_delta = sum(c.get("absolute_change", 0.0) for c in contribs) if contribs else 0.0
                    discrepancy = abs(sum_contrib_delta - (target_total or 0.0)) if target_total is not None else calc.get("reconciliation_diff", 0.0)

                    if discrepancy > 0.05 or is_flagged_unreconciled:
                        has_blocking = True
                        failed_result_ids.append(res.result_id)
                        res.validation_status = "rejected"
                        res.rejection_details = (
                            f"Mathematical reconciliation failure: sum of contributions ({sum_contrib_delta:.2f}) "
                            f"does not equal expected total change ({target_total}). Discrepancy: {discrepancy:.2f}"
                        )
                        checks.append(ValidationCheck(
                            check_name=f"Reconciliation_{res.result_id}",
                            description="100% additive dimensional decomposition reconciliation",
                            passed=False,
                            severity="blocking",
                            evidence=res.rejection_details,
                            remedy_action="Ensure categories are mutually exclusive and no dimension records are missing."
                        ))
                    else:
                        checks.append(ValidationCheck(
                            check_name=f"Reconciliation_{res.result_id}",
                            description="100% additive dimensional decomposition reconciliation",
                            passed=True,
                            severity="info",
                            evidence=f"Perfect reconciliation: sum of deltas ({sum_contrib_delta:.2f}) equals total delta ({target_total:.2f})."
                        ))

            # Check 2C: Forecast Eligibility Warning
            if res.step_id == "OP_07_BASELINE_FORECAST_EVAL":
                if not calc.get("eligible", False):
                    has_warnings = True
                    checks.append(ValidationCheck(
                        check_name="Forecast_Sample_Size_Audit",
                        description="Sample size audit for forecasting",
                        passed=True,  # Passing check of constraint
                        severity="warning",
                        evidence=calc.get("reason", "Insufficient historical depth"),
                        remedy_action="Disable complex forecast; maintain descriptive retrospective analysis only."
                    ))

            # If not rejected, mark approved or approved_with_warnings
            if res.result_id not in failed_result_ids:
                if len(res.warnings) > 0:
                    res.validation_status = "approved_with_warnings"
                else:
                    res.validation_status = "approved"

        # Determine overall report status
        if has_blocking:
            overall_status = "rejected"
            can_retry = (repair_attempt < 2)
            repair_instruction = "Review data joins and filters to ensure consistency and additive reconciliation."
        elif has_warnings:
            overall_status = "approved_with_warnings"
            can_retry = False
        else:
            overall_status = "approved"
            can_retry = False

        return ValidationReport(
            schema_version="1.0",
            run_id=run_id,
            overall_status=overall_status,
            checks_executed=checks,
            failed_results=failed_result_ids,
            repair_attempt=repair_attempt,
            repair_instruction=repair_instruction,
            can_retry=can_retry
        )
