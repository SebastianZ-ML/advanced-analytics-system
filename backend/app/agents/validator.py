"""
Agent F: Analytical Validator (Agente Validador Analítico).
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
        super().__init__(name="Agente Validador Analítico", role="Auditoría matemática independiente y control de calidad de resultados")

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
                        description=f"Verificación de cardinalidad y no inflación en la unión con {jc.right_table}",
                        passed=False,
                        severity="blocking",
                        evidence=f"Factor de multiplicación: {jc.multiplication_factor}. Variación no reconciliada.",
                        remedy_action=f"Deduplicar la dimensión {jc.right_table} antes de la unión."
                    ))
                else:
                    checks.append(ValidationCheck(
                        check_name=f"Join_Safety_{jc.right_table}",
                        description=f"Verificación de cardinalidad en unión con {jc.right_table}",
                        passed=True,
                        severity="info",
                        evidence=f"Factor exacto: {jc.multiplication_factor}x. Métrica perfectamente reconciliada."
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
                res.rejection_details = "Contiene valores NaN o infinitos en las métricas calculadas."
                checks.append(ValidationCheck(
                    check_name=f"Finite_Values_{res.result_id}",
                    description="Comprobación de ausencia de NaN e infinitos",
                    passed=False,
                    severity="blocking",
                    evidence=f"El resultado {res.result_id} contiene valores no finitos.",
                    remedy_action="Filtrar valores nulos o divisiones por cero en el cálculo."
                ))
            else:
                checks.append(ValidationCheck(
                    check_name=f"Finite_Values_{res.result_id}",
                    description="Comprobación de ausencia de NaN e infinitos",
                    passed=True,
                    severity="info",
                    evidence=f"Todos los valores calculados en {res.result_id} son finitos y definidos."
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
                            f"Fallo de reconciliación matemática: la suma de contribuciones ({sum_contrib_delta:.2f}) "
                            f"no iguala la variación total esperada ({target_total}). Discrepancia: {discrepancy:.2f}"
                        )
                        checks.append(ValidationCheck(
                            check_name=f"Reconciliation_{res.result_id}",
                            description="Reconciliación aditiva de descomposición dimensional al 100%",
                            passed=False,
                            severity="blocking",
                            evidence=res.rejection_details,
                            remedy_action="Asegurar que las categorías sean mutuamente excluyentes y que no falten registros en la dimensión."
                        ))
                    else:
                        checks.append(ValidationCheck(
                            check_name=f"Reconciliation_{res.result_id}",
                            description="Reconciliación aditiva de descomposición dimensional al 100%",
                            passed=True,
                            severity="info",
                            evidence=f"Reconciliación perfecta: suma de deltas ({sum_contrib_delta:.2f}) iguala delta total ({target_total:.2f})."
                        ))

            # Check 2C: Forecast Eligibility Warning
            if res.step_id == "OP_07_BASELINE_FORECAST_EVAL":
                if not calc.get("eligible", False):
                    has_warnings = True
                    checks.append(ValidationCheck(
                        check_name="Forecast_Sample_Size_Audit",
                        description="Auditoría de tamaño muestral para pronóstico",
                        passed=True,  # Passing check of constraint
                        severity="warning",
                        evidence=calc.get("reason", "Historial insuficiente"),
                        remedy_action="Deshabilitar pronóstico complejo; mantener sólo análisis retrospectivo descriptivo."
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
            repair_instruction = "Revisar uniones y filtros de datos para garantizar consistencia y reconciliación aditiva."
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
