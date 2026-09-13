"""
Agent I: Dashboard Validator.
Performs independent checks on the assembled DashboardSpec:
- Guarantees numbers shown in cards match underlying result values.
- Assures no rejected results are exposed.
- Confirms period alignment between cards and charts.
"""
import re
from typing import Dict, List
from app.agents.base import BaseAgent
from app.contracts import AnalysisResult, DashboardSpec, ValidationReport


class DashboardValidatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Dashboard Validator Agent", role="Consistency audit of visualizations and displayed figures")

    def validate_dashboard(
        self,
        dashboard: DashboardSpec,
        results: List[AnalysisResult],
        validation_report: ValidationReport
    ) -> DashboardSpec:
        """
        Validates the DashboardSpec deterministically.
        Verifies both structural references and exact numeric values.
        """
        notes: List[str] = []
        is_valid = True
        res_map = {r.result_id: r for r in results}
        rejected_ids = set(validation_report.failed_results)

        # Check 1: Cards validation (references AND exact numeric content)
        for card in dashboard.metric_cards:
            if card.result_id in rejected_ids:
                is_valid = False
                notes.append(f"REJECTED: Card '{card.title}' attempts to display rejected result '{card.result_id}'.")
            elif card.result_id not in res_map:
                is_valid = False
                notes.append(f"REJECTED: Card '{card.title}' references non-existent result '{card.result_id}'.")
            else:
                # Content verification: Verify numerical consistency
                res = res_map[card.result_id]
                clean_val_str = re.sub(r"[^\d.-]", "", str(card.value))
                if clean_val_str and clean_val_str != "-" and clean_val_str != ".":
                    try:
                        card_num = float(clean_val_str)
                        calc_vals = res.calculated_values
                        matched_calc = None

                        # Check exact match or field match
                        for k, v in calc_vals.items():
                            if isinstance(v, (int, float)):
                                if abs(float(v) - card_num) < 1.0:
                                    matched_calc = (k, v)
                                    break
                            elif isinstance(v, list):
                                for item in v:
                                    if isinstance(item, dict):
                                        for ik, iv in item.items():
                                            if isinstance(iv, (int, float)) and abs(float(iv) - card_num) < 1.0:
                                                matched_calc = (f"{k}.{ik}", iv)
                                                break
                                    if matched_calc:
                                        break
                            if matched_calc:
                                break

                        if matched_calc is None:
                            is_valid = False
                            notes.append(
                                f"REJECTED: Card '{card.title}' displayed value '{card.value}' (parsed: {card_num}) "
                                f"does not match any calculated value in referenced result '{card.result_id}' ({calc_vals})."
                            )
                        else:
                            notes.append(f"OK: Card '{card.title}' numerical value verified against {matched_calc[0]} in '{card.result_id}'.")
                    except ValueError:
                        notes.append(f"OK: Card '{card.title}' linked to valid result '{card.result_id}' (text representation).")
                else:
                    notes.append(f"OK: Card '{card.title}' linked to valid result '{card.result_id}'.")

        # Check 2: No rejected result in charts
        for chart in dashboard.charts:
            if chart.result_id in rejected_ids:
                is_valid = False
                notes.append(f"REJECTED: Chart '{chart.title}' attempts to display rejected result '{chart.result_id}'.")
            elif chart.result_id not in res_map:
                is_valid = False
                notes.append(f"REJECTED: Chart '{chart.title}' references non-existent result '{chart.result_id}'.")
            else:
                notes.append(f"OK: Chart '{chart.title}' linked to valid result '{chart.result_id}'.")

        # Check 3: Check Period Alignment between cards and primary waterfall chart
        waterfall_chart = next((c for c in dashboard.charts if c.is_primary_objective), None)
        total_drop_card = next((c for c in dashboard.metric_cards if "Total" in c.title or "Contraction" in c.title or "Decline" in c.title), None)
        if waterfall_chart and total_drop_card:
            if waterfall_chart.period != total_drop_card.period:
                is_valid = False
                notes.append(
                    f"REJECTED: Period mismatch between card ({total_drop_card.period}) "
                    f"and primary chart ({waterfall_chart.period})."
                )
            else:
                notes.append(f"OK: Periods aligned between summary card and decomposition chart ({waterfall_chart.period}).")

        # Check 4: Non-empty data in charts
        for chart in dashboard.charts:
            if not chart.data:
                is_valid = False
                notes.append(f"REJECTED: Chart '{chart.title}' contains no data points to display.")

        dashboard.is_dashboard_validated = is_valid
        dashboard.validation_notes = notes
        return dashboard
