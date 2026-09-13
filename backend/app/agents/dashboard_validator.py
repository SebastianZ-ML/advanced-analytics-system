"""
Agent I: Dashboard Validator.
Performs independent checks on the assembled DashboardSpec:
- Guarantees numbers shown in cards match underlying results.
- Assures no rejected results are exposed.
- Confirms period alignment between cards and charts.
"""
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
        Updates validation_notes and is_dashboard_validated flag.
        """
        notes: List[str] = []
        is_valid = True
        res_map = {r.result_id: r for r in results}
        rejected_ids = set(validation_report.failed_results)

        # Check 1: No rejected result in cards
        for card in dashboard.metric_cards:
            if card.result_id in rejected_ids:
                is_valid = False
                notes.append(f"REJECTED: Card '{card.title}' attempts to display rejected result '{card.result_id}'.")
            elif card.result_id not in res_map:
                is_valid = False
                notes.append(f"REJECTED: Card '{card.title}' references non-existent result '{card.result_id}'.")
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
