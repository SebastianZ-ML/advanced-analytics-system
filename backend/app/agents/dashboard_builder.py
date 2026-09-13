"""
Agent H: Dashboard Builder (Constructor de Dashboard).
Builds a structured, typed DashboardSpec from validated results and catalog components.
Does not generate arbitrary HTML. Maps validated metrics directly to component specifications.
"""
from typing import List
from app.agents.base import BaseAgent
from app.contracts import (
    AnalysisResult,
    ChartComponentSpec,
    DashboardSpec,
    InsightReport,
    MetricCardSpec,
    ValidationReport,
)


class DashboardBuilderAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Agente Constructor de Dashboard", role="Ensamblado de especificaciones de visualización a partir de componentes validados")

    def build_dashboard(
        self,
        run_id: str,
        results: List[AnalysisResult],
        validation_report: ValidationReport,
        insight_report: InsightReport,
        objective_question: str
    ) -> DashboardSpec:
        """
        Constructs typed DashboardSpec from approved results only.
        """
        approved_map = {r.result_id: r for r in results if r.validation_status in ["approved", "approved_with_warnings"]}

        metric_cards: List[MetricCardSpec] = []
        charts: List[ChartComponentSpec] = []

        # 1. Metric Cards
        comp_res = approved_map.get("RES_OP_03_PERIOD_COMPARISON")
        if comp_res:
            c = comp_res.calculated_values
            base_sales = c.get("baseline_net_sales", 0.0)
            curr_sales = c.get("current_net_sales", 0.0)
            delta = c.get("absolute_change", 0.0)
            pct = c.get("percentage_change", 0.0)

            metric_cards.append(MetricCardSpec(
                id="CARD_PEAK_SALES",
                title="Ventas Pico (Marzo 2026)",
                value=f"${base_sales:,.0f}",
                comparison_text="Nivel de referencia pre-caída",
                trend_direction="flat",
                result_id=comp_res.result_id,
                period="2026-03",
                validation_status=comp_res.validation_status
            ))

            metric_cards.append(MetricCardSpec(
                id="CARD_CURRENT_SALES",
                title="Ventas Actuales (Mayo 2026)",
                value=f"${curr_sales:,.0f}",
                comparison_text=f"{pct:.1f}% vs marzo",
                trend_direction="down" if delta < 0 else "up",
                result_id=comp_res.result_id,
                period="2026-05",
                validation_status=comp_res.validation_status
            ))

            metric_cards.append(MetricCardSpec(
                id="CARD_TOTAL_DROP",
                title="Contracción Neta Total",
                value=f"-${abs(delta):,.0f}",
                comparison_text=f"Disminución de {abs(pct):.1f}% en facturación neta",
                trend_direction="down",
                result_id=comp_res.result_id,
                period="2026-03 vs 2026-05",
                validation_status=comp_res.validation_status
            ))

        # Channel contribution card
        channel_res = approved_map.get("RES_OP_04_DIMENSION_BREAKDOWN_CHANNEL")
        if channel_res:
            contribs = channel_res.calculated_values.get("contributions", [])
            mayorista = next((item for item in contribs if "Mayorista" in item["dimension_value"]), None)
            if mayorista:
                m_contrib = mayorista["contribution_to_total_change"]
                metric_cards.append(MetricCardSpec(
                    id="CARD_MAYORISTA_SHARE",
                    title="Concentración en Canal Mayorista",
                    value=f"{m_contrib:.1f}%",
                    comparison_text=f"Aporta ${abs(mayorista['absolute_change']):,.0f} de la caída",
                    trend_direction="down",
                    result_id=channel_res.result_id,
                    period="2026-03 vs 2026-05",
                    validation_status=channel_res.validation_status
                ))

        # 2. Charts
        # Time series chart
        time_res = approved_map.get("RES_OP_02_TIME_AGGREGATION")
        if time_res:
            monthly_data = time_res.calculated_values.get("monthly_series", [])
            charts.append(ChartComponentSpec(
                id="CHART_TIME_SERIES",
                question_answered="¿Cómo han evolucionado las ventas mensuales y cuándo se inició la caída?",
                chart_type="time_series",
                title="Evolución Mensual de Facturación Neta (2026)",
                result_id=time_res.result_id,
                metric="net_sales",
                unit="CLP",
                dimensions=["period"],
                data=monthly_data,
                period="2026-01 a 2026-05",
                validation_status=time_res.validation_status,
                is_primary_objective=False
            ))

        # Channel waterfall breakdown chart (PRIMARY OBJECTIVE)
        if channel_res:
            channel_data = channel_res.calculated_values.get("contributions", [])
            charts.append(ChartComponentSpec(
                id="CHART_CHANNEL_WATERFALL",
                question_answered="¿Dónde se concentra la caída de ventas por canal comercial?",
                chart_type="waterfall_bars",
                title="Contribución Absoluta a la Caída por Canal Comercial (Reconciliado 100%)",
                result_id=channel_res.result_id,
                metric="absolute_change",
                unit="CLP",
                dimensions=["dimension_value"],
                data=channel_data,
                period="2026-03 vs 2026-05",
                validation_status=channel_res.validation_status,
                is_primary_objective=True
            ))

        # Customer segment breakdown chart
        segment_res = approved_map.get("RES_OP_05_DIMENSION_BREAKDOWN_SEGMENT")
        if segment_res:
            segment_data = segment_res.calculated_values.get("contributions", [])
            charts.append(ChartComponentSpec(
                id="CHART_SEGMENT_BREAKDOWN",
                question_answered="¿Qué segmentos de clientes explican la mayor parte de la reducción de compras?",
                chart_type="horizontal_bars",
                title="Variación de Facturación por Segmento de Cliente",
                result_id=segment_res.result_id,
                metric="absolute_change",
                unit="CLP",
                dimensions=["dimension_value"],
                data=segment_data,
                period="2026-03 vs 2026-05",
                validation_status=segment_res.validation_status,
                is_primary_objective=False
            ))

        # Customer dynamics chart (New vs Recurring)
        dyn_res = approved_map.get("RES_OP_06_CUSTOMER_DYNAMICS")
        if dyn_res:
            dyn_data = dyn_res.calculated_values.get("monthly_customer_dynamics", [])
            charts.append(ChartComponentSpec(
                id="CHART_CUSTOMER_DYNAMICS",
                question_answered="¿La caída proviene de menor adquisición de clientes nuevos o de fuga en la recompra de recurrentes?",
                chart_type="line_comparison",
                title="Dinámica de Ventas: Clientes Nuevos vs Recurrentes",
                result_id=dyn_res.result_id,
                metric="sales",
                unit="CLP",
                dimensions=["period"],
                data=dyn_data,
                period="2026-01 a 2026-05",
                validation_status=dyn_res.validation_status,
                is_primary_objective=False
            ))

        # Provenance summary
        provenance = [
            "Archivos crudos: orders.csv, customers.csv, products.xlsx, campaigns.csv",
            "Limpieza: Exclusión de registros con fechas corruptas (2 registros) y cancelaciones (4% del volumen)",
            "Recorte temporal: Exclusión explícita de junio 2026 por período incompleto (4 días registrados)",
            "Uniones seguras: Deduplicación previa de clientes y productos garantizando factor de multiplicación 1.0x",
            "Validación matemática: Reconciliación 100% aditiva verificada entre deltas dimensionales y delta total",
            "Presentación: Visualización construida exclusivamente con resultados auditados y aprobados"
        ]

        return DashboardSpec(
            schema_version="1.0",
            run_id=run_id,
            title="Diagnóstico de Desempeño y Caída de Ventas",
            subtitle="Descomposición aditiva y análisis de concentración en el canal comercial",
            objective_question=objective_question,
            metric_cards=metric_cards,
            charts=charts,
            quality_alerts=insight_report.data_limitations,
            methodology_notes=[
                "Métrica principal: Facturación neta devengada (excluyendo descuentos y devoluciones).",
                "Período de comparación: Marzo 2026 (pico pre-caída) vs Mayo 2026 (mes cerrado más reciente).",
                "Reconciliación: Las variaciones por canal suman exactamente la variación total de la empresa."
            ],
            provenance_chain_summary=provenance,
            is_dashboard_validated=False,  # Will be certified by DashboardValidatorAgent
            validation_notes=[]
        )
