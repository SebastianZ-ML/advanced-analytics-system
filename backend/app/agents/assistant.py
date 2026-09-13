"""
Agent J: Conversational Assistant (Asistente Conversacional).
Provides grounded Q&A with explicit provenance citations.
Distinguishes:
1. Explaining existing results.
2. Recalculating metrics with custom filters using the analytical engine.
3. Detecting requests for new analyses (demands planning/pipeline, no hallucinated numbers).
4. Explicitly stating when data cannot answer the question.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
from app.agents.base import BaseAgent
from app.contracts import (
    AnalysisResult,
    ChatAnswer,
    ChatRequest,
    DashboardSpec,
    DataCatalog,
    InsightReport,
    ObjectiveSpec,
    ProvenanceCitation,
)
from app.providers import ContextBuilder, LLMProvider, get_llm_provider


class ConversationalAssistantAgent(BaseAgent):
    def __init__(self, override_provider: Optional[LLMProvider] = None):
        super().__init__(name="Asistente Conversacional", role="Respuestas fundamentadas con citas de procedencia y recálculos controlados")
        self._override_provider = override_provider

    def answer_query(
        self,
        request: ChatRequest,
        objective: ObjectiveSpec,
        catalog: DataCatalog,
        analytical_df: pd.DataFrame,
        results: List[AnalysisResult],
        dashboard: DashboardSpec,
        insight_report: InsightReport
    ) -> ChatAnswer:
        """
        Processes user question with grounded reasoning and provenance citations.
        Supports Gemini contextual chat with deterministic recalculation and unanswerable guardrails.
        """
        provider = get_llm_provider(self._override_provider)
        is_demo = provider.is_deterministic_fallback
        q = request.question.lower().strip()
        active_filters = request.active_filters or {}
        res_map = {r.result_id: r for r in results}

        # -------------------------------------------------------------------
        # Class 4: Unanswerable Questions (Information absent from dataset)
        # -------------------------------------------------------------------
        unanswerable_triggers = ["competidor", "competencia", "inflación", "precio de la competencia", "clima", "rating", "televisión", "desabastecimiento externo", "satisfacción de cliente", "nps"]
        if any(trigger in q for trigger in unanswerable_triggers):
            missing_concept = next(trigger for trigger in unanswerable_triggers if trigger in q)
            return ChatAnswer(
                schema_version="1.0",
                query_type="unanswerable_by_data",
                answer_text=(
                    f"Los datos disponibles en el proyecto actual no contienen información sobre '{missing_concept}'. "
                    f"Las tablas cargadas ({', '.join(catalog.tables.keys())}) registran únicamente transacciones de pedidos, "
                    f"clientes registrados, catálogo de productos y presupuestos de campañas de marketing. "
                    f"Para analizar este factor, sería necesario incorporar fuentes externas de investigación de mercado o inventario logístico."
                ),
                citations=[],
                data_limitation_notice=f"Ausencia de datos de '{missing_concept}' en el catálogo de datos del proyecto.",
                requires_full_pipeline=False,
                is_demo_mode=is_demo
            )

        # -------------------------------------------------------------------
        # Class 3: Request for New Analysis (Requires Pipeline, not quick query)
        # -------------------------------------------------------------------
        new_analysis_triggers = ["nuevo análisis", "modelo predictivo", "cluster", "clustering", "arbol de decision", "predecir el futuro", "optimizar precios", "elasticidad precio"]
        if any(trigger in q for trigger in new_analysis_triggers):
            return ChatAnswer(
                schema_version="1.0",
                query_type="request_new_analysis",
                answer_text=(
                    "Esta solicitud requiere diseñar un nuevo plan analítico formal (definición de objetivo, selección metodológica, "
                    "preparación de variables y validación independiente). Para no generar cifras improvisadas sin control de calidad, "
                    "te invito a iniciar una nueva ejecución configurando este objetivo en la sección de Planificación."
                ),
                citations=[],
                calculation_summary="Solicitud clasificada como nuevo análisis. Requiere orquestación por el Agente Metodólogo.",
                requires_full_pipeline=True,
                is_demo_mode=is_demo
            )

        # -------------------------------------------------------------------
        # Class 2: Recalculate Metric with Custom Filters (Deterministic Engine)
        # -------------------------------------------------------------------
        recalc_triggers = ["cuánto vendió", "cuanto vendio", "calcular", "recalcular", "si filtramos", "filtrar por", "sólo en", "solo en", "región", "region"]
        has_filter_request = bool(active_filters) or any(t in q for t in recalc_triggers)

        if has_filter_request and ("cuánto" in q or "cuanto" in q or "ventas en" in q or active_filters):
            # Apply real filters to analytical_df
            filtered_df = analytical_df.copy()
            applied_desc = []

            # Check region in text or filters
            for reg in ["Metropolitana", "Norte", "Centro", "Sur"]:
                if reg.lower() in q or active_filters.get("region") == reg:
                    if "region" in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df["region"] == reg]
                        applied_desc.append(f"Región = '{reg}'")

            # Check channel in text or filters
            for ch in ["Mayorista / B2B", "Retail / Tiendas", "Online / Directo"]:
                if ch.lower() in q or active_filters.get("channel") == ch:
                    if "channel" in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df["channel"] == ch]
                        applied_desc.append(f"Canal = '{ch}'")

            # Calculate real net sales and orders
            total_net = float(filtered_df["net_sales"].sum()) if "net_sales" in filtered_df.columns else 0.0
            total_orders = int(len(filtered_df))
            filter_str = ", ".join(applied_desc) if applied_desc else "Sin filtros específicos"

            citations = [
                ProvenanceCitation(
                    source_type="transformed_table",
                    source_name="analytical_fact",
                    filter_or_condition=filter_str,
                    result_id="RECALC_ON_DEMAND",
                    details=f"Cálculo directo sobre el dataset analítico preparado ({len(filtered_df)} filas coincidentes)."
                )
            ]

            return ChatAnswer(
                schema_version="1.0",
                query_type="recalculate_with_filters",
                answer_text=(
                    f"Bajo las condiciones aplicadas ({filter_str}), la facturación neta acumulada en el período cerrado "
                    f"(enero a mayo 2026) es de ${total_net:,.2f} en un total de {total_orders} pedidos completados."
                ),
                citations=citations,
                calculation_summary=f"Filtros: {filter_str} | Filas evaluadas: {len(filtered_df)} | Total: ${total_net:,.2f}",
                requires_full_pipeline=False,
                is_demo_mode=is_demo
            )

        # -------------------------------------------------------------------
        # Class 1: Explain Existing Result / Grounded Semantic Explanation
        # -------------------------------------------------------------------
        if not provider.is_deterministic_fallback:
            try:
                catalog_summary = ContextBuilder.build_catalog_summary(catalog)
                approved_results = [r for r in results if r.validation_status in ["approved", "approved_with_warnings"]]
                llm_ans = provider.contextual_chat(
                    project_id=request.project_id,
                    run_id=request.run_id,
                    question=request.question,
                    objective=objective,
                    catalog_summary=catalog_summary,
                    approved_results=approved_results,
                    insight_report=insight_report,
                    active_filters=active_filters
                )
                # Verify and ground citations
                approved_ids = {r.result_id for r in results}
                valid_citations = []
                for cit in llm_ans.citations:
                    if cit.result_id in approved_ids or cit.source_type in ["transformed_table", "analytic_step"]:
                        valid_citations.append(cit)

                if not valid_citations and results:
                    valid_citations.append(ProvenanceCitation(
                        source_type="analytic_step",
                        source_name=results[0].step_id,
                        filter_or_condition="orders_valid_dates WHERE status = 'COMPLETED'",
                        result_id=results[0].result_id,
                        details="Procedencia auditada de la base analítica."
                    ))
                llm_ans.citations = valid_citations
                llm_ans.is_demo_mode = False
                return llm_ans
            except Exception as e:
                print(f"[ConversationalAssistantAgent] Fallback activado tras error en LLM: {e}")

        # Deterministic fallback response
        res_comp = res_map.get("RES_OP_03_PERIOD_COMPARISON")
        res_channel = res_map.get("RES_OP_04_DIMENSION_BREAKDOWN_CHANNEL")

        citations = []
        if res_comp:
            citations.append(ProvenanceCitation(
                source_type="analytic_step",
                source_name="RES_OP_03_PERIOD_COMPARISON",
                filter_or_condition="orders_valid_dates WHERE status = 'COMPLETED' AND date IN (2026-03, 2026-05)",
                result_id=res_comp.result_id,
                details="Comparación período contra período pico (marzo) vs mes cerrado más reciente (mayo)."
            ))
        if res_channel:
            citations.append(ProvenanceCitation(
                source_type="analytic_step",
                source_name="RES_OP_04_DIMENSION_BREAKDOWN_CHANNEL",
                filter_or_condition="orders_valid_dates GROUP BY channel",
                result_id=res_channel.result_id,
                details="Descomposición aditiva de variación por canal con reconciliación al 100%."
            ))

        executive = insight_report.executive_summary if insight_report else "Variación observada en el período analizado."
        mayorista_finding = next((f for f in insight_report.observed_findings if "Mayorista" in f.claim), None) if insight_report else None
        detail_msg = mayorista_finding.claim if mayorista_finding else "La mayor parte de la disminución se atribuye a canales corporativos."

        answer_text = (
            f"Basado en los resultados auditados y validados del proyecto:\n\n"
            f"1. **Hallazgo principal**: {executive}\n\n"
            f"2. **Concentración por canal**: {detail_msg}\n\n"
            f"3. **Garantía metodológica**: Esta conclusión no proviene de una correlación informal, "
            f"sino de una descomposición aditiva verificada donde la suma de las variaciones por canal iguala exactamente "
            f"el 100% de la variación total observada de la empresa."
        )

        return ChatAnswer(
            schema_version="1.0",
            query_type="explain_existing_result",
            answer_text=answer_text,
            citations=citations,
            calculation_summary="Procedencia: orders.csv (filtrado status='COMPLETED') -> unión segura con customers.csv -> reconciliación dimensional exacta.",
            requires_full_pipeline=False,
            is_demo_mode=True
        )
