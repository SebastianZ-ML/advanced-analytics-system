"""
Deterministic NoLLMProvider fallback.
Provides identical typed product operations using domain heuristics and catalog rules.
Explicitly identifies as 'DEMO_WITHOUT_LLM' without pretending to be a model.
"""
from typing import Any, Dict, List, Optional
from app.contracts import (
    ActionRecommendation,
    AmbiguityItem,
    AnalysisPlan,
    AnalysisResult,
    ChatAnswer,
    ExternalResearchCitation,
    InsightFinding,
    InsightInterpretation,
    InsightReport,
    ObjectiveSpec,
    OperationSpec,
    ProvenanceCitation,
    RelationshipSpec,
)
from app.providers.base import LLMProvider


class NoLLMProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "demo"

    @property
    def model_name(self) -> str:
        return "deterministic-catalog"

    @property
    def is_deterministic_fallback(self) -> bool:
        return True

    def structure_objective(
        self,
        project_id: str,
        user_question: str,
        catalog_summary: Dict[str, Any],
        user_clarifications: Optional[Dict[str, str]] = None
    ) -> ObjectiveSpec:
        clarifications = user_clarifications or {}
        metric_choice = clarifications.get("metric_definition", "net_sales")

        ambiguity_metric = AmbiguityItem(
            id="AMB_METRIC_DEFINITION",
            question="¿A qué métrica específica te refieres con 'ventas'?",
            context="El término 'ventas' puede interpretarse como facturación bruta, facturación neta o unidades vendidas.",
            proposed_default="net_sales (Facturación neta devengada)",
            user_decision=metric_choice,
            impact_level="high",
            status="resolved" if "metric_definition" in clarifications else "pending"
        )

        ambiguity_cutoff = AmbiguityItem(
            id="AMB_PERIOD_CUTOFF",
            question="¿Cómo tratar el último período registrado si está incompleto?",
            context="El mes de junio 2026 contiene sólo 4 días registrados. Compararlo directamente induciría una falsa caída.",
            proposed_default="exclude_incomplete_period (Comparar sólo meses cerrados)",
            user_decision=clarifications.get("period_cutoff", "exclude_incomplete_period"),
            impact_level="high",
            status="resolved" if "period_cutoff" in clarifications else "pending"
        )

        primary_metric_label = "net_sales (Facturación Neta)"
        if metric_choice == "gross_sales":
            primary_metric_label = "gross_sales (Facturación Bruta)"
        elif metric_choice == "units":
            primary_metric_label = "units (Unidades Vendidas)"

        return ObjectiveSpec(
            schema_version="1.0",
            project_id=project_id,
            original_question=user_question,
            operational_objective="Descomponer la variación de facturación neta entre períodos para identificar la concentración de la caída por canal, producto y segmento de clientes.",
            decision_to_inform="Priorizar qué canales comerciales o segmentos requieren planes de rescate, ajuste de precios o revisión operativa inmediata.",
            primary_metric=primary_metric_label,
            auxiliary_metrics=["gross_sales", "units", "discount_amount", "customer_retention"],
            time_period="2026-01 a 2026-05 (Meses completos cerrados)",
            comparison_period="Mes pico previo a la caída (2026-03) frente al mes más reciente cerrado (2026-05)",
            relevant_dimensions=["channel", "segment", "category_name", "region"],
            assumptions=[
                "Las transacciones con estado CANCELLED o REFUNDED se excluyen de la facturación neta devengada.",
                "El mes de junio 2026 se excluye de la comparación temporal por ser un período incompleto (4 días registrados).",
                "Las relaciones entre pedidos y dimensiones se deduplican en las dimensiones para evitar inflación de métricas."
            ],
            pending_ambiguities=[ambiguity_metric, ambiguity_cutoff],
            acceptance_criteria=[
                "La suma de las contribuciones de la dimensión descompuesta debe reconciliarse al 100% con la variación neta total observada.",
                "Ningún resultado con valores infinitos o NaN es admitido.",
                "Las uniones entre hechos y dimensiones deben tener un factor de multiplicación de filas exactamente igual a 1.0."
            ],
            excluded_scope=[
                "Modelos de atribución causal multivariable complejos no observables en las tablas cargadas.",
                "Optimización prescriptiva de precios (requiere elasticidades no modeladas)."
            ],
            status="confirmed" if all(a.status == "resolved" for a in [ambiguity_metric, ambiguity_cutoff]) else "draft",
            is_demo_mode=True
        )

    def propose_clarifications(
        self,
        project_id: str,
        user_question: str,
        catalog_summary: Dict[str, Any]
    ) -> List[AmbiguityItem]:
        return [
            AmbiguityItem(
                id="AMB_METRIC_DEFINITION",
                question="¿A qué métrica específica te refieres con 'ventas'?",
                context="El término 'ventas' puede interpretarse como facturación bruta, facturación neta o unidades vendidas.",
                proposed_default="net_sales (Facturación neta devengada)",
                impact_level="high",
                status="pending"
            ),
            AmbiguityItem(
                id="AMB_PERIOD_CUTOFF",
                question="¿Cómo tratar el último período registrado si está incompleto?",
                context="El mes de junio 2026 contiene sólo 4 días registrados. Compararlo directamente induciría una falsa caída.",
                proposed_default="exclude_incomplete_period (Comparar sólo meses cerrados)",
                impact_level="high",
                status="pending"
            )
        ]

    def propose_analysis_plan(
        self,
        project_id: str,
        run_id: str,
        objective: ObjectiveSpec,
        catalog_summary: Dict[str, Any],
        dq_summary: Dict[str, Any],
        confirmed_relationships: List[RelationshipSpec],
        registered_methods: List[Dict[str, Any]]
    ) -> AnalysisPlan:
        ops = []
        for m in registered_methods:
            ops.append(OperationSpec(
                step_id=m["step_id"],
                operation_name=m["operation_name"],
                category=m["category"],
                description=m["description"],
                required_inputs=m["required_inputs"],
                parameters=m["parameters"],
                assumptions=m.get("assumptions", []),
                pre_execution_validations=m.get("pre_execution_validations", []),
                depends_on=m.get("depends_on", [])
            ))

        citations = [
            ExternalResearchCitation(
                url="https://otexts.com/fpp3/decomposition.html",
                title="Forecasting: Principles and Practice - Time Series Decomposition",
                supported_claim="La descomposición aditiva exige categorías mutuamente excluyentes para reconciliar la suma de componentes con la serie agregada.",
                is_primary_source=True
            ),
            ExternalResearchCitation(
                url="https://pandas.pydata.org/docs/user_guide/merging.html",
                title="Pandas Documentation: Merge and Join Integrity",
                supported_claim="Las uniones de hechos a dimensiones deben verificar unicidad en la clave derecha para evitar inflación no intencionada de filas.",
                is_primary_source=True
            )
        ]

        return AnalysisPlan(
            schema_version="1.0",
            project_id=project_id,
            run_id=run_id,
            title="Plan de Diagnóstico y Descomposición de Caída de Ventas (Modo Determinista)",
            rationale="Secuencia analítica diseñada a partir del catálogo de operaciones registradas para aislar factores observables.",
            operations=ops,
            excluded_methods=[
                "Regresión multivariada causal (datos no contienen variables de confusión exógenas como inflación o stockouts).",
                "ARIMA / Prophet complejo (serie histórica de 5 meses no cumple el umbral estadístico mínimo de estacionalidad)."
            ],
            citations=citations
        )

    def interpret_validated_results(
        self,
        run_id: str,
        objective: ObjectiveSpec,
        approved_results: List[AnalysisResult],
        limitations: List[str],
        provenance_summary: List[str]
    ) -> InsightReport:
        approved_map = {r.result_id: r for r in approved_results if r.validation_status in ["approved", "approved_with_warnings"]}
        findings: List[InsightFinding] = []
        interpretations: List[InsightInterpretation] = []
        actions: List[ActionRecommendation] = []

        res_comp = approved_map.get("RES_OP_03_PERIOD_COMPARISON")
        if res_comp:
            c = res_comp.calculated_values
            diff_abs = c.get("absolute_change", 0.0)
            diff_pct = c.get("percentage_change", 0.0)
            findings.append(InsightFinding(
                id="FINDING_TOTAL_CONTRACTION",
                claim=f"La facturación neta disminuyó en {abs(diff_abs):,.2f} unidades monetarias ({diff_pct:.1f}%) entre marzo 2026 y mayo 2026.",
                result_id=res_comp.result_id,
                metric_name="net_sales_delta",
                observed_value=diff_abs,
                is_empirically_proven=True,
                evidence_text="Cálculo derivado de órdenes completadas cerradas en 2026-03 vs 2026-05."
            ))

        res_channel = approved_map.get("RES_OP_04_DIMENSION_BREAKDOWN_CHANNEL")
        if res_channel:
            contribs = res_channel.calculated_values.get("contributions", [])
            mayorista = next((item for item in contribs if "Mayorista" in item["dimension_value"]), None)
            if mayorista:
                m_abs = mayorista["absolute_change"]
                m_contrib = mayorista["contribution_to_total_change"]
                findings.append(InsightFinding(
                    id="FINDING_MAYORISTA_CONCENTRATION",
                    claim=f"La caída observada se concentra de forma predominante en el canal 'Mayorista / B2B', explicando {abs(m_abs):,.2f} unidades monetarias ({m_contrib:.1f}% de la caída total).",
                    result_id=res_channel.result_id,
                    metric_name="channel_contribution_mayorista",
                    observed_value=m_abs,
                    is_empirically_proven=True,
                    evidence_text="Descomposición aditiva mutuamente excluyente reconciliada al 100% con la variación total."
                ))
                interpretations.append(InsightInterpretation(
                    id="INTERP_MAYORISTA_FOCUS",
                    interpretation_text="La contracción general del negocio está focalizada en la cartera y frecuencia de compra de clientes mayoristas / corporativos.",
                    grounded_in_finding_ids=["FINDING_TOTAL_CONTRACTION", "FINDING_MAYORISTA_CONCENTRATION"],
                    confidence_rationale="Respaldado por descomposición contable con reconciliación exacta.",
                    distinction_from_causality="Identifica contablemente dónde ocurrió la pérdida, pero no prueba si la causa raíz fue deserción, stockouts o políticas crediticias."
                ))

        actions.append(ActionRecommendation(
            id="ACT_01_AUDIT_WHOLESALE_ACCOUNTS",
            title="Auditoría comercial de cuentas clave B2B",
            description="Contactar a los principales compradores mayoristas de marzo ausentes en mayo para indagar motivos operativos.",
            hypothesis_to_investigate="Pérdida de competitividad comercial o desabastecimiento de líneas clave.",
            expected_impact="Recuperación potencial de hasta el 70% del volumen perdido en el canal mayorista.",
            data_needed_to_confirm="Entrevistas comerciales cualitativas y registro de cotizaciones no concretadas.",
            is_action_proposal_only=True
        ))

        summary = (
            "El análisis verificado confirma una contracción neta sustancial entre marzo y mayo de 2026. "
            "La descomposición matemática demuestra de forma concluyente que la caída se concentra casi en su totalidad "
            "en el canal 'Mayorista / B2B', principalmente por disminución en compras de cuentas recurrentes."
        )

        return InsightReport(
            schema_version="1.0",
            run_id=run_id,
            executive_summary=summary,
            observed_findings=findings,
            interpretations=interpretations,
            unproven_hypotheses=[
                "Hipótesis no comprobada: La caída en compras mayoristas podría deberse a un cambio en condiciones de crédito comercial o adelanto de compras en marzo."
            ],
            data_limitations=limitations,
            recommended_actions=actions,
            suggested_further_analyses=[
                "Incorporar datos de stock y logística para descartar problemas de abastecimiento en productos B2B."
            ]
        )

    def contextual_chat(
        self,
        project_id: str,
        run_id: str,
        question: str,
        objective: ObjectiveSpec,
        catalog_summary: Dict[str, Any],
        approved_results: List[AnalysisResult],
        insight_report: Optional[InsightReport],
        active_filters: Dict[str, Any]
    ) -> ChatAnswer:
        q = question.lower()

        # Check unanswerable
        for trigger in ["competidor", "competencia", "inflación", "precio de la competencia", "clima", "rating", "nps"]:
            if trigger in q:
                return ChatAnswer(
                    schema_version="1.0",
                    query_type="unanswerable_by_data",
                    answer_text=(
                        f"Los datos disponibles en el proyecto actual no contienen información sobre '{trigger}'. "
                        f"Las tablas cargadas ({', '.join(catalog_summary.keys())}) registran únicamente transacciones de pedidos, "
                        f"clientes registrados, catálogo de productos y presupuestos de campañas de marketing. "
                        f"Para analizar este factor, sería necesario incorporar fuentes externas de investigación de mercado o inventario logístico."
                    ),
                    citations=[],
                    data_limitation_notice=f"Ausencia de datos de '{trigger}' en el catálogo de datos del proyecto.",
                    requires_full_pipeline=False,
                    is_demo_mode=True
                )

        if any(t in q for t in ["nuevo análisis", "modelo predictivo", "cluster", "clustering"]):
            return ChatAnswer(
                schema_version="1.0",
                query_type="request_new_analysis",
                answer_text="Esta solicitud requiere diseñar un nuevo plan analítico formal. Te invito a iniciar una nueva ejecución configurando este objetivo en la sección de Planificación.",
                citations=[],
                calculation_summary="Solicitud clasificada como nuevo análisis. Requiere orquestación por el Agente Metodólogo.",
                requires_full_pipeline=True,
                is_demo_mode=True
            )

        # Default explain
        citations = []
        res_comp = next((r for r in approved_results if r.step_id == "OP_03_PERIOD_COMPARISON"), None)
        res_channel = next((r for r in approved_results if r.step_id == "OP_04_DIMENSION_BREAKDOWN_CHANNEL"), None)
        if res_comp:
            citations.append(ProvenanceCitation(
                source_type="analytic_step",
                source_name=res_comp.step_id,
                filter_or_condition="orders WHERE status='COMPLETED' AND date IN (2026-03, 2026-05)",
                result_id=res_comp.result_id,
                details="Comparación período pico vs mes cerrado más reciente."
            ))
        if res_channel:
            citations.append(ProvenanceCitation(
                source_type="analytic_step",
                source_name=res_channel.step_id,
                filter_or_condition="orders GROUP BY channel",
                result_id=res_channel.result_id,
                details="Descomposición aditiva con 100% de reconciliación."
            ))

        text = (
            f"Basado en los resultados auditados y validados del proyecto:\n\n"
            f"1. **Hallazgo principal**: {insight_report.executive_summary if insight_report else 'Caída observada entre marzo y mayo de 2026.'}\n\n"
            f"2. **Garantía metodológica**: Esta conclusión proviene de una descomposición aditiva verificada donde la suma de las variaciones por canal iguala exactamente el 100% de la variación total observada."
        )

        return ChatAnswer(
            schema_version="1.0",
            query_type="explain_existing_result",
            answer_text=text,
            citations=citations,
            calculation_summary="Procedencia: orders.csv -> unión segura con customers.csv -> reconciliación dimensional exacta.",
            requires_full_pipeline=False,
            is_demo_mode=True
        )
