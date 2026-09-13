"""
Agent A: Organizer (Agente Organizador).
Translates business questions into formal operational objectives (ObjectiveSpec).
Supports Gemini LLM provider with graceful fallback to deterministic logic.
"""
from typing import Any, Dict, List, Optional
from app.agents.base import BaseAgent
from app.contracts import AmbiguityItem, DataCatalog, ObjectiveSpec
from app.providers import ContextBuilder, LLMProvider, get_llm_provider


class OrganizerAgent(BaseAgent):
    def __init__(self, override_provider: Optional[LLMProvider] = None):
        super().__init__(name="Agente Organizador", role="Estructuración de objetivos analíticos y resolución de ambigüedades")
        self._override_provider = override_provider

    def process_objective(
        self,
        project_id: str,
        user_question: str,
        catalog: Optional[DataCatalog] = None,
        user_clarifications: Optional[Dict[str, str]] = None
    ) -> ObjectiveSpec:
        """
        Produce a formal ObjectiveSpec using Gemini when available,
        or deterministic catalog heuristics in DEMO_WITHOUT_LLM mode.
        """
        provider = get_llm_provider(self._override_provider)
        catalog_summary = ContextBuilder.build_catalog_summary(catalog) if catalog else {}

        if not provider.is_deterministic_fallback:
            try:
                spec = provider.structure_objective(
                    project_id=project_id,
                    user_question=user_question,
                    catalog_summary=catalog_summary,
                    user_clarifications=user_clarifications
                )
                return spec
            except Exception as e:
                # Sanitized fallback with transparent indication
                print(f"[OrganizerAgent] Fallback activado tras error en proveedor LLM: {e}")

        # Deterministic fallback (DEMO_WITHOUT_LLM)
        clarifications = user_clarifications or {}
        metric_choice = clarifications.get("metric_definition", "net_sales")

        ambiguity_metric = AmbiguityItem(
            id="AMB_METRIC_DEFINITION",
            question="¿A qué métrica específica te refieres con 'ventas'?",
            context="El término 'ventas' puede interpretarse como facturación bruta, facturación neta (descontando descuentos y devoluciones) o volumen de unidades vendidas.",
            proposed_default="net_sales (Facturación neta devengada)",
            user_decision=metric_choice,
            impact_level="high",
            status="resolved" if "metric_definition" in clarifications else "pending"
        )

        ambiguity_cutoff = AmbiguityItem(
            id="AMB_PERIOD_CUTOFF",
            question="¿Cómo tratar el último período registrado si está incompleto?",
            context="Los datos registrados contienen un último mes parcial. Incluirlo directamente distorsionaría la comparación temporal.",
            proposed_default="exclude_incomplete_period (Comparar solo meses completos cerrados)",
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
