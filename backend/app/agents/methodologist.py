"""
Agent C: Methodologist & Researcher (Agente Metodólogo e Investigador).
Selects mathematically and empirically appropriate methods from a registered internal catalog,
documents assumptions, and produces a structured AnalysisPlan.
Supports Gemini LLM proposal with deterministic validation and fallback.
"""
from typing import Any, Dict, List, Optional
from app.agents.base import BaseAgent
from app.contracts import (
    AnalysisPlan,
    DataCatalog,
    DataQualityReport,
    ExternalResearchCitation,
    ObjectiveSpec,
    OperationSpec,
    RelationshipSpec,
)
from app.providers import ContextBuilder, LLMProvider, get_llm_provider

REGISTERED_METHODS_CATALOG: List[Dict[str, Any]] = [
    {
        "step_id": "OP_01_METRIC_SUMMARY",
        "operation_name": "Resumen descriptivo de métricas globales",
        "category": "metric_summary",
        "description": "Cálculo de volumen total de facturación bruta, neta, unidades y pedidos en el período cerrado.",
        "required_inputs": ["orders_analytical"],
        "parameters": {"metric_columns": ["net_sales", "gross_sales", "units", "discount_amount"]},
        "assumptions": ["Los montos corresponden a transacciones cerradas y completadas."],
        "pre_execution_validations": ["Verificar que las columnas numéricas no contengan valores NaN."],
        "depends_on": []
    },
    {
        "step_id": "OP_02_TIME_AGGREGATION",
        "operation_name": "Agregación temporal mensual",
        "category": "time_aggregation",
        "description": "Evolución mensual de facturación neta devengada y recuento de pedidos entre enero y mayo de 2026.",
        "required_inputs": ["orders_analytical"],
        "parameters": {"date_column": "order_date_clean", "freq": "M"},
        "assumptions": ["Se excluye el mes incompleto de junio para garantizar comparabilidad."],
        "pre_execution_validations": ["Verificar cobertura temporal continua."],
        "depends_on": ["OP_01_METRIC_SUMMARY"]
    },
    {
        "step_id": "OP_03_PERIOD_COMPARISON",
        "operation_name": "Comparación de variación entre períodos pico y valle",
        "category": "period_comparison",
        "description": "Medición de la variación absoluta y porcentual entre marzo 2026 (mes pico pre-caída) y mayo 2026.",
        "required_inputs": ["orders_analytical"],
        "parameters": {"baseline_period": "2026-03", "current_period": "2026-05", "metric": "net_sales"},
        "assumptions": ["Marzo representa el nivel operativo normal previo al inicio del declive."],
        "pre_execution_validations": ["Verificar que ambos meses tengan datos completos."],
        "depends_on": ["OP_02_TIME_AGGREGATION"]
    },
    {
        "step_id": "OP_04_DIMENSION_BREAKDOWN_CHANNEL",
        "operation_name": "Descomposición aditiva por canal comercial (Waterfall)",
        "category": "dimension_breakdown",
        "description": "Descomposición de la variación total en contribuciones mutuamente excluyentes por canal, reconciliadas al 100%.",
        "required_inputs": ["orders_analytical"],
        "parameters": {"dimension": "channel", "baseline_period": "2026-03", "current_period": "2026-05", "metric": "net_sales"},
        "assumptions": ["Los canales comerciales son exhaustivos y mutuamente excluyentes."],
        "pre_execution_validations": ["Comprobar que la suma de variaciones por canal iguala exactamente la variación total."],
        "depends_on": ["OP_03_PERIOD_COMPARISON"]
    },
    {
        "step_id": "OP_05_DIMENSION_BREAKDOWN_SEGMENT",
        "operation_name": "Descomposición aditiva por segmento de cliente",
        "category": "dimension_breakdown",
        "description": "Descomposición de la variación total por segmento de clientes para evaluar si la caída es corporativa o retail.",
        "required_inputs": ["orders_analytical"],
        "parameters": {"dimension": "segment", "baseline_period": "2026-03", "current_period": "2026-05", "metric": "net_sales"},
        "assumptions": ["Cada cliente pertenece a un único segmento en la dimensión."],
        "pre_execution_validations": ["Comprobar reconciliación total."],
        "depends_on": ["OP_03_PERIOD_COMPARISON"]
    },
    {
        "step_id": "OP_06_CUSTOMER_DYNAMICS",
        "operation_name": "Dinámica de clientes nuevos vs recurrentes",
        "category": "customer_dynamics",
        "description": "Evaluación de si la pérdida de ventas proviene de menor adquisición de clientes nuevos o caída en la recompra de recurrentes.",
        "required_inputs": ["orders_analytical"],
        "parameters": {"customer_id_col": "customer_id", "date_col": "order_date_clean"},
        "assumptions": ["Cliente nuevo definido formalmente como aquel cuya primera orden ocurre en el mes analizado."],
        "pre_execution_validations": ["Verificar consistencia de identificadores de clientes."],
        "depends_on": ["OP_02_TIME_AGGREGATION"]
    },
    {
        "step_id": "OP_07_BASELINE_FORECAST_EVAL",
        "operation_name": "Evaluación de viabilidad para pronóstico estadístico",
        "category": "baseline_forecast",
        "description": "Comprobación de condiciones estadísticas mínimas (historial >= 12 períodos) para modelado predictivo.",
        "required_inputs": ["orders_analytical"],
        "parameters": {"min_periods": 12},
        "assumptions": ["Un pronóstico sin suficiente historial introduce sobreajuste y falsas expectativas."],
        "pre_execution_validations": ["Revisar longitud de serie mensual."],
        "depends_on": ["OP_02_TIME_AGGREGATION"]
    }
]


class MethodologistAgent(BaseAgent):
    def __init__(self, override_provider: Optional[LLMProvider] = None):
        super().__init__(name="Agente Metodólogo", role="Selección de métodos analíticos registrados, definición de supuestos y diseño del plan")
        self._override_provider = override_provider

    def build_plan(
        self,
        project_id: str,
        run_id: str,
        objective: ObjectiveSpec,
        catalog: DataCatalog,
        dq_report: DataQualityReport,
        confirmed_relationships: Optional[List[RelationshipSpec]] = None
    ) -> AnalysisPlan:
        """
        Creates an AnalysisPlan tailored to decomposing sales variation.
        Uses Gemini to select operations if available, strictly validating against
        the registered catalog of implemented methods.
        """
        provider = get_llm_provider(self._override_provider)
        catalog_summary = ContextBuilder.build_catalog_summary(catalog)
        dq_summary = ContextBuilder.build_quality_summary(dq_report)
        relationships = confirmed_relationships or []

        if not provider.is_deterministic_fallback:
            try:
                proposed_plan = provider.propose_analysis_plan(
                    project_id=project_id,
                    run_id=run_id,
                    objective=objective,
                    catalog_summary=catalog_summary,
                    dq_summary=dq_summary,
                    confirmed_relationships=relationships,
                    registered_methods=REGISTERED_METHODS_CATALOG
                )

                # Deterministic Gatekeeper: Validate that all proposed operations are in the registered catalog
                # and do not reference non-existent columns
                registered_step_ids = {m["step_id"] for m in REGISTERED_METHODS_CATALOG}
                all_catalog_cols = set()
                if catalog:
                    for t in catalog.tables.values():
                        if isinstance(t.columns, dict):
                            for col_name in t.columns.keys():
                                all_catalog_cols.add(str(col_name).lower())
                        elif isinstance(t.columns, list):
                            for c in t.columns:
                                all_catalog_cols.add(getattr(c, "name", str(c)).lower())
                # Add known analytical derived columns
                all_catalog_cols.update([
                    "order_date_clean", "net_sales", "gross_sales", "units", 
                    "discount_amount", "channel", "region", "customer_name", "category_name"
                ])

                valid_ops = []
                for op in proposed_plan.operations:
                    if op.step_id not in registered_step_ids:
                        print(f"[MethodologistAgent] Operación no registrada rechazada: {op.step_id}")
                        continue

                    # Check for non-existent columns in parameters
                    params = op.parameters or {}
                    cols_to_check = []
                    if "metric_columns" in params and isinstance(params["metric_columns"], list):
                        cols_to_check.extend([str(c).lower() for c in params["metric_columns"]])
                    if "dimension" in params and isinstance(params["dimension"], str):
                        cols_to_check.append(str(params["dimension"]).lower())
                    if "metric" in params and isinstance(params["metric"], str):
                        cols_to_check.append(str(params["metric"]).lower())
                    if "date_column" in params and isinstance(params["date_column"], str):
                        cols_to_check.append(str(params["date_column"]).lower())

                    invalid_cols = [c for c in cols_to_check if c not in all_catalog_cols]
                    if invalid_cols:
                        print(f"[MethodologistAgent] Operación rechazada por referencia a columnas inexistentes: {invalid_cols}")
                        continue

                    valid_ops.append(op)

                if len(valid_ops) >= 3:
                    proposed_plan.operations = valid_ops
                    return proposed_plan
                else:
                    print("[MethodologistAgent] El plan propuesto por Gemini tenía operaciones insuficientes o no registradas. Aplicando plan determinista.")

            except Exception as e:
                print(f"[MethodologistAgent] Fallback activado tras error en propuesta de plan: {e}")

        # Deterministic fallback plan
        return self._build_deterministic_plan(project_id, run_id)

    def _build_deterministic_plan(self, project_id: str, run_id: str) -> AnalysisPlan:
        ops = []
        for m in REGISTERED_METHODS_CATALOG:
            ops.append(OperationSpec(
                step_id=m["step_id"],
                operation_name=m["operation_name"],
                category=m["category"],
                description=m["description"],
                required_inputs=m["required_inputs"],
                parameters=m["parameters"],
                assumptions=m["assumptions"],
                pre_execution_validations=m["pre_execution_validations"],
                depends_on=m["depends_on"]
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
            title="Plan de Diagnóstico y Descomposición de Caída de Ventas",
            rationale="Secuencia analítica diseñada para aislar el segmento y canal que explican la contracción observada, validando integridad matemática en cada paso.",
            operations=ops,
            excluded_methods=[
                "Regresión multivariada causal (datos no contienen variables de confusión exógenas como inflación o stockouts).",
                "ARIMA / Prophet complejo (serie histórica de 5 meses no cumple el umbral estadístico mínimo de estacionalidad)."
            ],
            citations=citations
        )
