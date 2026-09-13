"""
Real Gemini LLM provider implementation using the official google-genai SDK.
Enforces structured outputs, schema validation, transient retries,
timeouts, and sanitized audit logging.
"""
import json
import time
import uuid
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

from app.config import settings
from app.contracts import (
    AmbiguityItem,
    AnalysisPlan,
    AnalysisResult,
    ChatAnswer,
    InsightReport,
    ObjectiveSpec,
    RelationshipSpec,
)
from app.providers.base import (
    LLMAuthenticationError,
    LLMProvider,
    LLMProviderError,
    LLMTimeoutError,
    LLMValidationError,
)
from app.providers.context_builder import ContextBuilder
from app.storage.db import DatabaseService


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None
    ):
        self._api_key = api_key or settings.gemini_api_key
        self._model_name = model_name or settings.gemini_model
        self._timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.gemini_timeout_seconds
        self._max_retries = max_retries if max_retries is not None else settings.gemini_max_retries

        if not self._api_key:
            raise LLMAuthenticationError("GEMINI_API_KEY no está configurada.")

        self._client = genai.Client(api_key=self._api_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_deterministic_fallback(self) -> bool:
        return False

    def _execute_with_retry(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Any] = None,
        caller_agent: str = "UnknownAgent",
        project_id: str = "default",
        run_id: Optional[str] = None,
        context_artifacts: Optional[List[str]] = None
    ) -> str:
        """
        Executes Gemini call with timeout, retries on transient errors, and audit logging.
        """
        interaction_id = f"llm_{uuid.uuid4().hex[:10]}"
        start_time = time.time()
        last_exception: Optional[Exception] = None

        config_args = {
            "response_mime_type": "application/json"
        }
        if system_instruction:
            config_args["system_instruction"] = system_instruction
        if response_schema:
            config_args["response_schema"] = response_schema

        config = types.GenerateContentConfig(**config_args)

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=config
                )

                duration_ms = int((time.time() - start_time) * 1000)
                text = response.text or ""
                if not text.strip():
                    raise LLMProviderError("Respuesta vacía recibida de Gemini.")

                # Audit log success (no secrets)
                input_toks = getattr(response.usage_metadata, "prompt_token_count", 0) if hasattr(response, "usage_metadata") else 0
                output_toks = getattr(response.usage_metadata, "candidates_token_count", 0) if hasattr(response, "usage_metadata") else 0

                DatabaseService.log_llm_interaction(
                    interaction_id=interaction_id,
                    project_id=project_id,
                    run_id=run_id,
                    caller_agent=caller_agent,
                    provider=self.provider_name,
                    model=self._model_name,
                    status="success",
                    duration_ms=duration_ms,
                    input_tokens=input_toks,
                    output_tokens=output_toks,
                    context_artifacts=context_artifacts or [],
                    produced_artifacts=[]
                )
                return text

            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                # Check for rate limits or server errors
                is_transient = any(code in err_str for code in ["429", "503", "500", "resource_exhausted", "timeout"])
                if is_transient and attempt < self._max_retries:
                    backoff = (2 ** attempt) * 1.5
                    time.sleep(backoff)
                    continue
                break

        duration_ms = int((time.time() - start_time) * 1000)
        sanitized_err = str(last_exception).replace(self._api_key, "[REDACTED_API_KEY]") if self._api_key else str(last_exception)

        DatabaseService.log_llm_interaction(
            interaction_id=interaction_id,
            project_id=project_id,
            run_id=run_id,
            caller_agent=caller_agent,
            provider=self.provider_name,
            model=self._model_name,
            status="failed",
            duration_ms=duration_ms,
            error_sanitized=sanitized_err,
            context_artifacts=context_artifacts or []
        )

        if "timeout" in str(last_exception).lower():
            raise LLMTimeoutError(f"Tiempo de espera agotado al conectar con Gemini: {sanitized_err}") from last_exception
        raise LLMProviderError(f"Fallo en llamada a Gemini ({sanitized_err})") from last_exception

    # -----------------------------------------------------------------------
    # 1. Structure Objective
    # -----------------------------------------------------------------------
    def structure_objective(
        self,
        project_id: str,
        user_question: str,
        catalog_summary: Dict[str, Any],
        user_clarifications: Optional[Dict[str, str]] = None
    ) -> ObjectiveSpec:
        clarifications = user_clarifications or {}
        system_instruction = ContextBuilder.get_security_instruction() + "\n" + (
            "Tu tarea es actuar como Agente Organizador y estructurar la pregunta del usuario en un ObjectiveSpec. "
            "Debes seleccionar únicamente métricas y dimensiones que existan en las tablas del catálogo. "
            "Si la pregunta es ambigua respecto a 'ventas', propone una pregunta de aclaración o un supuesto editable."
        )

        prompt = f"""
Pregunta del usuario: "{user_question}"

Aclaraciones del usuario previamente registradas:
{json.dumps(clarifications, ensure_ascii=False)}

Resumen del catálogo de tablas disponibles:
{json.dumps(catalog_summary, ensure_ascii=False, indent=2)}

Genera un JSON conforme al contrato ObjectiveSpec.
"""
        raw_json = self._execute_with_retry(
            prompt=prompt,
            system_instruction=system_instruction,
            response_schema=ObjectiveSpec,
            caller_agent="OrganizerAgent",
            project_id=project_id,
            context_artifacts=["DataCatalog"]
        )

        try:
            spec = ObjectiveSpec.model_validate_json(raw_json)
        except Exception as e:
            # Single structural repair attempt
            repair_prompt = f"El JSON anterior falló la validación con error: {str(e)}. Corrige el JSON para que coincida exactamente con ObjectiveSpec."
            raw_json_repaired = self._execute_with_retry(
                prompt=repair_prompt,
                system_instruction=system_instruction,
                response_schema=ObjectiveSpec,
                caller_agent="OrganizerAgent",
                project_id=project_id
            )
            spec = ObjectiveSpec.model_validate_json(raw_json_repaired)

        # Semantic check: Verify columns mentioned exist in catalog
        all_catalog_cols = set()
        for tbl in catalog_summary.values():
            cols = tbl.get("columns", {})
            if isinstance(cols, dict):
                all_catalog_cols.update([str(k).lower() for k in cols.keys()])
            elif isinstance(cols, list):
                all_catalog_cols.update([str(c).lower() for c in cols])

        # Validate that relevant dimensions exist in at least one table
        for dim in spec.relevant_dimensions:
            dim_clean = dim.split(".")[-1].strip().lower()
            if dim_clean not in [c.lower() for c in all_catalog_cols]:
                raise LLMValidationError(f"El modelo propuso una dimensión inexistente en los datos cargados: '{dim}'")

        return spec

    # -----------------------------------------------------------------------
    # 2. Propose Clarifications
    # -----------------------------------------------------------------------
    def propose_clarifications(
        self,
        project_id: str,
        user_question: str,
        catalog_summary: Dict[str, Any]
    ) -> List[AmbiguityItem]:
        system_instruction = ContextBuilder.get_security_instruction() + "\n" + (
            "Identifica si existen ambigüedades de alto impacto en la pregunta (por ejemplo, definir 'ventas' como "
            "facturación neta vs facturación bruta, o cómo tratar períodos incompletos). "
            "Genera como máximo 2 preguntas de aclaración con opciones y un valor por defecto."
        )

        prompt = f"""
Pregunta del usuario: "{user_question}"
Catálogo: {json.dumps(catalog_summary, ensure_ascii=False)}

Devuelve una lista de AmbiguityItem en formato JSON.
"""
        raw_json = self._execute_with_retry(
            prompt=prompt,
            system_instruction=system_instruction,
            caller_agent="OrganizerAgent",
            project_id=project_id
        )

        try:
            data = json.loads(raw_json)
            if isinstance(data, dict) and "items" in data:
                items = data["items"]
            elif isinstance(data, list):
                items = data
            else:
                items = [data]
            return [AmbiguityItem.model_validate(it) for it in items]
        except Exception as e:
            raise LLMValidationError(f"No se pudieron deserializar las preguntas de aclaración: {e}")

    # -----------------------------------------------------------------------
    # 3. Propose Analysis Plan
    # -----------------------------------------------------------------------
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
        system_instruction = ContextBuilder.get_security_instruction() + "\n" + (
            "Tu tarea es seleccionar operaciones del catálogo analítico registrado para conformar un AnalysisPlan. "
            "NO inventes operaciones, librerías, código SQL ni código Python. "
            "Usa estrictamente los step_id y parámetros descritos en el catálogo de métodos implementados."
        )

        prompt = f"""
Objetivo Operativo: {objective.operational_objective}
Métrica Principal: {objective.primary_metric}
Dimensiones: {objective.relevant_dimensions}

Problemas de Calidad Identificados:
{json.dumps(dq_summary, ensure_ascii=False)}

Catálogo de Métodos Implementados:
{json.dumps(registered_methods, ensure_ascii=False, indent=2)}

Genera un JSON conforme a AnalysisPlan utilizando únicamente las operaciones anteriores.
"""
        raw_json = self._execute_with_retry(
            prompt=prompt,
            system_instruction=system_instruction,
            response_schema=None,
            caller_agent="MethodologistAgent",
            project_id=project_id,
            run_id=run_id,
            context_artifacts=["ObjectiveSpec", "DataCatalog", "DataQualityReport"]
        )

        try:
            plan = AnalysisPlan.model_validate_json(raw_json)
        except Exception as e:
            repair_prompt = f"El plan anterior falló la validación con error: {str(e)}. Corrige el JSON para cumplir el contrato AnalysisPlan."
            raw_json_repaired = self._execute_with_retry(
                prompt=repair_prompt,
                system_instruction=system_instruction,
                response_schema=None,
                caller_agent="MethodologistAgent",
                project_id=project_id,
                run_id=run_id
            )
            plan = AnalysisPlan.model_validate_json(raw_json_repaired)

        # Semantic check: Verify all operation categories exist in registered methods
        registered_categories = {m["category"] for m in registered_methods}
        for op in plan.operations:
            if op.category not in registered_categories:
                raise LLMValidationError(f"El modelo intentó incluir una operación no registrada: '{op.category}'")

        return plan

    # -----------------------------------------------------------------------
    # 4. Interpret Validated Results
    # -----------------------------------------------------------------------
    def interpret_validated_results(
        self,
        run_id: str,
        objective: ObjectiveSpec,
        approved_results: List[AnalysisResult],
        limitations: List[str],
        provenance_summary: List[str]
    ) -> InsightReport:
        system_instruction = ContextBuilder.get_security_instruction() + "\n" + (
            "Tu tarea es redactar el InsightReport a partir de resultados validados y aprobados. "
            "REGLAS OBLIGATORIAS: "
            "1. Toda afirmación cuantitativa debe enlazar a un 'result_id' real de los resultados entregados. "
            "2. Separa hechos empíricamente probados de interpretaciones, hipótesis y recomendaciones. "
            "3. Las recomendaciones deben tener 'is_action_proposal_only=True' y no presentarse como causas demostradas. "
            "4. Conserva visibles las limitaciones metodológicas."
        )

        results_summary = ContextBuilder.build_results_summary(approved_results)
        approved_ids = {r.result_id for r in approved_results}

        prompt = f"""
Objetivo del análisis: {objective.operational_objective}

Resultados Aprobados y Validados:
{json.dumps(results_summary, ensure_ascii=False, indent=2)}

Limitaciones Conocidas:
{json.dumps(limitations, ensure_ascii=False)}

Genera un JSON conforme a InsightReport.
"""
        raw_json = self._execute_with_retry(
            prompt=prompt,
            system_instruction=system_instruction,
            response_schema=None,
            caller_agent="InterpreterAgent",
            run_id=run_id,
            context_artifacts=["AnalysisResults", "ValidationReport"]
        )

        try:
            report = InsightReport.model_validate_json(raw_json)
        except Exception as e:
            repair_prompt = f"El informe falló la validación con error: {str(e)}. Corrige el JSON conforme al contrato InsightReport."
            raw_json_repaired = self._execute_with_retry(
                prompt=repair_prompt,
                system_instruction=system_instruction,
                response_schema=None,
                caller_agent="InterpreterAgent",
                run_id=run_id
            )
            report = InsightReport.model_validate_json(raw_json_repaired)

        # Semantic check: Verify every finding cites a valid approved result_id
        for finding in report.observed_findings:
            if finding.result_id not in approved_ids:
                raise LLMValidationError(f"El hallazgo '{finding.id}' cita un result_id inexistente o no aprobado: '{finding.result_id}'")

        # Enforce is_action_proposal_only on all actions
        for act in report.recommended_actions:
            act.is_action_proposal_only = True

        return report

    # -----------------------------------------------------------------------
    # 5. Contextual Chat
    # -----------------------------------------------------------------------
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
        system_instruction = ContextBuilder.get_security_instruction() + "\n" + (
            "Clasifica la consulta del usuario en una de 4 categorías: "
            "'explain_existing_result', 'recalculate_with_filters', 'request_new_analysis', o 'unanswerable_by_data'. "
            "Si la pregunta indaga por información ausente (competencia, factores externos), clasifícala como 'unanswerable_by_data' y explica qué falta. "
            "Si solicita un cálculo con filtros o una explicación, fundamenta con citas internas de procedencia. "
            "NO inventes cifras numéricas no presentes en los resultados."
        )

        results_summary = ContextBuilder.build_results_summary(approved_results)
        approved_ids = [r.result_id for r in approved_results]

        prompt = f"""
Pregunta del usuario: "{question}"
Filtros activos en interfaz: {json.dumps(active_filters, ensure_ascii=False)}

Objetivo: {objective.operational_objective}
Resultados Validados: {json.dumps(results_summary, ensure_ascii=False)}
Resumen Ejecutivo: {insight_report.executive_summary if insight_report else 'No disponible'}
Limitaciones: {json.dumps(insight_report.data_limitations if insight_report else [], ensure_ascii=False)}
Tablas disponibles: {list(catalog_summary.keys())}

Genera un JSON conforme al contrato ChatAnswer.
"""
        raw_json = self._execute_with_retry(
            prompt=prompt,
            system_instruction=system_instruction,
            response_schema=None,
            caller_agent="ConversationalAssistantAgent",
            project_id=project_id,
            run_id=run_id,
            context_artifacts=["DashboardSpec", "InsightReport"]
        )

        try:
            ans = ChatAnswer.model_validate_json(raw_json)
            ans.is_demo_mode = False
            return ans
        except Exception as e:
            raise LLMValidationError(f"Error deserializando respuesta del asistente: {e}")
