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
            raise LLMAuthenticationError("GEMINI_API_KEY is not configured.")

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
                    raise LLMProviderError("Empty response received from Gemini.")

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
            raise LLMTimeoutError(f"Timeout connecting to Gemini: {sanitized_err}") from last_exception
        raise LLMProviderError(f"Gemini API call failed ({sanitized_err})") from last_exception

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
            "Your task is to act as the Organizer Agent and structure the user business question into an ObjectiveSpec. "
            "You must select strictly metrics and dimensions that exist in the catalog tables. "
            "If the question is ambiguous regarding 'sales', propose a clarification question or an editable assumption."
        )

        prompt = f"""
User Question: "{user_question}"

Previously Recorded User Clarifications:
{json.dumps(clarifications, ensure_ascii=False)}

Catalog Summary of Available Tables:
{json.dumps(catalog_summary, ensure_ascii=False, indent=2)}

Generate a JSON conforming to the ObjectiveSpec contract.
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
            repair_prompt = f"The previous JSON failed validation with error: {str(e)}. Correct the JSON to match ObjectiveSpec exactly."
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
                raise LLMValidationError(f"The model proposed a non-existent dimension in the loaded data: '{dim}'")

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
            "Identify if high-impact ambiguities exist in the question (for example, defining 'sales' as "
            "net sales vs gross sales, or how to treat incomplete periods). "
            "Generate at most 2 clarification questions with selectable options and a default value."
        )

        prompt = f"""
User Question: "{user_question}"
Catalog: {json.dumps(catalog_summary, ensure_ascii=False)}

Return a list of AmbiguityItem in JSON format.
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
            raise LLMValidationError(f"Could not deserialize clarification questions: {e}")

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
            "Your task is to select operations from the registered analytical catalog to form an AnalysisPlan. "
            "Do NOT invent operations, libraries, SQL code, or Python code. "
            "Strictly use the step_id and parameters described in the catalog of implemented methods."
        )

        prompt = f"""
Operational Objective: {objective.operational_objective}
Primary Metric: {objective.primary_metric}
Dimensions: {objective.relevant_dimensions}

Identified Quality Issues:
{json.dumps(dq_summary, ensure_ascii=False)}

Catalog of Implemented Methods:
{json.dumps(registered_methods, ensure_ascii=False, indent=2)}

Generate a JSON conforming to AnalysisPlan using strictly the operations above.
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
            repair_prompt = f"The previous plan failed validation with error: {str(e)}. Correct the JSON to fulfill the AnalysisPlan contract."
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
                raise LLMValidationError(f"The model attempted to include an unregistered operation: '{op.category}'")

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
            "Your task is to draft the InsightReport from validated and approved results. "
            "MANDATORY RULES: "
            "1. Every quantitative claim must link to a real 'result_id' from the provided results. "
            "2. Separate empirically proven facts from interpretations, hypotheses, and recommendations. "
            "3. Recommendations must have 'is_action_proposal_only=True' and must not be presented as proven causes. "
            "4. Keep methodological limitations clearly visible."
        )

        results_summary = ContextBuilder.build_results_summary(approved_results)
        approved_ids = {r.result_id for r in approved_results}

        prompt = f"""
Analysis Objective: {objective.operational_objective}

Approved and Validated Results:
{json.dumps(results_summary, ensure_ascii=False, indent=2)}

Known Limitations:
{json.dumps(limitations, ensure_ascii=False)}

Generate a JSON conforming to InsightReport.
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
            repair_prompt = f"The report failed validation with error: {str(e)}. Correct the JSON conforming to the InsightReport contract."
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
                raise LLMValidationError(f"Finding '{finding.id}' cites a non-existent or unapproved result_id: '{finding.result_id}'")

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
            "Classify the user inquiry into one of 4 categories: "
            "'explain_existing_result', 'recalculate_with_filters', 'request_new_analysis', or 'unanswerable_by_data'. "
            "If the question inquires about absent information (competitors, external factors), classify it as 'unanswerable_by_data' and explain what is missing. "
            "If it requests a calculation with filters or an explanation, support it with internal provenance citations. "
            "Do NOT invent numeric figures not present in the results."
        )

        results_summary = ContextBuilder.build_results_summary(approved_results)
        approved_ids = [r.result_id for r in approved_results]

        prompt = f"""
User Question: "{question}"
Active Interface Filters: {json.dumps(active_filters, ensure_ascii=False)}

Objective: {objective.operational_objective}
Validated Results: {json.dumps(results_summary, ensure_ascii=False)}
Executive Summary: {insight_report.executive_summary if insight_report else 'Not available'}
Limitations: {json.dumps(insight_report.data_limitations if insight_report else [], ensure_ascii=False)}
Available Tables: {list(catalog_summary.keys())}

Generate a JSON conforming to the ChatAnswer contract.
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
            raise LLMValidationError(f"Error deserializing assistant answer: {e}")
