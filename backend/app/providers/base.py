"""
Abstract Base Provider and typed exceptions for LLM integration.
Enforces product-oriented operations, schema validation, and decoupled design.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.contracts import (
    AmbiguityItem,
    AnalysisPlan,
    AnalysisResult,
    ChatAnswer,
    DataCatalog,
    DataQualityReport,
    InsightReport,
    ObjectiveSpec,
    RelationshipSpec,
)


class LLMProviderError(Exception):
    """Base exception for all LLM provider failures."""
    pass


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM call exceeds the configured timeout."""
    pass


class LLMValidationError(LLMProviderError):
    """Raised when an LLM output fails schema validation or semantic checks."""
    pass


class LLMAuthenticationError(LLMProviderError):
    """Raised when the API key is missing or rejected by the provider."""
    pass


@dataclass
class LLMInteractionRecord:
    interaction_id: str
    project_id: str
    caller_agent: str
    provider: str
    model: str
    status: str
    duration_ms: int = 0
    run_id: Optional[str] = None
    schema_version: str = "1.0"
    input_tokens: int = 0
    output_tokens: int = 0
    error_sanitized: Optional[str] = None
    context_artifacts: List[str] = field(default_factory=list)
    produced_artifacts: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LLMProvider(ABC):
    """Product-oriented abstract interface for LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def is_deterministic_fallback(self) -> bool:
        pass

    @abstractmethod
    def structure_objective(
        self,
        project_id: str,
        user_question: str,
        catalog_summary: Dict[str, Any],
        user_clarifications: Optional[Dict[str, str]] = None
    ) -> ObjectiveSpec:
        """Converts user question and catalog summary into an ObjectiveSpec."""
        pass

    @abstractmethod
    def propose_clarifications(
        self,
        project_id: str,
        user_question: str,
        catalog_summary: Dict[str, Any]
    ) -> List[AmbiguityItem]:
        """Identifies impactful ambiguities requiring user confirmation."""
        pass

    @abstractmethod
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
        """Selects valid registered analytical operations to form an AnalysisPlan."""
        pass

    @abstractmethod
    def interpret_validated_results(
        self,
        run_id: str,
        objective: ObjectiveSpec,
        approved_results: List[AnalysisResult],
        limitations: List[str],
        provenance_summary: List[str]
    ) -> InsightReport:
        """Drafts InsightReport linking claims strictly to approved result_ids."""
        pass

    @abstractmethod
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
        """Classifies and answers conversational queries with provenance citations."""
        pass
