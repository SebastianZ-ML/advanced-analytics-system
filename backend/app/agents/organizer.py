"""
Agent A: Organizer.
Translates business questions into formal operational objectives (ObjectiveSpec).
Supports Gemini LLM provider with graceful fallback to deterministic logic.
"""
from typing import Any, Dict, List, Optional
from app.agents.base import BaseAgent
from app.contracts import AmbiguityItem, DataCatalog, ObjectiveSpec
from app.providers import ContextBuilder, LLMProvider, get_llm_provider


class OrganizerAgent(BaseAgent):
    def __init__(self, override_provider: Optional[LLMProvider] = None):
        super().__init__(name="Organizer Agent", role="Analytical objective structuring and ambiguity resolution")
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
                print(f"[OrganizerAgent] Fallback activated after LLM provider error: {e}")

        # Deterministic fallback (DEMO_WITHOUT_LLM)
        clarifications = user_clarifications or {}
        metric_choice = clarifications.get("metric_definition", "net_sales")

        ambiguity_metric = AmbiguityItem(
            id="AMB_METRIC_DEFINITION",
            question="Which specific metric do you mean by 'sales'?",
            context="The term 'sales' can be interpreted as gross sales, net sales (deducting discounts and returns), or unit sales volume.",
            proposed_default="net_sales (Accrued Net Sales)",
            user_decision=metric_choice,
            impact_level="high",
            status="resolved" if "metric_definition" in clarifications else "pending"
        )

        ambiguity_cutoff = AmbiguityItem(
            id="AMB_PERIOD_CUTOFF",
            question="How should the last recorded period be handled if incomplete?",
            context="The recorded data contains a partial final month. Including it directly would distort the period comparison.",
            proposed_default="exclude_incomplete_period (Compare only closed complete months)",
            user_decision=clarifications.get("period_cutoff", "exclude_incomplete_period"),
            impact_level="high",
            status="resolved" if "period_cutoff" in clarifications else "pending"
        )

        primary_metric_label = "net_sales (Net Sales)"
        if metric_choice == "gross_sales":
            primary_metric_label = "gross_sales (Gross Sales)"
        elif metric_choice == "units":
            primary_metric_label = "units (Units Sold)"

        return ObjectiveSpec(
            schema_version="1.0",
            project_id=project_id,
            original_question=user_question,
            operational_objective="Decompose net sales variation across periods to identify contraction concentration by channel, product, and customer segment.",
            decision_to_inform="Prioritize which commercial channels or segments require retention plans, price adjustments, or immediate operational review.",
            primary_metric=primary_metric_label,
            auxiliary_metrics=["gross_sales", "units", "discount_amount", "customer_retention"],
            time_period="2026-01 to 2026-05 (Closed complete months)",
            comparison_period="Pre-drop peak month (2026-03) versus most recent closed month (2026-05)",
            relevant_dimensions=["channel", "segment", "category_name", "region"],
            assumptions=[
                "Transactions with CANCELLED or REFUNDED status are excluded from accrued net sales.",
                "June 2026 is excluded from period comparisons due to incomplete coverage (4 recorded days).",
                "Relationships between orders and dimensions are deduplicated to avoid metric inflation."
            ],
            pending_ambiguities=[ambiguity_metric, ambiguity_cutoff],
            acceptance_criteria=[
                "The sum of decomposed dimensional contributions must reconcile 100% with total observed net variation.",
                "No results containing infinite or NaN values are permitted.",
                "Fact-to-dimension joins must have a row multiplication factor of exactly 1.0."
            ],
            excluded_scope=[
                "Complex multivariate causal attribution models not observable in uploaded tables.",
                "Prescriptive price optimization (requires unmodeled elasticities)."
            ],
            status="confirmed" if all(a.status == "resolved" for a in [ambiguity_metric, ambiguity_cutoff]) else "draft",
            is_demo_mode=True
        )
