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

        # Dynamic discovery from catalog when available
        clarifications = user_clarifications or {}
        detected_metrics = []
        detected_dates = []
        detected_dims = []

        if catalog and catalog.tables:
            for tbl_id, tbl in catalog.tables.items():
                for cname, cprof in tbl.columns.items():
                    low = cname.lower()
                    if cprof.inferred_type in ["float", "int", "numeric", "double"] and "id" not in low and "cod" not in low:
                        if cname not in detected_metrics:
                            detected_metrics.append(cname)
                    elif cprof.inferred_type in ["date", "datetime", "timestamp"] or "date" in low or "fecha" in low:
                        if cname not in detected_dates:
                            detected_dates.append(cname)
                    elif cprof.inferred_type in ["string", "object", "category"] and "id" not in low and cprof.unique_count <= 100:
                        if cname not in detected_dims:
                            detected_dims.append(cname)

        # Primary metric selection
        metric_choice = clarifications.get("metric_definition")
        if not metric_choice:
            q_lower = user_question.lower()
            # Prioritize net_sales if present and relevant to sales/revenue questions
            if "net_sales" in detected_metrics and any(kw in q_lower for kw in ["sales", "venta", "drop", "caida", "decline", "ingreso"]):
                metric_choice = "net_sales"
            else:
                # Match keywords from user question with metric parts
                matched = []
                for m in detected_metrics:
                    m_parts = [p for p in m.lower().split("_") if len(p) > 2]
                    if any(part in q_lower for part in m_parts):
                        matched.append(m)
                if matched:
                    pref = next((m for m in matched if ("net" in m or "total" in m or "sales" in m or "factur" in m or "monto" in m) and "discount" not in m), matched[0])
                    metric_choice = pref
                elif "net_sales" in detected_metrics:
                    metric_choice = "net_sales"
                elif detected_metrics:
                    pref = next((m for m in detected_metrics if ("sales" in m or "total" in m or "factur" in m or "monto" in m or "amount" in m) and "discount" not in m), detected_metrics[0])
                    metric_choice = pref
                else:
                    metric_choice = "net_sales"

        primary_metric_label = f"{metric_choice} ({metric_choice.replace('_', ' ').title()})"
        aux_metrics = [m for m in detected_metrics if m != metric_choice][:4]

        # Ambiguities
        ambiguity_metric = AmbiguityItem(
            id="AMB_METRIC_DEFINITION",
            question=f"Which metric should be the primary objective focus? (Detected: {', '.join(detected_metrics[:4])})",
            context=f"The dataset contains numeric measures: {', '.join(detected_metrics[:5])}.",
            proposed_default=metric_choice,
            user_decision=metric_choice,
            impact_level="high",
            status="resolved" if "metric_definition" in clarifications else "pending"
        )

        ambiguity_cutoff = AmbiguityItem(
            id="AMB_PERIOD_CUTOFF",
            question="How should the last recorded period be handled if incomplete?",
            context="Comparing a partial month directly with preceding full months can cause artificial contraction distortions.",
            proposed_default="exclude_incomplete_period",
            user_decision=clarifications.get("period_cutoff", "exclude_incomplete_period"),
            impact_level="medium",
            status="resolved" if "period_cutoff" in clarifications else "pending"
        )

        dims = detected_dims[:5] if detected_dims else ["channel", "segment", "region"]

        return ObjectiveSpec(
            schema_version="1.0",
            project_id=project_id,
            original_question=user_question,
            operational_objective=f"Analyze {metric_choice} variation and decompose behavior across {', '.join(dims[:3])} dimensions.",
            decision_to_inform=f"Identify drivers of {metric_choice} changes to inform operational and strategic allocation decisions.",
            primary_metric=primary_metric_label,
            auxiliary_metrics=aux_metrics,
            time_period="Observed historical periods (complete closed intervals)",
            comparison_period="Baseline comparative window versus most recent complete period",
            relevant_dimensions=dims,
            assumptions=[
                "Records with CANCELLED or REFUNDED status are excluded where applicable.",
                "Partial end periods are audited to avoid false contraction bias.",
                "Fact-to-dimension relationships are deduplicated to guarantee an exact 1.0x factor."
            ],
            pending_ambiguities=[ambiguity_metric, ambiguity_cutoff],
            acceptance_criteria=[
                "Additive decompositions must reconcile 100% with observed total variation.",
                "No results containing infinite or NaN values are permitted.",
                "Fact-to-dimension joins must preserve cardinalities without artificial multiplication."
            ],
            excluded_scope=[
                "Multivariate causal factors not captured in registered project tables.",
                "Prescriptive optimizations requiring unmodeled behavioral elasticities."
            ],
            status="confirmed" if all(a.status == "resolved" for a in [ambiguity_metric, ambiguity_cutoff]) else "draft",
            is_demo_mode=True
        )
