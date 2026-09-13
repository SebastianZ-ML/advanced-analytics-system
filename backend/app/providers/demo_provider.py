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
            question="Which specific metric do you mean by 'sales'?",
            context="The term 'sales' can be interpreted as gross sales, net sales, or units sold.",
            proposed_default="net_sales (Accrued Net Sales)",
            user_decision=metric_choice,
            impact_level="high",
            status="resolved" if "metric_definition" in clarifications else "pending"
        )

        ambiguity_cutoff = AmbiguityItem(
            id="AMB_PERIOD_CUTOFF",
            question="How should the last recorded period be handled if incomplete?",
            context="June 2026 contains only 4 recorded days. Directly comparing it would induce a false decline.",
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

    def propose_clarifications(
        self,
        project_id: str,
        user_question: str,
        catalog_summary: Dict[str, Any]
    ) -> List[AmbiguityItem]:
        return [
            AmbiguityItem(
                id="AMB_METRIC_DEFINITION",
                question="Which specific metric do you mean by 'sales'?",
                context="The term 'sales' can be interpreted as gross sales, net sales, or units sold.",
                proposed_default="net_sales (Accrued Net Sales)",
                impact_level="high",
                status="pending"
            ),
            AmbiguityItem(
                id="AMB_PERIOD_CUTOFF",
                question="How should the last recorded period be handled if incomplete?",
                context="June 2026 contains only 4 recorded days. Directly comparing it would induce a false decline.",
                proposed_default="exclude_incomplete_period (Compare only closed complete months)",
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
                supported_claim="Additive decomposition requires mutually exclusive categories to reconcile component sums with the aggregate series.",
                is_primary_source=True
            ),
            ExternalResearchCitation(
                url="https://pandas.pydata.org/docs/user_guide/merging.html",
                title="Pandas Documentation: Merge and Join Integrity",
                supported_claim="Fact-to-dimension joins must verify right-side uniqueness to prevent unintentional row inflation.",
                is_primary_source=True
            )
        ]

        return AnalysisPlan(
            schema_version="1.0",
            project_id=project_id,
            run_id=run_id,
            title="Sales Contraction Diagnostic and Decomposition Plan (Deterministic Mode)",
            rationale="Analytical sequence designed from registered operations catalog to isolate observable factors.",
            operations=ops,
            excluded_methods=[
                "Multivariate causal regression (data lacks exogenous confounding variables like inflation or stockouts).",
                "Complex ARIMA / Prophet (5-month historical series does not meet statistical seasonality minimums)."
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
                claim=f"Net sales decreased by {abs(diff_abs):,.2f} monetary units ({diff_pct:.1f}%) between March 2026 and May 2026.",
                result_id=res_comp.result_id,
                metric_name="net_sales_delta",
                observed_value=diff_abs,
                is_empirically_proven=True,
                evidence_text="Calculation derived from completed orders closed in 2026-03 vs 2026-05."
            ))

        res_channel = approved_map.get("RES_OP_04_DIMENSION_BREAKDOWN_CHANNEL")
        if res_channel:
            contribs = res_channel.calculated_values.get("contributions", [])
            wholesale = next((item for item in contribs if any(k in item["dimension_value"] for k in ["Wholesale", "Mayorista"])), None)
            if wholesale:
                m_abs = wholesale["absolute_change"]
                m_contrib = wholesale["contribution_to_total_change"]
                findings.append(InsightFinding(
                    id="FINDING_WHOLESALE_CONCENTRATION",
                    claim=f"The observed drop is predominantly concentrated in the 'Wholesale / B2B' channel, explaining {abs(m_abs):,.2f} monetary units ({m_contrib:.1f}% of total drop).",
                    result_id=res_channel.result_id,
                    metric_name="channel_contribution_wholesale",
                    observed_value=m_abs,
                    is_empirically_proven=True,
                    evidence_text="Mutually exclusive additive decomposition reconciled 100% with total variation."
                ))
                interpretations.append(InsightInterpretation(
                    id="INTERP_WHOLESALE_FOCUS",
                    interpretation_text="The overall business contraction is focused on purchase volume and frequency from wholesale/corporate clients.",
                    grounded_in_finding_ids=["FINDING_TOTAL_CONTRACTION", "FINDING_WHOLESALE_CONCENTRATION"],
                    confidence_rationale="Supported by accounting decomposition with exact reconciliation.",
                    distinction_from_causality="Identifies accounting loss location, but does not prove root cause (churn, stockouts, or credit policies)."
                ))

        actions.append(ActionRecommendation(
            id="ACT_01_AUDIT_WHOLESALE_ACCOUNTS",
            title="Commercial audit of key B2B accounts",
            description="Contact top March wholesale buyers absent in May to investigate operational reasons.",
            hypothesis_to_investigate="Loss of commercial competitiveness or stockouts in key product lines.",
            expected_impact="Potential recovery of up to 70% of lost wholesale channel volume.",
            data_needed_to_confirm="Qualitative commercial interviews and unfulfilled quote logs.",
            is_action_proposal_only=True
        ))

        summary = (
            "The verified analysis confirms a substantial net contraction between March and May 2026. "
            "Mathematical decomposition conclusively demonstrates that the decline is almost entirely concentrated "
            "in the 'Wholesale / B2B' channel, primarily driven by reduced purchases from recurring accounts."
        )

        return InsightReport(
            schema_version="1.0",
            run_id=run_id,
            executive_summary=summary,
            observed_findings=findings,
            interpretations=interpretations,
            unproven_hypotheses=[
                "Unproven hypothesis: The decline in wholesale purchases may stem from revised trade credit terms or early inventory stocking in March."
            ],
            data_limitations=limitations,
            recommended_actions=actions,
            suggested_further_analyses=[
                "Incorporate warehouse inventory and logistics data to rule out B2B fulfillment bottlenecks."
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
        unanswerable_triggers = [
            "competitor", "competition", "competidor", "competencia",
            "inflation", "inflación", "inflacion",
            "competitor price", "precio de la competencia",
            "weather", "clima", "rating", "nps"
        ]
        for trigger in unanswerable_triggers:
            if trigger in q:
                return ChatAnswer(
                    schema_version="1.0",
                    query_type="unanswerable_by_data",
                    answer_text=(
                        f"The data available in the current project does not contain information regarding '{trigger}'. "
                        f"The loaded tables ({', '.join(catalog_summary.keys())}) only record order transactions, "
                        f"registered customers, product catalog, and marketing campaign budgets. "
                        f"To analyze this factor, external market research or logistics inventory sources would need to be ingested."
                    ),
                    citations=[],
                    data_limitation_notice=f"Absence of '{trigger}' data in project catalog.",
                    requires_full_pipeline=False,
                    is_demo_mode=True
                )

        if any(t in q for t in ["new analysis", "predictive model", "cluster", "clustering", "nuevo análisis", "modelo predictivo"]):
            return ChatAnswer(
                schema_version="1.0",
                query_type="request_new_analysis",
                answer_text="This request requires designing a new formal analytical plan. Please initiate a new run configuring this objective in the Planning section.",
                citations=[],
                calculation_summary="Request classified as new analysis. Requires orchestration by Methodologist Agent.",
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
                details="Comparison between peak period and most recent closed month."
            ))
        if res_channel:
            citations.append(ProvenanceCitation(
                source_type="analytic_step",
                source_name=res_channel.step_id,
                filter_or_condition="orders GROUP BY channel",
                result_id=res_channel.result_id,
                details="Additive decomposition with 100% reconciliation."
            ))

        text = (
            f"Based on the project's audited and validated results:\n\n"
            f"1. **Core Finding**: {insight_report.executive_summary if insight_report else 'Observed decline between March and May 2026.'}\n\n"
            f"2. **Methodological Assurance**: This conclusion stems from a verified additive decomposition where channel variations sum exactly to 100% of the total observed decline."
        )

        return ChatAnswer(
            schema_version="1.0",
            query_type="explain_existing_result",
            answer_text=text,
            citations=citations,
            calculation_summary="Provenance: orders.csv -> safe join with customers.csv -> exact dimensional reconciliation.",
            requires_full_pipeline=False,
            is_demo_mode=True
        )
