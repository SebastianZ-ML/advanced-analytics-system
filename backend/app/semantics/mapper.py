"""
Semantic Mapper.
Inspects arbitrary tabular schemas and infers an executable SemanticModel.
Detects semantic ambiguities and provides default resolutions.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
from app.contracts import DataCatalog, TableProfile, RelationshipSpec
from app.semantics.models import (
    BusinessStateMapping,
    CalendarSpec,
    DimensionSpec,
    EntityGranularity,
    MetricDefinition,
    SemanticAmbiguity,
    SemanticModel,
)


class SemanticMapper:
    @classmethod
    def infer_semantic_model(
        cls,
        project_id: str,
        catalog: DataCatalog,
        relationships: List[RelationshipSpec],
        user_overrides: Optional[Dict[str, Any]] = None
    ) -> SemanticModel:
        """
        Infer an executable SemanticModel from profiled tables without hardcoding retail names.
        """
        user_overrides = user_overrides or {}
        tables = catalog.tables
        if not tables:
            raise ValueError("Cannot infer semantic model from empty data catalog.")

        # 1. Identify primary fact table: largest table with date and numeric metrics
        fact_table_name = user_overrides.get("fact_table")
        if not fact_table_name:
            candidates = sorted(
                tables.values(),
                key=lambda t: (len(t.date_columns) > 0, len(t.numeric_columns), t.row_count),
                reverse=True
            )
            fact_table_profile = candidates[0]
            fact_table_name = fact_table_profile.table_id
        else:
            fact_table_profile = tables.get(fact_table_name, list(tables.values())[0])

        # 2. Identify primary date column
        date_candidates = fact_table_profile.date_columns
        ambiguities: List[SemanticAmbiguity] = []
        if len(date_candidates) > 1 and "date_column" not in user_overrides:
            ambiguities.append(SemanticAmbiguity(
                id=f"AMB_DATE_{project_id}",
                concept="primary_date",
                question=f"Table '{fact_table_name}' contains multiple date fields. Which should govern temporal aggregation?",
                context=f"Candidates found: {', '.join(date_candidates)}",
                candidates=date_candidates,
                selected_choice=date_candidates[0],
                is_resolved=True
            ))
            selected_date = date_candidates[0]
        elif len(date_candidates) == 1:
            selected_date = date_candidates[0]
        else:
            # Fallback scan columns with 'date', 'fecha', 'time', 'timestamp'
            fallback_col = next((c for c in fact_table_profile.columns if any(k in c.lower() for k in ["date", "fecha", "time", "day", "month"])), "date")
            selected_date = fallback_col

        selected_date = user_overrides.get("date_column", selected_date)

        # 3. Identify metrics (numeric columns in fact table)
        metrics: List[MetricDefinition] = []
        numeric_candidates = fact_table_profile.numeric_columns
        primary_metric_name = user_overrides.get("primary_metric")

        for num_col in numeric_candidates:
            col_prof = fact_table_profile.columns.get(num_col)
            label = num_col.replace("_", " ").title()
            is_prim = (num_col == primary_metric_name) if primary_metric_name else ("net" in num_col.lower() or "amount" in num_col.lower() or "sales" in num_col.lower() or "charge" in num_col.lower() or "revenue" in num_col.lower())
            
            unit = "USD" if any(k in num_col.lower() for k in ["sales", "amount", "charge", "price", "cost", "revenue", "mrr", "arr"]) else "Count / Units"

            metrics.append(MetricDefinition(
                name=num_col,
                label=label,
                formula=f"SUM({num_col})",
                aggregation="sum",
                unit=unit,
                currency="USD" if unit == "USD" else None,
                is_primary=is_prim
            ))

        if metrics and not any(m.is_primary for m in metrics):
            metrics[0].is_primary = True

        # 4. Identify granularity (primary key)
        id_candidates = fact_table_profile.id_columns
        pk = id_candidates[0] if id_candidates else (list(fact_table_profile.columns.keys())[0])

        granularity = EntityGranularity(
            entity_name=fact_table_name,
            primary_key=pk,
            row_semantic=fact_table_profile.row_semantic_meaning or f"One record per {fact_table_name} event"
        )

        # 5. Identify status column
        status_col = next(
            (c for c in fact_table_profile.columns if any(k in c.lower() for k in ["status", "state", "estado"])),
            None
        )
        business_states = BusinessStateMapping(status_column=status_col)

        # 6. Extract dimensions (categorical columns from fact and joined dimensions)
        dimensions: List[DimensionSpec] = []
        for cat_col in fact_table_profile.categorical_columns:
            if cat_col != status_col and cat_col != pk:
                dimensions.append(DimensionSpec(
                    dimension_name=cat_col,
                    source_column=cat_col,
                    table_name=fact_table_name
                ))

        for dt_name, dt_prof in tables.items():
            if dt_name != fact_table_name:
                for cat_col in dt_prof.categorical_columns:
                    dimensions.append(DimensionSpec(
                        dimension_name=cat_col,
                        source_column=cat_col,
                        table_name=dt_name
                    ))

        # 7. Calendar
        calendar = CalendarSpec(
            date_column=selected_date,
            frequency="M",
            timezone="UTC"
        )

        return SemanticModel(
            project_id=project_id,
            fact_table=fact_table_name,
            granularity=granularity,
            calendar=calendar,
            metrics=metrics,
            dimensions=dimensions,
            business_states=business_states,
            dimension_relationships=relationships,
            ambiguities=ambiguities
        )
