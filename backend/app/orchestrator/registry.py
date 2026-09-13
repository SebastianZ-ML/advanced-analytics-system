"""
Analytical Method Registry.
Central registry linking every analytical operation to its parameter schema,
preconditions, execution handler, and post-validation invariants.
"""
from typing import Any, Callable, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field
from app.contracts import AnalysisResult, OperationSpec


class MethodSpecification(BaseModel):
    category: str
    name: str
    description: str
    required_columns: List[str] = Field(default_factory=list)
    optional_columns: List[str] = Field(default_factory=list)
    supports_dimensions: bool = False
    requires_time_dimension: bool = False


class MethodRegistry:
    _registry: Dict[str, MethodSpecification] = {}
    _handlers: Dict[str, Callable[..., AnalysisResult]] = {}

    @classmethod
    def register_method(
        cls,
        category: str,
        spec: MethodSpecification,
        handler: Optional[Callable[..., AnalysisResult]] = None
    ) -> None:
        cls._registry[category] = spec
        if handler:
            cls._handlers[category] = handler

    @classmethod
    def get_spec(cls, category: str) -> Optional[MethodSpecification]:
        return cls._registry.get(category)

    @classmethod
    def get_handler(cls, category: str) -> Optional[Callable[..., AnalysisResult]]:
        return cls._handlers.get(category)

    @classmethod
    def list_available_categories(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def validate_operation(cls, op: OperationSpec, df_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate operation preconditions against registered method specification.
        """
        spec = cls.get_spec(op.category)
        if not spec:
            return False, [f"Unregistered analytical method category: '{op.category}'"]

        errors: List[str] = []
        for req_col in spec.required_columns:
            # Check if required column exists or is in parameters
            param_col = op.parameters.get(req_col) or op.parameters.get("metric") or op.parameters.get("date_column")
            if param_col and param_col not in df_columns:
                errors.append(f"Operation '{op.step_id}' requires column '{param_col}' which is not in the dataset.")

        return len(errors) == 0, errors


# Populate default registered analytical methods
MethodRegistry.register_method(
    "metric_summary",
    MethodSpecification(
        category="metric_summary",
        name="Global Metric Summary",
        description="Aggregates core quantitative metrics across all eligible rows.",
        required_columns=[]
    )
)

MethodRegistry.register_method(
    "time_aggregation",
    MethodSpecification(
        category="time_aggregation",
        name="Time Series Aggregation",
        description="Aggregates metrics across uniform time intervals.",
        requires_time_dimension=True
    )
)

MethodRegistry.register_method(
    "period_comparison",
    MethodSpecification(
        category="period_comparison",
        name="Period-over-Period Variance Analysis",
        description="Calculates absolute and relative changes between two distinct temporal intervals.",
        requires_time_dimension=True
    )
)

MethodRegistry.register_method(
    "dimension_breakdown",
    MethodSpecification(
        category="dimension_breakdown",
        name="Additive Dimensional Decomposition (Waterfall)",
        description="Decomposes total change into mutually exclusive categorical contributions.",
        supports_dimensions=True,
        requires_time_dimension=True
    )
)

MethodRegistry.register_method(
    "customer_dynamics",
    MethodSpecification(
        category="customer_dynamics",
        name="Customer Repurchase & Churn Dynamics",
        description="Classifies volume between new vs recurring entity occurrences over time.",
        requires_time_dimension=True
    )
)

MethodRegistry.register_method(
    "baseline_forecast",
    MethodSpecification(
        category="baseline_forecast",
        name="Statistical Baseline Projection",
        description="Projects future periods using baseline models with out-of-sample evaluation.",
        requires_time_dimension=True
    )
)
