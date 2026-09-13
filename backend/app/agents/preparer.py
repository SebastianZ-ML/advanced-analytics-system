"""
Agent D: Data Preparer.
Executes reproducible, strictly tracked transformations and safe joins,
logging all row exclusions and validating join integrity.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
from app.agents.base import BaseAgent
from app.contracts import RelationshipSpec, TransformationRecord
from app.engine.duckdb_engine import DuckDBAnalyticsEngine


class DataPreparerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Data Preparer Agent", role="Auditable transformations, reproducible cleaning, and controlled joins")

    def prepare_data(
        self,
        tables: Dict[str, pd.DataFrame],
        relationships: List[RelationshipSpec],
        exclude_cancelled: bool = True,
        cutoff_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[TransformationRecord]]:
        """
        Executes clean data preparation through the deterministic engine.
        Returns the analytical table and the list of TransformationRecords.
        """
        return DuckDBAnalyticsEngine.prepare_analytical_dataset(
            tables=tables,
            relationships=relationships,
            exclude_cancelled=exclude_cancelled,
            cutoff_date=cutoff_date
        )
