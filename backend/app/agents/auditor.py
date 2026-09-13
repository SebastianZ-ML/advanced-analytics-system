"""
Agent B: Data Auditor (Agente Auditor de Datos).
Inventories tables, profiles columns, detects quality anomalies,
evaluates foreign keys, and proposes confirmed/unconfirmed relationships.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
from app.agents.base import BaseAgent
from app.contracts import DataCatalog, DataQualityReport, RelationshipSpec, TableProfile
from app.engine.duckdb_engine import DuckDBAnalyticsEngine


class DataAuditorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Data Auditor Agent", role="Table inventory, schema profiling, quality anomaly detection, and relationship evaluation")

    def audit_project_tables(
        self,
        project_id: str,
        loaded_tables: Dict[str, pd.DataFrame],
        file_metadata: Dict[str, Dict[str, str]]
    ) -> Tuple[DataCatalog, DataQualityReport, List[RelationshipSpec]]:
        """
        Profiles loaded DataFrames, generates a DataCatalog,
        runs deterministic quality checks, and infers relationships.
        """
        tables_profile: Dict[str, TableProfile] = {}

        for table_id, df in loaded_tables.items():
            meta = file_metadata.get(table_id, {})
            filename = meta.get("filename", f"{table_id}.csv")
            sheet_name = meta.get("sheet_name")
            file_hash = meta.get("file_hash", "UNKNOWN_HASH")

            profile = DuckDBAnalyticsEngine.profile_dataframe(
                df=df,
                table_id=table_id,
                filename=filename,
                file_hash=file_hash,
                sheet_name=sheet_name
            )
            tables_profile[table_id] = profile

        catalog = DataCatalog(
            schema_version="1.0",
            project_id=project_id,
            tables=tables_profile,
            total_tables=len(tables_profile)
        )

        # Run quality audit and relation proposals
        dq_report, relationships = DuckDBAnalyticsEngine.audit_quality(
            tables=loaded_tables,
            catalog=catalog,
            project_id=project_id
        )

        return catalog, dq_report, relationships
