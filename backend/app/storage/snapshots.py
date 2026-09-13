"""
Snapshot Storage Service.
Persists immutable analytical snapshots per pipeline run using DuckDB-backed Parquet files.
Calculates SHA-256 hashes, records schema metadata, and guarantees integrity.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import duckdb
import pandas as pd
from pydantic import BaseModel, Field

from app.config import settings
from app.storage.db import DatabaseService
from app.storage.files import FileManager


class SnapshotMetadata(BaseModel):
    snapshot_id: str
    run_id: str
    project_id: str
    file_path: str
    file_hash: str
    row_count: int
    column_count: int
    schema_metadata: Dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SnapshotManager:
    @staticmethod
    def get_snapshots_dir() -> Path:
        sdir = getattr(settings, "snapshots_dir", settings.storage_dir / "snapshots")
        sdir.mkdir(parents=True, exist_ok=True)
        return sdir

    @classmethod
    def save_snapshot(
        cls,
        run_id: str,
        project_id: str,
        df: pd.DataFrame,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> SnapshotMetadata:
        """
        Persist an immutable analytical dataset snapshot as a Parquet file.
        Records SHA-256 hash and schema for deterministic provenance.
        """
        sdir = cls.get_snapshots_dir()
        snapshot_id = f"SNAP_{run_id}"
        file_path = sdir / f"{run_id}.parquet"

        # Export to Parquet via DuckDB native writer
        con = duckdb.connect()
        try:
            con.register("snap_df_view", df)
            # Use forward slashes for DuckDB SQL path
            sql_path = str(file_path).replace("\\", "/")
            con.execute(f"COPY snap_df_view TO '{sql_path}' (FORMAT PARQUET)")
        finally:
            con.close()

        # Calculate exact SHA-256 hash
        file_hash = FileManager.calculate_sha256(file_path)

        # Build schema metadata
        schema_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
        if custom_metadata:
            schema_dict["_custom_meta"] = json.dumps(custom_metadata)

        metadata = SnapshotMetadata(
            snapshot_id=snapshot_id,
            run_id=run_id,
            project_id=project_id,
            file_path=str(file_path),
            file_hash=file_hash,
            row_count=len(df),
            column_count=len(df.columns),
            schema_metadata=schema_dict
        )

        # Persist to database
        DatabaseService.save_snapshot_record(metadata.model_dump())
        # Also persist as run artifact
        DatabaseService.save_artifact(snapshot_id, run_id, "DataSnapshot", metadata.model_dump())

        return metadata

    @classmethod
    def load_snapshot(cls, run_id: str, verify_integrity: bool = True) -> pd.DataFrame:
        """
        Load an immutable analytical snapshot for a run and verify SHA-256 integrity.
        """
        sdir = cls.get_snapshots_dir()
        file_path = sdir / f"{run_id}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"Analytical snapshot not found for run: {run_id} at {file_path}")

        # Check recorded hash in DB
        record = DatabaseService.get_snapshot_record(run_id)
        if verify_integrity and record:
            current_hash = FileManager.calculate_sha256(file_path)
            expected_hash = record["file_hash"]
            if current_hash != expected_hash:
                raise ValueError(
                    f"CRITICAL: Snapshot integrity compromised for run {run_id}. "
                    f"Expected hash {expected_hash}, found {current_hash}."
                )

        # Read back via DuckDB
        con = duckdb.connect()
        try:
            sql_path = str(file_path).replace("\\", "/")
            df = con.execute(f"SELECT * FROM read_parquet('{sql_path}')").df()
            return df
        finally:
            con.close()
