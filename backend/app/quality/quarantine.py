"""
Data Quality & Quarantine Service.
Provides auditable handling of conflicting duplicates, invalid numbers, and unparseable dates.
Replaces silent conversions (.fillna(0.0)) with transparent quarantine isolation.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, Field


class QuarantinedRecord(BaseModel):
    quarantine_id: str
    table_name: str
    record_key: str
    reason: str  # "conflicting_duplicate", "corrupt_numeric", "invalid_date", "orphan_record"
    details: str
    affected_column: str
    raw_payload: Dict[str, Any] = Field(default_factory=dict)
    quarantined_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class QuarantineAuditSummary(BaseModel):
    total_quarantined_rows: int
    reasons_breakdown: Dict[str, int] = Field(default_factory=dict)
    excluded_metric_impact: Dict[str, float] = Field(default_factory=dict)
    quarantined_records_sample: List[QuarantinedRecord] = Field(default_factory=list)


class QuarantineManager:
    @classmethod
    def isolate_conflicting_duplicates(
        cls,
        df: pd.DataFrame,
        table_name: str,
        key_column: str,
        ignore_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[QuarantinedRecord]]:
        """
        Differentiates identical duplicates from conflicting duplicates.
        - Identical duplicates: Safe to deduplicate automatically.
        - Conflicting duplicates: Same key but different attribute values.
          Route all conflicting occurrences to quarantine and log the issue!
        """
        ignore_columns = ignore_columns or []
        cols_to_check = [c for c in df.columns if c not in ignore_columns]
        
        # Check duplicate keys
        dup_keys = df[df.duplicated(subset=[key_column], keep=False)]
        if len(dup_keys) == 0:
            return df.copy(), []

        clean_records = []
        quarantined: List[QuarantinedRecord] = []

        # Group by key
        for key_val, group in df.groupby(key_column):
            if len(group) == 1:
                clean_records.append(group)
            else:
                # Check if group is identical across all checked columns
                is_identical = len(group.drop_duplicates(subset=cols_to_check)) == 1
                if is_identical:
                    # Keep one instance cleanly
                    clean_records.append(group.head(1))
                else:
                    # Conflict! Attributes disagree on the same key!
                    for idx, row in group.iterrows():
                        q_rec = QuarantinedRecord(
                            quarantine_id=f"Q_DUP_{table_name}_{key_val}_{idx}",
                            table_name=table_name,
                            record_key=str(key_val),
                            reason="conflicting_duplicate",
                            details=f"Key '{key_val}' has conflicting attribute values across {len(group)} duplicate records.",
                            affected_column=key_column,
                            raw_payload=row.to_dict()
                        )
                        quarantined.append(q_rec)

        clean_df = pd.concat(clean_records, ignore_index=True) if clean_records else pd.DataFrame(columns=df.columns)
        return clean_df, quarantined

    @classmethod
    def audit_and_clean_numerics(
        cls,
        df: pd.DataFrame,
        table_name: str,
        numeric_columns: List[str]
    ) -> Tuple[pd.DataFrame, List[QuarantinedRecord]]:
        """
        Audits numeric columns: does NOT silently coerce corrupt strings to 0.0.
        Isolates corrupt text into quarantine and preserves legitimate zeros.
        """
        cleaned_df = df.copy()
        quarantined: List[QuarantinedRecord] = []

        for col in numeric_columns:
            if col not in cleaned_df.columns:
                continue

            series = cleaned_df[col]
            # Convert commas to dots if string
            str_series = series.astype(str).str.strip().str.replace(",", ".")
            parsed_numeric = pd.to_numeric(str_series, errors="coerce")

            # Identify corrupt rows: was non-null/non-empty string, but became NaN upon conversion
            is_corrupt = parsed_numeric.isna() & series.notna() & (str_series != "") & (str_series != "nan") & (str_series != "none")

            if is_corrupt.any():
                corrupt_indices = cleaned_df[is_corrupt].index
                for idx in corrupt_indices:
                    raw_row = cleaned_df.loc[idx].to_dict()
                    q_rec = QuarantinedRecord(
                        quarantine_id=f"Q_NUM_{table_name}_{col}_{idx}",
                        table_name=table_name,
                        record_key=str(raw_row.get(cleaned_df.columns[0], idx)),
                        reason="corrupt_numeric",
                        details=f"Column '{col}' contains unparseable non-numeric value: '{cleaned_df.loc[idx, col]}'.",
                        affected_column=col,
                        raw_payload=raw_row
                    )
                    quarantined.append(q_rec)

                # Remove corrupt rows from analytical dataset rather than corrupting totals with arbitrary zero
                cleaned_df = cleaned_df[~is_corrupt].copy()
                parsed_numeric = parsed_numeric[~is_corrupt]

            cleaned_df[col] = parsed_numeric

        return cleaned_df, quarantined
