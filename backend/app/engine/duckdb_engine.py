"""
Analytical calculation engine using DuckDB and pandas.
Provides deterministic profiling, data quality auditing, safe transformation,
decomposition, and validation.
"""
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import duckdb
import numpy as np
import pandas as pd
from app.contracts import (
    ColumnProfile,
    DataCatalog,
    DataQualityReport,
    DimensionContribution,
    JoinCheckResult,
    QualityIssue,
    RelationshipSpec,
    TableProfile,
    TransformationRecord,
)
from app.quality.quarantine import QuarantineManager



class DuckDBAnalyticsEngine:
    def __init__(self):
        self.conn = duckdb.connect(":memory:")

    # -----------------------------------------------------------------------
    # Profiling & Data Quality Audit
    # -----------------------------------------------------------------------
    @staticmethod
    def profile_dataframe(df: pd.DataFrame, table_id: str, filename: str, file_hash: str, sheet_name: Optional[str] = None) -> TableProfile:
        """Deterministically profile columns, detect types, leading zeros, and nulls."""
        row_count = len(df)
        column_count = len(df.columns)
        columns_profile: Dict[str, ColumnProfile] = {}

        date_cols = []
        id_cols = []
        num_cols = []
        cat_cols = []

        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum()) + int((series == "").sum())
            null_pct = round((null_count / max(row_count, 1)) * 100, 2)
            non_null = series[series.notna() & (series != "")].astype(str)
            unique_count = int(non_null.nunique())

            # Detect leading zeros: string with length > 1, starts with '0', contains only digits
            has_leading_zeros = bool(non_null.str.match(r"^0\d+$").any())

            # Type inference
            inferred_type = "string"
            min_val = None
            max_val = None
            mean_val = None
            std_val = None

            # Test numeric
            numeric_converted = pd.to_numeric(non_null.str.replace(",", "."), errors="coerce")
            num_valid = numeric_converted.notna().sum()

            # Test date
            date_converted = pd.to_datetime(non_null, errors="coerce", format="mixed")
            date_valid = date_converted.notna().sum()

            if has_leading_zeros or "id" in col.lower() or "code" in col.lower():
                inferred_type = "string"
                id_cols.append(col)
            elif num_valid == len(non_null) and len(non_null) > 0 and not has_leading_zeros:
                if (numeric_converted % 1 == 0).all():
                    inferred_type = "integer"
                else:
                    inferred_type = "float"
                num_cols.append(col)
                min_val = float(numeric_converted.min())
                max_val = float(numeric_converted.max())
                mean_val = float(numeric_converted.mean())
                std_val = float(numeric_converted.std()) if len(numeric_converted) > 1 else 0.0
            elif date_valid == len(non_null) and len(non_null) > 0 and ("date" in col.lower() or "fecha" in col.lower() or date_valid > 5):
                inferred_type = "date"
                date_cols.append(col)
                min_val = str(date_converted.min().date())
                max_val = str(date_converted.max().date())
            else:
                inferred_type = "string"
                if unique_count < 50:
                    cat_cols.append(col)

            # Sample values up to 5
            samples = non_null.head(5).tolist()

            columns_profile[col] = ColumnProfile(
                name=col,
                inferred_type=inferred_type,
                total_count=row_count,
                null_count=null_count,
                null_percentage=null_pct,
                unique_count=unique_count,
                has_leading_zeros=has_leading_zeros,
                sample_values=samples,
                min_value=min_val,
                max_value=max_val,
                mean_value=mean_val,
                std_dev=std_val
            )

        # Semantic meaning
        meaning = "General table"
        if "order_id" in df.columns or "orden_id" in df.columns or "venta" in table_id.lower():
            meaning = "Transactional order/sales facts (one record per order line or order)."
        elif "customer_id" in df.columns or "cliente" in table_id.lower():
            meaning = "Customer dimension (one record per customer)."
        elif "product_id" in df.columns or "producto" in table_id.lower():
            meaning = "Product dimension (one record per product)."
        elif "campaign_id" in df.columns or "campaña" in table_id.lower():
            meaning = "Marketing campaign dimension."

        return TableProfile(
            table_id=table_id,
            source_filename=filename,
            sheet_name=sheet_name,
            display_name=table_id.replace("_", " ").title(),
            row_count=row_count,
            column_count=column_count,
            columns=columns_profile,
            file_hash_sha256=file_hash,
            row_semantic_meaning=meaning,
            date_columns=date_cols,
            id_columns=id_cols,
            numeric_columns=num_cols,
            categorical_columns=cat_cols
        )

    @staticmethod
    def audit_quality(
        tables: Dict[str, pd.DataFrame],
        catalog: DataCatalog,
        project_id: str
    ) -> Tuple[DataQualityReport, List[RelationshipSpec]]:
        """
        Audit tables for quality issues:
        - invalid dates
        - leading zeros preserved
        - duplicate dimension keys
        - orphan records across relations
        - incomplete final period
        - returns and cancellations
        - missing values
        """
        issues: List[QualityIssue] = []
        relationships: List[RelationshipSpec] = []
        has_blocking = False

        coverage_start = None
        coverage_end = None
        is_last_period_incomplete = False
        incomplete_details = None

        # 1. Audit individual tables
        for table_id, df in tables.items():
            total_rows = len(df)
            if total_rows == 0:
                issues.append(QualityIssue(
                    id=f"EMPTY_{table_id}",
                    issue_type="empty_file",
                    severity="blocking",
                    table_id=table_id,
                    affected_rows=0,
                    total_rows=0,
                    affected_percentage=100.0,
                    description=f"Table {table_id} is empty.",
                    recommended_action="Upload a file with data."
                ))
                has_blocking = True
                continue

            # Check duplicate primary keys in dimension tables
            profile = catalog.tables.get(table_id)
            if profile and ("customer" in table_id.lower() or "product" in table_id.lower()):
                id_col = [c for c in df.columns if "id" in c.lower()]
                if id_col:
                    pk = id_col[0]
                    dup_counts = df[pk].value_counts()
                    dups = dup_counts[dup_counts > 1]
                    if len(dups) > 0:
                        affected = int(dups.sum())
                        dup_samples = dups.head(3).index.tolist()
                        issues.append(QualityIssue(
                            id=f"DUP_KEY_{table_id}_{pk}",
                            issue_type="duplicate_keys",
                            severity="warning",
                            table_id=table_id,
                            column=pk,
                            affected_rows=affected,
                            total_rows=total_rows,
                            affected_percentage=round((affected / total_rows) * 100, 2),
                            evidence_samples=dup_samples,
                            description=f"Duplicate keys found in dimension {table_id} under column {pk}.",
                            recommended_action="Deduplicate dimension preserving most recent record before joining."
                        ))

            # Check invalid dates & time coverage
            for col in df.columns:
                col_lower = col.lower()
                if "date" in col_lower or "fecha" in col_lower:
                    non_null = df[col][df[col].notna() & (df[col] != "")].astype(str)
                    parsed = pd.to_datetime(non_null, errors="coerce")
                    invalid_mask = parsed.isna()
                    invalid_count = int(invalid_mask.sum())
                    if invalid_count > 0:
                        invalid_samples = non_null[invalid_mask].head(3).tolist()
                        issues.append(QualityIssue(
                            id=f"INVALID_DATE_{table_id}_{col}",
                            issue_type="invalid_date",
                            severity="warning",
                            table_id=table_id,
                            column=col,
                            affected_rows=invalid_count,
                            total_rows=total_rows,
                            affected_percentage=round((invalid_count / total_rows) * 100, 2),
                            evidence_samples=invalid_samples,
                            description=f"Detected {invalid_count} invalid or corrupt dates in {table_id}.{col}.",
                            recommended_action="Exclude rows with invalid dates from time series analysis and record their exclusion."
                        ))

                    valid_dates = parsed.dropna()
                    if len(valid_dates) > 0 and ("order" in table_id.lower() or "venta" in table_id.lower()):
                        start_str = valid_dates.min().strftime("%Y-%m-%d")
                        end_str = valid_dates.max().strftime("%Y-%m-%d")
                        coverage_start = start_str if not coverage_start else min(coverage_start, start_str)
                        coverage_end = end_str if not coverage_end else max(coverage_end, end_str)

                        # Check if last month is incomplete
                        dt_series = valid_dates.dt.to_period("M")
                        last_month = dt_series.max()
                        last_month_dates = valid_dates[dt_series == last_month]
                        days_in_last_month = last_month_dates.dt.day.nunique()
                        total_days_month = last_month.end_time.day

                        if dt_series.nunique() > 1 and days_in_last_month < 15 and days_in_last_month < total_days_month:
                            is_last_period_incomplete = True
                            incomplete_details = (
                                f"The final month ({last_month}) contains only {days_in_last_month} days of records. "
                                f"Comparing this month directly with preceding full months will cause a false contraction distortion."
                            )
                            issues.append(QualityIssue(
                                id=f"INCOMPLETE_PERIOD_{table_id}_{col}",
                                issue_type="incomplete_period",
                                severity="warning",
                                table_id=table_id,
                                column=col,
                                affected_rows=len(last_month_dates),
                                total_rows=total_rows,
                                affected_percentage=round((len(last_month_dates) / total_rows) * 100, 2),
                                evidence_samples=[f"Days recorded in {last_month}: {days_in_last_month} of {total_days_month}"],
                                description=incomplete_details,
                                recommended_action="Bound comparative analytical window to closed full months (e.g., January to May)."
                            ))

            # Check returns / cancellations
            if "status" in df.columns or "estado" in df.columns:
                st_col = "status" if "status" in df.columns else "estado"
                non_active = df[df[st_col].isin(["CANCELLED", "REFUNDED", "CANCELADO", "DEVUELTO"])]
                if len(non_active) > 0:
                    issues.append(QualityIssue(
                        id=f"RETURNS_CANCELLATIONS_{table_id}",
                        issue_type="returns_cancellations",
                        severity="info",
                        table_id=table_id,
                        column=st_col,
                        affected_rows=len(non_active),
                        total_rows=total_rows,
                        affected_percentage=round((len(non_active) / total_rows) * 100, 2),
                        evidence_samples=[f"{k}: {v}" for k, v in non_active[st_col].value_counts().items()],
                        description=f"There are {len(non_active)} orders with cancelled or refunded status.",
                        recommended_action="Filter completed orders to calculate actual effective accrued net sales."
                    ))

            # Check preserved leading zeros
            for col, cprof in (profile.columns.items() if profile else []):
                if cprof.has_leading_zeros:
                    issues.append(QualityIssue(
                        id=f"LEADING_ZEROS_{table_id}_{col}",
                        issue_type="leading_zeros_preserved",
                        severity="info",
                        table_id=table_id,
                        column=col,
                        affected_rows=cprof.total_count - cprof.null_count,
                        total_rows=total_rows,
                        affected_percentage=100.0 - cprof.null_percentage,
                        evidence_samples=cprof.sample_values,
                        description=f"Column {col} contains identifiers with leading zeros (e.g., {cprof.sample_values[:2]}).",
                        recommended_action="Safely preserved text type to prevent loss of leading digits in joins."
                    ))

        # 2. Check Relationships and Orphans
        table_names = list(tables.keys())
        fact_tables = [t for t in table_names if "order" in t.lower() or "venta" in t.lower()]
        dim_tables = [t for t in table_names if t not in fact_tables]

        for ft in fact_tables:
            df_fact = tables[ft]
            for dt in dim_tables:
                df_dim = tables[dt]
                # Find candidate joining keys
                common_cols = set(df_fact.columns).intersection(set(df_dim.columns))
                candidate_keys = [c for c in common_cols if "id" in c.lower() or "cod" in c.lower()]

                for key in candidate_keys:
                    fact_keys = df_fact[key].dropna().astype(str)
                    dim_keys = set(df_dim[key].dropna().astype(str))

                    # Orphans in fact table (fact rows with no matching dimension row)
                    orphan_mask = ~fact_keys.isin(dim_keys)
                    orphan_count_left = int(orphan_mask.sum())
                    match_rate_left = round(((len(fact_keys) - orphan_count_left) / max(len(fact_keys), 1)) * 100, 2)

                    # Orphans in dim (dim records never used in fact)
                    fact_keys_set = set(fact_keys)
                    orphan_count_right = sum(1 for dk in dim_keys if dk not in fact_keys_set)
                    match_rate_right = round(((len(dim_keys) - orphan_count_right) / max(len(dim_keys), 1)) * 100, 2)

                    # Check right table cardinality (is key unique in dimension?)
                    dim_key_series = df_dim[key].dropna().astype(str)
                    is_dim_unique = dim_key_series.nunique() == len(dim_key_series)
                    cardinality = "many_to_one" if is_dim_unique else "many_to_many"
                    risk = "high" if cardinality == "many_to_many" else ("medium" if orphan_count_left > 0 else "low")

                    rel_id = f"REL_{ft}_{dt}_{key}"
                    relationships.append(RelationshipSpec(
                        id=rel_id,
                        left_table=ft,
                        left_key=key,
                        right_table=dt,
                        right_key=key,
                        cardinality=cardinality,
                        match_rate_left=match_rate_left,
                        orphan_count_left=orphan_count_left,
                        match_rate_right=match_rate_right,
                        orphan_count_right=orphan_count_right,
                        risk_level=risk,
                        is_confirmed=False
                    ))

                    if orphan_count_left > 0:
                        orphan_samples = fact_keys[orphan_mask].head(3).tolist()
                        issues.append(QualityIssue(
                            id=f"ORPHAN_{ft}_{dt}_{key}",
                            issue_type="orphan_records",
                            severity="warning",
                            table_id=ft,
                            column=key,
                            affected_rows=orphan_count_left,
                            total_rows=len(df_fact),
                            affected_percentage=round((orphan_count_left / len(df_fact)) * 100, 2),
                            evidence_samples=orphan_samples,
                            description=f"There are {orphan_count_left} records in {ft} with key '{key}' not found in {dt} (e.g., {orphan_samples}).",
                            recommended_action=f"Perform LEFT JOIN preserving orphan orders with dimension marked as 'Unclassified / Unknown'."
                        ))

                    if cardinality == "many_to_many":
                        issues.append(QualityIssue(
                            id=f"M2M_{ft}_{dt}_{key}",
                            issue_type="many_to_many",
                            severity="warning",
                            table_id=dt,
                            column=key,
                            affected_rows=len(df_dim) - dim_key_series.nunique(),
                            total_rows=len(df_dim),
                            affected_percentage=round(((len(df_dim) - dim_key_series.nunique()) / len(df_dim)) * 100, 2),
                            evidence_samples=df_dim[key].value_counts().head(2).index.tolist(),
                            description=f"Key {key} in {dt} is not unique. A direct join will artificially multiply sales figures.",
                            recommended_action=f"Deduplicate {dt} by {key} prior to joining to guarantee exact many_to_one cardinality."
                        ))

        report = DataQualityReport(
            schema_version="1.0",
            project_id=project_id,
            issues=issues,
            has_blocking_issues=has_blocking,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            is_last_period_incomplete=is_last_period_incomplete,
            incomplete_period_details=incomplete_details
        )
        return report, relationships

    # -----------------------------------------------------------------------
    # Data Preparation & Safe Joins
    # -----------------------------------------------------------------------
    @staticmethod
    def prepare_analytical_dataset(
        tables: Dict[str, pd.DataFrame],
        relationships: List[RelationshipSpec],
        exclude_cancelled: bool = True,
        cutoff_date: Optional[str] = None,
        semantic_model: Optional[Any] = None
    ) -> Tuple[pd.DataFrame, List[TransformationRecord]]:
        """
        Execute clean, fully tracked and reproducible data preparation:
        1. Clean orders: parse dates, exclude invalid dates, record exclusions.
        2. Filter order status (keep completed).
        3. Exclude incomplete periods if configured.
        4. Deduplicate dimensions (e.g. customers, products) to prevent many-to-many multiplication.
        5. Safe join with metric reconciliation before and after.
        """
        records: List[TransformationRecord] = []
        step_idx = 1

        # Identify fact table
        fact_key = next((t for t in tables.keys() if "order" in t.lower() or "venta" in t.lower()), list(tables.keys())[0])
        df_orders = tables[fact_key].copy()
        initial_order_count = len(df_orders)

        # 1. Clean Dates
        date_col = next((c for c in df_orders.columns if "date" in c.lower() or "fecha" in c.lower()), None)
        if date_col:
            parsed_dates = pd.to_datetime(df_orders[date_col], errors="coerce")
            invalid_date_mask = parsed_dates.isna()
            invalid_date_count = int(invalid_date_mask.sum())
            excluded_samples = df_orders[invalid_date_mask][[df_orders.columns[0], date_col]].to_dict(orient="records")

            df_orders = df_orders[~invalid_date_mask].copy()
            df_orders["order_date_clean"] = parsed_dates[~invalid_date_mask]

            records.append(TransformationRecord(
                transformation_id="TRF_01_CLEAN_DATES",
                step_index=step_idx,
                input_table=fact_key,
                output_table="orders_valid_dates",
                operation="filter_invalid_dates",
                columns_affected=[date_col],
                rationale="Removal of records with corrupt or invalid dates to ensure time series integrity.",
                parameters={"date_column": date_col},
                rows_before=initial_order_count,
                rows_after=len(df_orders),
                rows_excluded=invalid_date_count,
                exclusion_reason="Invalid or corrupt date",
                exclusion_samples=excluded_samples[:5]
            ))
            step_idx += 1

        # 2. Status filter
        status_col = next((c for c in df_orders.columns if c.lower() in ["status", "estado"]), None)
        if exclude_cancelled and status_col:
            count_before = len(df_orders)
            cancelled_mask = df_orders[status_col].isin(["CANCELLED", "REFUNDED", "CANCELADO", "DEVUELTO"])
            cancelled_count = int(cancelled_mask.sum())
            cancelled_samples = df_orders[cancelled_mask][[df_orders.columns[0], status_col]].head(5).to_dict(orient="records")

            df_orders = df_orders[~cancelled_mask].copy()
            records.append(TransformationRecord(
                transformation_id="TRF_02_FILTER_STATUS",
                step_index=step_idx,
                input_table="orders_valid_dates",
                output_table="orders_completed",
                operation="filter_completed_orders",
                columns_affected=[status_col],
                rationale="Exclusive retention of completed transactions to measure actual accrued net sales.",
                parameters={"status_column": status_col, "allowed_values": ["COMPLETED", "COMPLETADO"]},
                rows_before=count_before,
                rows_after=len(df_orders),
                rows_excluded=cancelled_count,
                exclusion_reason="Cancelled or refunded order",
                exclusion_samples=cancelled_samples
            ))
            step_idx += 1

        # 3. Cutoff incomplete period
        if cutoff_date and "order_date_clean" in df_orders.columns:
            count_before = len(df_orders)
            cutoff_dt = pd.to_datetime(cutoff_date)
            incomplete_mask = df_orders["order_date_clean"] > cutoff_dt
            incomplete_count = int(incomplete_mask.sum())

            if incomplete_count > 0:
                incomplete_samples = df_orders[incomplete_mask][[df_orders.columns[0], "order_date_clean"]].head(5).to_dict(orient="records")
                df_orders = df_orders[~incomplete_mask].copy()
                records.append(TransformationRecord(
                    transformation_id="TRF_03_CUTOFF_INCOMPLETE_PERIOD",
                    step_index=step_idx,
                    input_table="orders_completed",
                    output_table="orders_closed_periods",
                    operation="filter_date_range",
                    columns_affected=["order_date_clean"],
                    rationale=f"Exclusion of days from incomplete period after {cutoff_date} to prevent false contraction bias.",
                    parameters={"cutoff_date": cutoff_date},
                    rows_before=count_before,
                    rows_after=len(df_orders),
                    rows_excluded=incomplete_count,
                    exclusion_reason=f"Record post-dates closed monthly period ({cutoff_date})",
                    exclusion_samples=incomplete_samples
                ))
                step_idx += 1

        # Numeric casts and auditable quarantine on fact table
        numeric_cols_to_audit = [
            c for c in df_orders.columns
            if c in ["units", "unit_price", "discount_amount", "gross_sales", "net_sales"]
            or (semantic_model and hasattr(semantic_model, "metrics") and any(m.name == c for m in semantic_model.metrics))
            or pd.api.types.is_numeric_dtype(df_orders[c])
        ]
        df_orders, num_quarantined = QuarantineManager.audit_and_clean_numerics(
            df=df_orders,
            table_name=fact_key,
            numeric_columns=numeric_cols_to_audit
        )
        if num_quarantined:
            records.append(TransformationRecord(
                transformation_id=f"TRF_{step_idx:02d}_QUARANTINE_NUMERICS",
                step_index=step_idx,
                input_table=fact_key,
                output_table=fact_key,
                operation="quarantine_corrupt_numerics",
                columns_affected=list({q.affected_column for q in num_quarantined}),
                rationale="Isolate non-numeric corrupt values into quarantine to preserve mathematical integrity.",
                rows_before=len(df_orders) + len(num_quarantined),
                rows_after=len(df_orders),
                rows_excluded=len(num_quarantined),
                exclusion_reason="Unparseable non-numeric characters detected"
            ))
            step_idx += 1

        # Baseline net_sales metric to reconcile across joins
        baseline_net_sales = float(df_orders["net_sales"].sum()) if "net_sales" in df_orders.columns else 0.0

        # 4. Safe Joins with Dimensions
        # Deduplicate dimension tables before join
        analytical_df = df_orders.copy()

        for rel in relationships:
            dim_name = rel.right_table
            if dim_name not in tables:
                continue

            dim_df = tables[dim_name].copy()
            right_key = rel.right_key
            left_key = rel.left_key

            if left_key not in analytical_df.columns or right_key not in dim_df.columns:
                continue

            # Ensure right key is deduplicated with conflict isolation
            dim_rows_before = len(dim_df)
            dim_df_dedup, conflict_quarantined = QuarantineManager.isolate_conflicting_duplicates(
                df=dim_df,
                table_name=dim_name,
                key_column=right_key
            )
            dups_removed = dim_rows_before - len(dim_df_dedup)
            if conflict_quarantined:
                records.append(TransformationRecord(
                    transformation_id=f"TRF_{step_idx:02d}_QUARANTINE_CONFLICTING_DUPS",
                    step_index=step_idx,
                    input_table=dim_name,
                    output_table=f"{dim_name}_clean",
                    operation="quarantine_conflicting_duplicates",
                    columns_affected=[right_key],
                    rationale=f"Isolate conflicting duplicate records for {right_key} into quarantine rather than picking arbitrary first row.",
                    rows_before=dim_rows_before,
                    rows_after=len(dim_df_dedup),
                    rows_excluded=len(conflict_quarantined),
                    exclusion_reason="Conflicting attribute values for identical dimension key"
                ))
                step_idx += 1

            # Left join
            rows_before_join = len(analytical_df)
            sales_before_join = float(analytical_df["net_sales"].sum()) if "net_sales" in analytical_df.columns else 0.0

            # Disambiguate overlapping columns
            overlap_cols = [c for c in dim_df_dedup.columns if c in analytical_df.columns and c != right_key]
            if overlap_cols:
                dim_df_dedup = dim_df_dedup.rename(columns={c: f"{dim_name}_{c}" for c in overlap_cols})

            merged = pd.merge(
                analytical_df,
                dim_df_dedup,
                left_on=left_key,
                right_on=right_key,
                how="left"
            )

            rows_after_join = len(merged)
            sales_after_join = float(merged["net_sales"].sum()) if "net_sales" in merged.columns else 0.0
            multiplication_factor = round(rows_after_join / max(rows_before_join, 1), 4)

            # Check safety
            is_safe = (multiplication_factor == 1.0) and (abs(sales_after_join - sales_before_join) < 0.01)
            join_warning = None
            if not is_safe:
                join_warning = f"Join alert: multiplication factor = {multiplication_factor}, sales discrepancy = {sales_after_join - sales_before_join:.2f}"

            join_check = JoinCheckResult(
                left_table="analytical_fact",
                right_table=dim_name,
                join_type="LEFT",
                join_keys=[f"{left_key} = {right_key}"],
                left_rows=rows_before_join,
                right_rows=len(dim_df_dedup),
                result_rows=rows_after_join,
                multiplication_factor=multiplication_factor,
                orphans_left=rel.orphan_count_left,
                orphans_right=rel.orphan_count_right,
                metric_reconciled=abs(sales_after_join - sales_before_join) < 0.01,
                is_safe=is_safe,
                warning=join_warning
            )

            records.append(TransformationRecord(
                transformation_id=f"TRF_JOIN_{dim_name.upper()}",
                step_index=step_idx,
                input_table="analytical_fact",
                output_table=f"analytical_with_{dim_name}",
                operation="safe_left_join",
                columns_affected=[left_key, right_key],
                rationale=f"Controlled join with dimension {dim_name} following key deduplication to prevent artificial sales inflation.",
                parameters={"join_type": "left", "left_key": left_key, "right_key": right_key, "dups_removed_in_dim": dups_removed},
                rows_before=rows_before_join,
                rows_after=rows_after_join,
                rows_excluded=0,
                join_check=join_check
            ))
            step_idx += 1
            analytical_df = merged

        # Fill missing dimension values as 'Sin clasificar / No atribuido'
        for col in analytical_df.select_dtypes(include=["object"]).columns:
            analytical_df[col] = analytical_df[col].fillna("Unclassified")

        return analytical_df, records

    # -----------------------------------------------------------------------
    # Analytical Operations (Decomposition, Dynamics, Metrics)
    # -----------------------------------------------------------------------
    @staticmethod
    def calculate_period_breakdown(
        df: pd.DataFrame,
        dimension: str,
        baseline_period: str = "2026-03",
        current_period: str = "2026-05",
        metric: str = "net_sales"
    ) -> Dict[str, Any]:
        """
        Decompose variation across a mutually exclusive dimension.
        Enforces 100% mathematical reconciliation:
        sum(absolute_change) == total_change
        """
        df_copy = df.copy()
        if dimension not in df_copy.columns:
            df_copy[dimension] = "Unclassified"
        df_copy["year_month"] = pd.to_datetime(df_copy["order_date_clean"]).dt.strftime("%Y-%m")

        df_base = df_copy[df_copy["year_month"] == baseline_period]
        df_curr = df_copy[df_copy["year_month"] == current_period]

        total_base = float(df_base[metric].sum())
        total_curr = float(df_curr[metric].sum())
        total_change = float(total_curr - total_base)
        total_pct_change = round((total_change / total_base) * 100, 2) if total_base != 0 else 0.0

        # Group by dimension
        dim_base = df_base.groupby(dimension, observed=False)[metric].sum().to_dict()
        dim_curr = df_curr.groupby(dimension, observed=False)[metric].sum().to_dict()

        all_keys = sorted(list(set(dim_base.keys()).union(set(dim_curr.keys()))))
        contributions: List[DimensionContribution] = []

        reconciled_sum = 0.0
        for k in all_keys:
            v_base = float(dim_base.get(k, 0.0))
            v_curr = float(dim_curr.get(k, 0.0))
            delta = float(v_curr - v_base)
            pct_chg = round((delta / v_base) * 100, 2) if v_base != 0 else 0.0
            # Contribution to total decline: delta / total_change
            contrib_pct = round((delta / total_change) * 100, 2) if total_change != 0 else 0.0

            reconciled_sum += delta

            contributions.append(DimensionContribution(
                dimension_value=str(k),
                baseline_value=round(v_base, 2),
                current_value=round(v_curr, 2),
                absolute_change=round(delta, 2),
                percentage_change=pct_chg,
                contribution_to_total_change=contrib_pct
            ))

        # Check reconciliation difference
        reconciliation_diff = round(abs(total_change - reconciled_sum), 4)

        return {
            "dimension": dimension,
            "metric": metric,
            "baseline_period": baseline_period,
            "current_period": current_period,
            "total_baseline": round(total_base, 2),
            "total_current": round(total_curr, 2),
            "total_change": round(total_change, 2),
            "total_percentage_change": total_pct_change,
            "reconciliation_diff": reconciliation_diff,
            "is_perfectly_reconciled": reconciliation_diff < 0.01,
            "contributions": [c.model_dump() for c in contributions]
        }

    @staticmethod
    def calculate_customer_dynamics(
        df: pd.DataFrame,
        customer_id_col: Optional[str] = None,
        date_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculates new vs recurring customer sales and order counts per month.
        Definition:
        - Customer first order date = min(date_col) across all history.
        - An order is 'new_customer' if its date == customer_first_order_date.
        - Otherwise, 'recurring_customer'.
        """
        df_copy = df.copy()
        cust_col = customer_id_col if (customer_id_col and customer_id_col in df_copy.columns) else next((c for c in df_copy.columns if "cust" in c.lower() or "client" in c.lower() or "user" in c.lower()), "customer_id")
        dt_col = date_col if (date_col and date_col in df_copy.columns) else ("order_date_clean" if "order_date_clean" in df_copy.columns else next((c for c in df_copy.columns if "date" in c.lower()), df_copy.columns[0]))

        if cust_col not in df_copy.columns or dt_col not in df_copy.columns:
            return {"monthly_customer_dynamics": []}

        first_dates = df_copy.groupby(cust_col)[dt_col].min().to_dict()
        df_copy["customer_first_date"] = df_copy[cust_col].map(first_dates)
        try:
            df_copy["is_new_customer"] = df_copy[dt_col].dt.date == df_copy["customer_first_date"].dt.date
            df_copy["year_month"] = df_copy[dt_col].dt.strftime("%Y-%m")
        except Exception:
            df_copy["is_new_customer"] = df_copy[dt_col] == df_copy["customer_first_date"]
            df_copy["year_month"] = df_copy[dt_col].astype(str).str.slice(0, 7)

        metric_col = "net_sales" if "net_sales" in df_copy.columns else next((c for c in df_copy.columns if pd.api.types.is_numeric_dtype(df_copy[c]) and "id" not in c.lower()), None)

        monthly_dyn = []
        for ym, group in df_copy.groupby("year_month"):
            new_orders = group[group["is_new_customer"]]
            rec_orders = group[~group["is_new_customer"]]

            new_sales = round(float(new_orders[metric_col].sum()), 2) if metric_col else 0.0
            rec_sales = round(float(rec_orders[metric_col].sum()), 2) if metric_col else 0.0
            tot_sales = round(float(group[metric_col].sum()), 2) if metric_col else 0.0

            monthly_dyn.append({
                "period": str(ym),
                "new_customers_sales": new_sales,
                "recurring_customers_sales": rec_sales,
                "new_customers_orders": int(len(new_orders)),
                "recurring_customers_orders": int(len(rec_orders)),
                "total_sales": tot_sales
            })

        return {"monthly_customer_dynamics": monthly_dyn}

    @staticmethod
    def calculate_monthly_trend(
        df: pd.DataFrame,
        date_col: Optional[str] = None,
        metric_col: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Monthly aggregated metrics without hardcoding column names."""
        if df.empty:
            return []

        df_copy = df.copy()

        # Resolve date column
        if not date_col or date_col not in df_copy.columns:
            date_candidates = [c for c in df_copy.columns if "date" in c.lower() or "fecha" in c.lower() or "time" in c.lower()]
            date_col = date_candidates[0] if date_candidates else df_copy.columns[0]

        parsed_dt = pd.to_datetime(df_copy[date_col], errors="coerce")
        df_copy["year_month"] = parsed_dt.dt.strftime("%Y-%m")
        df_copy = df_copy[df_copy["year_month"].notna()]
        if df_copy.empty:
            return []

        # Resolve primary metric
        if not metric_col or metric_col not in df_copy.columns:
            if "net_sales" in df_copy.columns:
                metric_col = "net_sales"
            else:
                num_cols = [c for c in df_copy.columns if pd.api.types.is_numeric_dtype(df_copy[c]) and "id" not in c.lower() and c != "year_month"]
                metric_col = num_cols[0] if num_cols else df_copy.columns[0]

        res = []
        for ym, grp in df_copy.groupby("year_month"):
            entry = {
                "period": ym,
                metric_col: round(float(grp[metric_col].sum()), 2) if metric_col in grp.columns and pd.api.types.is_numeric_dtype(grp[metric_col]) else 0.0,
                "gross_sales": round(float(grp["gross_sales"].sum()), 2) if "gross_sales" in grp.columns else 0.0,
                "units": int(grp["units"].sum()) if "units" in grp.columns else 0,
                "order_count": int(len(grp)),
                "avg_order_value": round(float(grp[metric_col].mean()), 2) if metric_col in grp.columns and pd.api.types.is_numeric_dtype(grp[metric_col]) else 0.0
            }
            # Maintain backward compatibility alias
            if metric_col != "net_sales":
                entry["net_sales"] = entry[metric_col]
            res.append(entry)
        return sorted(res, key=lambda x: x["period"])

    @staticmethod
    def evaluate_forecast_eligibility(monthly_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates whether historical series meets minimum statistical requirements
        for a reliable forecast.
        Requirement: at least 12 periods for monthly seasonality, or baseline comparison.
        """
        n_periods = len(monthly_series)
        if n_periods < 12:
            return {
                "eligible": False,
                "reason": f"Insufficient historical depth: {n_periods} closed months available (at least 12 months required to model annual seasonality without overfitting).",
                "recommended_action": "Maintain moving average or naive baseline projection without claiming seasonal certainty."
            }
        return {"eligible": True}
