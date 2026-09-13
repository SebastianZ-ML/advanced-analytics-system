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
        meaning = "Tabla general"
        if "order_id" in df.columns or "orden_id" in df.columns or "venta" in table_id.lower():
            meaning = "Hechos transaccionales de pedidos/ventas (un registro por línea de pedido u orden)."
        elif "customer_id" in df.columns or "cliente" in table_id.lower():
            meaning = "Dimensión de clientes (un registro por cliente)."
        elif "product_id" in df.columns or "producto" in table_id.lower():
            meaning = "Dimensión de productos (un registro por producto)."
        elif "campaign_id" in df.columns or "campaña" in table_id.lower():
            meaning = "Dimensión de campañas de marketing."

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
                    description=f"La tabla {table_id} está vacía.",
                    recommended_action="Cargar un archivo con datos."
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
                            description=f"Claves duplicadas encontradas en la dimensión {table_id} en la columna {pk}.",
                            recommended_action="Deduplicar la dimensión conservando el registro más reciente antes de hacer uniones."
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
                            description=f"Se detectaron {invalid_count} fechas inválidas o corruptas en {table_id}.{col}.",
                            recommended_action="Excluir filas con fechas inválidas del análisis temporal y registrar su exclusión."
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

                        if days_in_last_month < 15 and days_in_last_month < total_days_month:
                            is_last_period_incomplete = True
                            incomplete_details = (
                                f"El mes final ({last_month}) contiene solo {days_in_last_month} días de registros. "
                                f"Comparar este mes directamente con meses completos anteriores causará una distorsión falsa de caída."
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
                                evidence_samples=[f"Días registrados en {last_month}: {days_in_last_month} de {total_days_month}"],
                                description=incomplete_details,
                                recommended_action="Acotar la comparación analítica a meses enteros cerrados (ej. enero a mayo)."
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
                        description=f"Existen {len(non_active)} pedidos con estado cancelado o devuelto.",
                        recommended_action="Filtrar pedidos completados para calcular facturación neta real efectiva."
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
                        description=f"La columna {col} contiene identificadores con ceros a la izquierda (ej. {cprof.sample_values[:2]}).",
                        recommended_action="Se conservó el tipo texto de forma segura para evitar pérdida de dígitos en uniones."
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
                            description=f"Existen {orphan_count_left} registros en {ft} con claves '{key}' que no existen en {dt} (ej. {orphan_samples}).",
                            recommended_action=f"Realizar LEFT JOIN preservando los pedidos huérfanos con dimensión marcada como 'Sin clasificar / Desconocido'."
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
                            description=f"La clave {key} en {dt} no es única. Una unión directa multiplicará artificialmente los montos de venta.",
                            recommended_action=f"Deduplicar {dt} por {key} antes de unir para garantizar cardinalidad many_to_one exacta."
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
        cutoff_date: Optional[str] = "2026-05-31"  # Cutoff incomplete June
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
                rationale="Eliminación de registros con fechas corruptas o inválidas para asegurar consistencia de la serie temporal.",
                parameters={"date_column": date_col},
                rows_before=initial_order_count,
                rows_after=len(df_orders),
                rows_excluded=invalid_date_count,
                exclusion_reason="Fecha inválida o corrupta",
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
                rationale="Conservación exclusiva de transacciones completadas para medir facturación neta real devengada.",
                parameters={"status_column": status_col, "allowed_values": ["COMPLETED", "COMPLETADO"]},
                rows_before=count_before,
                rows_after=len(df_orders),
                rows_excluded=cancelled_count,
                exclusion_reason="Pedido cancelado o reembolsado",
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
                    rationale=f"Exclusión de días del período incompleto posterior al {cutoff_date} para evitar sesgo de estacionalidad o falsa caída.",
                    parameters={"cutoff_date": cutoff_date},
                    rows_before=count_before,
                    rows_after=len(df_orders),
                    rows_excluded=incomplete_count,
                    exclusion_reason=f"Registro posterior al período mensual cerrado ({cutoff_date})",
                    exclusion_samples=incomplete_samples
                ))
                step_idx += 1

        # Numeric casts on fact table
        for num_col in ["units", "unit_price", "discount_amount", "gross_sales", "net_sales"]:
            if num_col in df_orders.columns:
                df_orders[num_col] = pd.to_numeric(df_orders[num_col].astype(str).str.replace(",", "."), errors="coerce").fillna(0.0)

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

            # Ensure right key is deduplicated
            dim_rows_before = len(dim_df)
            dim_df_dedup = dim_df.drop_duplicates(subset=[right_key], keep="first").copy()
            dups_removed = dim_rows_before - len(dim_df_dedup)

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
                join_warning = f"Alerta de unión: factor de multiplicación = {multiplication_factor}, variación de ventas = {sales_after_join - sales_before_join:.2f}"

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
                rationale=f"Unión controlada con dimensión {dim_name} tras deduplicación de claves para evitar multiplicación artificial de facturación.",
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
            analytical_df[col] = analytical_df[col].fillna("Sin clasificar")

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
            df_copy[dimension] = "Sin clasificar"
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
    def calculate_customer_dynamics(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates new vs recurring customer sales and order counts per month.
        Definition:
        - Customer first order date = min(order_date_clean) across all history.
        - An order is 'new_customer' if its order_date == customer_first_order_date.
        - Otherwise, 'recurring_customer'.
        """
        df_copy = df.copy()
        first_dates = df_copy.groupby("customer_id")["order_date_clean"].min().to_dict()
        df_copy["customer_first_date"] = df_copy["customer_id"].map(first_dates)
        df_copy["is_new_customer"] = df_copy["order_date_clean"].dt.date == df_copy["customer_first_date"].dt.date

        df_copy["year_month"] = df_copy["order_date_clean"].dt.strftime("%Y-%m")

        monthly_dyn = []
        for ym, group in df_copy.groupby("year_month"):
            new_orders = group[group["is_new_customer"]]
            rec_orders = group[~group["is_new_customer"]]

            monthly_dyn.append({
                "period": ym,
                "new_customers_sales": round(float(new_orders["net_sales"].sum()), 2),
                "recurring_customers_sales": round(float(rec_orders["net_sales"].sum()), 2),
                "new_customers_orders": int(len(new_orders)),
                "recurring_customers_orders": int(len(rec_orders)),
                "total_sales": round(float(group["net_sales"].sum()), 2)
            })

        return {"monthly_customer_dynamics": monthly_dyn}

    @staticmethod
    def calculate_monthly_trend(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Monthly aggregated net sales, gross sales, units and order count."""
        df_copy = df.copy()
        df_copy["year_month"] = pd.to_datetime(df_copy["order_date_clean"]).dt.strftime("%Y-%m")

        res = []
        for ym, grp in df_copy.groupby("year_month"):
            res.append({
                "period": ym,
                "net_sales": round(float(grp["net_sales"].sum()), 2),
                "gross_sales": round(float(grp["gross_sales"].sum()), 2),
                "units": int(grp["units"].sum()) if "units" in grp.columns else 0,
                "order_count": int(len(grp)),
                "avg_order_value": round(float(grp["net_sales"].mean()), 2)
            })
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
                "reason": f"Historial insuficiente: se disponen de {n_periods} meses cerrados (se requieren al menos 12 meses para modelar estacionalidad anual sin sobreajuste).",
                "recommended_action": "Mantener proyección basada en promedio móvil o baseline ingenuo sin prometer certeza estacional."
            }
        return {"eligible": True}
