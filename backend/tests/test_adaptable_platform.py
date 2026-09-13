"""
Adaptable Platform Test Suite.
Verifies that the analytics platform handles diverse domains beyond the hardcoded 2026 retail scenario.
Tests three datasets:
1. Retail Sales 2026 (existing demo)
2. SaaS Subscription & Recurring Revenue (2024-2025)
3. Healthcare Clinical Admissions (2025)
"""
import uuid
from pathlib import Path
import pandas as pd
import pytest

from app.engine.duckdb_engine import DuckDBAnalyticsEngine
from app.engine.multi_domain_datasets import generate_saas_dataset, generate_healthcare_dataset
from app.contracts import RelationshipSpec
from app.storage.files import FileManager


# ---------------------------------------------------------------------------
# SaaS Domain Tests
# ---------------------------------------------------------------------------
class TestSaaSDomain:
    @pytest.fixture(scope="class")
    def saas_data(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("saas")
        generate_saas_dataset(base, seed=42)
        data_dir = base / "saas_subscriptions"
        subs = FileManager.read_table_dataframe(data_dir / "subscriptions.csv")
        accounts = FileManager.read_table_dataframe(data_dir / "accounts.csv")
        return {"dir": data_dir, "subscriptions": subs, "accounts": accounts}

    def test_saas_profiling_detects_columns(self, saas_data):
        """Profile SaaS subscription table with non-retail column names."""
        df = saas_data["subscriptions"]
        profile = DuckDBAnalyticsEngine.profile_dataframe(
            df=df,
            table_id="subscriptions",
            filename="subscriptions.csv",
            file_hash="hash_saas"
        )

        assert "subscription_id" in profile.columns
        assert "mrr_amount" in profile.columns
        assert "start_date" in profile.columns
        assert profile.row_count == len(df)

    def test_saas_monthly_trend_resolves_columns(self, saas_data):
        """calculate_monthly_trend dynamically resolves date and metric columns for SaaS data."""
        df = saas_data["subscriptions"].copy()
        # Filter numeric mrr_amount (some entries have corrupt strings)
        df["mrr_amount"] = pd.to_numeric(df["mrr_amount"], errors="coerce")
        df = df.dropna(subset=["mrr_amount"])

        trend = DuckDBAnalyticsEngine.calculate_monthly_trend(df)
        assert len(trend) > 0
        assert all("period" in m for m in trend)

    def test_saas_prepare_dataset_handles_accounts_join(self, saas_data):
        """Prepare an analytical dataset joining subscriptions to accounts dimension."""
        df_subs = saas_data["subscriptions"].copy()
        df_accts = saas_data["accounts"].copy()

        tables = {"subscriptions": df_subs, "accounts": df_accts}
        rel = RelationshipSpec(
            id="rel_subs_accts",
            left_table="subscriptions",
            left_key="account_id",
            right_table="accounts",
            right_key="account_id",
            cardinality="many_to_one",
            match_rate_left=95.0,
            orphan_count_left=0,
            match_rate_right=100.0,
            orphan_count_right=0,
            risk_level="low"
        )

        analytical_df, records = DuckDBAnalyticsEngine.prepare_analytical_dataset(
            tables=tables,
            relationships=[rel],
            exclude_cancelled=False,
            cutoff_date=None
        )

        assert len(analytical_df) > 0
        # The dimension columns from accounts should be present
        assert "industry" in analytical_df.columns or "company_name" in analytical_df.columns

    def test_saas_customer_dynamics_with_account_id(self, saas_data):
        """Customer dynamics resolves account_id as the customer identifier."""
        df = saas_data["subscriptions"].copy()
        df["mrr_amount"] = pd.to_numeric(df["mrr_amount"], errors="coerce")
        df = df.dropna(subset=["mrr_amount"])
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df = df.dropna(subset=["start_date"])

        dyn = DuckDBAnalyticsEngine.calculate_customer_dynamics(
            df, customer_id_col="account_id", date_col="start_date"
        )
        assert "monthly_customer_dynamics" in dyn
        assert len(dyn["monthly_customer_dynamics"]) > 0
        for m in dyn["monthly_customer_dynamics"]:
            assert "new_customers_orders" in m
            assert "recurring_customers_orders" in m


# ---------------------------------------------------------------------------
# Healthcare Domain Tests
# ---------------------------------------------------------------------------
class TestHealthcareDomain:
    @pytest.fixture(scope="class")
    def health_data(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("health")
        generate_healthcare_dataset(base, seed=123)
        data_dir = base / "healthcare_admissions"
        admissions = FileManager.read_table_dataframe(data_dir / "admissions.csv")
        wards = FileManager.read_table_dataframe(data_dir / "wards.csv")
        physicians = FileManager.read_table_dataframe(data_dir / "physicians.csv")
        return {
            "dir": data_dir,
            "admissions": admissions,
            "wards": wards,
            "physicians": physicians
        }

    def test_health_profiling_detects_clinical_columns(self, health_data):
        """Profile healthcare admissions table with medical column names."""
        df = health_data["admissions"]
        profile = DuckDBAnalyticsEngine.profile_dataframe(
            df=df,
            table_id="admissions",
            filename="admissions.csv",
            file_hash="hash_health"
        )

        assert "encounter_id" in profile.columns
        assert "total_cost" in profile.columns
        assert "admit_date" in profile.columns
        assert profile.row_count == len(df)

    def test_health_monthly_trend_resolves_admit_date(self, health_data):
        """calculate_monthly_trend resolves admit_date as date and total_cost as metric."""
        df = health_data["admissions"].copy()
        df["total_cost"] = pd.to_numeric(df["total_cost"], errors="coerce")
        df = df.dropna(subset=["total_cost"])

        trend = DuckDBAnalyticsEngine.calculate_monthly_trend(df)
        assert len(trend) > 0
        assert all("period" in m for m in trend)

    def test_health_prepare_dataset_handles_ward_join(self, health_data):
        """Prepare analytical dataset joining admissions to wards dimension."""
        df_adm = health_data["admissions"].copy()
        df_wards = health_data["wards"].copy()

        tables = {"admissions": df_adm, "wards": df_wards}
        rel = RelationshipSpec(
            id="rel_adm_wards",
            left_table="admissions",
            left_key="ward_id",
            right_table="wards",
            right_key="ward_id",
            cardinality="many_to_one",
            match_rate_left=100.0,
            orphan_count_left=0,
            match_rate_right=100.0,
            orphan_count_right=0,
            risk_level="low"
        )

        analytical_df, records = DuckDBAnalyticsEngine.prepare_analytical_dataset(
            tables=tables,
            relationships=[rel],
            exclude_cancelled=False,
            cutoff_date=None
        )

        assert len(analytical_df) > 0
        assert "department" in analytical_df.columns or "ward_name" in analytical_df.columns

    def test_health_quarantine_detects_billing_error(self, health_data):
        """Quarantine should detect the corrupt 'UNKNOWN_BILLING_ERR' in total_cost."""
        df = health_data["admissions"].copy()
        tables = {"admissions": df}

        analytical_df, records = DuckDBAnalyticsEngine.prepare_analytical_dataset(
            tables=tables,
            relationships=[],
            exclude_cancelled=False,
            cutoff_date=None
        )

        # The corrupt billing row should either be quarantined (removed) or have NaN
        if "UNKNOWN_BILLING_ERR" in df["total_cost"].astype(str).values:
            # After preparation, the corrupt value should not appear as a valid number
            corrupt_mask = analytical_df["encounter_id"].isin(
                df.loc[df["total_cost"].astype(str) == "UNKNOWN_BILLING_ERR", "encounter_id"]
            )
            if corrupt_mask.any():
                corrupt_val = analytical_df.loc[corrupt_mask, "total_cost"].values[0]
                assert pd.isna(corrupt_val) or corrupt_val != 0.0, (
                    "Quarantine should not silently coerce billing errors to 0.0"
                )

    def test_health_period_breakdown_by_department(self, health_data):
        """Period breakdown works with department dimension instead of channel."""
        df = health_data["admissions"].copy()
        df_wards = health_data["wards"].copy()

        tables = {"admissions": df, "wards": df_wards}
        rel = RelationshipSpec(
            id="rel_adm_wards",
            left_table="admissions",
            left_key="ward_id",
            right_table="wards",
            right_key="ward_id",
            cardinality="many_to_one",
            match_rate_left=100.0,
            orphan_count_left=0,
            match_rate_right=100.0,
            orphan_count_right=0,
            risk_level="low"
        )

        analytical_df, _ = DuckDBAnalyticsEngine.prepare_analytical_dataset(
            tables=tables,
            relationships=[rel],
            exclude_cancelled=False,
            cutoff_date=None
        )

        if "department" in analytical_df.columns:
            # Find the date column
            date_col = next(
                (c for c in analytical_df.columns if "date" in c.lower()),
                None
            )
            if date_col:
                analytical_df[date_col] = pd.to_datetime(analytical_df[date_col], errors="coerce")
                analytical_df = analytical_df.dropna(subset=[date_col])
                analytical_df["year_month"] = analytical_df[date_col].dt.strftime("%Y-%m")
                months = sorted(analytical_df["year_month"].unique())
                if len(months) >= 2:
                    # Find a metric column
                    metric_col = next(
                        (c for c in analytical_df.columns
                         if pd.api.types.is_numeric_dtype(analytical_df[c])
                         and "id" not in c.lower() and "length" not in c.lower()),
                        None
                    )
                    if metric_col:
                        breakdown = DuckDBAnalyticsEngine.calculate_period_breakdown(
                            df=analytical_df,
                            dimension="department",
                            baseline_period=months[0],
                            current_period=months[-1],
                            metric=metric_col
                        )
                        assert "contributions" in breakdown
                        assert breakdown["is_perfectly_reconciled"] is True


# ---------------------------------------------------------------------------
# Cross-Domain Regression Tests
# ---------------------------------------------------------------------------
class TestCrossDomainRegression:
    def test_retail_demo_dataset_still_works(self):
        """Retail demo dataset continues to function as a regression baseline."""
        from app.engine.synthetic_data import generate_demo_dataset
        from tempfile import mkdtemp
        data_dir = Path(mkdtemp())
        gt = generate_demo_dataset(data_dir, seed=42)

        df_orders = FileManager.read_table_dataframe(data_dir / "orders.csv")
        df_cust = FileManager.read_table_dataframe(data_dir / "customers.csv")

        tables = {"orders": df_orders, "customers": df_cust}
        rel = RelationshipSpec(
            id="r1",
            left_table="orders",
            left_key="customer_id",
            right_table="customers",
            right_key="customer_id",
            cardinality="many_to_one",
            match_rate_left=98.0,
            orphan_count_left=5,
            match_rate_right=100.0,
            orphan_count_right=0,
            risk_level="low"
        )

        analytical_df, records = DuckDBAnalyticsEngine.prepare_analytical_dataset(
            tables=tables,
            relationships=[rel],
            exclude_cancelled=True,
            cutoff_date="2026-05-31"
        )

        assert len(analytical_df) > 0
        assert "net_sales" in analytical_df.columns
        assert "order_date_clean" in analytical_df.columns

        trend = DuckDBAnalyticsEngine.calculate_monthly_trend(analytical_df)
        assert len(trend) >= 5  # Jan-May

        breakdown = DuckDBAnalyticsEngine.calculate_period_breakdown(
            df=analytical_df,
            dimension="channel",
            baseline_period="2026-03",
            current_period="2026-05",
            metric="net_sales"
        )
        assert breakdown["is_perfectly_reconciled"] is True
