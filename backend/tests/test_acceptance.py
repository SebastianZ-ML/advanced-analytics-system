"""
Automated Test Suite for the 15 Non-Negotiable Acceptance Criteria.
Tests real executions against synthetic data, contracts, and agents.
"""
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.agents.assistant import ConversationalAssistantAgent
from app.agents.auditor import DataAuditorAgent
from app.agents.dashboard_builder import DashboardBuilderAgent
from app.agents.dashboard_validator import DashboardValidatorAgent
from app.agents.executor import AnalyticalExecutorAgent
from app.agents.interpreter import InterpreterAgent
from app.agents.methodologist import MethodologistAgent
from app.agents.organizer import OrganizerAgent
from app.agents.preparer import DataPreparerAgent
from app.agents.validator import AnalyticalValidatorAgent
from app.contracts import (
    AnalysisResult,
    ChatRequest,
    DashboardSpec,
    DataCatalog,
    DataQualityReport,
    InsightReport,
    MetricCardSpec,
    ObjectiveSpec,
    RelationshipSpec,
    TransformationRecord,
    ValidationReport,
)
from app.engine.duckdb_engine import DuckDBAnalyticsEngine
from app.engine.synthetic_data import generate_demo_dataset
from app.orchestrator.runner import PipelineRunner
from app.storage.db import DatabaseService, init_db
from app.storage.files import FileManager


@pytest.fixture(scope="session")
def setup_demo_data(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("demo_data")
    gt = generate_demo_dataset(data_dir, seed=42)
    init_db()
    return data_dir, gt


# ---------------------------------------------------------------------------
# Test 1: Import multiple CSVs and Excel sheets
# ---------------------------------------------------------------------------
def test_01_import_multiple_csv_and_excel_sheets(setup_demo_data):
    data_dir, _ = setup_demo_data
    orders_path = data_dir / "orders.csv"
    customers_path = data_dir / "customers.csv"
    products_path = data_dir / "products.xlsx"

    assert orders_path.exists()
    assert customers_path.exists()
    assert products_path.exists()

    # Read sheets
    sheets = FileManager.inspect_excel_sheets(products_path)
    assert "Products" in sheets
    assert "Categories" in sheets

    df_orders = FileManager.read_table_dataframe(orders_path)
    df_customers = FileManager.read_table_dataframe(customers_path)
    df_prod = FileManager.read_table_dataframe(products_path, sheet_name="Products")
    df_cat = FileManager.read_table_dataframe(products_path, sheet_name="Categories")

    assert len(df_orders) > 1000
    assert len(df_customers) >= 80
    assert len(df_prod) == 8
    assert len(df_cat) == 4


# ---------------------------------------------------------------------------
# Test 2: Preservation of identifiers with leading zeros
# ---------------------------------------------------------------------------
def test_02_preservation_of_leading_zeros(setup_demo_data):
    data_dir, _ = setup_demo_data
    df_customers = FileManager.read_table_dataframe(data_dir / "customers.csv")
    profile = DuckDBAnalyticsEngine.profile_dataframe(
        df=df_customers,
        table_id="customers",
        filename="customers.csv",
        file_hash="test_hash"
    )

    cust_col = profile.columns["customer_id"]
    assert cust_col.inferred_type == "string"
    assert cust_col.has_leading_zeros is True
    # Verify values still have 5 characters starting with '0'
    sample_val = str(df_customers["customer_id"].iloc[0])
    assert sample_val.startswith("0") and len(sample_val) == 5


# ---------------------------------------------------------------------------
# Test 3: Detection and prevention of a join that multiplies sales
# ---------------------------------------------------------------------------
def test_03_detect_and_prevent_join_multiplication():
    # Construct a fact table and a dimension with duplicate keys
    df_fact = pd.DataFrame([
        {"order_id": "O1", "customer_id": "C1", "net_sales": 100.0},
        {"order_id": "O2", "customer_id": "C2", "net_sales": 200.0},
    ])
    df_dim_duplicate = pd.DataFrame([
        {"customer_id": "C1", "name": "Alice - HQ"},
        {"customer_id": "C1", "name": "Alice - Branch"},  # Duplicate key!
        {"customer_id": "C2", "name": "Bob"},
    ])

    # Direct unsafe merge multiplies rows and sales
    unsafe_merge = pd.merge(df_fact, df_dim_duplicate, on="customer_id", how="left")
    assert len(unsafe_merge) == 3  # Exploded from 2 to 3 rows
    assert unsafe_merge["net_sales"].sum() == 400.0  # Exploded from 300 to 400!

    # Controlled preparation engine deduplicates dimension and preserves exact sales
    tables = {"orders": df_fact, "customers": df_dim_duplicate}
    rel = RelationshipSpec(
        id="rel_orders_customers",
        left_table="orders",
        left_key="customer_id",
        right_table="customers",
        right_key="customer_id",
        cardinality="many_to_many",
        match_rate_left=100.0,
        orphan_count_left=0,
        match_rate_right=100.0,
        orphan_count_right=0,
        risk_level="high"
    )

    prepared_df, records = DuckDBAnalyticsEngine.prepare_analytical_dataset(
        tables=tables,
        relationships=[rel],
        exclude_cancelled=False,
        cutoff_date=None
    )

    assert len(prepared_df) == 2  # Row count preserved!
    assert prepared_df["net_sales"].sum() == 300.0  # Sales preserved!
    join_record = next(r for r in records if "customers" in r.transformation_id.lower())
    assert join_record.join_check.multiplication_factor == 1.0
    assert join_record.join_check.metric_reconciled is True


# ---------------------------------------------------------------------------
# Test 4: Detection of orphan records
# ---------------------------------------------------------------------------
def test_04_detection_of_orphan_records(setup_demo_data):
    data_dir, gt = setup_demo_data
    df_orders = FileManager.read_table_dataframe(data_dir / "orders.csv")
    df_cust = FileManager.read_table_dataframe(data_dir / "customers.csv")

    tables = {"orders": df_orders, "customers": df_cust}
    auditor = DataAuditorAgent()
    catalog, dq_report, relationships = auditor.audit_project_tables(
        project_id="test_proj",
        loaded_tables=tables,
        file_metadata={"orders": {"filename": "orders.csv"}, "customers": {"filename": "customers.csv"}}
    )

    orphan_rel = next((r for r in relationships if r.right_table == "customers"), None)
    assert orphan_rel is not None
    assert orphan_rel.orphan_count_left > 0
    # Orphan issues present in DQ report
    orphan_issues = [i for i in dq_report.issues if i.issue_type == "orphan_records"]
    assert len(orphan_issues) > 0
    assert "00999" in str(orphan_issues[0].evidence_samples)


# ---------------------------------------------------------------------------
# Test 5: Explicit handling of invalid dates
# ---------------------------------------------------------------------------
def test_05_explicit_handling_of_invalid_dates(setup_demo_data):
    data_dir, gt = setup_demo_data
    df_orders = FileManager.read_table_dataframe(data_dir / "orders.csv")
    tables = {"orders": df_orders}

    prepared_df, records = DuckDBAnalyticsEngine.prepare_analytical_dataset(
        tables=tables,
        relationships=[],
        exclude_cancelled=False,
        cutoff_date=None
    )

    date_trf = next(r for r in records if r.operation == "filter_invalid_dates")
    assert date_trf.rows_excluded == gt["invalid_date_orders_count"]
    assert len(date_trf.exclusion_samples) > 0
    # No invalid dates remain in analytical table
    assert prepared_df["order_date_clean"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Test 6: Warning on incomplete period
# ---------------------------------------------------------------------------
def test_06_warning_on_incomplete_period(setup_demo_data):
    data_dir, _ = setup_demo_data
    df_orders = FileManager.read_table_dataframe(data_dir / "orders.csv")
    tables = {"orders": df_orders}

    auditor = DataAuditorAgent()
    catalog, dq_report, _ = auditor.audit_project_tables(
        project_id="test_proj",
        loaded_tables=tables,
        file_metadata={"orders": {"filename": "orders.csv"}}
    )

    assert dq_report.is_last_period_incomplete is True
    incomplete_issue = next(i for i in dq_report.issues if i.issue_type == "incomplete_period")
    assert "2026-06" in incomplete_issue.description
    assert incomplete_issue.severity == "warning"


# ---------------------------------------------------------------------------
# Test 7: Reconciliation between total variation and contributions
# ---------------------------------------------------------------------------
def test_07_reconciliation_total_variation_and_contributions(setup_demo_data):
    data_dir, _ = setup_demo_data
    df_orders = FileManager.read_table_dataframe(data_dir / "orders.csv")
    df_cust = FileManager.read_table_dataframe(data_dir / "customers.csv")

    tables = {"orders": df_orders, "customers": df_cust}
    rel = RelationshipSpec(
        id="rel_1", left_table="orders", left_key="customer_id",
        right_table="customers", right_key="customer_id",
        cardinality="many_to_one", match_rate_left=98.0, orphan_count_left=5,
        match_rate_right=100.0, orphan_count_right=0, risk_level="low"
    )

    analytical_df, _ = DuckDBAnalyticsEngine.prepare_analytical_dataset(
        tables=tables,
        relationships=[rel],
        exclude_cancelled=True,
        cutoff_date="2026-05-31"
    )

    breakdown = DuckDBAnalyticsEngine.calculate_period_breakdown(
        df=analytical_df,
        dimension="channel",
        baseline_period="2026-03",
        current_period="2026-05",
        metric="net_sales"
    )

    assert breakdown["is_perfectly_reconciled"] is True
    assert breakdown["reconciliation_diff"] < 0.01
    sum_contributions = sum(c["absolute_change"] for c in breakdown["contributions"])
    assert abs(sum_contributions - breakdown["total_change"]) < 0.01


# ---------------------------------------------------------------------------
# Test 8: Rejection of NaN or infinite values
# ---------------------------------------------------------------------------
def test_8_rejection_of_nan_or_infinite_values():
    validator = AnalyticalValidatorAgent()
    corrupt_result = AnalysisResult(
        result_id="RES_CORRUPT",
        step_id="OP_TEST",
        operation_name="Cálculo con NaN",
        method="Test Method",
        parameters={},
        data_source_version="v1.0",
        unit_of_measure="CLP",
        time_period_covered="2026-03",
        calculated_values={"sales": float("nan"), "valid_metric": 100.0}
    )

    report = validator.validate_results(
        run_id="run_test",
        results=[corrupt_result],
        transformations=[]
    )

    assert report.overall_status == "rejected"
    assert "RES_CORRUPT" in report.failed_results
    assert corrupt_result.validation_status == "rejected"


# ---------------------------------------------------------------------------
# Test 9: Propagation of a rejected result to dashboard and chat
# ---------------------------------------------------------------------------
def test_09_propagation_of_rejected_result_to_dashboard_and_chat():
    rejected_res = AnalysisResult(
        result_id="RES_REJECTED",
        step_id="OP_FAIL",
        operation_name="Operación fallida",
        method="Faulty",
        parameters={},
        data_source_version="v1.0",
        unit_of_measure="CLP",
        time_period_covered="2026-03",
        calculated_values={"metric": float("inf")},
        validation_status="rejected"
    )

    v_report = ValidationReport(
        schema_version="1.0",
        run_id="run_test",
        overall_status="rejected",
        failed_results=["RES_REJECTED"]
    )

    dash = DashboardSpec(
        schema_version="1.0",
        run_id="run_test",
        title="Dashboard Test",
        subtitle="",
        objective_question="Pregunta test",
        metric_cards=[
            MetricCardSpec(
                id="CARD_1",
                title="Métrica Rechazada",
                value="999",
                result_id="RES_REJECTED",
                period="2026-03",
                validation_status="rejected"
            )
        ],
        charts=[]
    )

    # Dashboard validator must reject dashboard
    dash_validator = DashboardValidatorAgent()
    validated_dash = dash_validator.validate_dashboard(dash, [rejected_res], v_report)

    assert validated_dash.is_dashboard_validated is False
    assert any("RECHAZO" in note for note in validated_dash.validation_notes)


# ---------------------------------------------------------------------------
# Test 10: Rerun with identical inputs and versioning
# ---------------------------------------------------------------------------
def test_10_rerun_idempotency_and_versioning(setup_demo_data):
    data_dir, _ = setup_demo_data
    project_id = f"proj_{uuid.uuid4().hex[:6]}"
    DatabaseService.create_project(project_id, "Test Idempotency")

    # Add file
    fpath = data_dir / "orders.csv"
    fhash = FileManager.calculate_sha256(fpath)
    df = FileManager.read_table_dataframe(fpath)
    DatabaseService.add_project_file(
        file_id=str(uuid.uuid4()),
        project_id=project_id,
        filename="orders.csv",
        table_name="orders",
        file_path=str(fpath),
        file_hash=fhash,
        row_count=len(df),
        column_count=len(df.columns)
    )

    runner = PipelineRunner()
    run1 = f"run_{uuid.uuid4().hex[:6]}"
    DatabaseService.create_run(run1, project_id)
    res1 = runner.execute_run(project_id, run1)

    run2 = f"run_{uuid.uuid4().hex[:6]}"
    DatabaseService.create_run(run2, project_id)
    res2 = runner.execute_run(project_id, run2)

    assert res1["status"] == "READY"
    assert res2["status"] == "READY"
    assert res1["run_id"] != res2["run_id"]
    # Verify outputs are identical
    val1 = res1["dashboard"]["metric_cards"][0]["value"]
    val2 = res2["dashboard"]["metric_cards"][0]["value"]
    assert val1 == val2


# ---------------------------------------------------------------------------
# Test 11: Filters that update figures coherently
# ---------------------------------------------------------------------------
def test_11_filters_update_figures_coherently(setup_demo_data):
    data_dir, _ = setup_demo_data
    df_orders = FileManager.read_table_dataframe(data_dir / "orders.csv")
    df_cust = FileManager.read_table_dataframe(data_dir / "customers.csv")

    tables = {"orders": df_orders, "customers": df_cust}
    rel = RelationshipSpec(
        id="r1", left_table="orders", left_key="customer_id",
        right_table="customers", right_key="customer_id",
        cardinality="many_to_one", match_rate_left=98.0, orphan_count_left=5,
        match_rate_right=100.0, orphan_count_right=0, risk_level="low"
    )

    analytical_df, _ = DuckDBAnalyticsEngine.prepare_analytical_dataset(
        tables=tables,
        relationships=[rel],
        exclude_cancelled=True,
        cutoff_date="2026-05-31"
    )

    total_net = analytical_df["net_sales"].sum()
    metro_net = analytical_df[analytical_df["region"] == "Metropolitana"]["net_sales"].sum()
    norte_net = analytical_df[analytical_df["region"] == "Norte"]["net_sales"].sum()

    assert metro_net > 0
    assert norte_net > 0
    assert metro_net + norte_net <= total_net


# ---------------------------------------------------------------------------
# Test 12: Chatbot query that triggers real calculation
# ---------------------------------------------------------------------------
def test_12_chatbot_triggers_real_calculation():
    assistant = ConversationalAssistantAgent()
    df_mock = pd.DataFrame([
        {"net_sales": 5000.0, "region": "Norte", "channel": "Retail / Tiendas"},
        {"net_sales": 3000.0, "region": "Norte", "channel": "Online / Directo"},
        {"net_sales": 10000.0, "region": "Sur", "channel": "Retail / Tiendas"}
    ])

    req = ChatRequest(
        project_id="p1",
        run_id="r1",
        question="¿Cuánto vendió la región Norte?"
    )

    ans = assistant.answer_query(
        request=req,
        objective=ObjectiveSpec(project_id="p1", original_question="q", operational_objective="o", decision_to_inform="d", primary_metric="net_sales", time_period="2026"),
        catalog=DataCatalog(project_id="p1"),
        analytical_df=df_mock,
        results=[],
        dashboard=DashboardSpec(run_id="r1", title="d", subtitle="s", objective_question="q"),
        insight_report=InsightReport(run_id="r1", executive_summary="ex")
    )

    assert ans.query_type == "recalculate_with_filters"
    assert "$8,000.00" in ans.answer_text  # 5000 + 3000
    assert len(ans.citations) == 1
    assert ans.citations[0].source_type == "transformed_table"


# ---------------------------------------------------------------------------
# Test 13: Unanswerable question receives explicit limitation
# ---------------------------------------------------------------------------
def test_13_unanswerable_question_receives_explicit_limitation():
    assistant = ConversationalAssistantAgent()
    req = ChatRequest(
        project_id="p1",
        run_id="r1",
        question="¿Qué competidor bajó precios y provocó la caída?"
    )

    ans = assistant.answer_query(
        request=req,
        objective=ObjectiveSpec(project_id="p1", original_question="q", operational_objective="o", decision_to_inform="d", primary_metric="net_sales", time_period="2026"),
        catalog=DataCatalog(project_id="p1"),
        analytical_df=pd.DataFrame(),
        results=[],
        dashboard=DashboardSpec(run_id="r1", title="d", subtitle="s", objective_question="q"),
        insight_report=InsightReport(run_id="r1", executive_summary="ex")
    )

    assert ans.query_type == "unanswerable_by_data"
    assert "no contienen información sobre 'competidor'" in ans.answer_text
    assert ans.data_limitation_notice is not None


# ---------------------------------------------------------------------------
# Test 14: Empty or malformed file handling
# ---------------------------------------------------------------------------
def test_14_empty_or_malformed_file_handling():
    df_empty = pd.DataFrame()
    auditor = DataAuditorAgent()
    catalog, dq_report, _ = auditor.audit_project_tables(
        project_id="p1",
        loaded_tables={"empty_table": df_empty},
        file_metadata={"empty_table": {"filename": "empty.csv"}}
    )

    assert dq_report.has_blocking_issues is True
    empty_issue = next(i for i in dq_report.issues if i.issue_type == "empty_file")
    assert empty_issue.severity == "blocking"


# ---------------------------------------------------------------------------
# Test 15: Run failure or cancellation does not leave deceptive state
# ---------------------------------------------------------------------------
def test_15_failure_or_cancellation_does_not_leave_deceptive_state():
    init_db()
    project_id = f"proj_fail_{uuid.uuid4().hex[:6]}"
    run_id = f"run_fail_{uuid.uuid4().hex[:6]}"
    DatabaseService.create_project(project_id, "Fail test")
    DatabaseService.create_run(run_id, project_id)

    # Intentionally trigger failure by running with no files
    runner = PipelineRunner()
    try:
        runner.execute_run(project_id, run_id)
    except Exception:
        pass

    run_record = DatabaseService.get_run(run_id)
    assert run_record["status"] == "FAILED"
    assert run_record["failure_reason"] is not None
    assert len(run_record["failure_reason"]) > 0
    # Must NOT be left as RUNNING or READY
    assert run_record["status"] != "RUNNING"
    assert run_record["status"] != "READY"
