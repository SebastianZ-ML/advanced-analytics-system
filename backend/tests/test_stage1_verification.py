"""
Verification test suite for Stage 1 enhancements:
- Parquet snapshot generation and SHA-256 integrity verification.
- Content-level card verification rejecting tampered numbers.
- Persistence of final validated statuses (approved/rejected).
- Grounded chat assistant executing against the run snapshot.
"""
import uuid
from pathlib import Path
import pandas as pd
import pytest

from app.agents.dashboard_validator import DashboardValidatorAgent
from app.contracts import (
    AnalysisResult,
    ChatRequest,
    DashboardSpec,
    MetricCardSpec,
    ValidationReport,
)
from app.orchestrator.runner import PipelineRunner
from app.storage.db import DatabaseService
from app.storage.snapshots import SnapshotManager
from app.api.chat import chat_with_assistant


def test_stage1_snapshot_creation_and_integrity(tmp_path):
    """Verify that every pipeline run generates an immutable Parquet snapshot with SHA-256."""
    project_id = f"proj_snap_{uuid.uuid4().hex[:6]}"
    run_id = f"run_snap_{uuid.uuid4().hex[:6]}"
    DatabaseService.create_project(project_id, "Snapshot Test")

    df_orders = pd.DataFrame({
        "order_id": ["O-1", "O-2", "O-3", "O-4", "O-5", "O-6"],
        "customer_id": ["C1", "C2", "C3", "C1", "C2", "C3"],
        "order_date": ["2026-01-05", "2026-01-15", "2026-01-25", "2026-02-05", "2026-02-15", "2026-02-25"],
        "channel": ["Wholesale / B2B", "Retail / Stores", "Online / Direct", "Wholesale / B2B", "Retail / Stores", "Online / Direct"],
        "units": [1, 2, 3, 4, 5, 6],
        "gross_sales": [120.0, 240.0, 360.0, 480.0, 600.0, 720.0],
        "discount_amount": [20.0, 40.0, 60.0, 80.0, 100.0, 120.0],
        "net_sales": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        "status": ["COMPLETED", "COMPLETED", "COMPLETED", "COMPLETED", "COMPLETED", "COMPLETED"]
    })
    fpath = tmp_path / "orders.csv"
    df_orders.to_csv(fpath, index=False)

    DatabaseService.add_project_file(
        file_id=str(uuid.uuid4()),
        project_id=project_id,
        filename="orders.csv",
        table_name="orders",
        file_path=str(fpath),
        file_hash="hash_snap_test",
        row_count=6,
        column_count=len(df_orders.columns)
    )
    DatabaseService.create_run(run_id, project_id)

    runner = PipelineRunner()
    res = runner.execute_run(project_id, run_id)
    assert res["status"] == "READY"

    # 1. Verify snapshot record in DB
    snap_record = DatabaseService.get_snapshot_record(run_id)
    assert snap_record is not None
    assert snap_record["run_id"] == run_id
    assert len(snap_record["file_hash"]) == 64  # Valid SHA-256

    # 2. Verify snapshot load and integrity
    loaded_df = SnapshotManager.load_snapshot(run_id)
    assert len(loaded_df) >= 3  # At least one complete month retained
    assert "net_sales" in loaded_df.columns
    assert loaded_df["net_sales"].sum() > 0


def test_stage1_persisted_results_are_approved_not_pending(tmp_path):
    """Verify that AnalysisResults artifacts in DB have validation_status='approved' upon completion."""
    project_id = f"proj_status_{uuid.uuid4().hex[:6]}"
    run_id = f"run_status_{uuid.uuid4().hex[:6]}"
    DatabaseService.create_project(project_id, "Status Test")

    df_orders = pd.DataFrame({
        "order_id": ["O-1"],
        "customer_id": ["C1"],
        "order_date": ["2026-01-10"],
        "channel": ["Wholesale / B2B"],
        "units": [1],
        "gross_sales": [120.0],
        "discount_amount": [20.0],
        "net_sales": [100.0],
        "status": ["COMPLETED"]
    })
    fpath = tmp_path / "orders.csv"
    df_orders.to_csv(fpath, index=False)

    DatabaseService.add_project_file(
        file_id=str(uuid.uuid4()),
        project_id=project_id,
        filename="orders.csv",
        table_name="orders",
        file_path=str(fpath),
        file_hash="hash_stat",
        row_count=1,
        column_count=len(df_orders.columns)
    )
    DatabaseService.create_run(run_id, project_id)

    runner = PipelineRunner()
    res = runner.execute_run(project_id, run_id)
    assert res["status"] == "READY"

    artifacts = DatabaseService.get_all_run_artifacts(run_id)
    persisted_results = artifacts.get("AnalysisResults", [])
    assert len(persisted_results) > 0
    for r in persisted_results:
        assert r["validation_status"] in ["approved", "approved_with_warnings"], f"Result {r['result_id']} was not approved!"


def test_stage1_dashboard_validator_rejects_tampered_value():
    """Verify that DashboardValidatorAgent strictly rejects metric cards with fabricated numbers."""
    validator = DashboardValidatorAgent()
    real_res = AnalysisResult(
        result_id="RES_01",
        step_id="OP_01",
        operation_name="Sales Total",
        method="Sum",
        parameters={},
        data_source_version="v1.0",
        unit_of_measure="USD",
        time_period_covered="2026-05",
        calculated_values={"net_sales": 10500.0},
        validation_status="approved"
    )

    v_report = ValidationReport(schema_version="1.0", run_id="r1", overall_status="approved", failed_results=[])

    # Tampered card with $999,999,999
    tampered_dash = DashboardSpec(
        schema_version="1.0",
        run_id="r1",
        title="Test",
        subtitle="",
        objective_question="Q",
        metric_cards=[
            MetricCardSpec(
                id="C1",
                title="Sales",
                value="$999,999,999.00",
                result_id="RES_01",
                period="2026-05",
                validation_status="approved"
            )
        ],
        charts=[]
    )

    checked = validator.validate_dashboard(tampered_dash, [real_res], v_report)
    assert checked.is_dashboard_validated is False
    assert any("REJECTED" in n and "does not match" in n for n in checked.validation_notes)

    # Legitimate card with $10,500.00
    valid_dash = DashboardSpec(
        schema_version="1.0",
        run_id="r1",
        title="Test",
        subtitle="",
        objective_question="Q",
        metric_cards=[
            MetricCardSpec(
                id="C1",
                title="Sales",
                value="$10,500.00",
                result_id="RES_01",
                period="2026-05",
                validation_status="approved"
            )
        ],
        charts=[]
    )
    checked_valid = validator.validate_dashboard(valid_dash, [real_res], v_report)
    assert checked_valid.is_dashboard_validated is True


def test_stage1_chat_uses_snapshot_for_recalculation(tmp_path):
    """Verify that chat endpoint loads and recalculates from the run snapshot."""
    project_id = f"proj_chat_{uuid.uuid4().hex[:6]}"
    run_id = f"run_chat_{uuid.uuid4().hex[:6]}"
    DatabaseService.create_project(project_id, "Chat Test")

    df_orders = pd.DataFrame({
        "order_id": ["O-1", "O-2"],
        "customer_id": ["C1", "C2"],
        "order_date": ["2026-01-10", "2026-02-15"],
        "channel": ["Wholesale / B2B", "Retail / Stores"],
        "region": ["Metropolitan", "North"],
        "units": [1, 2],
        "gross_sales": [100.0, 200.0],
        "discount_amount": [0.0, 0.0],
        "net_sales": [100.0, 200.0],
        "status": ["COMPLETED", "COMPLETED"]
    })
    fpath = tmp_path / "orders.csv"
    df_orders.to_csv(fpath, index=False)

    DatabaseService.add_project_file(
        file_id=str(uuid.uuid4()),
        project_id=project_id,
        filename="orders.csv",
        table_name="orders",
        file_path=str(fpath),
        file_hash="hash_chat",
        row_count=2,
        column_count=len(df_orders.columns)
    )
    DatabaseService.create_run(run_id, project_id)

    runner = PipelineRunner()
    runner.execute_run(project_id, run_id)

    # Call chat endpoint requesting recalculation
    req = ChatRequest(
        project_id=project_id,
        run_id=run_id,
        question="How much was sold in the Metropolitan region?",
        active_filters={}
    )
    answer = chat_with_assistant(req)
    assert answer.query_type == "recalculate_with_filters"
    assert "$100.00" in answer.answer_text
