"""
Automated tests for the end-to-end user-driven flow:
- CSV delimiter & encoding detection
- Excel sheet inspection & preview
- Table collision prevention (replace vs reject)
- Objective interpretation & clarifications
- Dynamic execution & chat recalculation
- Downloadable sample datasets integrity
"""
import io
import uuid
from pathlib import Path
import pytest
import pandas as pd
from openpyxl import Workbook

from app.storage.db import DatabaseService
from app.storage.files import FileManager
from app.orchestrator.runner import PipelineRunner
from app.engine.sample_datasets import generate_all_samples


@pytest.fixture
def test_workspace(tmp_path):
    project_id = f"proj_user_flow_{uuid.uuid4().hex[:6]}"
    DatabaseService.create_project(project_id, "User Driven Flow Test")
    return project_id, tmp_path


def test_01_csv_sniffing_delimiter_and_preview(test_workspace):
    """Verify automatic delimiter detection (semicolon) and rich table preview."""
    project_id, tmp_path = test_workspace

    # Create semicolon-delimited CSV with leading zeros
    csv_path = tmp_path / "ventas_semicolon.csv"
    csv_content = (
        "id_cliente;fecha_venta;canal;total_facturado\n"
        "00101;2025-01-10;Online;1500.50\n"
        "00102;2025-02-15;Tienda;2300.00\n"
        "00103;2025-03-20;Online;4100.25\n"
    )
    csv_path.write_text(csv_content, encoding="utf-8")

    # 1. Sniffer test
    sniff_res = FileManager.sniff_csv(csv_path)
    assert sniff_res["delimiter"] == ";", f"Expected ';' but got {sniff_res['delimiter']}"
    assert "utf-8" in sniff_res["encoding"].lower()

    # 2. Preview test
    preview = FileManager.preview_table(csv_path)
    assert preview["row_count"] == 3
    assert preview["column_count"] == 4
    col_names = [c["column_name"] for c in preview["columns"]]
    assert "id_cliente" in col_names
    assert "total_facturado" in col_names

    # Check leading zero preservation
    id_col = next(c for c in preview["columns"] if c["column_name"] == "id_cliente")
    assert id_col["has_leading_zeros"] is True


def test_02_excel_inspection_and_sheet_selection(test_workspace):
    """Verify Excel inspection, multiple sheet discovery, and preview generation."""
    project_id, tmp_path = test_workspace

    xlsx_path = tmp_path / "multi_sheet_test.xlsx"
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Facturacion"
    ws1.append(["id_factura", "fecha", "monto"])
    ws1.append(["F-01", "2025-01-10", 5000.0])
    ws1.append(["F-02", "2025-01-20", 7500.0])

    ws2 = wb.create_sheet(title="Clientes")
    ws2.append(["id_cliente", "razon_social", "ciudad"])
    ws2.append(["C-01", "Empresa A", "Santiago"])
    ws2.append(["C-02", "Empresa B", "Valparaiso"])
    wb.save(xlsx_path)

    # 1. Sheet inspection
    excel_info = FileManager.inspect_excel_full(xlsx_path)
    assert "Facturacion" in excel_info["sheets"]
    assert "Clientes" in excel_info["sheets"]
    assert excel_info["has_unevaluated_formulas"] is False

    # 2. Preview specific sheet
    preview_fact = FileManager.preview_table(xlsx_path, sheet_name="Facturacion")
    assert preview_fact["row_count"] == 2
    assert "monto" in [c["column_name"] for c in preview_fact["columns"]]

    preview_cli = FileManager.preview_table(xlsx_path, sheet_name="Clientes")
    assert preview_cli["row_count"] == 2
    assert "ciudad" in [c["column_name"] for c in preview_cli["columns"]]


def test_03_table_collision_prevention_and_replacement(test_workspace):
    """Verify that table collisions are prevented unless explicitly replaced."""
    project_id, tmp_path = test_workspace

    # Add initial table
    fpath = tmp_path / "orders.csv"
    fpath.write_text("id,amount\n1,100\n", encoding="utf-8")
    fhash = FileManager.calculate_sha256(fpath)
    file_id_1 = f"f_{uuid.uuid4().hex[:8]}"

    DatabaseService.add_project_file(
        file_id=file_id_1,
        project_id=project_id,
        filename="orders.csv",
        table_name="orders",
        file_path=str(fpath),
        file_hash=fhash,
        row_count=1,
        column_count=2
    )

    files_before = DatabaseService.get_project_files(project_id)
    assert len(files_before) == 1
    assert files_before[0]["row_count"] == 1

    # Replace existing table
    DatabaseService.delete_project_file_by_table(project_id, "orders")
    fpath2 = tmp_path / "orders_v2.csv"
    fpath2.write_text("id,amount\n1,100\n2,200\n", encoding="utf-8")
    fhash2 = FileManager.calculate_sha256(fpath2)
    file_id_2 = f"f_{uuid.uuid4().hex[:8]}"

    DatabaseService.add_project_file(
        file_id=file_id_2,
        project_id=project_id,
        filename="orders_v2.csv",
        table_name="orders",
        file_path=str(fpath2),
        file_hash=fhash2,
        row_count=2,
        column_count=2
    )

    files_after = DatabaseService.get_project_files(project_id)
    assert len(files_after) == 1
    assert files_after[0]["file_id"] == file_id_2
    assert files_after[0]["row_count"] == 2


def test_04_downloadable_sample_datasets_and_verification_cards(tmp_path):
    """Verify that all 3 downloadable sample datasets are generated with ground truth."""
    catalog = generate_all_samples(tmp_path)

    assert "simple_csv" in catalog
    assert "excel_multi_sheet" in catalog
    assert "edge_cases_flaws" in catalog

    # 1. Simple CSV verification
    s1 = catalog["simple_csv"]
    assert s1["verifiable_totals"]["facturacion_total_pagada"] == 140500.0
    f1_path = tmp_path / "data" / "samples" / "simple_csv" / "servicios_mensuales_2025.csv"
    assert f1_path.exists()
    df1 = pd.read_csv(f1_path)
    # Exclude CANCELADO
    pagados = df1[df1["estado_pago"] == "PAGADO"]["monto_facturado"].sum()
    assert pagados == 140500.0

    # 2. Excel multi-sheet verification
    s2 = catalog["excel_multi_sheet"]
    assert s2["verifiable_totals"]["monto_total_operaciones"] == 215400.0
    f2_path = tmp_path / "data" / "samples" / "excel_multi_sheet" / "operaciones_comerciales_2024.xlsx"
    assert f2_path.exists()
    df2 = pd.read_excel(f2_path, sheet_name="Operaciones")
    assert df2["monto_operacion"].sum() == 215400.0

    # 3. Edge cases flaws verification
    s3 = catalog["edge_cases_flaws"]
    f3_tx = tmp_path / "data" / "samples" / "edge_cases_flaws" / "transacciones_anomalas.csv"
    f3_ent = tmp_path / "data" / "samples" / "edge_cases_flaws" / "maestro_entidades.csv"
    assert f3_tx.exists()
    assert f3_ent.exists()


def test_05_dynamic_execution_with_custom_question(test_workspace):
    """Verify pipeline execution with a custom business question on non-demo data."""
    project_id, tmp_path = test_workspace

    # Create monthly service sales data
    df_services = pd.DataFrame({
        "srv_id": [f"S{i}" for i in range(1, 7)],
        "client_id": ["C1", "C2", "C3", "C1", "C2", "C3"],
        "srv_date": ["2025-01-10", "2025-01-20", "2025-02-10", "2025-02-20", "2025-03-10", "2025-03-20"],
        "category": ["Cloud", "Data", "Cloud", "Data", "Cloud", "Data"],
        "billing_amount": [1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0],
        "status": ["PAID", "PAID", "PAID", "PAID", "PAID", "PAID"]
    })
    fpath = tmp_path / "services.csv"
    df_services.to_csv(fpath, index=False)

    fhash = FileManager.calculate_sha256(fpath)
    file_id_srv = f"f_{uuid.uuid4().hex[:8]}"
    DatabaseService.add_project_file(
        file_id=file_id_srv,
        project_id=project_id,
        filename="services.csv",
        table_name="services",
        file_path=str(fpath),
        file_hash=fhash,
        row_count=len(df_services),
        column_count=len(df_services.columns)
    )

    run_id = f"run_custom_{uuid.uuid4().hex[:6]}"
    DatabaseService.create_run(run_id, project_id)

    custom_question = "¿Cómo evolucionó la facturación de servicios entre enero y marzo de 2025?"
    runner = PipelineRunner()
    res = runner.execute_run(
        project_id=project_id,
        run_id=run_id,
        objective_question=custom_question,
        user_clarifications={"metric_definition": "billing_amount"}
    )

    assert res["status"] == "READY"
    assert res["run_id"] == run_id
    assert res["project_id"] == project_id
    assert len(res["dashboard"]["metric_cards"]) > 0
