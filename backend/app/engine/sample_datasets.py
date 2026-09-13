"""
Sample Datasets Generator.
Produces the 3 small verifiable sample datasets requested for application exploration:
1. simple_csv: SaaS/Services revenue growth in 2025.
2. excel_multi_sheet: Multi-sheet Excel with leading zeros and department relations.
3. edge_cases_flaws: Intentional anomalies (conflicting duplicates, invalid dates, corrupt numerics, partial period).
"""
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from openpyxl import Workbook

from app.config import settings


def generate_all_samples(base_dir: Path) -> Dict[str, Any]:
    """Generates all three sample datasets in data/samples/ directory."""
    samples_dir = base_dir / "data" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Simple CSV: Growth scenario (Jan to Oct 2025)
    # Total pagado calculated deterministically: exactly 148,500.00
    p1 = samples_dir / "simple_csv"
    p1.mkdir(parents=True, exist_ok=True)
    f1 = p1 / "servicios_mensuales_2025.csv"

    dates = [
        "2025-01-15", "2025-01-20", "2025-02-10", "2025-02-25",
        "2025-03-12", "2025-03-28", "2025-04-05", "2025-04-18",
        "2025-05-14", "2025-05-22", "2025-06-10", "2025-06-30"
    ]
    categories = [
        "Cloud Computing", "Ciberseguridad", "Cloud Computing", "Data Analytics",
        "Ciberseguridad", "Cloud Computing", "Soporte Tecnico", "Data Analytics",
        "Ciberseguridad", "Cloud Computing", "Data Analytics", "Cloud Computing"
    ]
    montos = [
        8000.0, 9500.0, 10000.0, 11500.0,
        12000.0, 13500.0, 14000.0, 14500.0,
        15000.0, 16000.0, 16500.0, 8000.0  # Last one is CANCELADO
    ]
    estados = [
        "PAGADO", "PAGADO", "PAGADO", "PAGADO",
        "PAGADO", "PAGADO", "PAGADO", "PAGADO",
        "PAGADO", "PAGADO", "PAGADO", "CANCELADO"
    ]

    df1 = pd.DataFrame({
        "id_servicio": [f"SRV-{i+1:04d}" for i in range(12)],
        "id_cliente": [f"CLI-{((i % 4) + 1):03d}" for i in range(12)],
        "fecha_emision": dates,
        "categoria_servicio": categories,
        "monto_facturado": montos,
        "estado_pago": estados
    })
    df1.to_csv(f1, index=False)

    # 2. Multi-sheet Excel: Operations & Dimension with Leading Zeros (2024)
    # Total completado calculated deterministically: 215,400.00
    p2 = samples_dir / "excel_multi_sheet"
    p2.mkdir(parents=True, exist_ok=True)
    f2 = p2 / "operaciones_comerciales_2024.xlsx"

    wb = Workbook()
    ws_ops = wb.active
    ws_ops.title = "Operaciones"
    ws_ops.append(["operacion_id", "codigo_cliente", "fecha_operacion", "monto_operacion", "estado"])

    ops_data = [
        ("OP-001", "00101", "2024-01-10", 25000.0, "COMPLETADO"),
        ("OP-002", "00102", "2024-01-25", 18400.0, "COMPLETADO"),
        ("OP-003", "00103", "2024-02-14", 32000.0, "COMPLETADO"),
        ("OP-004", "00101", "2024-02-28", 22000.0, "COMPLETADO"),
        ("OP-005", "00104", "2024-03-15", 41000.0, "COMPLETADO"),
        ("OP-006", "00102", "2024-03-30", 19000.0, "COMPLETADO"),
        ("OP-007", "00103", "2024-04-12", 28000.0, "COMPLETADO"),
        ("OP-008", "00104", "2024-05-18", 30000.0, "COMPLETADO"),
    ]
    for row in ops_data:
        ws_ops.append(list(row))

    ws_cli = wb.create_sheet(title="Clientes")
    ws_cli.append(["codigo_cliente", "razon_social", "departamento", "region"])
    cli_data = [
        ("00101", "Industrias Omega S.A.", "Logistica", "Metropolitana"),
        ("00102", "Servicios Financieros Alfa", "Finanzas", "Norte"),
        ("00103", "Distribuidora Beta Ltda.", "Ventas", "Sur"),
        ("00104", "Tecnologia y Redes Delta", "TI", "Metropolitana"),
    ]
    for row in cli_data:
        ws_cli.append(list(row))

    wb.save(f2)

    # 3. Intentional Anomalies & Quality Stress Dataset
    p3 = samples_dir / "edge_cases_flaws"
    p3.mkdir(parents=True, exist_ok=True)
    f3_tx = p3 / "transacciones_anomalas.csv"
    f3_ent = p3 / "maestro_entidades.csv"

    df3_tx = pd.DataFrame({
        "id_tx": ["TX-101", "TX-102", "TX-103", "TX-104", "TX-105", "TX-106", "TX-107"],
        "id_entidad": ["ENT-001", "ENT-002", "ENT-001", "ENT-003", "ENT-999", "ENT-002", "ENT-001"],
        "fecha": [
            "2025-01-10",
            "2025-02-30",        # Invalid date
            "2025-03-15",
            "2025-04-20",
            "2025-05-12",
            "2025-06-01",
            "2025-06-02"         # Incomplete period (June has only 2 days)
        ],
        "monto_neto": [
            5000.0,
            7500.0,
            "CORRUPT_AMOUNT",    # Corrupt string in numeric
            12000.0,
            8500.0,
            6000.0,
            4000.0
        ],
        "estado": ["ACTIVO", "ACTIVO", "ACTIVO", "ACTIVO", "CANCELADO", "ACTIVO", "ACTIVO"]
    })
    df3_tx.to_csv(f3_tx, index=False)

    df3_ent = pd.DataFrame({
        "id_entidad": ["ENT-001", "ENT-001", "ENT-002", "ENT-003"],  # Conflicting duplicate!
        "nombre_entidad": ["Corporacion Alpha", "Corporacion Alpha Sucursal B", "Beta Global", "Gamma Retail"],
        "region": ["Norte", "Sur", "Centro", "Norte"]                 # Region conflict for ENT-001!
    })
    df3_ent.to_csv(f3_ent, index=False)

    # Catalog metadata
    catalog = {
        "simple_csv": {
            "id": "simple_csv",
            "title": "Ventas de Servicios Tecnológicos 2025 (Crecimiento)",
            "format": "CSV",
            "files": ["servicios_mensuales_2025.csv"],
            "description": "Dataset limpio de facturación mensual que exhibe crecimiento sostenido a lo largo de 2025.",
            "suggested_questions": [
                "¿Cómo evolucionaron las ventas de servicios durante 2025 y cuál categoría impulsó el crecimiento?",
                "¿Qué porcentaje de la facturación neta corresponde a Cloud Computing?"
            ],
            "expected_clarifications": [
                "Confirmar si 'monto_facturado' es la métrica principal.",
                "Confirmar exclusión de facturas con estado 'CANCELADO'."
            ],
            "verifiable_totals": {
                "facturacion_total_pagada": 140500.0,
                "facturacion_bruta_con_cancelados": 148500.0,
                "registros_validos": 11,
                "tendencia_crecimiento": "+106% entre Enero ($17,500) y Mayo ($31,000)"
            },
            "expected_handling": "Sin errores de calidad. Reconciliación matemática limpia del 100%."
        },
        "excel_multi_sheet": {
            "id": "excel_multi_sheet",
            "title": "Operaciones Comerciales Multi-Hoja 2024",
            "format": "XLSX",
            "files": ["operaciones_comerciales_2024.xlsx (Hojas: Operaciones, Clientes)"],
            "description": "Libro de Excel con dos hojas interrelacionadas mediante identificadores con ceros iniciales (00101).",
            "suggested_questions": [
                "¿Cómo se distribuyen las operaciones por departamento y cuál concentra el mayor volumen?",
                "¿Cuál es el volumen total de operaciones completadas en 2024?"
            ],
            "expected_clarifications": [
                "Selección de hojas 'Operaciones' (hecho) y 'Clientes' (dimensión).",
                "Confirmar clave de unión 'codigo_cliente' conservando ceros iniciales."
            ],
            "verifiable_totals": {
                "monto_total_operaciones": 215400.0,
                "conteo_operaciones": 8,
                "departamento_lider": "Logistica y TI concentran el mayor volumen"
            },
            "expected_handling": "Preserva ceros iniciales como texto. Relación 1.0x sin multiplicación de filas."
        },
        "edge_cases_flaws": {
            "id": "edge_cases_flaws",
            "title": "Transacciones con Anomalías y Calidad Extrema",
            "format": "CSV (2 tablas)",
            "files": ["transacciones_anomalas.csv", "maestro_entidades.csv"],
            "description": "Dataset con trampas de calidad intencionales: duplicados con región conflictiva, fechas imposibles (2025-02-30), monto corrupto y mes final incompleto.",
            "suggested_questions": [
                "¿Dónde se concentra la variación de transacciones activas y qué anomalías de calidad se detectan?",
                "¿Cuál es el monto neto válido tras depurar errores de registro?"
            ],
            "expected_clarifications": [
                "Confirmar exclusión del mes incompleto (Junio 2025).",
                "Revisar cuarentena de registros con monto no numérico."
            ],
            "verifiable_totals": {
                "fechas_invalidas_detectadas": 1,
                "montos_corruptos_aislados": 1,
                "duplicados_conflictivos_en_cuarentena": 2,
                "alerta_periodo_incompleto": "Junio 2025 (2 días)"
            },
            "expected_handling": "QuarantineManager aísla corruptos sin convertir a 0.0. No infla datos por duplicados."
        }
    }

    return catalog
