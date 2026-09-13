"""
FastAPI router for grounded contextual Q&A with provenance.
"""
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
import pandas as pd

from app.agents.assistant import ConversationalAssistantAgent
from app.contracts import (
    AnalysisResult,
    ChatAnswer,
    ChatRequest,
    DashboardSpec,
    DataCatalog,
    InsightReport,
    ObjectiveSpec,
)
from app.storage.db import DatabaseService
from app.storage.files import FileManager

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("")
def chat_with_assistant(req: ChatRequest) -> ChatAnswer:
    run = DatabaseService.get_run(req.run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Ejecución no encontrada")

    artifacts = DatabaseService.get_all_run_artifacts(req.run_id)
    if "DashboardSpec" not in artifacts or "InsightReport" not in artifacts:
        raise HTTPException(status_code=400, detail="La ejecución aún no ha completado el análisis.")

    objective = ObjectiveSpec(**artifacts["ObjectiveSpec"])
    catalog = DataCatalog(**artifacts["DataCatalog"])
    dashboard = DashboardSpec(**artifacts["DashboardSpec"])
    insight_report = InsightReport(**artifacts["InsightReport"])
    results = [AnalysisResult(**r) for r in artifacts.get("AnalysisResults", [])]

    # Load analytical fact data if available or reconstruct from project files
    files = DatabaseService.get_project_files(req.project_id)
    fact_file = next((f for f in files if "order" in f["table_name"] or "venta" in f["table_name"]), None)
    if fact_file:
        df = FileManager.read_table_dataframe(fact_file["file_path"], sheet_name=fact_file["sheet_name"])
        # Quick clean for filtering
        for col in ["net_sales", "gross_sales", "units"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce").fillna(0.0)
    else:
        df = pd.DataFrame()

    assistant = ConversationalAssistantAgent()
    return assistant.answer_query(
        request=req,
        objective=objective,
        catalog=catalog,
        analytical_df=df,
        results=results,
        dashboard=dashboard,
        insight_report=insight_report
    )
