"""
FastAPI router for pipeline execution, run lifecycle, and artifact inspection.
"""
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.orchestrator.runner import PipelineRunner
from app.storage.db import DatabaseService

router = APIRouter(prefix="/api", tags=["pipeline"])


class RunCreateRequest(BaseModel):
    objective_question: Optional[str] = (
        "Las ventas bajaron durante los últimos meses. "
        "Quiero entender dónde se concentra la caída, qué factores observables la explican y qué acciones conviene investigar."
    )
    user_clarifications: Optional[Dict[str, str]] = None
    force_error_for_test: Optional[bool] = False


@router.post("/projects/{project_id}/runs")
def start_pipeline_run(project_id: str, req: RunCreateRequest) -> Dict[str, Any]:
    proj = DatabaseService.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    DatabaseService.create_run(run_id, project_id)

    # In this local version, run synchronously to return full result,
    # or progress can be tracked through DatabaseService events.
    runner = PipelineRunner()
    try:
        res = runner.execute_run(
            project_id=project_id,
            run_id=run_id,
            objective_question=req.objective_question,
            user_clarifications=req.user_clarifications or {"metric_definition": "net_sales", "period_cutoff": "exclude_incomplete_period"},
            force_error_for_test=bool(req.force_error_for_test)
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}")
def get_run_status(run_id: str) -> Dict[str, Any]:
    run = DatabaseService.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Ejecución no encontrada")
    events = DatabaseService.get_run_events(run_id)
    return {**run, "events": events}


@router.get("/runs/{run_id}/artifacts")
def get_run_artifacts(run_id: str) -> Dict[str, Any]:
    run = DatabaseService.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Ejecución no encontrada")
    return DatabaseService.get_all_run_artifacts(run_id)


@router.get("/runs/{run_id}/events")
def get_run_events(run_id: str) -> List[Dict[str, Any]]:
    return DatabaseService.get_run_events(run_id)
