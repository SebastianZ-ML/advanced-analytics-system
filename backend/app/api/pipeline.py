"""
FastAPI router for pipeline execution, objective pre-interpretation,
run lifecycle, cancellations, and artifact inspection.
"""
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.auditor import DataAuditorAgent
from app.agents.organizer import OrganizerAgent
from app.orchestrator.runner import PipelineRunner
from app.storage.db import DatabaseService
from app.storage.files import FileManager

router = APIRouter(prefix="/api", tags=["pipeline"])


class InterpretObjectiveRequest(BaseModel):
    objective_question: str


class RunCreateRequest(BaseModel):
    objective_question: str
    user_clarifications: Optional[Dict[str, str]] = None
    force_error_for_test: Optional[bool] = False


@router.post("/projects/{project_id}/interpret_objective")
def interpret_business_objective(project_id: str, req: InterpretObjectiveRequest) -> Dict[str, Any]:
    """
    Evaluates the user's business question against the project's registered tables
    and returns a structured pre-interpretation before full execution:
    - Interpreted operational objective
    - Primary and auxiliary metrics proposed
    - Time periods detected
    - Tables and columns utilized
    - Assumptions made
    - Ambiguities / clarifications needed (if any)
    - Method feasibility check
    """
    proj = DatabaseService.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    files_info = DatabaseService.get_project_files(project_id)
    if not files_info:
        raise HTTPException(
            status_code=400,
            detail="No tables registered in this project yet. Please upload data tables before interpreting an objective."
        )

    # Load tables for profiling
    loaded_tables = {}
    for finfo in files_info:
        tname = finfo["table_name"]
        df = FileManager.read_table_dataframe(finfo["file_path"], sheet_name=finfo.get("sheet_name"))
        loaded_tables[tname] = df

    # Audit & Profile tables
    auditor = DataAuditorAgent()
    catalog, dq_report, relationships = auditor.audit_dataset(
        project_id=project_id,
        tables=loaded_tables,
        files_info=files_info
    )

    # Structure objective through Organizer
    organizer = OrganizerAgent()
    objective_spec = organizer.process_objective(
        project_id=project_id,
        user_question=req.objective_question,
        catalog=catalog,
        user_clarifications=None
    )

    # Check method support
    q_lower = req.objective_question.lower()
    unsupported_reasons = []
    if "predecir el clima" in q_lower or "weather forecast" in q_lower:
        unsupported_reasons.append("La pregunta solicita predicción meteorológica, ausente en los datos cargados.")
    if "machine learning clustering" in q_lower and "clustering" not in [m.name for m in getattr(catalog, "methods", [])]:
        unsupported_reasons.append("El método de clustering avanzado no se encuentra registrado en el catálogo de análisis tabular actual.")

    is_supported = len(unsupported_reasons) == 0

    # Build selectable clarification options for the UI
    clarification_options = []
    for amb in objective_spec.pending_ambiguities:
        options = []
        if amb.id == "AMB_METRIC_DEFINITION":
            options = [objective_spec.primary_metric.split(" ")[0]] + objective_spec.auxiliary_metrics
        elif amb.id == "AMB_PERIOD_CUTOFF":
            options = ["exclude_incomplete_period", "include_all_periods"]

        clarification_options.append({
            "id": amb.id,
            "question": amb.question,
            "context": amb.context,
            "proposed_default": amb.proposed_default,
            "selectable_options": options,
            "impact_level": amb.impact_level
        })

    return {
        "project_id": project_id,
        "original_question": req.objective_question,
        "is_supported": is_supported,
        "unsupported_reasons": unsupported_reasons,
        "operational_objective": objective_spec.operational_objective,
        "decision_to_inform": objective_spec.decision_to_inform,
        "primary_metric": objective_spec.primary_metric,
        "auxiliary_metrics": objective_spec.auxiliary_metrics,
        "time_period": objective_spec.time_period,
        "comparison_period": objective_spec.comparison_period,
        "relevant_dimensions": objective_spec.relevant_dimensions,
        "tables_used": list(loaded_tables.keys()),
        "assumptions": objective_spec.assumptions,
        "clarifications_needed": clarification_options,
        "acceptance_criteria": objective_spec.acceptance_criteria
    }


@router.post("/projects/{project_id}/runs")
def start_pipeline_run(project_id: str, req: RunCreateRequest) -> Dict[str, Any]:
    """Initiates execution using the exact user-specified question and resolved clarifications."""
    proj = DatabaseService.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    files_info = DatabaseService.get_project_files(project_id)
    if not files_info:
        raise HTTPException(status_code=400, detail="Cannot run analysis without registered tables in project.")

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    DatabaseService.create_run(run_id, project_id)

    runner = PipelineRunner()
    try:
        res = runner.execute_run(
            project_id=project_id,
            run_id=run_id,
            objective_question=req.objective_question,
            user_clarifications=req.user_clarifications or {},
            force_error_for_test=bool(req.force_error_for_test)
        )
        return res
    except Exception as e:
        DatabaseService.update_run_stage(run_id, "FAILED", "FAILED")
        DatabaseService.log_event(run_id, "FAILED", "ERROR", f"Execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> Dict[str, Any]:
    run = DatabaseService.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    DatabaseService.update_run_stage(run_id, "CANCELLED", "CANCELLED")
    DatabaseService.log_event(run_id, "CANCELLED", "WARN", "Pipeline run cancelled by user request.")
    return {"status": "cancelled", "run_id": run_id}


@router.get("/runs/{run_id}")
def get_run_status(run_id: str) -> Dict[str, Any]:
    run = DatabaseService.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    events = DatabaseService.get_run_events(run_id)
    return {**run, "events": events}


@router.get("/runs/{run_id}/artifacts")
def get_run_artifacts(run_id: str) -> Dict[str, Any]:
    run = DatabaseService.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return DatabaseService.get_all_run_artifacts(run_id)


@router.get("/runs/{run_id}/events")
def get_run_events(run_id: str) -> List[Dict[str, Any]]:
    return DatabaseService.get_run_events(run_id)
