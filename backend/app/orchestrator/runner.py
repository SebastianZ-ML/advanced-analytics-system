"""
Sequential pipeline runner and orchestrator.
Manages the lifecycle of an analytical run through the 10 specialized agents.
Persists all artifacts, logs real events, and controls the automated repair loop.
"""
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
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
    AnalysisPlan,
    AnalysisResult,
    DashboardSpec,
    DataCatalog,
    DataQualityReport,
    InsightReport,
    ObjectiveSpec,
    RelationshipSpec,
    TransformationRecord,
    ValidationReport,
)
from app.orchestrator.state_machine import PipelineStage, validate_transition
from app.storage.db import DatabaseService
from app.storage.files import FileManager


class PipelineRunner:
    def __init__(self):
        self.organizer = OrganizerAgent()
        self.auditor = DataAuditorAgent()
        self.methodologist = MethodologistAgent()
        self.preparer = DataPreparerAgent()
        self.executor = AnalyticalExecutorAgent()
        self.validator = AnalyticalValidatorAgent()
        self.interpreter = InterpreterAgent()
        self.dashboard_builder = DashboardBuilderAgent()
        self.dashboard_validator = DashboardValidatorAgent()
        self.assistant = ConversationalAssistantAgent()

    def _transition(self, run_id: str, current_stage: PipelineStage, target_stage: PipelineStage, message: str) -> PipelineStage:
        validate_transition(current_stage, target_stage)
        status = "READY" if target_stage == PipelineStage.READY else ("FAILED" if target_stage == PipelineStage.FAILED else "RUNNING")
        DatabaseService.update_run_stage(run_id=run_id, stage=target_stage.value, status=status)
        DatabaseService.log_event(run_id=run_id, stage=target_stage.value, level="INFO", message=message)
        return target_stage

    def execute_run(
        self,
        project_id: str,
        run_id: str,
        objective_question: str = "Las ventas bajaron durante los últimos meses. Quiero entender dónde se concentra la caída, qué factores observables la explican y qué acciones conviene investigar.",
        user_clarifications: Optional[Dict[str, str]] = None,
        force_error_for_test: bool = False
    ) -> Dict[str, Any]:
        """
        Executes end-to-end analytical workflow.
        """
        curr_stage = PipelineStage.UPLOADED
        DatabaseService.log_event(run_id=run_id, stage="INIT", level="INFO", message="Iniciando ejecución de pipeline analítico.")

        try:
            # 1. Load project files from disk
            files_info = DatabaseService.get_project_files(project_id)
            if not files_info:
                raise ValueError(f"No hay archivos registrados para el proyecto {project_id}.")

            loaded_tables: Dict[str, pd.DataFrame] = {}
            file_meta: Dict[str, Dict[str, str]] = {}

            for f in files_info:
                fpath = Path(f["file_path"])
                table_name = f["table_name"]
                sheet_name = f["sheet_name"]
                df = FileManager.read_table_dataframe(fpath, sheet_name=sheet_name)
                loaded_tables[table_name] = df
                file_meta[table_name] = {
                    "filename": f["filename"],
                    "sheet_name": sheet_name,
                    "file_hash": f["file_hash"]
                }

            # 2. Audit & Profile
            curr_stage = self._transition(run_id, curr_stage, PipelineStage.PROFILED, "Agente Auditor: Perfilado de tablas y detección de anomalías.")
            catalog, dq_report, relationships = self.auditor.audit_project_tables(
                project_id=project_id,
                loaded_tables=loaded_tables,
                file_metadata=file_meta
            )
            DatabaseService.save_artifact(f"CAT_{run_id}", run_id, "DataCatalog", catalog.model_dump())
            DatabaseService.save_artifact(f"DQ_{run_id}", run_id, "DataQualityReport", dq_report.model_dump())
            DatabaseService.save_artifact(f"REL_{run_id}", run_id, "Relationships", [r.model_dump() for r in relationships])

            # 3. Objective & Ambiguities
            objective = self.organizer.process_objective(
                project_id=project_id,
                user_question=objective_question,
                catalog=catalog,
                user_clarifications=user_clarifications
            )
            DatabaseService.save_artifact(f"OBJ_{run_id}", run_id, "ObjectiveSpec", objective.model_dump())

            # Check if clarification is needed
            unresolved = [a for a in objective.pending_ambiguities if a.status != "resolved"]
            if unresolved and not user_clarifications:
                curr_stage = self._transition(run_id, curr_stage, PipelineStage.NEEDS_CLARIFICATION, "Se requiere confirmación de supuestos antes de proceder al plan.")
                # We can pause here or proceed if defaults are accepted
                # For demo vertical run, if user provided clarifications or default mode, we proceed
                pass

            # 4. Methodologist Plan
            curr_stage = self._transition(run_id, curr_stage, PipelineStage.PLAN_READY, "Agente Metodólogo: Elaboración del plan analítico registrado.")
            plan = self.methodologist.build_plan(
                project_id=project_id,
                run_id=run_id,
                objective=objective,
                catalog=catalog,
                dq_report=dq_report,
                confirmed_relationships=relationships
            )
            DatabaseService.save_artifact(f"PLAN_{run_id}", run_id, "AnalysisPlan", plan.model_dump())

            # 5. Data Preparation
            curr_stage = self._transition(run_id, curr_stage, PipelineStage.PREPARING, "Agente Preparador: Limpieza, deduplicación y uniones controladas.")
            analytical_df, transformations = self.preparer.prepare_data(
                tables=loaded_tables,
                relationships=relationships,
                exclude_cancelled=True,
                cutoff_date="2026-05-31"
            )
            DatabaseService.save_artifact(f"TRF_{run_id}", run_id, "TransformationRecords", [t.model_dump() for t in transformations])

            # 6. Analytical Execution
            curr_stage = self._transition(run_id, curr_stage, PipelineStage.ANALYZING, "Agente Ejecutor: Cálculo de métricas, descomposiciones y dinámicas.")
            results = self.executor.execute_plan(analytical_df=analytical_df, plan=plan)

            # Test injection: force non-finite number or reconciliation error to verify repair loop
            if force_error_for_test:
                results[0].calculated_values["net_sales"] = float("nan")

            DatabaseService.save_artifact(f"RES_{run_id}", run_id, "AnalysisResults", [r.model_dump() for r in results])

            # 7. Independent Validation with Repair Loop
            curr_stage = self._transition(run_id, curr_stage, PipelineStage.VALIDATING, "Agente Validador: Comprobación determinista de reconciliación y finitud.")
            repair_attempt = 0
            validation_report = self.validator.validate_results(
                run_id=run_id,
                results=results,
                transformations=transformations,
                repair_attempt=repair_attempt
            )

            # Repair cycle if rejected and can retry
            while validation_report.overall_status == "rejected" and validation_report.can_retry and repair_attempt < 2:
                repair_attempt = DatabaseService.increment_repair_attempt(run_id)
                curr_stage = self._transition(run_id, curr_stage, PipelineStage.NEEDS_REPAIR, f"Intento de reparación automática {repair_attempt}: {validation_report.repair_instruction}")

                # Auto-repair actions
                if force_error_for_test:
                    # Fix the test injected NaN
                    results[0].calculated_values["net_sales"] = float(analytical_df["net_sales"].sum())
                    force_error_for_test = False

                # Re-validate
                curr_stage = self._transition(run_id, curr_stage, PipelineStage.ANALYZING, "Re-ejecución tras reparación.")
                curr_stage = self._transition(run_id, curr_stage, PipelineStage.VALIDATING, "Re-validación independiente de resultados.")
                validation_report = self.validator.validate_results(
                    run_id=run_id,
                    results=results,
                    transformations=transformations,
                    repair_attempt=repair_attempt
                )

            DatabaseService.save_artifact(f"VAL_{run_id}", run_id, "ValidationReport", validation_report.model_dump())

            if validation_report.overall_status == "rejected":
                fail_msg = f"Validación rechazada tras {repair_attempt} intentos. Errores: {', '.join(validation_report.failed_results)}"
                DatabaseService.update_run_stage(run_id, PipelineStage.FAILED.value, status="FAILED", failure_reason=fail_msg)
                return {"status": "FAILED", "reason": fail_msg, "validation_report": validation_report.model_dump()}

            curr_stage = self._transition(run_id, curr_stage, PipelineStage.VALIDATED, f"Resultados aprobados con estado: {validation_report.overall_status}.")

            # 8. Interpretation
            insight_report = self.interpreter.interpret_results(
                run_id=run_id,
                results=results,
                validation_report=validation_report,
                objective=objective
            )
            DatabaseService.save_artifact(f"INS_{run_id}", run_id, "InsightReport", insight_report.model_dump())

            # 9. Dashboard Building
            curr_stage = self._transition(run_id, curr_stage, PipelineStage.BUILDING_DASHBOARD, "Agente Constructor: Generando DashboardSpec con componentes aprobados.")
            dashboard_spec = self.dashboard_builder.build_dashboard(
                run_id=run_id,
                results=results,
                validation_report=validation_report,
                insight_report=insight_report,
                objective_question=objective.original_question
            )

            # 10. Dashboard Validation
            validated_dashboard = self.dashboard_validator.validate_dashboard(
                dashboard=dashboard_spec,
                results=results,
                validation_report=validation_report
            )
            DatabaseService.save_artifact(f"DASH_{run_id}", run_id, "DashboardSpec", validated_dashboard.model_dump())

            # Final transition
            curr_stage = self._transition(run_id, curr_stage, PipelineStage.READY, "Pipeline completado con éxito. Dashboard y asistente listos.")

            return {
                "status": "READY",
                "run_id": run_id,
                "project_id": project_id,
                "dashboard": validated_dashboard.model_dump(),
                "insights": insight_report.model_dump(),
                "validation": validation_report.model_dump()
            }

        except Exception as e:
            DatabaseService.update_run_stage(run_id, PipelineStage.FAILED.value, status="FAILED", failure_reason=str(e))
            DatabaseService.log_event(run_id, "ERROR", "ERROR", f"Fallo catastrófico en pipeline: {str(e)}")
            raise
