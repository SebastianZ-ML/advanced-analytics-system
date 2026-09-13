"""
Automated Test Suite for Gemini LLM Integration and Deterministic Guardrails.
Covers 16 required verification points:
1. Startup without API key (resolves DEMO_WITHOUT_LLM).
2. DEMO_WITHOUT_LLM operation without external calls.
3. Structuring agent valid spec generation via mock LLM.
4. Structuring agent graceful handling of malformed LLM responses.
5. Methodologist agent rejection of non-existent columns.
6. Methodologist agent rejection of unregistered operations.
7. Independent validator rejection of inconsistent results regardless of LLM.
8. Validation status immutability (LLM cannot alter approved/rejected status).
9. Interpreter agent rejection of hallucinated / non-existent result_ids.
10. Conversational assistant deterministic recalculation with custom filters.
11. Conversational assistant explicit refusal for factors absent from data.
12. API key redaction in logs, database, and exception traces.
13. Timeout and bounded retry handling.
14. Graceful degradation to DEMO_WITHOUT_LLM on fatal API errors.
15. Strict equality between dashboard numerical figures and deterministic engine.
16. Real Gemini API call verification (conditional on GEMINI_API_KEY).
"""
import os
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from app.agents.assistant import ConversationalAssistantAgent
from app.agents.auditor import DataAuditorAgent
from app.agents.dashboard_builder import DashboardBuilderAgent
from app.agents.dashboard_validator import DashboardValidatorAgent
from app.agents.executor import AnalyticalExecutorAgent
from app.agents.interpreter import InterpreterAgent
from app.agents.methodologist import REGISTERED_METHODS_CATALOG, MethodologistAgent
from app.agents.organizer import OrganizerAgent
from app.agents.preparer import DataPreparerAgent
from app.agents.validator import AnalyticalValidatorAgent
from app.config import Settings
from app.contracts import (
    ActionRecommendation,
    AmbiguityItem,
    AnalysisPlan,
    AnalysisResult,
    ChatRequest,
    ColumnProfile,
    DashboardSpec,
    DataCatalog,
    DataQualityReport,
    ExternalResearchCitation,
    InsightFinding,
    InsightInterpretation,
    InsightReport,
    MetricCardSpec,
    ObjectiveSpec,
    OperationSpec,
    RelationshipSpec,
    TableProfile,
    TransformationRecord,
    ValidationReport,
)
from app.engine.duckdb_engine import DuckDBAnalyticsEngine
from app.engine.synthetic_data import generate_demo_dataset
from app.orchestrator.runner import PipelineRunner
from app.providers.base import (
    LLMAuthenticationError,
    LLMProvider,
    LLMProviderError,
    LLMTimeoutError,
    LLMValidationError,
)
from app.providers.context_builder import ContextBuilder
from app.providers.demo_provider import NoLLMProvider
from app.providers.factory import get_llm_provider
from app.providers.gemini_provider import GeminiProvider
from app.storage.db import DatabaseService, get_db_connection, init_db
from app.storage.files import FileManager


@pytest.fixture(scope="session")
def setup_gemini_test_env(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("gemini_test_data")
    gt = generate_demo_dataset(data_dir, seed=123)
    init_db()
    return data_dir, gt


# ---------------------------------------------------------------------------
# Test 1: Successful startup without API key
# ---------------------------------------------------------------------------
def test_01_startup_without_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    st = Settings(gemini_api_key="")
    assert st.operating_mode == "DEMO_WITHOUT_LLM"
    assert st.is_demo_without_llm is True

    provider = get_llm_provider(override_provider=NoLLMProvider())
    assert isinstance(provider, NoLLMProvider)
    assert provider.is_deterministic_fallback is True
    assert provider.provider_name in ["demo", "Demostración sin LLM"]


# ---------------------------------------------------------------------------
# Test 2: DEMO_WITHOUT_LLM operates completely deterministically
# ---------------------------------------------------------------------------
def test_02_demo_without_llm_mode(setup_gemini_test_env):
    data_dir, _ = setup_gemini_test_env
    provider = NoLLMProvider()
    assert provider.is_deterministic_fallback is True

    # Test structuring
    obj = provider.structure_objective(
        project_id="P_TEST_DEMO",
        user_question="¿Por qué cayeron las ventas?",
        catalog_summary={}
    )
    assert isinstance(obj, ObjectiveSpec)
    assert "net_sales" in obj.primary_metric
    assert obj.is_demo_mode is True

    # Test plan proposal
    plan = provider.propose_analysis_plan(
        project_id="P_TEST_DEMO",
        run_id="R_TEST_DEMO",
        objective=obj,
        catalog_summary={},
        dq_summary={},
        confirmed_relationships=[],
        registered_methods=REGISTERED_METHODS_CATALOG
    )
    assert isinstance(plan, AnalysisPlan)
    assert len(plan.operations) >= 4
    for op in plan.operations:
        assert any(m["step_id"] == op.step_id for m in REGISTERED_METHODS_CATALOG)


# ---------------------------------------------------------------------------
# Test 3: Structuring agent produces valid spec via Mock LLM
# ---------------------------------------------------------------------------
def test_03_structuring_agent_valid_spec_with_mock():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.is_deterministic_fallback = False
    mock_provider.provider_name = "gemini"
    mock_provider.model_name = "gemini-3.6-flash"

    expected_spec = ObjectiveSpec(
        schema_version="1.0",
        project_id="P_MOCK_1",
        original_question="Ventas cayeron en mayo",
        operational_objective="Cuantificar la variación de facturación neta entre marzo y mayo de 2026",
        decision_to_inform="Focalizar plan de recuperación de clientes",
        primary_metric="net_sales",
        time_period="2026-01 a 2026-05",
        relevant_dimensions=["channel", "region"],
        is_demo_mode=False
    )
    mock_provider.structure_objective.return_value = expected_spec

    organizer = OrganizerAgent(override_provider=mock_provider)
    result = organizer.process_objective(
        project_id="P_MOCK_1",
        user_question="Ventas cayeron en mayo"
    )

    assert result == expected_spec
    assert result.is_demo_mode is False
    assert result.primary_metric == "net_sales"
    mock_provider.structure_objective.assert_called_once()


# ---------------------------------------------------------------------------
# Test 4: Structuring agent rejects malformed JSON and falls back safely
# ---------------------------------------------------------------------------
def test_04_structuring_agent_rejects_malformed_json_and_handles_gracefully():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.is_deterministic_fallback = False
    # Simulate LLM returning malformed JSON / validation error
    mock_provider.structure_objective.side_effect = LLMValidationError("JSON malformado o campos faltantes")

    organizer = OrganizerAgent(override_provider=mock_provider)
    # Must not raise an exception; must gracefully fall back to deterministic spec
    result = organizer.process_objective(
        project_id="P_MOCK_ERR",
        user_question="Ventas cayeron en mayo"
    )

    assert isinstance(result, ObjectiveSpec)
    assert "net_sales" in result.primary_metric
    assert result.is_demo_mode is True


# ---------------------------------------------------------------------------
# Test 5: Methodologist rejects non-existent columns proposed by LLM
# ---------------------------------------------------------------------------
def test_05_methodologist_rejects_nonexistent_columns():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.is_deterministic_fallback = False

    # Propose an operation requesting a column that doesn't exist
    illegal_op = OperationSpec(
        step_id="OP_01_METRIC_SUMMARY",
        operation_name="Resumen descriptivo",
        category="metric_summary",
        description="Operación con columna fantasma",
        required_inputs=["orders_analytical"],
        parameters={"metric_columns": ["non_existent_fake_column_xyz"]},
        assumptions=[],
        pre_execution_validations=[],
        depends_on=[]
    )

    mock_provider.propose_analysis_plan.return_value = AnalysisPlan(
        schema_version="1.0",
        project_id="P_TEST",
        run_id="R_TEST",
        title="Plan Ilícito",
        rationale="Prueba de columnas inexistentes",
        operations=[illegal_op],
        excluded_methods=[],
        citations=[]
    )

    # Valid catalog with standard columns
    catalog = DataCatalog(
        schema_version="1.0",
        project_id="P_TEST",
        tables={
            "orders": TableProfile(
                table_id="orders",
                source_filename="orders.csv",
                display_name="Orders",
                row_count=100,
                column_count=2,
                file_hash_sha256="hash123",
                row_semantic_meaning="Transacción",
                columns={
                    "order_id": ColumnProfile(name="order_id", inferred_type="string", total_count=100, null_count=0, null_percentage=0.0, unique_count=100),
                    "net_sales": ColumnProfile(name="net_sales", inferred_type="float", total_count=100, null_count=0, null_percentage=0.0, unique_count=90)
                }
            )
        }
    )

    methodologist = MethodologistAgent(override_provider=mock_provider)
    objective = ObjectiveSpec(
        schema_version="1.0",
        project_id="P_TEST",
        original_question="Ventas",
        operational_objective="Diagnóstico",
        decision_to_inform="Decisión comercial",
        primary_metric="net_sales",
        time_period="2026-01 a 2026-05"
    )

    plan = methodologist.build_plan(
        project_id="P_TEST",
        run_id="R_TEST",
        objective=objective,
        catalog=catalog,
        dq_report=DataQualityReport(schema_version="1.0", project_id="P_TEST", has_critical_issues=False, issues=[], global_score=100.0)
    )

    # The illegal op must be rejected; methodologist falls back to registered deterministic plan
    for op in plan.operations:
        params = op.parameters or {}
        if "metric_columns" in params:
            assert "non_existent_fake_column_xyz" not in params["metric_columns"]


# ---------------------------------------------------------------------------
# Test 6: Methodologist rejects unregistered operation methods
# ---------------------------------------------------------------------------
def test_06_methodologist_rejects_unregistered_method():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.is_deterministic_fallback = False

    unregistered_op = OperationSpec(
        step_id="OP_BLACKBOX_NEURAL_PREDICT",
        operation_name="Red Neuronal No Registrada",
        category="metric_summary",
        description="Método no implementado en el backend",
        required_inputs=["orders_analytical"],
        parameters={},
        assumptions=[],
        pre_execution_validations=[],
        depends_on=[]
    )

    mock_provider.propose_analysis_plan.return_value = AnalysisPlan(
        schema_version="1.0",
        project_id="P_TEST",
        run_id="R_TEST",
        title="Plan con Operación No Registrada",
        rationale="Prueba",
        operations=[unregistered_op],
        excluded_methods=[],
        citations=[]
    )

    methodologist = MethodologistAgent(override_provider=mock_provider)
    objective = ObjectiveSpec(
        schema_version="1.0",
        project_id="P_TEST",
        original_question="Ventas",
        operational_objective="Diagnóstico",
        decision_to_inform="Decisión comercial",
        primary_metric="net_sales",
        time_period="2026-01 a 2026-05"
    )

    plan = methodologist.build_plan(
        project_id="P_TEST",
        run_id="R_TEST",
        objective=objective,
        catalog=DataCatalog(schema_version="1.0", project_id="P_TEST", tables={}),
        dq_report=DataQualityReport(schema_version="1.0", project_id="P_TEST", has_critical_issues=False, issues=[], global_score=100.0)
    )

    # OP_BLACKBOX_NEURAL_PREDICT must not be in the approved plan
    assert all(op.step_id != "OP_BLACKBOX_NEURAL_PREDICT" for op in plan.operations)


# ---------------------------------------------------------------------------
# Test 7: Independent validator rejects inconsistent result regardless of LLM
# ---------------------------------------------------------------------------
def test_07_validator_rejects_inconsistent_result_regardless_of_llm():
    validator = AnalyticalValidatorAgent()

    inconsistent_result = AnalysisResult(
        result_id="RES_INCONSISTENT_DELTA",
        step_id="OP_04_DIMENSION_BREAKDOWN_CHANNEL",
        operation_name="Descomposición aditiva con error matemático",
        method="Waterfall Breakdown",
        parameters={},
        data_source_version="v1.0",
        unit_of_measure="CLP",
        time_period_covered="2026-03 vs 2026-05",
        calculated_values={
            "total_change": -100000.0,
            "contributions": [
                {"dimension_value": "Canal A", "absolute_change": -50000.0, "contribution_to_total_change": 50.0},
                {"dimension_value": "Canal B", "absolute_change": -20000.0, "contribution_to_total_change": 20.0}
                # Missing 30,000 CLP -> fails 100% reconciliation check
            ],
            "is_perfectly_reconciled": False,
            "reconciliation_diff": 30000.0
        }
    )

    v_report = validator.validate_results(
        run_id="R_TEST_VAL",
        results=[inconsistent_result],
        transformations=[]
    )

    assert v_report.overall_status == "rejected"
    assert inconsistent_result.validation_status == "rejected"
    assert "RES_INCONSISTENT_DELTA" in v_report.failed_results


# ---------------------------------------------------------------------------
# Test 8: Validation status cannot be overwritten by LLM
# ---------------------------------------------------------------------------
def test_08_validation_status_unalterable_by_llm():
    # Attempt to pass a rejected result to InterpreterAgent
    rejected_res = AnalysisResult(
        result_id="RES_REJECTED_BY_MATH",
        step_id="OP_FAIL",
        operation_name="Cálculo Fallido",
        method="Method",
        parameters={},
        data_source_version="v1.0",
        unit_of_measure="CLP",
        time_period_covered="2026-03",
        calculated_values={"val": float("nan")},
        validation_status="rejected"
    )

    v_report = ValidationReport(
        schema_version="1.0",
        run_id="R_TEST_IMMUTABLE",
        overall_status="rejected",
        failed_results=["RES_REJECTED_BY_MATH"]
    )

    interpreter = InterpreterAgent()
    report = interpreter.interpret_results(
        run_id="R_TEST_IMMUTABLE",
        results=[rejected_res],
        validation_report=v_report
    )

    # Rejected results must NEVER become findings or be approved
    assert all(f.result_id != "RES_REJECTED_BY_MATH" for f in report.observed_findings)


# ---------------------------------------------------------------------------
# Test 9: Interpreter rejects hallucinated result_id
# ---------------------------------------------------------------------------
def test_09_interpreter_rejects_nonexistent_result_id():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.is_deterministic_fallback = False

    approved_res = AnalysisResult(
        result_id="RES_APPROVED_REAL",
        step_id="OP_01_METRIC_SUMMARY",
        operation_name="Resumen",
        method="Aggregation",
        parameters={},
        data_source_version="v1.0",
        unit_of_measure="CLP",
        time_period_covered="2026-01 a 2026-05",
        calculated_values={"net_sales": 100000.0},
        validation_status="approved"
    )

    hallucinated_finding = InsightFinding(
        id="FINDING_HALLUCINATED",
        claim="Las ventas se desplomaron por factores no probados",
        result_id="RES_HALLUCINATED_DOES_NOT_EXIST",
        metric_name="net_sales",
        observed_value=-99999.0,
        is_empirically_proven=True,
        evidence_text="Sin sustento real"
    )

    mock_provider.interpret_validated_results.return_value = InsightReport(
        schema_version="1.0",
        run_id="R_TEST_HALLUCINATION",
        executive_summary="Resumen con alucinación",
        observed_findings=[hallucinated_finding],
        interpretations=[],
        unproven_hypotheses=[],
        data_limitations=[],
        recommended_actions=[]
    )

    interpreter = InterpreterAgent(override_provider=mock_provider)
    v_report = ValidationReport(
        schema_version="1.0",
        run_id="R_TEST_HALLUCINATION",
        overall_status="approved",
        approved_results=["RES_APPROVED_REAL"]
    )

    insight_report = interpreter.interpret_results(
        run_id="R_TEST_HALLUCINATION",
        results=[approved_res],
        validation_report=v_report
    )

    # Finding citing non-existent result_id must be discarded
    assert all(f.result_id != "RES_HALLUCINATED_DOES_NOT_EXIST" for f in insight_report.observed_findings)


# ---------------------------------------------------------------------------
# Test 10: Assistant executes deterministic recalculation tool for filters
# ---------------------------------------------------------------------------
def test_10_assistant_executes_recalc_tool_for_filters(setup_gemini_test_env):
    data_dir, _ = setup_gemini_test_env
    df_orders = FileManager.read_table_dataframe(data_dir / "orders.csv")
    df_orders["net_sales"] = pd.to_numeric(df_orders["net_sales"], errors="coerce").fillna(0.0)

    # Filter Mayorista / B2B channel
    expected_sum = float(df_orders[df_orders["channel"] == "Mayorista / B2B"]["net_sales"].sum())
    expected_count = int(len(df_orders[df_orders["channel"] == "Mayorista / B2B"]))

    assistant = ConversationalAssistantAgent()
    req = ChatRequest(
        project_id="P_TEST",
        run_id="R_TEST",
        question="¿Cuánto vendió el canal Mayorista / B2B?",
        active_filters={"channel": "Mayorista / B2B"}
    )

    test_obj = ObjectiveSpec(
        schema_version="1.0",
        project_id="P_TEST",
        original_question="Ventas",
        operational_objective="Diagnóstico",
        decision_to_inform="Decisión comercial",
        primary_metric="net_sales",
        time_period="2026-01 a 2026-05"
    )

    ans = assistant.answer_query(
        request=req,
        objective=test_obj,
        catalog=DataCatalog(schema_version="1.0", project_id="P_TEST", tables={}),
        analytical_df=df_orders,
        results=[],
        dashboard=DashboardSpec(schema_version="1.0", run_id="R_TEST", title="", subtitle="", objective_question="", metric_cards=[], charts=[]),
        insight_report=InsightReport(schema_version="1.0", run_id="R_TEST", executive_summary="")
    )

    assert ans.query_type == "recalculate_with_filters"
    assert len(ans.citations) > 0
    assert ans.citations[0].result_id == "RECALC_ON_DEMAND"
    assert f"${expected_sum:,.2f}" in ans.answer_text
    assert f"{expected_count} pedidos" in ans.answer_text


# ---------------------------------------------------------------------------
# Test 11: Assistant responds with explicit refusal for factors absent from data
# ---------------------------------------------------------------------------
def test_11_assistant_explicit_refusal_for_missing_factors():
    assistant = ConversationalAssistantAgent()
    catalog = DataCatalog(
        schema_version="1.0",
        project_id="P_TEST",
        tables={
            "orders": TableProfile(
                table_id="orders",
                source_filename="orders.csv",
                display_name="Orders",
                row_count=10,
                column_count=1,
                file_hash_sha256="h1",
                row_semantic_meaning="Order",
                columns={"order_id": ColumnProfile(name="order_id", inferred_type="string", total_count=10, null_count=0, null_percentage=0.0, unique_count=10)}
            )
        }
    )

    test_obj = ObjectiveSpec(
        schema_version="1.0",
        project_id="P_TEST",
        original_question="Ventas",
        operational_objective="Diagnóstico",
        decision_to_inform="Decisión comercial",
        primary_metric="net_sales",
        time_period="2026-01 a 2026-05"
    )

    for unanswerable_q in [
        "¿Qué impacto tuvieron los precios de la competencia en mayo?",
        "¿El clima afectó la caída de ventas?",
        "¿Cómo influyó la inflación nacional en el resultado?"
    ]:
        req = ChatRequest(
            project_id="P_TEST",
            run_id="R_TEST",
            question=unanswerable_q
        )
        ans = assistant.answer_query(
            request=req,
            objective=test_obj,
            catalog=catalog,
            analytical_df=pd.DataFrame(),
            results=[],
            dashboard=DashboardSpec(schema_version="1.0", run_id="R_TEST", title="", subtitle="", objective_question="", metric_cards=[], charts=[]),
            insight_report=InsightReport(schema_version="1.0", run_id="R_TEST", executive_summary="")
        )

        assert ans.query_type == "unanswerable_by_data"
        assert len(ans.citations) == 0
        assert ans.data_limitation_notice is not None
        assert "no contienen información sobre" in ans.answer_text


# ---------------------------------------------------------------------------
# Test 12: Gemini API key is NEVER leaked in logs or error traces
# ---------------------------------------------------------------------------
def test_12_api_key_never_leaked_in_logs_or_errors():
    secret_key = "AIzaSyFakeSecretKey_9876543210ABCDEF"
    provider = GeminiProvider(api_key=secret_key, model_name="gemini-3.6-flash")

    # Mock the client's generate_content to raise an exception embedding the key
    with patch.object(provider._client.models, "generate_content") as mock_gen:
        mock_gen.side_effect = Exception(f"HTTP 403 Forbidden with key {secret_key}")

        with pytest.raises(Exception) as exc_info:
            provider._execute_with_retry(
                prompt="Test prompt",
                system_instruction="System security instruction",
                caller_agent="TestAgent",
                project_id="P_TEST_SECRET",
                run_id="R_TEST_SECRET"
            )

        # Ensure the exception raised to caller doesn't contain raw key
        assert secret_key not in str(exc_info.value)
        assert "[REDACTED_API_KEY]" in str(exc_info.value)

    # Check the database audit table llm_interactions
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM llm_interactions WHERE project_id = 'P_TEST_SECRET'").fetchall()
    conn.close()

    assert len(rows) > 0
    for row in rows:
        row_dict = dict(row)
        err_msg = row_dict.get("error_sanitized") or ""
        assert secret_key not in err_msg
        assert "[REDACTED_API_KEY]" in err_msg


# ---------------------------------------------------------------------------
# Test 13: Timeout and retry handling with exponential backoff
# ---------------------------------------------------------------------------
def test_13_timeout_and_retry_handling():
    provider = GeminiProvider(api_key="fake_key_for_test", model_name="gemini-3.6-flash", max_retries=2)

    with patch.object(provider._client.models, "generate_content") as mock_gen:
        # Simulate transient error 429 then success
        mock_success_response = MagicMock()
        mock_success_response.text = '{"status": "ok"}'
        mock_success_response.usage_metadata.prompt_token_count = 10
        mock_success_response.usage_metadata.candidates_token_count = 5

        mock_gen.side_effect = [
            Exception("HTTP 429 Resource Exhausted"),
            mock_success_response
        ]

        start_t = time.time()
        res = provider._execute_with_retry(
            prompt="Hello",
            system_instruction="Security instruction",
            caller_agent="TestAgent",
            project_id="P_RETRY",
            run_id="R_RETRY"
        )
        elapsed = time.time() - start_t

        assert res == '{"status": "ok"}'
        assert mock_gen.call_count == 2
        # Backoff delay was applied
        assert elapsed >= 1.0


# ---------------------------------------------------------------------------
# Test 14: Graceful degradation to DEMO_WITHOUT_LLM on fatal API error
# ---------------------------------------------------------------------------
def test_14_graceful_degradation_to_demo_on_fatal_llm_error():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.is_deterministic_fallback = False
    mock_provider.structure_objective.side_effect = LLMAuthenticationError("Clave revocada o cuota excedida")

    organizer = OrganizerAgent(override_provider=mock_provider)
    spec = organizer.process_objective(
        project_id="P_FATAL",
        user_question="¿Por qué bajaron las ventas?"
    )

    # Must fall back gracefully to deterministic spec
    assert spec is not None
    assert spec.is_demo_mode is True
    assert "net_sales" in spec.primary_metric


# ---------------------------------------------------------------------------
# Test 15: Numeric values in dashboard match deterministic engine exactly
# ---------------------------------------------------------------------------
def test_15_numeric_values_unaltered_by_llm(setup_gemini_test_env):
    data_dir, _ = setup_gemini_test_env
    df_orders = FileManager.read_table_dataframe(data_dir / "orders.csv")
    df_orders["net_sales"] = pd.to_numeric(df_orders["net_sales"], errors="coerce").fillna(0.0)

    # Calculate real gross and net sums directly using pandas / DuckDB
    completed_orders = df_orders[df_orders["status"] == "COMPLETED"]
    raw_net_sum = float(completed_orders["net_sales"].sum())

    project_id = f"proj_num_check_{uuid.uuid4().hex[:6]}"
    run_id = f"run_num_check_{uuid.uuid4().hex[:6]}"
    DatabaseService.create_project(project_id, "Test Numerical Precision")
    DatabaseService.add_project_file(
        file_id=str(uuid.uuid4()),
        project_id=project_id,
        filename="orders.csv",
        table_name="orders",
        file_path=str(data_dir / "orders.csv"),
        file_hash="hash_orders",
        row_count=len(df_orders),
        column_count=len(df_orders.columns)
    )
    DatabaseService.create_run(run_id, project_id)

    runner = PipelineRunner()
    result = runner.execute_run(project_id, run_id)

    assert result["status"] == "READY"
    dash = result["dashboard"]

    # Verify current sales metric card matches May completed net sales exactly
    curr_sales_card = next(c for c in dash["metric_cards"] if "Actuales" in c["title"])
    extracted_val = float(curr_sales_card["value"].replace("$", "").replace(",", ""))

    may_orders = df_orders[(df_orders["status"] == "COMPLETED") & (df_orders["order_date"].str.startswith("2026-05"))]
    may_net_sum = float(may_orders["net_sales"].sum())

    # Must match within 1 dollar due to formatting round
    assert abs(extracted_val - may_net_sum) < 1.0


# ---------------------------------------------------------------------------
# Test 16: Real Gemini API call with structured output (Conditional)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requiere GEMINI_API_KEY en variables de entorno")
def test_16_real_gemini_call_when_key_present():
    real_api_key = os.getenv("GEMINI_API_KEY")
    assert real_api_key is not None and len(real_api_key) > 10

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    provider = GeminiProvider(api_key=real_api_key, model_name=model_name)
    assert provider.is_deterministic_fallback is False

    test_catalog = {
        "orders": {
            "row_count": 500,
            "columns": ["order_id", "order_date", "order_status", "net_sales", "channel", "region"]
        },
        "customers": {
            "row_count": 200,
            "columns": ["customer_id", "customer_name", "segment", "region"]
        }
    }

    spec = provider.structure_objective(
        project_id="P_REAL_GEMINI_TEST",
        user_question="Las ventas bajaron fuertemente entre marzo y mayo de 2026. Necesito diagnosticar en qué canal se concentró la pérdida.",
        catalog_summary=test_catalog
    )

    assert isinstance(spec, ObjectiveSpec)
    assert spec.schema_version in ["1.0", "1.0.0"]
    assert spec.primary_metric in ["net_sales", "ventas_netas", "sales"]
    assert spec.is_demo_mode is False
    assert len(spec.operational_objective) > 15
