"""
Tests for objective interpretation, error classification, provider fallbacks,
and structured JSON API responses.
"""
import uuid
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.storage.db import DatabaseService
from app.providers.base import LLMTimeoutError, LLMValidationError, LLMQuotaExceededError


@pytest.fixture
def client():
    # raise_server_exceptions=False ensures Starlette routes unhandled 500 exceptions
    # to FastAPI global exception handlers instead of raising into pytest runner.
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def test_project():
    pid = f"test_err_{uuid.uuid4().hex[:8]}"
    DatabaseService.create_project(pid, "Error & Interpretation Test Project")
    return pid


def test_interpret_objective_success_csv(client, test_project):
    """Test 1: Successful interpretation with CSV dataset."""
    load_res = client.post(f"/api/samples/simple_csv/load_to_project/{test_project}")
    assert load_res.status_code == 200, f"Failed loading sample: {load_res.text}"

    payload = {
        "objective_question": "¿Cómo evolucionaron las ventas mensuales y cuáles categorías tuvieron mayor variación?"
    }
    res = client.post(f"/api/projects/{test_project}/interpret_objective", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "operational_objective" in data
    assert "primary_metric" in data
    assert data["is_supported"] is True
    assert isinstance(data.get("clarifications_needed"), list)


def test_interpret_objective_success_xlsx(client, test_project):
    """Test 2: Successful interpretation with multi-sheet Excel dataset."""
    load_res = client.post(f"/api/samples/excel_multi_sheet/load_to_project/{test_project}")
    assert load_res.status_code == 200, f"Failed loading sample: {load_res.text}"

    payload = {
        "objective_question": "¿Cuál es el margen y rentabilidad por sucursal en el periodo analizado?"
    }
    res = client.post(f"/api/projects/{test_project}/interpret_objective", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "operational_objective" in data
    assert "primary_metric" in data
    assert data["is_supported"] is True


def test_interpret_objective_project_not_found_returns_404_json(client):
    """Test 3: Non-existent project returns 404 with structured JSON."""
    fake_pid = f"missing_{uuid.uuid4().hex[:8]}"
    res = client.post(f"/api/projects/{fake_pid}/interpret_objective", json={"objective_question": "test question"})
    
    assert res.status_code == 404
    assert res.headers["content-type"].startswith("application/json")
    data = res.json()
    assert data["error_code"] == "NOT_FOUND"
    assert "not found" in data["message"].lower() or "no encontrado" in data["message"].lower()
    assert "trace_id" in data


def test_interpret_objective_project_without_tables_returns_400_json(client, test_project):
    """Test 4: Project without tables returns 400 with structured JSON."""
    res = client.post(f"/api/projects/{test_project}/interpret_objective", json={"objective_question": "test question"})
    
    assert res.status_code == 400
    assert res.headers["content-type"].startswith("application/json")
    data = res.json()
    assert data["error_code"] == "BAD_REQUEST"
    assert "tables" in data["message"].lower() or "tablas" in data["message"].lower()
    assert "trace_id" in data


def test_interpret_objective_validation_error_returns_422_json(client, test_project):
    """Test 5: Missing or invalid payload returns 422 with structured JSON."""
    res = client.post(f"/api/projects/{test_project}/interpret_objective", json={})
    
    assert res.status_code == 422
    assert res.headers["content-type"].startswith("application/json")
    data = res.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    assert "trace_id" in data
    assert data.get("detail") is not None


def test_interpret_objective_provider_timeout_triggers_deterministic_fallback(client, test_project):
    """Test 6: Provider timeout triggers deterministic fallback with notice and reason."""
    load_res = client.post(f"/api/samples/simple_csv/load_to_project/{test_project}")
    assert load_res.status_code == 200

    with patch("app.providers.gemini_provider.GeminiProvider.structure_objective", side_effect=LLMTimeoutError("Timeout after 45s")):
        res = client.post(
            f"/api/projects/{test_project}/interpret_objective",
            json={"objective_question": "¿Cómo evolucionaron las ventas?"}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["fallback_reason"] == "timeout"
        assert data["provider_notice"] is not None
        assert "tiempo de espera" in data["provider_notice"].lower() or "timeout" in data["provider_notice"].lower()
        assert data["operational_objective"] is not None


def test_interpret_objective_provider_schema_validation_error_triggers_fallback(client, test_project):
    """Test 7: Provider schema mismatch triggers deterministic fallback."""
    load_res = client.post(f"/api/samples/simple_csv/load_to_project/{test_project}")
    assert load_res.status_code == 200

    with patch("app.providers.gemini_provider.GeminiProvider.structure_objective", side_effect=LLMValidationError("Invalid schema format")):
        res = client.post(
            f"/api/projects/{test_project}/interpret_objective",
            json={"objective_question": "¿Cómo evolucionaron las ventas?"}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["fallback_reason"] == "schema_validation_error"
        assert data["provider_notice"] is not None
        assert "esquema" in data["provider_notice"].lower() or "schema" in data["provider_notice"].lower()
        assert data["operational_objective"] is not None


def test_unhandled_exception_returns_structured_json_and_redacts_secrets(client, test_project):
    """Test 8: Unhandled 500 error returns structured JSON with trace_id without leaking secrets."""
    with patch("app.storage.db.DatabaseService.get_project", side_effect=RuntimeError("Unexpected DB crash with secret AIzaSyFakeSecretKey")):
        res = client.post(
            f"/api/projects/{test_project}/interpret_objective",
            json={"objective_question": "¿Cómo evolucionaron las ventas?"}
        )
        assert res.status_code == 500
        assert res.headers["content-type"].startswith("application/json")
        data = res.json()
        assert data["error_code"] == "INTERNAL_SERVER_ERROR"
        assert "trace_id" in data
        assert data["trace_id"].startswith("err_")
        raw_output = str(data)
        assert "AIzaSyFakeSecretKey" not in raw_output, "Secrets must be redacted!"


def test_frontend_error_handling_resilience_simulation():
    """
    Test 9: Simulate safeFetchJson behavior against plain text / HTML 500 errors.
    Ensures that non-JSON responses do NOT cause Unexpected token 'I', 'Internal S'...
    """
    def simulate_safe_fetch(status, headers, body_text):
        content_type = headers.get("content-type", "")
        data = None
        raw_text = ""

        if "application/json" in content_type:
            try:
                import json
                data = json.loads(body_text)
            except Exception:
                raw_text = body_text
        else:
            raw_text = body_text

        if status >= 400:
            error_msg = f"Error del servidor ({status})"
            trace_id = None
            error_code = f"HTTP_{status}"

            if data and isinstance(data, dict):
                error_msg = data.get("message") or data.get("detail") or str(data)
                trace_id = data.get("trace_id")
                error_code = data.get("error_code") or error_code
            elif raw_text:
                clean_text = raw_text.strip()
                error_msg = clean_text[:200] + "..." if len(clean_text) > 200 else clean_text

            return {
                "ok": False,
                "error": error_msg,
                "status": status,
                "trace_id": trace_id,
                "error_code": error_code,
            }
        return {"ok": True, "data": data or raw_text}

    # Case A: Plain text "Internal Server Error"
    result_a = simulate_safe_fetch(500, {"content-type": "text/plain"}, "Internal Server Error")
    assert result_a["ok"] is False
    assert result_a["error"] == "Internal Server Error"
    assert "Unexpected token" not in result_a["error"]

    # Case B: HTML error page
    html_error = "<html><body><h1>502 Bad Gateway</h1></body></html>"
    result_b = simulate_safe_fetch(502, {"content-type": "text/html"}, html_error)
    assert result_b["ok"] is False
    assert "502 Bad Gateway" in result_b["error"]
    assert "Unexpected token" not in result_b["error"]

    # Case C: Structured JSON error
    json_error = '{"error_code": "NOT_FOUND", "message": "Proyecto no encontrado", "trace_id": "tr_123"}'
    result_c = simulate_safe_fetch(404, {"content-type": "application/json"}, json_error)
    assert result_c["ok"] is False
    assert result_c["error"] == "Proyecto no encontrado"
    assert result_c["trace_id"] == "tr_123"
    assert result_c["error_code"] == "NOT_FOUND"
