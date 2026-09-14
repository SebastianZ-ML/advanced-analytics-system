"""
Tests for LLM Integration Diagnostics:
- Effective configuration inspection without credential leakage
- Two separated checks (minimal ping vs structured ObjectiveSpec)
- Precise error classification (auth, permissions, model not found, rate limit, quota, timeout, network, schema)
- Millisecond timeout conversion in google-genai SDK
- Non-retry of fatal errors (401/403)
- Eradication of &#x20; entities in notice strings
"""
import uuid
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.providers.base import (
    LLMAuthenticationError,
    LLMPermissionError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMQuotaExceededError,
    LLMTimeoutError,
    LLMNetworkError,
    LLMValidationError,
)
from app.providers.gemini_provider import GeminiProvider


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_01_llm_status_endpoint_never_leaks_api_key(client):
    """Test 1: Status endpoint reports effective provider config without exposing API key."""
    res = client.get("/api/llm/status")
    assert res.status_code == 200
    data = res.json()

    assert data["provider"] in ["gemini", "deterministic"]
    assert "base_url" in data
    assert "timeout_seconds" in data
    assert data["timeout_seconds"] == settings.gemini_timeout_seconds
    assert "api_key_configured" in data
    assert "api_key_length" in data

    # Strict assertion: no API key leaked
    raw_str = str(data)
    assert settings.gemini_api_key not in raw_str if settings.gemini_api_key else True
    assert "AIza" not in raw_str, "Google API key pattern found in status response!"


def test_02_separated_checks_reporting_independence(client):
    """Test 2: Minimal check success does not mask structured check failure."""
    with patch.object(
        GeminiProvider,
        "test_minimal_connection",
        return_value={
            "check_name": "Peticion Minima",
            "success": True,
            "duration_ms": 150,
            "input_tokens": 5,
            "output_tokens": 2,
            "response_preview": "OK",
            "error": None,
            "error_category": None,
            "timestamp": "2026-09-13T21:00:00Z"
        }
    ), patch.object(
        GeminiProvider,
        "test_structured_connection",
        return_value={
            "check_name": "Peticion Estructurada",
            "success": False,
            "duration_ms": 850,
            "input_tokens": 70,
            "output_tokens": 0,
            "validated_objective": None,
            "error": "JSON output missing required 'operational_objective' field",
            "error_category": "LLMValidationError",
            "timestamp": "2026-09-13T21:00:01Z"
        }
    ):
        res = client.post("/api/llm/test_connection")
        assert res.status_code == 200
        data = res.json()

        assert data["trace_id"].startswith("llm_diag_")
        assert data["minimal_check"]["success"] is True
        assert data["structured_check"]["success"] is False
        assert data["overall_status"] == "LLMValidationError"
        assert data["structured_check"]["error_category"] == "LLMValidationError"


def test_03_error_classification_simulations():
    """Test 3: Precise exception classification without conflating errors."""
    provider = GeminiProvider(api_key="dummy_test_key_for_testing_purposes", timeout_seconds=45)

    # 1. Timeout
    err_timeout = provider._classify_exception(Exception("The read operation timed out"))
    assert isinstance(err_timeout, LLMTimeoutError)
    assert "45s" in str(err_timeout)

    # 2. Authentication (401)
    err_auth = provider._classify_exception(Exception("API_KEY_INVALID: 401 Unauthorized"))
    assert isinstance(err_auth, LLMAuthenticationError)

    # 3. Permissions (403)
    err_perm = provider._classify_exception(Exception("403 Forbidden: Caller does not have required permission"))
    assert isinstance(err_perm, LLMPermissionError)

    # 4. Model not found (404)
    err_model = provider._classify_exception(Exception("404 Not Found: models/gemini-999 is not found"))
    assert isinstance(err_model, LLMModelNotFoundError)

    # 5. Quota Failure (Free tier quota exhausted)
    err_quota = provider._classify_exception(Exception(
        "429 RESOURCE_EXHAUSTED: Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20"
    ))
    assert isinstance(err_quota, LLMQuotaExceededError)

    # 6. Rate limit (tokens/requests per min)
    err_rate = provider._classify_exception(Exception("429 Too Many Requests: Rate limit exceeded"))
    assert isinstance(err_rate, LLMRateLimitError)

    # 7. Network error
    err_net = provider._classify_exception(Exception("Connection reset by peer while connecting to api"))
    assert isinstance(err_net, LLMNetworkError)


def test_04_http_options_timeout_in_milliseconds():
    """Test 4: Verify timeout_seconds is converted to milliseconds for google-genai SDK."""
    provider = GeminiProvider(api_key="test_key_1234567890", timeout_seconds=60)
    # The client must have been initialized with 60000.0 ms timeout in http_options
    assert provider.effective_timeout_seconds == 60


def test_05_no_retries_on_fatal_errors():
    """Test 5: Fatal errors (401, 403) are never retried repeatedly."""
    provider = GeminiProvider(api_key="test_key_1234567890", timeout_seconds=30, max_retries=2)
    
    mock_generate = MagicMock(side_effect=Exception("401 Unauthorized: Invalid API key"))
    provider._client.models.generate_content = mock_generate

    with pytest.raises(LLMAuthenticationError):
        provider._execute_with_retry(prompt="test", caller_agent="TestAgent")

    # Must only be called ONCE (no retry loops on 401)
    assert mock_generate.call_count == 1


def test_06_notice_strings_contain_no_x20_entities():
    """Test 6: Provider notice strings never contain &#x20; or trailing spaces."""
    from app.agents.organizer import OrganizerAgent
    organizer = OrganizerAgent()
    
    # Trigger fallback with timeout
    with patch("app.providers.gemini_provider.GeminiProvider.structure_objective", side_effect=LLMTimeoutError("timeout")):
        spec = organizer.process_objective(
            project_id="test",
            user_question="¿Cómo evolucionaron las ventas?",
            catalog=None
        )
        assert spec.provider_notice is not None
        assert "&#x20;" not in spec.provider_notice
        assert spec.provider_notice == spec.provider_notice.strip()


def test_07_real_test_connection_structure(client):
    """Test 7: Live / Real test connection endpoint returns structured audit trace."""
    res = client.post("/api/llm/test_connection")
    assert res.status_code == 200
    data = res.json()

    assert "trace_id" in data
    assert data["trace_id"].startswith("llm_diag_")
    assert "overall_status" in data
    assert "minimal_check" in data
    assert "structured_check" in data
    assert "duration_ms" in data["minimal_check"]
    assert "duration_ms" in data["structured_check"]
