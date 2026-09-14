"""
LLM Diagnostics and Healthcheck Endpoints.
Allows safe testing of LLM connectivity, latency, token consumption,
and schema validation under real and mock conditions without leaking API keys.
"""
import uuid
import time
from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.providers import get_llm_provider
from app.providers.gemini_provider import GeminiProvider

router = APIRouter(prefix="/api/llm", tags=["llm"])


@router.get("/status")
def get_llm_status() -> Dict[str, Any]:
    """
    Returns effective LLM configuration without ever revealing secret credentials.
    """
    provider = get_llm_provider()
    key_configured = bool(settings.gemini_api_key and len(settings.gemini_api_key) > 5)
    key_length = len(settings.gemini_api_key) if settings.gemini_api_key else 0

    base_url = "https://generativelanguage.googleapis.com"
    if hasattr(provider, "base_url"):
        base_url = provider.base_url

    effective_timeout = settings.gemini_timeout_seconds
    if hasattr(provider, "effective_timeout_seconds"):
        effective_timeout = provider.effective_timeout_seconds

    return {
        "provider": getattr(provider, "provider_name", "deterministic"),
        "model": getattr(provider, "model_name", "none"),
        "base_url": base_url,
        "timeout_seconds": effective_timeout,
        "max_retries": settings.gemini_max_retries,
        "api_key_configured": key_configured,
        "api_key_length": key_length,
        "operating_mode": settings.operating_mode,
        "mode_label": settings.mode_label,
        "is_demo_mode": settings.is_demo_mode
    }


@router.post("/test_connection")
def test_llm_connection() -> Dict[str, Any]:
    """
    Executes on-demand separated diagnostic checks:
    1. Minimal Check: verifies basic connectivity and short generation without user data.
    2. Structured Check: verifies schema adherence with ObjectiveSpec and small mock catalog.
    Categorizes errors precisely and records an audit trace_id.
    """
    trace_id = f"llm_diag_{uuid.uuid4().hex[:8]}"
    provider = get_llm_provider()

    # If in DEMO_WITHOUT_LLM mode
    if provider.is_deterministic_fallback or not isinstance(provider, GeminiProvider):
        return {
            "trace_id": trace_id,
            "overall_status": "DEMO_MODE",
            "provider": getattr(provider, "provider_name", "deterministic"),
            "model": getattr(provider, "model_name", "none"),
            "base_url": "N/A (Local Engine)",
            "timeout_seconds": settings.gemini_timeout_seconds,
            "minimal_check": {
                "check_name": "Peticion Minima",
                "success": True,
                "duration_ms": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "response_preview": "OK (Modo Local)",
                "error": None,
                "error_category": None
            },
            "structured_check": {
                "check_name": "Peticion Estructurada",
                "success": True,
                "duration_ms": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "validated_objective": "Objetivo interpretado mediante analizador determinista local",
                "error": None,
                "error_category": None
            },
            "notice": "El sistema se encuentra operando en modo DEMO_WITHOUT_LLM (motor analítico local determinista)."
        }

    # Execute Check 1: Minimal Connection
    minimal_res = provider.test_minimal_connection()

    # Execute Check 2: Structured Connection
    structured_res = provider.test_structured_connection()

    # Determine overall status
    if minimal_res["success"] and structured_res["success"]:
        overall_status = "HEALTHY"
    elif minimal_res["success"] and not structured_res["success"]:
        overall_status = structured_res.get("error_category") or "SCHEMA_ERROR"
    else:
        overall_status = minimal_res.get("error_category") or "FAILED"

    return {
        "trace_id": trace_id,
        "overall_status": overall_status,
        "provider": provider.provider_name,
        "model": provider.model_name,
        "base_url": provider.base_url,
        "timeout_seconds": provider.effective_timeout_seconds,
        "minimal_check": minimal_res,
        "structured_check": structured_res
    }
