"""
Factory for resolving and instantiating the active LLMProvider.
Supports transparent fallback to NoLLMProvider when credentials are unset or invalid.
"""
from typing import Optional
from app.config import settings
from app.providers.base import LLMProvider
from app.providers.demo_provider import NoLLMProvider
from app.providers.gemini_provider import GeminiProvider


def get_llm_provider(override_provider: Optional[LLMProvider] = None) -> LLMProvider:
    """
    Returns the active LLM provider based on system settings.
    If override_provider is passed (e.g., in unit tests), returns it directly.
    """
    if override_provider:
        return override_provider

    if settings.operating_mode == "LLM_ENABLED":
        try:
            return GeminiProvider(
                api_key=settings.gemini_api_key,
                model_name=settings.gemini_model
            )
        except Exception as e:
            # Sanitized error logging without secrets
            sanitized = str(e).replace(settings.gemini_api_key, "[REDACTED]") if settings.gemini_api_key else str(e)
            print(f"[WARN] No se pudo inicializar GeminiProvider ({sanitized}). Activando fallback NoLLMProvider.")
            return NoLLMProvider()

    return NoLLMProvider()
