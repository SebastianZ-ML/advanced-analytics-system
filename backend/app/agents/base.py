"""
Base agent definition and LLM provider abstraction.
Supports:
- Real LLM providers (Gemini, OpenAI) if configured.
- 'Demostración sin LLM' (Deterministic logic without simulating LLM reasoning).
- Strict validation of structured outputs.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from app.config import settings


class BaseAgent(ABC):
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    @property
    def is_demo_mode(self) -> bool:
        return settings.is_demo_mode

    @property
    def execution_mode_label(self) -> str:
        if self.is_demo_mode:
            return "Demo Without LLM (Deterministic Analytics Engine)"
        return f"LLM-Assisted Mode ({settings.llm_provider.capitalize()})"
