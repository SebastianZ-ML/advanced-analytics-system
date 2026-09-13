"""
Configuration settings for the Adaptive Multi-Agent Analytics System.
Supports LLM_ENABLED and DEMO_WITHOUT_LLM modes.
"""
import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
RUNS_DIR = STORAGE_DIR / "runs"
PROJECTS_DIR = STORAGE_DIR / "projects"
DB_PATH = STORAGE_DIR / "analytics_system.sqlite"

# Load local .env safely without requiring python-dotenv
env_file = BASE_DIR / ".env"
if env_file.exists():
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass

# Ensure essential directories exist
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    app_name: str = "Adaptive Multi-Agent Analytics Platform"
    environment: str = "development"
    debug: bool = True
    port: int = 8000
    host: str = "127.0.0.1"

    # Storage
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    storage_dir: Path = STORAGE_DIR
    runs_dir: Path = RUNS_DIR
    projects_dir: Path = PROJECTS_DIR
    db_path: Path = DB_PATH

    # Security & Execution limits
    allowed_extensions: set = {".csv", ".xlsx"}
    max_upload_size_bytes: int = 100 * 1024 * 1024  # 100MB
    execution_timeout_seconds: int = 60
    max_repair_attempts: int = 2
    max_context_chars: int = 15000

    # Gemini LLM Settings
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip())
    gemini_timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("GEMINI_TIMEOUT_SECONDS", "60")))
    gemini_max_retries: int = Field(default_factory=lambda: int(os.getenv("GEMINI_MAX_RETRIES", "2")))

    @property
    def operating_mode(self) -> Literal["LLM_ENABLED", "DEMO_WITHOUT_LLM"]:
        """Returns LLM_ENABLED only if a non-empty Gemini API key is configured."""
        if self.gemini_api_key and len(self.gemini_api_key) > 5:
            return "LLM_ENABLED"
        return "DEMO_WITHOUT_LLM"

    @property
    def is_demo_mode(self) -> bool:
        """Returns True if running in DEMO_WITHOUT_LLM mode."""
        return self.operating_mode == "DEMO_WITHOUT_LLM"

    @property
    def is_demo_without_llm(self) -> bool:
        """Alias for is_demo_mode."""
        return self.operating_mode == "DEMO_WITHOUT_LLM"

    @property
    def mode_label(self) -> str:
        if self.operating_mode == "LLM_ENABLED":
            return f"Gemini habilitado ({self.gemini_model})"
        return "Demostración sin LLM (Motor analítico determinista)"


settings = Settings()
