"""
Main entry point for the Adaptive Multi-Agent Analytics Platform FastAPI backend.
"""
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.storage.db import init_db
from app.api.projects import router as projects_router
from app.api.pipeline import router as pipeline_router
from app.api.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure SQLite tables exist
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Sistema multiagente de analítica avanzada y adaptativa",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(projects_router)
app.include_router(pipeline_router)
app.include_router(chat_router)


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "environment": settings.environment,
        "operating_mode": settings.operating_mode,
        "is_demo_mode": settings.is_demo_mode,
        "gemini_model": settings.gemini_model if not settings.is_demo_mode else None,
        "mode_label": settings.mode_label,
    }


# Frontend static files mounting if present
frontend_dist = settings.base_dir / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = frontend_dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")
else:
    # Also support direct frontend index.html if standalone
    frontend_raw = settings.base_dir / "frontend"
    if (frontend_raw / "index.html").exists():
        @app.get("/")
        async def serve_index():
            return FileResponse(frontend_raw / "index.html")
