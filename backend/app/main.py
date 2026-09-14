"""
Main entry point for the Adaptive Multi-Agent Analytics Platform FastAPI backend.
"""
import logging
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.storage.db import init_db
from app.api.projects import router as projects_router
from app.api.pipeline import router as pipeline_router
from app.api.chat import router as chat_router
from app.api.samples import router as samples_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure SQLite tables exist
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Adaptive Multi-Agent Advanced Analytics Platform",
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

# Global Error Normalization with Structured JSON and Trace ID
logger = logging.getLogger("app.main")


@app.exception_handler(HTTPException)
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    trace_id = f"err_{uuid.uuid4().hex[:12]}"
    status_code = exc.status_code
    error_code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
        429: "TOO_MANY_REQUESTS",
        500: "INTERNAL_SERVER_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
        504: "GATEWAY_TIMEOUT"
    }
    code_str = error_code_map.get(status_code, "HTTP_ERROR")
    detail = str(exc.detail) if exc.detail else "Error en la petición HTTP."

    # Redact potential sensitive tokens
    if settings.gemini_api_key and settings.gemini_api_key in detail:
        detail = detail.replace(settings.gemini_api_key, "[REDACTED_API_KEY]")

    return JSONResponse(
        status_code=status_code,
        content={
            "error_code": code_str,
            "message": detail,
            "trace_id": trace_id,
            "detail": detail
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    trace_id = f"err_{uuid.uuid4().hex[:12]}"
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "VALIDATION_ERROR",
            "message": "Solicitud con formato o parámetros inválidos.",
            "trace_id": trace_id,
            "detail": [
                {
                    "loc": [str(loc) for loc in err.get("loc", [])],
                    "msg": err.get("msg", ""),
                    "type": err.get("type", "")
                }
                for err in exc.errors()
            ]
        }
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    trace_id = f"err_{uuid.uuid4().hex[:12]}"
    logger.exception(f"Unhandled exception on {request.method} {request.url.path} [trace_id={trace_id}]: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "Ocurrió un error interno en el servidor al procesar la solicitud.",
            "trace_id": trace_id,
            "detail": "Error interno del servidor. Consulta los registros con el identificador de seguimiento para más información."
        }
    )


# Register API routers
app.include_router(projects_router)
app.include_router(pipeline_router)
app.include_router(chat_router)
app.include_router(samples_router)


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
