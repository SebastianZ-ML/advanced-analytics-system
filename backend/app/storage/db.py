"""
SQLite persistence layer for Projects, Runs, File Records, Artifacts, and Event Logs.
Thread-safe and lightweight using native sqlite3.
"""
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.config import settings


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS projects (
        project_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS project_files (
        file_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        sheet_name TEXT,
        table_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        row_count INTEGER,
        column_count INTEGER,
        uploaded_at TEXT NOT NULL,
        FOREIGN KEY (project_id) REFERENCES projects(project_id)
    );

    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        status TEXT NOT NULL,
        current_stage TEXT NOT NULL,
        repair_attempts INTEGER DEFAULT 0,
        failure_reason TEXT,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        FOREIGN KEY (project_id) REFERENCES projects(project_id)
    );

    CREATE TABLE IF NOT EXISTS run_artifacts (
        artifact_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        artifact_type TEXT NOT NULL,
        data_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    );

    CREATE TABLE IF NOT EXISTS run_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        stage TEXT NOT NULL,
        level TEXT NOT NULL,
        message TEXT NOT NULL,
        metadata_json TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    );

    CREATE TABLE IF NOT EXISTS llm_interactions (
        interaction_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        run_id TEXT,
        caller_agent TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        created_at TEXT NOT NULL,
        duration_ms INTEGER,
        status TEXT NOT NULL,
        schema_version TEXT,
        input_tokens INTEGER DEFAULT 0,
        output_tokens INTEGER DEFAULT 0,
        error_sanitized TEXT,
        context_artifacts TEXT,
        produced_artifacts TEXT,
        FOREIGN KEY (project_id) REFERENCES projects(project_id)
    );
    """)

    conn.commit()
    conn.close()


class DatabaseService:
    @staticmethod
    def create_project(project_id: str, name: str, description: str = "") -> Dict[str, Any]:
        conn = get_db_connection()
        now = datetime.now(timezone.utc).isoformat()
        with conn:
            conn.execute(
                "INSERT INTO projects (project_id, name, description, created_at) VALUES (?, ?, ?, ?)",
                (project_id, name, description, now)
            )
        conn.close()
        return {"project_id": project_id, "name": name, "description": description, "created_at": now}

    @staticmethod
    def get_project(project_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def list_projects() -> List[Dict[str, Any]]:
        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def add_project_file(
        file_id: str,
        project_id: str,
        filename: str,
        table_name: str,
        file_path: str,
        file_hash: str,
        sheet_name: Optional[str] = None,
        row_count: int = 0,
        column_count: int = 0
    ) -> None:
        conn = get_db_connection()
        now = datetime.now(timezone.utc).isoformat()
        with conn:
            conn.execute(
                """
                INSERT INTO project_files 
                (file_id, project_id, filename, sheet_name, table_name, file_path, file_hash, row_count, column_count, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (file_id, project_id, filename, sheet_name, table_name, file_path, file_hash, row_count, column_count, now)
            )
        conn.close()

    @staticmethod
    def get_project_files(project_id: str) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM project_files WHERE project_id = ?", (project_id,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def create_run(run_id: str, project_id: str, initial_stage: str = "UPLOADED") -> Dict[str, Any]:
        conn = get_db_connection()
        now = datetime.now(timezone.utc).isoformat()
        with conn:
            conn.execute(
                "INSERT INTO runs (run_id, project_id, status, current_stage, repair_attempts, started_at) VALUES (?, ?, ?, ?, 0, ?)",
                (run_id, project_id, "RUNNING", initial_stage, now)
            )
        conn.close()
        return {
            "run_id": run_id,
            "project_id": project_id,
            "status": "RUNNING",
            "current_stage": initial_stage,
            "repair_attempts": 0,
            "started_at": now
        }

    @staticmethod
    def update_run_stage(run_id: str, stage: str, status: str = "RUNNING", failure_reason: Optional[str] = None) -> None:
        conn = get_db_connection()
        now = datetime.now(timezone.utc).isoformat()
        completed_at = now if status in ["READY", "FAILED", "CANCELLED"] else None
        with conn:
            conn.execute(
                """
                UPDATE runs 
                SET current_stage = ?, status = ?, failure_reason = ?, completed_at = COALESCE(?, completed_at)
                WHERE run_id = ?
                """,
                (stage, status, failure_reason, completed_at, run_id)
            )
        conn.close()

    @staticmethod
    def increment_repair_attempt(run_id: str) -> int:
        conn = get_db_connection()
        with conn:
            conn.execute("UPDATE runs SET repair_attempts = repair_attempts + 1 WHERE run_id = ?", (run_id,))
            row = conn.execute("SELECT repair_attempts FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        conn.close()
        return row["repair_attempts"] if row else 1

    @staticmethod
    def get_run(run_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def save_artifact(artifact_id: str, run_id: str, artifact_type: str, data: Dict[str, Any]) -> None:
        conn = get_db_connection()
        now = datetime.now(timezone.utc).isoformat()
        data_json = json.dumps(data, default=str)
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO run_artifacts (artifact_id, run_id, artifact_type, data_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (artifact_id, run_id, artifact_type, data_json, now)
            )
        conn.close()

    @staticmethod
    def get_artifact(run_id: str, artifact_type: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        row = conn.execute(
            "SELECT data_json FROM run_artifacts WHERE run_id = ? AND artifact_type = ? ORDER BY created_at DESC LIMIT 1",
            (run_id, artifact_type)
        ).fetchone()
        conn.close()
        if row:
            return json.loads(row["data_json"])
        return None

    @staticmethod
    def get_all_run_artifacts(run_id: str) -> Dict[str, Any]:
        conn = get_db_connection()
        rows = conn.execute(
            "SELECT artifact_type, data_json FROM run_artifacts WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,)
        ).fetchall()
        conn.close()
        result = {}
        for r in rows:
            result[r["artifact_type"]] = json.loads(r["data_json"])
        return result

    @staticmethod
    def log_event(run_id: str, stage: str, level: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        conn = get_db_connection()
        now = datetime.now(timezone.utc).isoformat()
        meta_str = json.dumps(metadata or {}, default=str)
        with conn:
            conn.execute(
                "INSERT INTO run_events (run_id, stage, level, message, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, stage, level, message, meta_str, now)
            )
        conn.close()

    @staticmethod
    def get_run_events(run_id: str) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        rows = conn.execute(
            "SELECT * FROM run_events WHERE run_id = ? ORDER BY event_id ASC",
            (run_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def log_llm_interaction(
        interaction_id: str,
        project_id: str,
        caller_agent: str,
        provider: str,
        model: str,
        status: str,
        duration_ms: int = 0,
        run_id: Optional[str] = None,
        schema_version: Optional[str] = "1.0",
        input_tokens: int = 0,
        output_tokens: int = 0,
        error_sanitized: Optional[str] = None,
        context_artifacts: Optional[List[str]] = None,
        produced_artifacts: Optional[List[str]] = None
    ) -> None:
        conn = get_db_connection()
        now = datetime.now(timezone.utc).isoformat()
        ctx_str = json.dumps(context_artifacts or [])
        prod_str = json.dumps(produced_artifacts or [])
        with conn:
            conn.execute(
                """
                INSERT INTO llm_interactions
                (interaction_id, project_id, run_id, caller_agent, provider, model, created_at, duration_ms, status, schema_version, input_tokens, output_tokens, error_sanitized, context_artifacts, produced_artifacts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (interaction_id, project_id, run_id, caller_agent, provider, model, now, duration_ms, status, schema_version, input_tokens, output_tokens, error_sanitized, ctx_str, prod_str)
            )
        conn.close()

    @staticmethod
    def get_llm_interactions(project_id: Optional[str] = None, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        if run_id:
            rows = conn.execute(
                "SELECT * FROM llm_interactions WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,)
            ).fetchall()
        elif project_id:
            rows = conn.execute(
                "SELECT * FROM llm_interactions WHERE project_id = ? ORDER BY created_at ASC",
                (project_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM llm_interactions ORDER BY created_at DESC LIMIT 50").fetchall()
        conn.close()
        return [dict(r) for r in rows]

