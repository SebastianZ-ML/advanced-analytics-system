"""
File management service: safe file persistence, SHA-256 calculation,
Excel sheet inspection and secure local storage per project.
"""
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from openpyxl import load_workbook
from app.config import settings


class FileSecurityError(Exception):
    pass


class FileManager:
    @staticmethod
    def calculate_sha256(file_path: Path) -> str:
        """Calculate SHA-256 hash of a file for immutability and provenance tracking."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def get_project_dir(project_id: str) -> Path:
        """Get or create sanitized project directory."""
        if ".." in project_id or "/" in project_id or "\\" in project_id:
            raise FileSecurityError("Invalid project ID syntax")
        pdir = settings.projects_dir / project_id
        pdir.mkdir(parents=True, exist_ok=True)
        return pdir

    @staticmethod
    def save_uploaded_file(project_id: str, filename: str, content: bytes) -> Tuple[Path, str]:
        """Save file securely, check extension, calculate hash."""
        ext = Path(filename).suffix.lower()
        if ext not in settings.allowed_extensions:
            raise FileSecurityError(f"Extension {ext} is not allowed. Allowed: {settings.allowed_extensions}")

        if len(content) > settings.max_upload_size_bytes:
            raise FileSecurityError(f"File exceeds maximum allowed size of {settings.max_upload_size_bytes} bytes")

        pdir = FileManager.get_project_dir(project_id)
        # Clean filename to avoid path traversal
        clean_name = Path(filename).name
        target_path = pdir / clean_name

        with open(target_path, "wb") as f:
            f.write(content)

        file_hash = FileManager.calculate_sha256(target_path)
        return target_path, file_hash

    @staticmethod
    def inspect_excel_sheets(file_path: Path) -> List[str]:
        """Return sheet names from an Excel file without executing macros."""
        wb = load_workbook(file_path, read_only=True, keep_vba=False)
        sheets = wb.sheetnames
        wb.close()
        return sheets

    @staticmethod
    def read_table_dataframe(file_path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Safely load a table as pandas DataFrame.
        Preserves string values (e.g. leading zeros) by reading object columns accurately.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        ext = file_path.suffix.lower()
        if ext == ".csv":
            # First pass: read string representation to detect leading zeros
            df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
        elif ext == ".xlsx":
            df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str, keep_default_na=False)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        return df
