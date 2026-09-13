"""
FastAPI router for downloadable sample datasets and independent verification cards.
"""
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.engine.sample_datasets import generate_all_samples
from app.storage.db import DatabaseService
from app.storage.files import FileManager

router = APIRouter(prefix="/api/samples", tags=["samples"])


def get_samples_catalog() -> Dict[str, Any]:
    return generate_all_samples(settings.base_dir)


@router.get("")
def list_sample_datasets() -> Dict[str, Any]:
    """Returns the metadata catalog of the 3 small verifiable sample datasets."""
    return get_samples_catalog()


@router.get("/{dataset_id}/download/{filename}")
def download_sample_file(dataset_id: str, filename: str):
    """Downloads a raw CSV or XLSX sample file for user testing in the browser."""
    samples_dir = settings.base_dir / "data" / "samples" / dataset_id
    fpath = samples_dir / filename
    if not fpath.exists():
        # Ensure generated
        get_samples_catalog()

    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Sample file not found")

    media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if fpath.suffix == ".xlsx" else "text/csv"
    return FileResponse(
        path=fpath,
        filename=filename,
        media_type=media_type
    )


@router.post("/{dataset_id}/load_to_project/{project_id}")
def load_sample_to_project(dataset_id: str, project_id: str) -> Dict[str, Any]:
    """
    Directly incorporates the sample dataset files into the project,
    preserving exact standard auditing, SHA-256 calculation, and row counting.
    """
    proj = DatabaseService.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    catalog = get_samples_catalog()
    if dataset_id not in catalog:
        raise HTTPException(status_code=404, detail="Sample dataset not found")

    samples_dir = settings.base_dir / "data" / "samples" / dataset_id
    loaded_tables = []

    for file_path in samples_dir.glob("*.*"):
        if file_path.suffix.lower() == ".xlsx":
            sheets = FileManager.inspect_excel_sheets(file_path)
            for sname in sheets:
                tbl_name = f"{file_path.stem.lower()}_{sname.lower()}"
                fhash = FileManager.calculate_sha256(file_path)
                df = FileManager.read_table_dataframe(file_path, sheet_name=sname)
                fid = f"file_{dataset_id}_{tbl_name[:8]}"
                DatabaseService.delete_project_file_by_table(project_id, tbl_name)
                DatabaseService.add_project_file(
                    file_id=fid,
                    project_id=project_id,
                    filename=file_path.name,
                    sheet_name=sname,
                    table_name=tbl_name,
                    file_path=str(file_path),
                    file_hash=fhash,
                    row_count=len(df),
                    column_count=len(df.columns)
                )
                loaded_tables.append(tbl_name)
        elif file_path.suffix.lower() == ".csv":
            tbl_name = file_path.stem.lower().replace(" ", "_")
            fhash = FileManager.calculate_sha256(file_path)
            df = FileManager.read_table_dataframe(file_path)
            fid = f"file_{dataset_id}_{tbl_name[:8]}"
            DatabaseService.delete_project_file_by_table(project_id, tbl_name)
            DatabaseService.add_project_file(
                file_id=fid,
                project_id=project_id,
                filename=file_path.name,
                sheet_name=None,
                table_name=tbl_name,
                file_path=str(file_path),
                file_hash=fhash,
                row_count=len(df),
                column_count=len(df.columns)
            )
            loaded_tables.append(tbl_name)

    return {
        "status": "sample_loaded",
        "dataset_id": dataset_id,
        "project_id": project_id,
        "tables": loaded_tables,
        "sample_metadata": catalog[dataset_id]
    }
