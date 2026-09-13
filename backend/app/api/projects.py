"""
FastAPI router for Project management, file uploads, previews,
sheet inspections, delimiter/encoding detection, and demo dataset loading.
"""
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.storage.db import DatabaseService
from app.storage.files import FileManager

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""


@router.get("")
def list_projects() -> List[Dict[str, Any]]:
    return DatabaseService.list_projects()


@router.post("")
def create_project(req: ProjectCreateRequest) -> Dict[str, Any]:
    project_id = f"proj_{uuid.uuid4().hex[:8]}"
    return DatabaseService.create_project(
        project_id=project_id,
        name=req.name,
        description=req.description or ""
    )


@router.get("/{project_id}")
def get_project(project_id: str) -> Dict[str, Any]:
    proj = DatabaseService.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    files = DatabaseService.get_project_files(project_id)
    return {**proj, "files": files}


@router.post("/{project_id}/preview_upload")
async def preview_upload_file(
    project_id: str,
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None),
    delimiter: Optional[str] = Form(None),
    encoding: Optional[str] = Form(None),
    header_row: int = Form(0)
) -> Dict[str, Any]:
    """
    Saves the file to a staging path and returns an interactive preview:
    columns, inferred types, sample rows, sheet names, sniffed delimiter/encoding,
    and quality/formula warnings before the user confirms incorporation.
    """
    proj = DatabaseService.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    content = await file.read()
    try:
        fpath, fhash = FileManager.save_uploaded_file(project_id, file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ext = fpath.suffix.lower()
    sniff_info = {}
    excel_info = {}

    if ext == ".csv":
        sniff_info = FileManager.sniff_csv(fpath)
    elif ext == ".xlsx":
        excel_info = FileManager.inspect_excel_full(fpath)

    # Generate preview
    try:
        preview = FileManager.preview_table(
            file_path=fpath,
            sheet_name=sheet_name or (excel_info.get("sheets", [None])[0] if excel_info else None),
            delimiter=delimiter or sniff_info.get("delimiter", ","),
            encoding=encoding or sniff_info.get("encoding", "utf-8"),
            header_row=header_row,
            max_rows=8
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file preview: {e}")

    # Check if table_name would collide with existing tables
    default_table_name = Path(file.filename).stem.lower().replace(" ", "_").replace("-", "_")
    if sheet_name:
        default_table_name = f"{default_table_name}_{sheet_name.lower().replace(' ', '_')}"

    existing_files = DatabaseService.get_project_files(project_id)
    colliding_file = next((f for f in existing_files if f["table_name"] == default_table_name), None)

    return {
        "filename": file.filename,
        "staging_path": str(fpath),
        "file_hash": fhash,
        "default_table_name": default_table_name,
        "has_collision": colliding_file is not None,
        "colliding_table_name": colliding_file["table_name"] if colliding_file else None,
        "sniff_info": sniff_info,
        "excel_info": excel_info,
        "preview": preview
    }


@router.post("/{project_id}/upload")
async def upload_file(
    project_id: str,
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    delimiter: Optional[str] = Form(None),
    encoding: Optional[str] = Form(None),
    header_row: int = Form(0),
    replace_existing: bool = Form(False)
) -> Dict[str, Any]:
    """
    Confirms incorporation of a CSV or Excel table into the active project.
    Prevents unintentional overwrites and handles delimiter, encoding, sheet selection,
    and leading zero preservation.
    """
    proj = DatabaseService.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    content = await file.read()
    try:
        fpath, fhash = FileManager.save_uploaded_file(project_id, file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    sheets = []
    if fpath.suffix.lower() == ".xlsx":
        sheets = FileManager.inspect_excel_sheets(fpath)

    tbl_name = table_name or Path(file.filename).stem.lower().replace(" ", "_").replace("-", "_")
    target_sheet = sheet_name or (sheets[0] if sheets else None)
    if sheets and len(sheets) > 1 and not table_name and target_sheet:
        tbl_name = f"{tbl_name}_{target_sheet.lower().replace(' ', '_')}"

    # Check collision
    existing_files = DatabaseService.get_project_files(project_id)
    collision = next((f for f in existing_files if f["table_name"] == tbl_name), None)
    if collision:
        if not replace_existing:
            raise HTTPException(
                status_code=409,
                detail=f"Table '{tbl_name}' already exists in this project. Choose to replace it or provide a different table name."
            )
        # Delete old record
        DatabaseService.delete_project_file_by_table(project_id, tbl_name)

    # Read dataframe safely preserving string types
    try:
        df = FileManager.read_table_dataframe(
            file_path=fpath,
            sheet_name=target_sheet,
            delimiter=delimiter,
            encoding=encoding,
            header_row=header_row
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse table: {e}")

    file_id = f"file_{uuid.uuid4().hex[:8]}"

    DatabaseService.add_project_file(
        file_id=file_id,
        project_id=project_id,
        filename=file.filename,
        sheet_name=target_sheet,
        table_name=tbl_name,
        file_path=str(fpath),
        file_hash=fhash,
        row_count=len(df),
        column_count=len(df.columns)
    )

    return {
        "file_id": file_id,
        "filename": file.filename,
        "table_name": tbl_name,
        "sheet_name": target_sheet,
        "available_sheets": sheets,
        "file_hash": fhash,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns)
    }


@router.delete("/{project_id}/files/{file_id}")
def delete_project_file(project_id: str, file_id: str) -> Dict[str, Any]:
    proj = DatabaseService.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    deleted = DatabaseService.remove_project_file(file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")

    return {"status": "deleted", "file_id": file_id}


@router.post("/{project_id}/load_demo")
def load_demo_dataset(project_id: str) -> Dict[str, Any]:
    """Populates the project with the generated synthetic demonstration dataset."""
    proj = DatabaseService.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    demo_dir = settings.data_dir / "demo"
    if not (demo_dir / "orders.csv").exists():
        from app.engine.synthetic_data import generate_demo_dataset
        generate_demo_dataset(demo_dir)

    files_to_load = [
        ("orders.csv", "orders", None),
        ("customers.csv", "customers", None),
        ("products.xlsx", "products", "Products"),
        ("products.xlsx", "categories", "Categories"),
        ("campaigns.csv", "campaigns", None)
    ]

    loaded = []
    for fname, tbl, sheet in files_to_load:
        fpath = demo_dir / fname
        if fpath.exists():
            fhash = FileManager.calculate_sha256(fpath)
            df = FileManager.read_table_dataframe(fpath, sheet_name=sheet)
            file_id = f"demo_{tbl}_{uuid.uuid4().hex[:6]}"
            # Clear old if exists
            DatabaseService.delete_project_file_by_table(project_id, tbl)
            DatabaseService.add_project_file(
                file_id=file_id,
                project_id=project_id,
                filename=fname,
                sheet_name=sheet,
                table_name=tbl,
                file_path=str(fpath),
                file_hash=fhash,
                row_count=len(df),
                column_count=len(df.columns)
            )
            loaded.append(tbl)

    return {"status": "demo_loaded", "project_id": project_id, "tables_loaded": loaded}
