"""
FastAPI router for Project management, file uploads, and demo dataset loading.
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
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")
    files = DatabaseService.get_project_files(project_id)
    return {**proj, "files": files}


@router.post("/{project_id}/upload")
async def upload_file(
    project_id: str,
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None)
) -> Dict[str, Any]:
    proj = DatabaseService.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")

    content = await file.read()
    try:
        fpath, fhash = FileManager.save_uploaded_file(project_id, file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    sheets = []
    if fpath.suffix.lower() == ".xlsx":
        sheets = FileManager.inspect_excel_sheets(fpath)

    tbl_name = table_name or Path(file.filename).stem.lower().replace(" ", "_")
    target_sheet = sheet_name or (sheets[0] if sheets else None)

    # Read dataframe to count rows/cols
    df = FileManager.read_table_dataframe(fpath, sheet_name=target_sheet)
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
        "column_count": len(df.columns)
    }


@router.post("/{project_id}/load_demo")
def load_demo_dataset(project_id: str) -> Dict[str, Any]:
    """Populates the project with the generated synthetic demonstration dataset."""
    proj = DatabaseService.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")

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
        fhash = FileManager.calculate_sha256(fpath)
        df = FileManager.read_table_dataframe(fpath, sheet_name=sheet)
        file_id = f"file_{uuid.uuid4().hex[:8]}"
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
        loaded.append({"table_name": tbl, "rows": len(df), "columns": len(df.columns), "source": fname})

    return {"status": "success", "tables_loaded": loaded}
