"""
File management service: safe file persistence, SHA-256 calculation,
CSV sniffer, Excel sheet inspection with formula detection,
preview generation, and secure local storage per project.
"""
import csv
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
        clean_name = Path(filename).name
        target_path = pdir / clean_name

        with open(target_path, "wb") as f:
            f.write(content)

        file_hash = FileManager.calculate_sha256(target_path)
        return target_path, file_hash

    @staticmethod
    def detect_encoding(file_path: Path) -> str:
        """Detect encoding by attempting common encodings."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
        with open(file_path, "rb") as f:
            raw_data = f.read(16384)
        for enc in encodings:
            try:
                raw_data.decode(enc)
                return enc
            except UnicodeDecodeError:
                continue
        return "utf-8"

    @classmethod
    def sniff_csv(cls, file_path: Path) -> Dict[str, Any]:
        """Detect delimiter, encoding, and header presence for CSV files."""
        encoding = cls.detect_encoding(file_path)
        try:
            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                sample = f.read(8192)
                f.seek(0)
                first_lines = [f.readline() for _ in range(5) if f]

            delimiter = ","
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
                delimiter = dialect.delimiter
            except Exception:
                # Fallback: count occurrences in first line
                if first_lines:
                    line = first_lines[0]
                    counts = {d: line.count(d) for d in [",", ";", "\t", "|"]}
                    delimiter = max(counts, key=counts.get) if max(counts.values()) > 0 else ","

            has_header = True
            try:
                has_header = csv.Sniffer().has_header(sample)
            except Exception:
                pass

            return {
                "delimiter": delimiter,
                "encoding": encoding,
                "has_header": has_header,
                "supported_delimiters": [",", ";", "\t", "|"],
                "supported_encodings": ["utf-8", "latin-1", "cp1252", "utf-8-sig"]
            }
        except Exception as e:
            return {
                "delimiter": ",",
                "encoding": "utf-8",
                "has_header": True,
                "warning": f"Could not automatically detect CSV format: {e}"
            }

    @staticmethod
    def inspect_excel_sheets(file_path: Path) -> List[str]:
        """Return sheet names from an Excel file."""
        wb = load_workbook(file_path, read_only=True, keep_vba=False)
        sheets = wb.sheetnames
        wb.close()
        return sheets

    @classmethod
    def inspect_excel_full(cls, file_path: Path) -> Dict[str, Any]:
        """Inspect sheet names and detect unevaluated formulas."""
        wb_data = load_workbook(file_path, read_only=True, data_only=True)
        wb_formulas = load_workbook(file_path, read_only=True, data_only=False)

        sheets_meta = []
        has_unevaluated_formulas = False
        formula_warnings = []

        try:
            for sname in wb_data.sheetnames:
                ws_data = wb_data[sname]
                ws_form = wb_formulas[sname]

                formula_cells_count = 0
                none_formula_cells = 0

                # Sample up to 25 rows
                row_count = 0
                max_cols = 0
                for r_idx, (r_data, r_form) in enumerate(zip(ws_data.iter_rows(values_only=True), ws_form.iter_rows(values_only=True))):
                    if r_idx > 25:
                        break
                    row_count += 1
                    max_cols = max(max_cols, len(r_data))
                    for val_data, val_form in zip(r_data, r_form):
                        if isinstance(val_form, str) and val_form.startswith("="):
                            formula_cells_count += 1
                            if val_data is None:
                                none_formula_cells += 1

                if none_formula_cells > 0:
                    has_unevaluated_formulas = True
                    formula_warnings.append(
                        f"Sheet '{sname}' contains {none_formula_cells} formula cells without pre-calculated cached values."
                    )

                sheets_meta.append({
                    "sheet_name": sname,
                    "estimated_columns": max_cols,
                    "sample_rows_checked": row_count,
                    "has_formulas": formula_cells_count > 0,
                    "unevaluated_formula_cells": none_formula_cells
                })
        finally:
            wb_data.close()
            wb_formulas.close()

        return {
            "sheets": wb_data.sheetnames,
            "sheets_metadata": sheets_meta,
            "has_unevaluated_formulas": has_unevaluated_formulas,
            "formula_warnings": formula_warnings
        }

    @classmethod
    def preview_table(
        cls,
        file_path: Path,
        sheet_name: Optional[str] = None,
        delimiter: Optional[str] = None,
        encoding: Optional[str] = None,
        header_row: int = 0,
        max_rows: int = 8
    ) -> Dict[str, Any]:
        """
        Generate a rich preview before incorporating a table into the project.
        Returns columns, inferred types, preview rows, total row estimate, and quality warnings.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        ext = file_path.suffix.lower()
        warnings = []

        if ext == ".csv":
            sniff_res = cls.sniff_csv(file_path)
            sep = delimiter or sniff_res.get("delimiter", ",")
            enc = encoding or sniff_res.get("encoding", "utf-8")
            try:
                df = pd.read_csv(
                    file_path,
                    sep=sep,
                    encoding=enc,
                    header=header_row,
                    dtype=str,
                    keep_default_na=False
                )
            except Exception as e:
                # Retry with latin-1 fallback
                warnings.append(f"Initial read failed ({e}), falling back to latin-1 encoding.")
                df = pd.read_csv(
                    file_path,
                    sep=sep,
                    encoding="latin-1",
                    header=header_row,
                    dtype=str,
                    keep_default_na=False
                )
        elif ext == ".xlsx":
            excel_info = cls.inspect_excel_full(file_path)
            warnings.extend(excel_info.get("formula_warnings", []))
            target_sheet = sheet_name or excel_info["sheets"][0]
            df = pd.read_excel(
                file_path,
                sheet_name=target_sheet,
                header=header_row,
                dtype=str,
                keep_default_na=False
            )
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Column type detection & leading zeros
        columns_info = []
        for col in df.columns:
            series = df[col]
            non_empty = series[series != ""]
            has_leading_zeros = False
            inferred_type = "string"

            # Check leading zeros
            for val in non_empty.head(100):
                if isinstance(val, str) and len(val) > 1 and val.startswith("0") and val[1:].isdigit():
                    has_leading_zeros = True
                    break

            if has_leading_zeros:
                inferred_type = "identifier_with_leading_zeros"
            else:
                # Check numeric
                num_conv = pd.to_numeric(non_empty, errors="coerce")
                if len(non_empty) > 0 and num_conv.notna().sum() / len(non_empty) > 0.85:
                    inferred_type = "numeric"
                else:
                    # Check date
                    dt_conv = pd.to_datetime(non_empty, errors="coerce")
                    if len(non_empty) > 0 and dt_conv.notna().sum() / len(non_empty) > 0.85:
                        inferred_type = "datetime"

            columns_info.append({
                "column_name": str(col),
                "inferred_type": inferred_type,
                "has_leading_zeros": has_leading_zeros,
                "null_count": int((series == "").sum()),
                "sample_values": non_empty.head(3).tolist()
            })

        # Sample rows for UI preview table
        preview_rows = df.head(max_rows).to_dict(orient="records")

        return {
            "filename": file_path.name,
            "sheet_name": sheet_name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": columns_info,
            "preview_rows": preview_rows,
            "warnings": warnings
        }

    @classmethod
    def read_table_dataframe(
        cls,
        file_path: Path,
        sheet_name: Optional[str] = None,
        delimiter: Optional[str] = None,
        encoding: Optional[str] = None,
        header_row: int = 0
    ) -> pd.DataFrame:
        """
        Safely load a table as pandas DataFrame.
        Preserves string values (e.g. leading zeros) by reading object columns accurately.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        ext = file_path.suffix.lower()
        if ext == ".csv":
            sep = delimiter
            enc = encoding
            if not sep or not enc:
                sniffed = cls.sniff_csv(file_path)
                sep = sep or sniffed.get("delimiter", ",")
                enc = enc or sniffed.get("encoding", "utf-8")
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=enc, header=header_row, dtype=str, keep_default_na=False)
            except Exception:
                df = pd.read_csv(file_path, sep=sep, encoding="latin-1", header=header_row, dtype=str, keep_default_na=False)
        elif ext == ".xlsx":
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row, dtype=str, keep_default_na=False)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        return df
