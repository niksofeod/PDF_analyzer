"""Utilities for validating and storing PDF files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict


OUTPUT_SUBDIRS = ("json", "text", "tables", "visualizations")


def sanitize_filename(file_name: str) -> str:
    """Return a filesystem-safe file name while preserving the extension."""

    name = Path(file_name).name
    cleaned = re.sub(r"[^A-Za-zА-Яа-я0-9_. -]+", "_", name).strip(" .")
    return cleaned or "document.pdf"


def validate_pdf_path(pdf_path: str | Path) -> tuple[Path, list[str]]:
    """Validate a PDF path and return warnings instead of raising user-facing errors."""

    path = Path(pdf_path)
    warnings: list[str] = []
    if path.suffix.lower() != ".pdf":
        warnings.append("Input file does not have a .pdf extension")
    if not path.exists():
        warnings.append(f"PDF file was not found: {path}")
    elif not path.is_file():
        warnings.append(f"Input path is not a file: {path}")
    return path, warnings


def ensure_project_directories(project_root: Path) -> Dict[str, Path]:
    """Create data and output directories required by the project."""

    paths = {
        "raw": project_root / "data" / "raw",
        "processed": project_root / "data" / "processed",
        "outputs": project_root / "outputs",
    }
    for subdir in OUTPUT_SUBDIRS:
        paths[subdir] = project_root / "outputs" / subdir
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


async def save_upload_file(upload_file, raw_dir: Path) -> Path:
    """Persist a FastAPI UploadFile into data/raw and return the saved path."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / sanitize_filename(upload_file.filename or "document.pdf")
    suffix = destination.suffix
    stem = destination.stem
    counter = 1
    while destination.exists():
        destination = raw_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    with destination.open("wb") as handle:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return destination

