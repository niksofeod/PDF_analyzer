"""REST endpoints for document upload and result retrieval."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import DocumentResult, VisualizationResponse
from core.pdf_loader import save_upload_file
from core.pipeline import PROJECT_ROOT, process_document


router = APIRouter(prefix="/documents", tags=["documents"])


def _json_path(document_id: str) -> Path:
    return PROJECT_ROOT / "outputs" / "json" / f"{document_id}.json"


def _text_path(document_id: str) -> Path:
    return PROJECT_ROOT / "outputs" / "text" / f"{document_id}.txt"


def _load_result(document_id: str) -> dict:
    path = _json_path(document_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Document result '{document_id}' was not found")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/upload", response_model=DocumentResult)
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Save an uploaded PDF, run the pipeline, and return the JSON result."""

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        stored_path = await save_upload_file(file, PROJECT_ROOT / "data" / "raw")
        return process_document(stored_path)
    except Exception as exc:  # pragma: no cover - FastAPI safety boundary
        raise HTTPException(status_code=500, detail=f"Document processing failed: {exc}") from exc


@router.get("/{document_id}", response_model=DocumentResult)
def get_document(document_id: str) -> dict:
    """Return a saved processing result."""

    return _load_result(document_id)


@router.get("/{document_id}/text")
def get_document_text(document_id: str) -> dict:
    """Return extracted text for a processed document."""

    path = _text_path(document_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Text for document '{document_id}' was not found")
    return {"document_id": document_id, "text": path.read_text(encoding="utf-8")}


@router.get("/{document_id}/visualization", response_model=VisualizationResponse)
def get_document_visualization(document_id: str) -> dict:
    """Return saved visualization paths for a processed document."""

    result = _load_result(document_id)
    return {
        "document_id": document_id,
        "visualization_paths": result.get("visualization_paths", []),
    }

