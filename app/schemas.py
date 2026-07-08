"""Pydantic schemas used by the REST API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Simple health-check response."""

    status: str = "ok"


class FieldValue(BaseModel):
    """Extracted field value with a heuristic confidence score."""

    value: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DocumentType(BaseModel):
    """Classified business document type."""

    label: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VisualMark(BaseModel):
    """Detected visual object such as a signature or stamp."""

    label: Optional[str] = None
    page: int
    bbox: List[float]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    method: str
    method_label: Optional[str] = None
    near_text: Optional[str] = None


class VisualMarks(BaseModel):
    """Grouped signatures and stamps."""

    signatures: List[VisualMark] = Field(default_factory=list)
    stamps: List[VisualMark] = Field(default_factory=list)


class DocumentResult(BaseModel):
    """Full processing result returned by the pipeline and API."""

    document_id: str
    file_name: str
    pdf_type: str
    document_type: DocumentType
    fields: Dict[str, FieldValue]
    tables: List[dict] = Field(default_factory=list)
    visual_marks: VisualMarks
    text_path: str
    json_path: str
    visualization_paths: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class VisualizationResponse(BaseModel):
    """List of saved visualization files for one document."""

    document_id: str
    visualization_paths: List[str] = Field(default_factory=list)
