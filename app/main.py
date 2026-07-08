"""FastAPI entry point."""

from __future__ import annotations

from fastapi import FastAPI

from app.routes.documents import router as documents_router
from app.schemas import HealthResponse


app = FastAPI(
    title="PDF Document AI MVP",
    description="Local MVP for PDF type detection, OCR, document classification, field extraction, and visual mark detection.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health status."""

    return HealthResponse(status="ok")


app.include_router(documents_router)

