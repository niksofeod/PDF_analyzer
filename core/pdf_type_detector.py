"""PDF type detection based on text-layer coverage."""

from __future__ import annotations

from pathlib import Path


def detect_pdf_type(pdf_path: str | Path, min_text_chars: int = 20) -> dict:
    """Detect whether a PDF is text-based, scanned, mixed, or unknown."""

    path = Path(pdf_path)
    warnings: list[str] = []
    if not path.exists():
        return {
            "pdf_type": "unknown",
            "page_count": 0,
            "text_pages": 0,
            "scanned_pages": 0,
            "warnings": [f"PDF file was not found: {path}"],
        }

    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        return {
            "pdf_type": "unknown",
            "page_count": 0,
            "text_pages": 0,
            "scanned_pages": 0,
            "warnings": [f"PyMuPDF is unavailable, PDF type cannot be detected: {exc}"],
        }

    try:
        with fitz.open(path) as document:
            page_count = len(document)
            if page_count == 0:
                return {
                    "pdf_type": "unknown",
                    "page_count": 0,
                    "text_pages": 0,
                    "scanned_pages": 0,
                    "warnings": ["PDF has no pages"],
                }

            text_pages = 0
            for page in document:
                text = page.get_text("text").strip()
                if len(text) >= min_text_chars:
                    text_pages += 1

            scanned_pages = page_count - text_pages
            if text_pages == page_count:
                pdf_type = "text"
            elif text_pages == 0:
                pdf_type = "scanned"
            else:
                pdf_type = "mixed"

            return {
                "pdf_type": pdf_type,
                "page_count": page_count,
                "text_pages": text_pages,
                "scanned_pages": scanned_pages,
                "warnings": warnings,
            }
    except Exception as exc:
        return {
            "pdf_type": "unknown",
            "page_count": 0,
            "text_pages": 0,
            "scanned_pages": 0,
            "warnings": [f"PDF type detection failed: {exc}"],
        }

