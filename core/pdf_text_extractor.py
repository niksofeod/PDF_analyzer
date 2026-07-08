"""Text extraction from PDFs that contain a text layer."""

from __future__ import annotations

from pathlib import Path


def extract_text_from_pdf(pdf_path: str | Path) -> dict:
    """Extract full text and positioned text blocks with PyMuPDF."""

    path = Path(pdf_path)
    if not path.exists():
        return {"full_text": "", "blocks": [], "warnings": [f"PDF file was not found: {path}"]}

    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        return {"full_text": "", "blocks": [], "warnings": [f"PyMuPDF is unavailable, text extraction skipped: {exc}"]}

    pages: list[str] = []
    blocks: list[dict] = []
    warnings: list[str] = []
    try:
        with fitz.open(path) as document:
            for page_index, page in enumerate(document, start=1):
                page_text = page.get_text("text") or ""
                pages.append(page_text)
                for block in page.get_text("blocks"):
                    if len(block) < 5:
                        continue
                    x1, y1, x2, y2, text = block[:5]
                    clean_text = str(text).strip()
                    if clean_text:
                        blocks.append(
                            {
                                "page": page_index,
                                "text": clean_text,
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            }
                        )
    except Exception as exc:
        warnings.append(f"Text extraction failed: {exc}")

    return {"full_text": "\n".join(pages).strip(), "blocks": blocks, "warnings": warnings}

