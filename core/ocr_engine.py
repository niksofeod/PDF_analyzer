"""OCR extraction through Tesseract/pytesseract."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Iterable

from core.image_preprocessing import preprocess_for_ocr


def tesseract_status() -> tuple[bool, str | None]:
    """Check whether pytesseract and the Tesseract binary are available."""

    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True, None
    except Exception as exc:
        return False, f"Tesseract OCR is unavailable: {exc}"


def render_page_to_image(pdf_path: str | Path, page_index: int, dpi: int = 200):
    """Render one PDF page to a PIL image."""

    import fitz
    from PIL import Image

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as document:
        page = document[page_index]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)


def ocr_pdf(pdf_path: str | Path, pages: Iterable[int] | None = None, lang: str = "rus+eng") -> dict:
    """Run OCR for selected zero-based pages of a PDF."""

    path = Path(pdf_path)
    warnings: list[str] = []
    if not path.exists():
        return {"full_text": "", "blocks": [], "confidence": 0.0, "warnings": [f"PDF file was not found: {path}"]}

    available, warning = tesseract_status()
    if not available:
        return {"full_text": "", "blocks": [], "confidence": 0.0, "warnings": [warning or "Tesseract OCR is unavailable"]}

    try:
        import fitz
        import pytesseract
        from pytesseract import Output
    except Exception as exc:
        return {"full_text": "", "blocks": [], "confidence": 0.0, "warnings": [f"OCR dependencies are unavailable: {exc}"]}

    text_parts: list[str] = []
    blocks: list[dict] = []
    confidences: list[float] = []

    try:
        with fitz.open(path) as document:
            page_indexes = list(pages) if pages is not None else list(range(len(document)))

        for page_index in page_indexes:
            try:
                image = render_page_to_image(path, page_index)
                prepared = preprocess_for_ocr(image)
                page_text = pytesseract.image_to_string(prepared, lang=lang) or ""
                text_parts.append(page_text)

                data = pytesseract.image_to_data(prepared, lang=lang, output_type=Output.DICT)
                for i, word in enumerate(data.get("text", [])):
                    clean_word = str(word).strip()
                    if not clean_word:
                        continue
                    conf = float(data.get("conf", ["-1"])[i])
                    if conf >= 0:
                        confidences.append(conf / 100)
                    x = float(data.get("left", [0])[i])
                    y = float(data.get("top", [0])[i])
                    w = float(data.get("width", [0])[i])
                    h = float(data.get("height", [0])[i])
                    blocks.append(
                        {
                            "page": page_index + 1,
                            "text": clean_word,
                            "bbox": [x, y, x + w, y + h],
                            "confidence": max(0.0, min(1.0, conf / 100)) if conf >= 0 else 0.0,
                        }
                    )
            except Exception as exc:
                warnings.append(f"OCR failed on page {page_index + 1}: {exc}")
    except Exception as exc:
        warnings.append(f"OCR failed: {exc}")

    confidence = mean(confidences) if confidences else 0.0
    return {
        "full_text": "\n".join(text_parts).strip(),
        "blocks": blocks,
        "confidence": round(confidence, 3),
        "warnings": warnings,
    }

