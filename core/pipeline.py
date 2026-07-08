"""End-to-end PDF processing pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

from core.document_classifier import classify_document
from core.field_extractor import empty_fields, extract_fields
from core.pdf_loader import ensure_project_directories, validate_pdf_path
from core.pdf_text_extractor import extract_text_from_pdf
from core.pdf_type_detector import detect_pdf_type
from core.ocr_engine import ocr_pdf
from core.postprocessing import normalize_fields
from core.signature_stamp_detector import detect_visual_marks
from core.table_extractor import extract_tables
from core.visualization import draw_visual_marks


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_id(file_name: str) -> str:
    stem = Path(file_name).stem or "document"
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "document"
    return f"{stem}_{uuid4().hex[:10]}"


def _relative(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def process_document(pdf_path: str | Path, document_id: str | None = None) -> dict:
    """Run the complete PDF recognition pipeline and return a JSON-serializable result."""

    paths = ensure_project_directories(PROJECT_ROOT)
    path, validation_warnings = validate_pdf_path(pdf_path)
    doc_id = document_id or _safe_id(path.name)
    warnings: list[str] = list(validation_warnings)

    pdf_type_info = detect_pdf_type(path)
    warnings.extend(pdf_type_info.get("warnings", []))
    pdf_type = pdf_type_info.get("pdf_type", "unknown")

    text_info = {"full_text": "", "blocks": [], "warnings": []}
    ocr_info = {"full_text": "", "blocks": [], "confidence": 0.0, "warnings": []}
    if path.exists() and pdf_type in {"text", "mixed"}:
        text_info = extract_text_from_pdf(path)
        warnings.extend(text_info.get("warnings", []))
    if path.exists() and (pdf_type == "scanned" or (pdf_type == "mixed" and not text_info.get("full_text"))):
        ocr_info = ocr_pdf(path)
        warnings.extend(ocr_info.get("warnings", []))

    full_text = "\n\n".join(part for part in [text_info.get("full_text", ""), ocr_info.get("full_text", "")] if part).strip()
    text_blocks = list(text_info.get("blocks", [])) + list(ocr_info.get("blocks", []))
    if not full_text:
        warnings.append("No text was extracted from the document")

    document_type = classify_document(full_text, PROJECT_ROOT / "models" / "document_classifier.pkl")
    fields = extract_fields(full_text, text_blocks) if full_text else empty_fields()
    fields, post_warnings = normalize_fields(fields)
    warnings.extend(post_warnings)

    tables, table_warnings = extract_tables(path, paths["tables"], doc_id, full_text)
    warnings.extend(table_warnings)
    for table in tables:
        if table.get("csv_path"):
            table["csv_path"] = _relative(table["csv_path"])

    visual_marks, mark_warnings = detect_visual_marks(path, text_blocks)
    warnings.extend(mark_warnings)

    visualization_paths, visualization_warnings = draw_visual_marks(path, visual_marks, paths["visualizations"], doc_id)
    warnings.extend(visualization_warnings)

    text_path = paths["text"] / f"{doc_id}.txt"
    text_path.write_text(full_text, encoding="utf-8")
    json_path = paths["json"] / f"{doc_id}.json"

    result = {
        "document_id": doc_id,
        "file_name": path.name,
        "pdf_type": pdf_type,
        "document_type": {
            "label": document_type.get("label", "unknown"),
            "confidence": float(document_type.get("confidence", 0.0)),
        },
        "fields": fields,
        "tables": tables,
        "visual_marks": visual_marks,
        "text_path": _relative(text_path),
        "json_path": _relative(json_path),
        "visualization_paths": [_relative(item) for item in visualization_paths],
        "warnings": list(dict.fromkeys(str(warning) for warning in warnings if warning)),
    }
    _save_json(json_path, result)
    return result

