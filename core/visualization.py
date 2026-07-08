"""Drawing bounding boxes for detected signatures and stamps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.signature_stamp_detector import render_pdf_pages


def _draw_box(draw: Any, mark: dict, label: str, color: str) -> None:
    bbox = mark.get("bbox", [])
    if len(bbox) != 4:
        return
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
    draw.text((x1, max(0, y1 - 18)), label, fill=color)


def draw_visual_marks(
    pdf_path: str | Path,
    visual_marks: dict,
    output_dir: str | Path,
    document_id: str,
) -> tuple[list[str], list[str]]:
    """Save page images with bounding boxes for detected visual marks."""

    marks_by_page: dict[int, list[tuple[str, dict]]] = {}
    for mark in visual_marks.get("signatures", []):
        marks_by_page.setdefault(int(mark.get("page", 1)), []).append(("signature", mark))
    for mark in visual_marks.get("stamps", []):
        marks_by_page.setdefault(int(mark.get("page", 1)), []).append(("stamp", mark))
    try:
        from PIL import ImageDraw
    except Exception as exc:
        return [], [f"Pillow is unavailable, visualizations skipped: {exc}"]

    images, warnings = render_pdf_pages(pdf_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for page_number, image in enumerate(images, start=1):
        image = images[page_number - 1].copy()
        draw = ImageDraw.Draw(image)
        for label, mark in marks_by_page.get(page_number, []):
            color = "red" if label == "signature" else "blue"
            _draw_box(draw, mark, label, color)
        path = output / f"{document_id}_page_{page_number}.png"
        image.save(path)
        paths.append(str(path))

    for page_number in marks_by_page:
        if page_number < 1 or page_number > len(images):
            warnings.append(f"Visualization skipped for missing page {page_number}")
    return paths, warnings
