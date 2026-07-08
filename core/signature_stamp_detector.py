"""Heuristic OpenCV detection of signatures and stamps."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re


STAMP_KEYWORDS = ("м.п", "м.п.", "печать", "stamp", "seal")
SIGNATURE_KEYWORDS = ("подпись", "signature", "руководитель", "директор", "бухгалтер")


def _empty_result() -> dict:
    return {"signatures": [], "stamps": []}


def _to_array(image: Any):
    import numpy as np

    if image is None:
        return None
    if hasattr(image, "convert"):
        return np.array(image.convert("RGB"))
    return np.asarray(image)


def _nearest_text(
    page: int,
    bbox: list[float],
    text_blocks: list[dict] | None,
    keywords: tuple[str, ...],
    max_distance: float | None = None,
) -> str | None:
    if not text_blocks:
        return None
    x1, y1, x2, y2 = bbox
    best_text: str | None = None
    best_distance = float("inf")
    for block in text_blocks:
        if block.get("page") != page:
            continue
        text = str(block.get("text", ""))
        lowered = text.lower()
        if not any(keyword in lowered for keyword in keywords):
            continue
        raw_bbox = block.get("bbox", [0, 0, 0, 0])
        if len(raw_bbox) != 4:
            continue
        # PyMuPDF text blocks are usually in PDF points, while rendered pages
        # are pixels. OCR blocks may already be in pixels, so try both scales.
        for scale in (1.0, 150 / 72, 200 / 72):
            bx1, by1, bx2, by2 = [float(value) * scale for value in raw_bbox]
            distance = abs(((bx1 + bx2) / 2) - ((x1 + x2) / 2)) + abs(((by1 + by2) / 2) - ((y1 + y2) / 2))
            if distance < best_distance:
                best_distance = distance
                best_text = text[:80]
    if max_distance is not None and best_distance > max_distance:
        return None
    return best_text


def _stamp_color_score(rgb, bbox: list[float]) -> float:
    """Estimate whether a region contains blue, purple, or red stamp ink."""

    try:
        import cv2
        import numpy as np
    except Exception:
        return 0.0

    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    blue = cv2.inRange(hsv, np.array([85, 35, 35]), np.array([145, 255, 255]))
    purple = cv2.inRange(hsv, np.array([130, 30, 35]), np.array([170, 255, 255]))
    red_1 = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([12, 255, 255]))
    red_2 = cv2.inRange(hsv, np.array([170, 35, 35]), np.array([179, 255, 255]))
    colored = cv2.bitwise_or(cv2.bitwise_or(blue, purple), cv2.bitwise_or(red_1, red_2))
    return float(cv2.countNonZero(colored)) / float(crop.shape[0] * crop.shape[1])


def _ink_color_score(rgb, bbox: list[float]) -> float:
    """Estimate whether a region contains colored handwritten ink."""

    return _stamp_color_score(rgb, bbox)


def _circle_edge_score(gray, x: int, y: int, radius: int) -> float:
    """Estimate how much of a candidate circle perimeter contains ink."""

    try:
        import cv2
        import numpy as np
    except Exception:
        return 0.0

    if radius <= 0:
        return 0.0
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 255, 2)
    edges = cv2.Canny(gray, 60, 180)
    perimeter_pixels = cv2.countNonZero(mask)
    if perimeter_pixels == 0:
        return 0.0
    overlap = cv2.bitwise_and(edges, edges, mask=mask)
    return float(cv2.countNonZero(overlap)) / float(perimeter_pixels)


def _is_text_signature_value(text: str) -> bool:
    """Return True for short name-like text used as a typed signature."""

    cleaned = text.strip()
    if not cleaned or len(cleaned) > 40:
        return False
    if cleaned.lower() in {"подпись", "расшифровка подписи", "м.п.", "м.п", "должность"}:
        return False
    return bool(re.fullmatch(r"[А-ЯЁA-Z][А-ЯЁа-яёA-Za-z.-]+(?:\s+[А-ЯЁA-Z]\.){0,2}", cleaned))


def _detect_text_signature_marks(pdf_path: str | Path, dpi: int = 150) -> tuple[list[dict], list[str]]:
    """Detect typed/filled signature fields from colored PDF text spans."""

    path = Path(pdf_path)
    if not path.exists():
        return [], []
    try:
        import fitz
    except Exception as exc:
        return [], [f"Text signature detection skipped, PyMuPDF unavailable: {exc}"]

    scale = dpi / 72
    signatures: list[dict] = []
    warnings: list[str] = []
    try:
        with fitz.open(path) as document:
            for page_index, page in enumerate(document, start=1):
                page_height = float(page.rect.height)
                spans: list[dict] = []
                signature_labels: list[dict] = []
                for block in page.get_text("dict").get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = str(span.get("text", "")).strip()
                            if not text:
                                continue
                            item = {
                                "text": text,
                                "bbox": [float(value) for value in span.get("bbox", [0, 0, 0, 0])],
                                "color": int(span.get("color", 0)),
                                "font": str(span.get("font", "")),
                            }
                            spans.append(item)
                            if text.lower() in {"подпись", "signature"}:
                                signature_labels.append(item)

                for span in spans:
                    text = span["text"]
                    x1, y1, x2, y2 = span["bbox"]
                    if y1 < page_height * 0.45:
                        continue
                    if not _is_text_signature_value(text):
                        continue
                    is_colored_or_italic = span["color"] != 0 or "italic" in span["font"].lower()
                    if not is_colored_or_italic:
                        continue
                    center_x = (x1 + x2) / 2
                    label = None
                    for candidate in signature_labels:
                        lx1, ly1, lx2, ly2 = candidate["bbox"]
                        label_center_x = (lx1 + lx2) / 2
                        if abs(center_x - label_center_x) <= 60 and 0 <= ly1 - y2 <= 25:
                            label = candidate
                            break
                    if label is None:
                        continue
                    bbox = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
                    signatures.append(
                        {
                            "label": "Подпись",
                            "page": page_index,
                            "bbox": bbox,
                            "confidence": 0.82,
                            "method": "pdf_text_signature_field",
                            "method_label": "Заполненное поле подписи",
                            "near_text": "подпись",
                        }
                    )
    except Exception as exc:
        warnings.append(f"Text signature detection failed: {exc}")
    return signatures, warnings


def _is_signature_candidate(
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    area: float,
    page_width: int,
    page_height: int,
    near_text: str | None,
    color_score: float,
) -> bool:
    """Return True only for conservative signature-like candidates."""

    if y < page_height * 0.42 or y > page_height * 0.9:
        return False
    if w < max(55, page_width * 0.035) or h < max(12, page_height * 0.008):
        return False
    if w > page_width * 0.55 or h > page_height * 0.22:
        return False
    aspect = w / max(h, 1)
    if not 2.0 <= aspect <= 14.0:
        return False
    density = area / max(w * h, 1)
    if density > 0.42:
        return False
    # Without nearby signature text, require colored ink. This deliberately
    # avoids reporting ordinary black text, table borders, or separator lines.
    if near_text is None and color_score < 0.018:
        return False
    return True


def detect_signature_stamp_in_image(image: Any, page: int = 1, text_blocks: list[dict] | None = None) -> dict:
    """Detect likely signatures and stamps in a rendered page image."""

    try:
        import cv2
        import numpy as np
    except Exception:
        return _empty_result()

    array = _to_array(image)
    if array is None or array.size == 0:
        return _empty_result()

    if len(array.shape) == 2:
        rgb = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    else:
        rgb = array

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    result = _empty_result()

    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 120 or w < 25 or h < 8:
            continue
        bbox = [float(x), float(y), float(x + w), float(y + h)]
        near_text = _nearest_text(page, bbox, text_blocks, SIGNATURE_KEYWORDS, max_distance=max(width, height) * 0.22)
        color_score = _ink_color_score(rgb, bbox)
        if _is_signature_candidate(
            x=x,
            y=y,
            w=w,
            h=h,
            area=area,
            page_width=width,
            page_height=height,
            near_text=near_text,
            color_score=color_score,
        ):
            bbox = [float(x), float(y), float(x + w), float(y + h)]
            confidence = min(0.88, 0.42 + min(area / 5000, 0.22) + min(color_score * 4, 0.18))
            if near_text:
                confidence += 0.12
            result["signatures"].append(
                {
                    "label": "Подпись",
                    "page": page,
                    "bbox": bbox,
                    "confidence": round(min(confidence, 0.92), 3),
                    "method": "opencv_contours",
                    "method_label": "Контурный анализ изображения",
                    "near_text": near_text,
                }
            )

    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min(width, height) // 8),
        param1=110,
        param2=42,
        minRadius=max(14, min(width, height) // 45),
        maxRadius=max(18, min(width, height) // 7),
    )
    if circles is not None:
        for circle in np.round(circles[0, :]).astype("int"):
            x, y, radius = circle
            bbox = [float(x - radius), float(y - radius), float(x + radius), float(y + radius)]
            if x - radius < 0 or y - radius < 0 or x + radius > width or y + radius > height:
                continue
            near_text = _nearest_text(page, bbox, text_blocks, STAMP_KEYWORDS, max_distance=max(width, height) * 0.35)
            color_score = _stamp_color_score(rgb, bbox)
            edge_score = _circle_edge_score(gray, int(x), int(y), int(radius))
            lower_half_bonus = 0.08 if y > height * 0.4 else 0.0
            center_y = y
            has_strong_color_stamp = center_y > height * 0.52 and color_score >= 0.08 and edge_score >= 0.055
            has_stamp_text_and_color = near_text is not None and color_score >= 0.045 and edge_score >= 0.055
            if not (has_strong_color_stamp or has_stamp_text_and_color):
                continue
            if edge_score < 0.035:
                continue
            confidence = 0.45 + min(color_score * 4, 0.22) + min(edge_score * 2, 0.18) + lower_half_bonus
            if near_text:
                confidence += 0.12
            result["stamps"].append(
                {
                    "label": "Печать",
                    "page": page,
                    "bbox": bbox,
                    "confidence": round(min(0.92, confidence), 3),
                    "method": "opencv_contours_hough",
                    "method_label": "Поиск круглой печати",
                    "near_text": near_text,
                }
            )

    return result


def render_pdf_pages(pdf_path: str | Path, dpi: int = 150) -> tuple[list[Any], list[str]]:
    """Render all PDF pages to PIL images for visual analysis."""

    path = Path(pdf_path)
    if not path.exists():
        return [], [f"PDF file was not found: {path}"]
    try:
        import fitz
        from PIL import Image
    except Exception as exc:
        return [], [f"PDF rendering dependencies are unavailable: {exc}"]

    images: list[Any] = []
    warnings: list[str] = []
    try:
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        with fitz.open(path) as document:
            for page in document:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                images.append(Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples))
    except Exception as exc:
        warnings.append(f"PDF page rendering failed: {exc}")
    return images, warnings


def detect_visual_marks(pdf_path: str | Path, text_blocks: list[dict] | None = None) -> tuple[dict, list[str]]:
    """Detect signatures and stamps across all pages of a PDF."""

    images, warnings = render_pdf_pages(pdf_path)
    result = _empty_result()
    for page_index, image in enumerate(images, start=1):
        page_result = detect_signature_stamp_in_image(image, page=page_index, text_blocks=text_blocks)
        result["signatures"].extend(page_result["signatures"])
        result["stamps"].extend(page_result["stamps"])
    text_signatures, text_warnings = _detect_text_signature_marks(pdf_path)
    warnings.extend(text_warnings)
    for text_signature in text_signatures:
        sx1, sy1, sx2, sy2 = text_signature["bbox"]
        duplicate = False
        for existing in result["signatures"]:
            ex1, ey1, ex2, ey2 = existing.get("bbox", [0, 0, 0, 0])
            if existing.get("page") == text_signature["page"] and abs(((sx1 + sx2) / 2) - ((ex1 + ex2) / 2)) < 35 and abs(((sy1 + sy2) / 2) - ((ey1 + ey2) / 2)) < 35:
                duplicate = True
                break
        if not duplicate:
            result["signatures"].append(text_signature)
    return result, warnings
