"""Image preprocessing helpers for OCR."""

from __future__ import annotations

from typing import Any


def _pil_to_numpy(image: Any):
    import numpy as np

    if hasattr(image, "convert"):
        return np.array(image.convert("RGB"))
    return np.asarray(image)


def _numpy_to_pil(image: Any):
    from PIL import Image

    return Image.fromarray(image)


def to_grayscale(image: Any) -> Any:
    """Convert an image to grayscale."""

    try:
        import cv2

        array = _pil_to_numpy(image)
        if len(array.shape) == 2:
            return array
        return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    except Exception:
        try:
            from PIL import ImageOps

            return ImageOps.grayscale(image)
        except Exception:
            return image


def denoise(image: Any) -> Any:
    """Apply light denoising to an OCR image."""

    try:
        import cv2

        gray = to_grayscale(image)
        return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    except Exception:
        return image


def threshold_image(image: Any) -> Any:
    """Binarize an image for OCR."""

    try:
        import cv2

        gray = to_grayscale(image)
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    except Exception:
        try:
            gray = to_grayscale(image)
            return gray.point(lambda pixel: 255 if pixel > 180 else 0)
        except Exception:
            return image


def enhance_contrast(image: Any) -> Any:
    """Increase image contrast before OCR."""

    try:
        import cv2

        gray = to_grayscale(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    except Exception:
        try:
            from PIL import ImageEnhance

            return ImageEnhance.Contrast(image).enhance(1.5)
        except Exception:
            return image


def preprocess_for_ocr(image: Any) -> Any:
    """Run the full OCR preprocessing chain."""

    processed = to_grayscale(image)
    processed = enhance_contrast(processed)
    processed = denoise(processed)
    processed = threshold_image(processed)
    try:
        return _numpy_to_pil(processed)
    except Exception:
        return processed

