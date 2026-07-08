from core.signature_stamp_detector import detect_signature_stamp_in_image, detect_visual_marks


def test_detector_returns_expected_keys_for_none_image() -> None:
    result = detect_signature_stamp_in_image(None)
    assert set(result.keys()) == {"signatures", "stamps"}
    assert result["signatures"] == []
    assert result["stamps"] == []


def test_empty_image_does_not_raise() -> None:
    try:
        import numpy as np
    except Exception:
        return
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detect_signature_stamp_in_image(image)
    assert set(result.keys()) == {"signatures", "stamps"}


def test_detected_marks_have_bbox_and_confidence_shape() -> None:
    try:
        import cv2
        import numpy as np
    except Exception:
        return

    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.line(image, (40, 180), (220, 190), (0, 0, 0), 3)
    cv2.line(image, (50, 198), (200, 170), (0, 0, 0), 2)
    result = detect_signature_stamp_in_image(image)
    for mark in result["signatures"] + result["stamps"]:
        assert len(mark["bbox"]) == 4
        assert 0.0 <= mark["confidence"] <= 1.0


def test_plain_black_circle_is_not_stamp_without_context() -> None:
    try:
        import cv2
        import numpy as np
    except Exception:
        return

    image = np.full((260, 260, 3), 255, dtype=np.uint8)
    cv2.circle(image, (130, 130), 55, (0, 0, 0), 3)
    result = detect_signature_stamp_in_image(image)
    assert result["stamps"] == []


def test_plain_black_signature_line_is_not_signature_without_context() -> None:
    try:
        import cv2
        import numpy as np
    except Exception:
        return

    image = np.full((260, 420, 3), 255, dtype=np.uint8)
    cv2.line(image, (70, 195), (320, 205), (0, 0, 0), 3)
    cv2.line(image, (80, 215), (300, 215), (0, 0, 0), 2)
    result = detect_signature_stamp_in_image(image)
    assert result["signatures"] == []


def test_colored_circle_near_stamp_text_can_be_stamp() -> None:
    try:
        import cv2
        import numpy as np
    except Exception:
        return

    image = np.full((320, 360, 3), 255, dtype=np.uint8)
    cv2.circle(image, (190, 210), 55, (35, 90, 210), 5)
    cv2.circle(image, (190, 210), 38, (35, 90, 210), 2)
    text_blocks = [{"page": 1, "text": "М.П.", "bbox": [120, 130, 180, 160]}]
    result = detect_signature_stamp_in_image(image, text_blocks=text_blocks)
    assert result["stamps"]


def test_detect_text_signature_field_in_pdf(tmp_path) -> None:
    try:
        import fitz
    except Exception:
        return

    pdf_path = tmp_path / "text_signature.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((80, 380), "Executor")
    page.insert_text((80, 430), "Smith", color=(0, 0.35, 0.75))
    page.insert_text((85, 445), "signature", fontsize=7)
    document.save(pdf_path)
    document.close()

    result, warnings = detect_visual_marks(pdf_path)
    assert warnings == []
    assert result["signatures"]
    assert result["signatures"][0]["method"] == "pdf_text_signature_field"
