from pathlib import Path

from core.visualization import draw_visual_marks


def test_visualization_created_without_marks(tmp_path: Path) -> None:
    try:
        import fitz
    except Exception:
        return

    pdf_path = tmp_path / "empty_marks.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Document without signatures or stamps")
    document.save(pdf_path)
    document.close()

    paths, warnings = draw_visual_marks(
        pdf_path,
        {"signatures": [], "stamps": []},
        tmp_path / "visualizations",
        "empty_marks",
    )

    assert warnings == []
    assert len(paths) == 1
    assert Path(paths[0]).exists()

