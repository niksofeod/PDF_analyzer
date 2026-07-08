from pathlib import Path

from core.pdf_type_detector import detect_pdf_type


def test_invalid_path_returns_unknown(tmp_path: Path) -> None:
    result = detect_pdf_type(tmp_path / "missing.pdf")
    assert result["pdf_type"] == "unknown"
    assert result["warnings"]


def test_problem_pdf_returns_unknown(tmp_path: Path) -> None:
    bad_pdf = tmp_path / "broken.pdf"
    bad_pdf.write_bytes(b"not a real pdf")
    result = detect_pdf_type(bad_pdf)
    assert result["pdf_type"] == "unknown"
    assert result["warnings"]


def test_empty_or_problem_file_does_not_raise(tmp_path: Path) -> None:
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    result = detect_pdf_type(empty_pdf)
    assert result["pdf_type"] in {"unknown", "scanned", "text", "mixed"}
    assert isinstance(result["warnings"], list)

