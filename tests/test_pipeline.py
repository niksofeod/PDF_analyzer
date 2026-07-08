from pathlib import Path

from core.pipeline import process_document


def test_pipeline_returns_required_keys_for_missing_pdf(tmp_path: Path) -> None:
    result = process_document(tmp_path / "missing.pdf", document_id="test_missing_pdf")
    for key in [
        "document_id",
        "file_name",
        "pdf_type",
        "document_type",
        "fields",
        "tables",
        "visual_marks",
        "warnings",
    ]:
        assert key in result


def test_pipeline_does_not_fail_without_visual_marks(tmp_path: Path) -> None:
    result = process_document(tmp_path / "missing.pdf", document_id="test_no_visual_marks")
    assert result["visual_marks"]["signatures"] == []
    assert result["visual_marks"]["stamps"] == []


def test_pipeline_warnings_is_list(tmp_path: Path) -> None:
    result = process_document(tmp_path / "missing.pdf", document_id="test_warnings")
    assert isinstance(result["warnings"], list)

