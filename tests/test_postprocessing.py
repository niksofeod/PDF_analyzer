from core.postprocessing import normalize_fields


def test_normalize_numeric_date_to_dd_mm_yyyy() -> None:
    fields, warnings = normalize_fields({"document_date": {"value": "27.01.2020", "confidence": 0.82}})
    assert warnings == []
    assert fields["document_date"]["value"] == "27.01.2020"


def test_normalize_iso_date_to_dd_mm_yyyy() -> None:
    fields, warnings = normalize_fields({"document_date": {"value": "2026-02-12", "confidence": 0.82}})
    assert warnings == []
    assert fields["document_date"]["value"] == "12.02.2026"


def test_normalize_long_date_to_dd_mm_yyyy() -> None:
    fields, warnings = normalize_fields({"document_date": {"value": "«05» сентября 2023 г", "confidence": 0.82}})
    assert warnings == []
    assert fields["document_date"]["value"] == "05.09.2023"
