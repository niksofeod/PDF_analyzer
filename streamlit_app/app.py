"""Streamlit interface for the Document AI MVP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline import process_document  # noqa: E402


PDF_TYPE_LABELS = {
    "text": "Текстовый PDF",
    "scanned": "Скан / PDF-изображение",
    "mixed": "Смешанный PDF",
    "unknown": "Не удалось определить",
}

DOCUMENT_TYPE_LABELS = {
    "invoice": "Счет",
    "contract": "Договор",
    "act": "Акт",
    "waybill": "Накладная",
    "statement": "Выписка",
    "unknown": "Неизвестный тип",
}

FIELD_LABELS = {
    "document_number": "Номер документа",
    "document_date": "Дата документа",
    "organization_name": "Организация",
    "inn": "ИНН",
    "kpp": "КПП",
    "address": "Адрес",
    "person_name": "ФИО / ответственное лицо",
}

METHOD_LABELS = {
    "opencv_contours": "Контурный анализ изображения",
    "opencv_contours_hough": "Поиск круглой печати",
}


def label(mapping: dict[str, str], value: str | None) -> str:
    """Return a readable label for a technical value."""

    if not value:
        return "Не указано"
    return mapping.get(value, value)


def confidence_percent(value: float | int | None) -> str:
    """Format confidence as a readable percent value."""

    if value is None:
        return "0%"
    return f"{float(value) * 100:.0f}%"


def field_rows(fields: dict) -> list[dict]:
    """Build a readable table for extracted fields."""

    return [
        {
            "Поле": label(FIELD_LABELS, name),
            "Значение": fields.get(name, {}).get("value") or "Не найдено",
            "Уверенность": confidence_percent(fields.get(name, {}).get("confidence")),
        }
        for name in FIELD_LABELS
    ]


def mark_rows(kind: str, marks: list[dict]) -> list[dict]:
    """Build a readable table for signatures or stamps."""

    kind_label = "Подпись" if kind == "signature" else "Печать"
    return [
        {
            "Тип": mark.get("label") or kind_label,
            "Страница": mark.get("page"),
            "Уверенность": confidence_percent(mark.get("confidence")),
            "Как найдено": mark.get("method_label") or label(METHOD_LABELS, mark.get("method")),
            "Текст рядом": mark.get("near_text") or "Не найден",
            "Координаты": mark.get("bbox"),
        }
        for mark in marks
    ]


st.set_page_config(page_title="PDF Document AI MVP", layout="wide")
st.title("Система автоматического распознавания PDF-документов")
st.caption("Локальный MVP: определение типа PDF, извлечение текста, реквизитов, подписей и печатей.")

uploaded_file = st.file_uploader("Загрузите PDF-документ", type=["pdf"])

if uploaded_file and st.button("Обработать документ", type="primary"):
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = raw_dir / uploaded_file.name
    pdf_path.write_bytes(uploaded_file.getbuffer())

    with st.spinner("Идет обработка PDF..."):
        result = process_document(pdf_path)

    st.subheader("Краткий результат")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Файл", result["file_name"])
    col2.metric("Тип PDF", label(PDF_TYPE_LABELS, result["pdf_type"]))
    col3.metric("Тип документа", label(DOCUMENT_TYPE_LABELS, result["document_type"]["label"]))
    col4.metric("Уверенность", confidence_percent(result["document_type"]["confidence"]))

    st.subheader("Извлеченные поля")
    st.dataframe(pd.DataFrame(field_rows(result.get("fields", {}))), use_container_width=True, hide_index=True)

    signatures = result.get("visual_marks", {}).get("signatures", [])
    stamps = result.get("visual_marks", {}).get("stamps", [])
    st.subheader("Подписи и печати")
    st.caption("Детектор печатей работает консервативно: круг засчитывается только при наличии признаков печати.")
    mark_col1, mark_col2 = st.columns(2)
    mark_col1.metric("Подписи", len(signatures))
    mark_col2.metric("Печати", len(stamps))

    tabs = st.tabs(["Подписи", "Печати"])
    with tabs[0]:
        if signatures:
            st.dataframe(pd.DataFrame(mark_rows("signature", signatures)), use_container_width=True, hide_index=True)
        else:
            st.info("Подписи не обнаружены")
    with tabs[1]:
        if stamps:
            st.dataframe(pd.DataFrame(mark_rows("stamp", stamps)), use_container_width=True, hide_index=True)
        else:
            st.info("Печати не обнаружены")

    if result.get("visualization_paths"):
        st.subheader("Визуализации")
        for relative_path in result["visualization_paths"]:
            image_path = PROJECT_ROOT / relative_path
            if image_path.exists():
                st.image(str(image_path), caption=relative_path, use_container_width=True)

    st.subheader("Извлеченный текст")
    text_path = PROJECT_ROOT / result["text_path"]
    extracted_text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
    st.text_area("Текст документа", extracted_text or "Текст не найден", height=280)

    if result.get("warnings"):
        st.subheader("Предупреждения")
        for warning in result["warnings"]:
            st.warning(warning)

    st.subheader("JSON")
    st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
