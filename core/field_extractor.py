"""Hybrid extraction of key fields from document text."""

from __future__ import annotations

import re
from typing import Dict, Optional


FIELD_NAMES = (
    "document_number",
    "document_date",
    "organization_name",
    "inn",
    "kpp",
    "address",
    "person_name",
)

MONTHS_PATTERN = (
    "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря|"
    "january|february|march|april|may|june|july|august|september|october|november|december"
)


def empty_fields() -> Dict[str, dict]:
    """Return all supported fields with null values."""

    return {name: {"value": None, "confidence": 0.0} for name in FIELD_NAMES}


def _field(value: Optional[str], confidence: float) -> dict:
    value = " ".join(value.split()) if isinstance(value, str) else value
    if isinstance(value, str):
        value = value.strip(" \t\r\n.,;:")
    return {"value": value or None, "confidence": round(max(0.0, min(1.0, confidence)), 3)}


def _flat_text(text: str) -> str:
    """Return text flattened for regexes that need to cross PDF line breaks."""

    return re.sub(r"\s+", " ", text)


def _clean_org(value: str) -> str:
    """Trim common requisites that often follow organization names on one line."""

    value = value.replace("''", '"').replace("“", '"').replace("”", '"')
    value = re.split(r"\b(?:ИНН|КПП|ОГРН|ОГРНИП|адрес|тел\.?|телефон|р/с|к/с|банк)\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    return value.strip(" \t\r\n.,;:-")


def _clean_address(value: str) -> str:
    """Trim banking and phone details that often follow addresses."""

    value = re.split(r"\b(?:р/с|расчетный\s+счет|банк|бик|корр\.?/сч|тел\.?|телефон)\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    return value.strip(" \t\r\n.,;:-")


def _clean_person(value: str) -> str:
    """Normalize common slash-wrapped signature names."""

    return value.strip(" \t\r\n/.,;:-")


def _search(pattern: str, text: str, confidence: float, flags: int = re.IGNORECASE | re.MULTILINE) -> dict:
    match = re.search(pattern, text, flags)
    if not match:
        return _field(None, 0.0)
    return _field(match.group(1), confidence)


def _search_document_number(text: str) -> dict:
    """Extract a document number only when an explicit number marker exists."""

    patterns = (
        r"(?:счет|счёт|договор|акт|накладная|invoice|contract)(?:[^\n№#]{0,90})?(?:№|No\.?|N|#)\s*([A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9._\-\/]{0,40})",
        r"(?:номер\s+документа|document\s+number|номер)\s*[:№#-]+\s*([A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9._\-\/]{0,40})",
    )
    stop_words = {"от", "дата", "на", "оплата", "оплату", "услуг", "работ", "товар"}
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        candidate = match.group(1).strip(" .,:;")
        lowered = candidate.lower()
        has_number_signal = bool(re.search(r"\d", candidate) or re.search(r"[-_/]", candidate))
        if lowered in stop_words or not has_number_signal:
            continue
        return _field(candidate, 0.82)
    return _field(None, 0.0)


def _value_near_keyword(text: str, keywords: tuple[str, ...], confidence: float) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        lowered = line.lower()
        for keyword in keywords:
            if keyword in lowered:
                parts = re.split(r"[:\-]", line, maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    return _field(parts[1].strip(), confidence)
                if index + 1 < len(lines):
                    return _field(lines[index + 1].strip(), max(0.35, confidence - 0.15))
    return _field(None, 0.0)


def _search_date(text: str) -> dict:
    """Extract common numeric and long-form document dates."""

    flat = _flat_text(text)
    patterns = (
        r"(?:дата|дата\s+составления|составлен[ао]?|от|date|dated)\s*[:№#-]?\s*(\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}|\d{4}\s*-\s*\d{2}\s*-\s*\d{2})",
        rf"(?:дата|дата\s+составления|составлен[ао]?|от|date|dated)\s*[:№#-]?\s*((?:«|\"|')?\d{{1,2}}(?:»|\"|')?\s+(?:{MONTHS_PATTERN})\s+\d{{4}}\s*(?:г\.?|года)?)",
        rf"((?:«|\"|')?\d{{1,2}}(?:»|\"|')?\s+(?:{MONTHS_PATTERN})\s+\d{{4}}\s*(?:г\.?|года)?)",
        r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
    )
    for index, pattern in enumerate(patterns):
        result = _search(pattern, flat, 0.82 if index < 2 else 0.58)
        if result["value"]:
            return result
    return _field(None, 0.0)


def _search_organization(text: str) -> dict:
    """Extract a likely organization name from legal-form markers and nearby labels."""

    flat = _flat_text(text)
    patterns = (
        r"(?:исполнитель|поставщик|продавец|организация|контрагент|полное\s+наименование|company|supplier)\s*[:\-]?\s*([^\n;]{3,180})",
        r"\b((?:общество\s+с\s+ограниченной\s+ответственностью|акционерное\s+общество|публичное\s+акционерное\s+общество|закрытое\s+акционерное\s+общество|индивидуальный\s+предприниматель)\s+(?:['\"«]+)?[A-ZА-ЯЁ0-9][^,\n;]{1,160})",
        r"\b((?:ООО|АО|ПАО|ЗАО|НКО|АНО|ФГБУ|ГБУ|МБУ|ИП|LLC|JSC|OAO)\s+(?:['\"«]+)?[A-ZА-ЯЁ0-9][^,\n;]{1,140})",
        r"(?:покупатель|заказчик|клиент|customer)\s*[:\-]?\s*([^\n;]{3,160})",
    )
    for index, pattern in enumerate(patterns):
        result = _search(pattern, text, 0.78 if index < 2 else 0.62)
        if not result["value"]:
            result = _search(pattern, flat, 0.7 if index < 2 else 0.55)
        if result["value"]:
            return _field(_clean_org(result["value"]), result["confidence"])
    return _field(None, 0.0)


def _search_address(text: str) -> dict:
    """Extract postal or legal addresses using address keywords and city/index patterns."""

    flat = _flat_text(text)
    label_pattern = (
        r"(?:юридический\s+адрес|почтовый\s+адрес|фактический\s+адрес|адрес\s+места\s+нахождения|"
        r"место\s+нахождения|местонахождение|адрес|address)\s*[:\-]?\s*([^;\n]{8,220})"
    )
    label_result = _search(label_pattern, text, 0.76)
    if not label_result["value"]:
        label_result = _search(label_pattern, flat, 0.66)
    if label_result["value"]:
        return _field(_clean_address(label_result["value"]), label_result["confidence"])

    keyword_result = _value_near_keyword(
        text,
        ("юридический адрес", "почтовый адрес", "фактический адрес", "адрес", "местонахождение", "место нахождения", "address"),
        0.72,
    )
    if keyword_result["value"]:
        return _field(_clean_address(keyword_result["value"]), keyword_result["confidence"])

    patterns = (
        r"\b(\d{6},?\s*(?:Россия|РФ|Российская Федерация)?\s*[^;\n]{10,180})",
        r"\b((?:г\.|город)\s*[А-ЯЁA-Z][^;\n]{10,180})",
        r"\b((?:ул\.|улица|проспект|пр-т|пер\.|переулок|шоссе|д\.|дом)\s*[^;\n]{10,160})",
    )
    for pattern in patterns:
        result = _search(pattern, text, 0.58)
        if not result["value"]:
            result = _search(pattern, flat, 0.5)
        if result["value"]:
            return _field(_clean_address(result["value"]), result["confidence"])
    return _field(None, 0.0)


def _search_person_name(text: str) -> dict:
    """Extract a likely responsible person's full name."""

    flat = _flat_text(text)
    patterns = (
        r"(?:руководитель|директор|генеральный директор|главный\s+бухгалтер|бухгалтер)[^\n]{0,140}\(\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2}|[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.|[А-ЯЁ]\.\s*[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]+)\s*\)",
        r"(?:директор|генеральный директор|руководитель|главный\s+бухгалтер|бухгалтер|подписал|представитель|ФИО|расшифровка подписи|fio|name)[^А-ЯЁA-Z]{0,80}\(?\s*/?\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2}|[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.|[А-ЯЁ]\.\s*[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]+)",
        r"(?:подпись|signature)[^А-ЯЁA-Z]{0,80}\(?\s*/?\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2}|[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.|[А-ЯЁ]\.\s*[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]+)",
        r"/\s*([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.|[А-ЯЁ]\.\s*[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]+|[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})\s*/",
    )
    bad_fragments = ("подпись", "расшифровка", "должность", "м.п", "печать")
    for index, pattern in enumerate(patterns):
        result = _search(pattern, text, 0.74 if index < 2 else 0.52)
        if not result["value"]:
            result = _search(pattern, flat, 0.62 if index < 2 else 0.45)
        if result["value"]:
            cleaned = _clean_person(result["value"])
            if not any(fragment in cleaned.lower() for fragment in bad_fragments):
                return _field(cleaned, result["confidence"])

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    name_with_initials = re.compile(r"^[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.?$")
    for index, line in enumerate(lines):
        if not name_with_initials.fullmatch(line):
            continue
        window = " ".join(lines[max(0, index - 5) : min(len(lines), index + 6)]).lower()
        if "подпись" in window or "расшифровка" in window or "должность" in window:
            return _field(_clean_person(line), 0.58)
    return _field(None, 0.0)


def extract_fields(text: str, blocks: list[dict] | None = None) -> Dict[str, dict]:
    """Extract document number, dates, tax IDs, address, organization, and person names."""

    del blocks
    fields = empty_fields()
    if not text or not text.strip():
        return fields

    fields["inn"] = _search(r"\b(?:ИНН|INN)\s*[:№#-]?\s*(\d{10}|\d{12})\b", text, 0.95)
    fields["kpp"] = _search(r"\b(?:КПП|KPP)\s*[:№#-]?\s*(\d{9})\b", text, 0.95)
    fields["document_date"] = _search_date(text)
    fields["document_number"] = _search_document_number(text)

    fields["organization_name"] = _search_organization(text)
    fields["address"] = _search_address(text)
    fields["person_name"] = _search_person_name(text)
    return fields
